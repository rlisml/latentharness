"""Teacher cache read/write + integrity checking (E1-A/E1-C prerequisite).

The teacher cache stores, for every training example, the frozen-model outputs
``M_theta(H_behavior, x)`` needed for later latent-harness distillation (RESEARCH_PLAN
§1.2/§2.2). One JSONL line per (example, cache key); ALL outputs are cached — correct and
incorrect alike — with the correctness flag recorded. Whether distillation filters to
correct-only traces is a TRAINING-TIME choice made by `harness/build_distill_data.py`
(both modes must be supported there; default correct-only for generative imitation).
Nothing is filtered at cache time, and no `filtered_in`/`compliant` flag is stored:
filtering depends on the loss arm and is recomputed from the cached output text, so
storing it here would freeze a pre-registration decision into the data.

Cache key: (dataset, harness_sha256, model, split, decoding settings). Any change of
harness wording or decoding produces a NEW filename — never an overwrite (registry rule 2).

Per-line metadata is sufficient for reproducibility (task spec): example id / dataset /
split, model + tokenizer, harness id AND sha256 (re-verified against the registry by
`inspect`), teacher prediction/answer/gold label/correctness flag, answer position
metadata (match method + char/token span), decoding settings, dataset revision, git commit.

Integrity commands (no GPU / no model load):

    python harness/teacher_cache.py inspect  --cache-file <jsonl> [--num-examples N]
    python harness/teacher_cache.py validate --cache-file <jsonl>   # loads the dataset;
        # checks exact id alignment, base_prompt byte-identity with build_prompt, gold
        # labels, and recomputes the correctness flag with the loader's own evaluate().

`answer-position` extraction mirrors the per-dataset `evaluate` regexes of
`utils/dataloaders.py` (read-only reuse; the loaders are never modified). Equivalence
with `loader.evaluate` is enforced by tests/test_teacher_cache.py against the real E1-B
raw corpus, so the mirror cannot silently drift.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.append("./")

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_SCHEMA_VERSION = 1

# Per-dataset repo ids (for dataset-revision provenance; loaders keep loading their own
# way — this mapping is only used to record WHERE the data came from).
DATASET_HF_REPOS = {
    "BoolQ": "google/boolq",
    "ARC": "allenai/ai2_arc",
    "GSM8K": "openai/gsm8k",
    "SVAMP": "ChilleD/SVAMP",
    "MBPP": "mbpp",  # v3: google-research-datasets/mbpp, config `full`
}

# Pre-registered E1-B max_new_tokens (all arms alike; PROJECT_STATE key decisions).
# v3 adds MBPP = 512 (RESEARCH_PLAN §9.4).
E1B_MAX_NEW_TOKENS = {"GSM8K": 512, "ARC": 256, "BoolQ": 96, "MBPP": 512}

# Answer-position match methods (mirror of the loaders' evaluate() paths).
METHOD_ANSWER_LINE = "answer_line"
METHOD_FALLBACK = "fallback"
METHOD_NONE = "none"

REQUIRED_RECORD_FIELDS = [
    "schema_version", "example_id", "example_uid", "dataset", "split", "dataset_revision",
    "model", "tokenizer", "harness_id", "harness_sha256", "teacher_prompt", "base_prompt",
    "prompt_token_len", "decoding", "teacher_output", "completion_token_len",
    "teacher_answer", "teacher_prediction", "answer_match_method", "answer_char_start",
    "answer_char_end", "answer_token_index", "answer_extracted", "gold_label",
    "gold_completion", "teacher_correct", "option_scores", "git_commit", "generated_at",
    "cache_meta",
]

# ---------------------------------------------------------------------------
# Answer extraction with position metadata (mirror of utils/dataloaders.py evaluate)
# ---------------------------------------------------------------------------

_GSM8K_NUM_PAT = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def extract_answer_boolq(output: str) -> Dict[str, Any]:
    """Mirror of BoolQ.evaluate: strict `Answer: true/false` line, then first standalone
    true/false. Case-insensitive on the ORIGINAL string (evaluate lowercases; for ASCII
    true/false IGNORECASE matching is decision-equivalent and preserves offsets).
    `teacher_answer` is the raw matched span; `teacher_prediction` is normalized."""
    m = re.search(r"answer\s*:\s*(true|false)", output, re.IGNORECASE)
    if m:
        return _hit(m.group(1), m.group(1).lower(), METHOD_ANSWER_LINE, m.start(1), m.end(1))
    m2 = re.search(r"\b(true|false)\b", output, re.IGNORECASE)
    if m2:
        return _hit(m2.group(1), m2.group(1).lower(), METHOD_FALLBACK, m2.start(1), m2.end(1))
    return _miss()


def extract_answer_arc(output: str) -> Dict[str, Any]:
    """Mirror of ARC_Challenge.evaluate: `Answer: (C)` style, else a single standalone
    letter A-E / 1-5. Uppercased matching via IGNORECASE (offset-preserving)."""
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-E1-5])\)?", output, re.IGNORECASE)
    if m:
        return _hit(m.group(1), m.group(1).upper(), METHOD_ANSWER_LINE, m.start(1), m.end(1))
    letters = re.findall(r"\b([A-E1-5])\b", output, re.IGNORECASE)
    if len(letters) == 1:
        # finditer for offsets (findall has none)
        m2 = re.search(r"\b([A-E1-5])\b", output, re.IGNORECASE)
        return _hit(m2.group(1), m2.group(1).upper(), METHOD_FALLBACK, m2.start(1), m2.end(1))
    return _miss()


def extract_answer_gsm8k(output: str) -> Dict[str, Any]:
    """Mirror of GSM8K.evaluate: number inside the LAST `Answer:` line, else the last
    number anywhere. Offsets are reported on the original (unstripped) string."""
    if not output:
        return _miss()
    pad = len(output) - len(output.lstrip())  # leading-whitespace offset (strip() on
    s = output.strip()                        # both ends would shift every offset)
    matches = list(re.finditer(r"(?i)answer\s*:\s*(.+)", s))
    if matches:
        ans_raw = matches[-1].group(1)
        base = pad + matches[-1].start(1)
        mnum = _GSM8K_NUM_PAT.search(ans_raw)
        if mnum:
            try:
                pred = float(mnum.group(0).replace(",", ""))
                return {
                    "teacher_answer": mnum.group(0),
                    "teacher_prediction": pred,
                    "answer_match_method": METHOD_ANSWER_LINE,
                    "answer_char_start": base + mnum.start(),
                    "answer_char_end": base + mnum.end(),
                    "answer_extracted": True,
                }
            except ValueError:
                pass
    nums = list(_GSM8K_NUM_PAT.finditer(s))
    if nums:
        try:
            pred = float(nums[-1].group(0).replace(",", ""))
            return {
                "teacher_answer": nums[-1].group(0),
                "teacher_prediction": pred,
                "answer_match_method": METHOD_FALLBACK,
                "answer_char_start": pad + nums[-1].start(),
                "answer_char_end": pad + nums[-1].end(),
                "answer_extracted": True,
            }
        except ValueError:
            return _miss()
    return _miss()


def extract_answer_mbpp(output: str) -> Dict[str, Any]:
    """Mirror of MBPP.evaluate's extraction stage (v3): the [BEGIN]...[DONE] body or
    the first fenced code block. The loader's evaluate then runs the code in the
    sandbox, so the 'answer' here is the extracted code span itself; correctness is
    NOT decidable from strings and is always recomputed via the sandboxed
    `loader.evaluate` in `validate`. The extraction logic is delegated to the loader
    module's `extract_mbpp_code` (single source of truth; test-enforced)."""
    from utils.dataloaders_mbpp import (METHOD_BEGIN_DONE, METHOD_BEGIN_OPEN,
                                        METHOD_FENCED, extract_mbpp_code)
    code, method = extract_mbpp_code(output)
    if code is None:
        return _miss()
    start = output.find(code) if code else None
    end = start + len(code) if start is not None and start >= 0 else None
    return {"teacher_answer": code, "teacher_prediction": code,
            "answer_match_method": method,  # code_begin_done | code_begin_open | code_fenced
            "answer_char_start": start, "answer_char_end": end, "answer_extracted": True}


_EXTRACTORS = {"BoolQ": extract_answer_boolq, "ARC": extract_answer_arc,
               "GSM8K": extract_answer_gsm8k, "MBPP": extract_answer_mbpp}


def extract_answer_with_position(dataset_name: str, output: str) -> Dict[str, Any]:
    """Answer-position metadata for one teacher output. Decisions mirror the loader's
    evaluate() (test-enforced); raises for datasets without a mirror."""
    if dataset_name not in _EXTRACTORS:
        raise ValueError(
            f"No answer-position extractor registered for dataset {dataset_name!r}; "
            f"available: {sorted(_EXTRACTORS)}. Add one mirroring the loader's evaluate()."
        )
    return _EXTRACTORS[dataset_name](output)


def _hit(raw_answer: str, prediction: Any, method: str, start: int, end: int) -> Dict[str, Any]:
    return {"teacher_answer": raw_answer, "teacher_prediction": prediction,
            "answer_match_method": method,
            "answer_char_start": start, "answer_char_end": end, "answer_extracted": True}


def _miss() -> Dict[str, Any]:
    return {"teacher_answer": None, "teacher_prediction": None, "answer_match_method": METHOD_NONE,
            "answer_char_start": None, "answer_char_end": None, "answer_extracted": False}


def answer_token_index(tokenizer, output: str, char_start: Optional[int]) -> Optional[int]:
    """Index (into the generated token sequence, no BOS) of the token containing
    char_start, via fast-tokenizer offset mapping."""
    if char_start is None:
        return None
    try:
        enc = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
        for idx, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= char_start < e or (s == e == char_start):
                return idx
        # char_start at the very end (shouldn't happen for a real match)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Option-level scores (multiple-choice / binary tasks; requirement 4)
# ---------------------------------------------------------------------------

def option_strings(dataset_name: str, raw_example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """Answer-format continuation strings per option, in loader answer format.

    ARC -> 'Answer: (A) <text>.' per choice; BoolQ -> ['Answer: true.', 'Answer: false.'];
    GSM8K (free-form numeric) -> None (no enumerable options — output-level caching is
    sufficient there, per requirement 5)."""
    if dataset_name == "ARC":
        choices = raw_example["choices"]
        return [{"label": str(l),
                 "string": f"Answer: ({l}) {text}."}
                for l, text in zip(choices["label"], choices["text"])]
    if dataset_name == "BoolQ":
        return [{"label": "true", "string": "Answer: true."},
                {"label": "false", "string": "Answer: false."}]
    return None


@torch.no_grad()
def score_options_batch(model, tokenizer, prompts: List[str],
                        options_per_prompt: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    """Batched option scoring for a list of examples. Returns per example:
        {labels, sum_logprobs, mean_logprobs, distribution, argmax_label}."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    seqs = []
    for pi, (prompt, options) in enumerate(zip(prompts, options_per_prompt)):
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        for k, opt in enumerate(options):
            o_ids = tokenizer(opt["string"], add_special_tokens=False)["input_ids"]
            seqs.append((p_ids + o_ids, len(p_ids), len(o_ids), (pi, k)))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]), reverse=True)
    results: List[List[float]] = [[] for _ in prompts]
    sum_lp = [[0.0] * len(opts) for opts in options_per_prompt]
    for b0 in range(0, len(order), 32):
        chunk = order[b0:b0 + 32]
        maxlen = max(len(seqs[i][0]) for i in chunk)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids, _, _, _ = seqs[i]
            input_ids[r, :len(ids)] = torch.tensor(ids)
            attention[r, :len(ids)] = 1
        input_ids, attention = input_ids.to(device), attention.to(device)
        hidden = model.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        pos_b, pos_t, tok = [], [], []
        for r, i in enumerate(chunk):
            ids, plen, olen, (pi, k) = seqs[i]
            for j in range(olen):
                pos_b.append(r)
                pos_t.append(plen + j - 1)
                tok.append(ids[plen + j])
        sel = hidden[pos_b, pos_t]
        logits = model.lm_head(sel)
        lp = torch.log_softmax(logits.float(), dim=-1)
        gathered = lp[torch.arange(len(tok)), torch.tensor(tok, device=device)].tolist()
        cursor = 0
        for r, i in enumerate(chunk):
            ids, plen, olen, (pi, k) = seqs[i]
            s = sum(gathered[cursor:cursor + olen])
            sum_lp[pi][k] = float(s)
            cursor += olen
    out = []
    import math
    for pi, options in enumerate(options_per_prompt):
        lps = sum_lp[pi]
        mx = max(lps)
        exps = [math.exp(v - mx) for v in lps]
        z = sum(exps)
        dist = [e / z for e in exps]
        argmax = dist.index(max(dist))
        out.append({
            "labels": [o["label"] for o in options],
            "sum_logprobs": lps,
            "mean_logprobs": [lps[k] / max(1, len(tokenizer(options[k]["string"],
                                                 add_special_tokens=False)["input_ids"]))
                              for k in range(len(options))],
            "distribution": dist,
            "argmax_label": options[argmax]["label"],
        })
    return out


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def dataset_revision(loader, dataset_name: str) -> Dict[str, Any]:
    """Dataset provenance: HF repo id, config, version, and the revision hash of the
    local snapshot (offline-safe). `revision` = hub sha when resolvable (best-effort,
    network-guarded) else the local cache snapshot hash, else 'unknown'."""
    prov: Dict[str, Any] = {"hf_repo": DATASET_HF_REPOS.get(dataset_name, "unknown"),
                            "hf_config": None, "hf_version": None,
                            "local_snapshot_hash": None, "hub_sha": None,
                            "revision": "unknown"}
    try:
        split = "validation" if "validation" in loader.raw_dict else next(
            iter(loader.raw_dict))
        cache_files = loader.raw_dict[split].cache_files
        if cache_files:
            parts = os.path.normpath(cache_files[0]["filename"]).split(os.sep)
            # .../datasets/<ns>___<repo>/<config>/<version>/<hash>/...
            idx = next(i for i, p in enumerate(parts) if "___" in p)
            ns_repo = parts[idx].split("___")
            prov["hf_repo"] = f"{ns_repo[0]}/{ns_repo[1]}"
            prov["hf_config"] = parts[idx + 1]
            prov["hf_version"] = parts[idx + 2]
            prov["local_snapshot_hash"] = parts[idx + 3]
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda: HfApi().dataset_info(prov["hf_repo"]).sha)
            prov["hub_sha"] = fut.result(timeout=10)
    except Exception:
        prov["hub_sha"] = None
    prov["revision"] = prov["hub_sha"] or prov["local_snapshot_hash"] or "unknown"
    return prov


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def model_slug(model_name: str) -> str:
    return model_name.replace("/", "--")


def cache_filename(dataset_name: str, harness_id: str, split: str, model_name: str,
                   max_new_tokens: int, limit: Optional[int] = None) -> str:
    """Deterministic filename encoding the cache key (dataset, harness id, split, model,
    decoding). Hash changes with harness wording change the harness id/hash — the id is
    versioned (e.g. VERIFY-001), so new wordings land in new files, never overwrites."""
    name = f"{dataset_name}__{harness_id}__{split}__{model_slug(model_name)}__nmt{max_new_tokens}"
    if limit is not None:
        name += f"__limit{limit}"
    return name + ".jsonl"


def make_record(*, schema_version: int, example_id: int, dataset: str, split: str,
                dataset_revision: Dict[str, Any], model: str, tokenizer_name: str,
                tokenizer_class: str, harness_id: str, harness_sha256: str,
                teacher_prompt: str, base_prompt: str, prompt_token_len: int,
                decoding: Dict[str, Any], teacher_output: str, completion_token_len: int,
                extraction: Dict[str, Any], answer_token_index: Optional[int],
                gold_label: Any, gold_completion: str, teacher_correct: bool,
                option_scores: Optional[Dict[str, Any]], git_commit: str,
                generated_at: str, cache_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One cache line. Field order mirrors REQUIRED_RECORD_FIELDS (readability only)."""
    rec = {
        "schema_version": schema_version,
        "example_id": example_id,
        "example_uid": f"{dataset}:{split}:{example_id}",
        "dataset": dataset,
        "split": split,
        "dataset_revision": dataset_revision,
        "model": model,
        "tokenizer": {"name": tokenizer_name, "class": tokenizer_class},
        "harness_id": harness_id,
        "harness_sha256": harness_sha256,
        "teacher_prompt": teacher_prompt,
        "base_prompt": base_prompt,
        "prompt_token_len": prompt_token_len,
        "decoding": decoding,
        "teacher_output": teacher_output,
        "completion_token_len": completion_token_len,
        "teacher_answer": extraction["teacher_answer"],
        "teacher_prediction": extraction["teacher_prediction"],
        "answer_match_method": extraction["answer_match_method"],
        "answer_char_start": extraction["answer_char_start"],
        "answer_char_end": extraction["answer_char_end"],
        "answer_extracted": extraction["answer_extracted"],
        "answer_token_index": answer_token_index,
        "gold_label": gold_label,
        "gold_completion": gold_completion,
        "teacher_correct": bool(teacher_correct),
        "option_scores": option_scores,
        "git_commit": git_commit,
        "generated_at": generated_at,
        "cache_meta": cache_meta if cache_meta is not None else {},
    }
    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in rec]
    if missing:
        raise ValueError(f"Cache record missing fields: {missing}")
    return rec


def read_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def existing_ids(path: str) -> set:
    """Example ids already in the cache (for resumable caching)."""
    return {r["example_id"] for r in read_records(path)}


def append_records(path: str, records: List[Dict[str, Any]]) -> None:
    """Append records durably: open in append mode, flush + fsync per record so a killed
    run leaves a valid prefix (resume scans whatever landed)."""
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_internal_consistency(records: List[Dict[str, Any]],
                               registry: Optional[Dict[str, Any]] = None) -> List[str]:
    """Offline checks: schema fields present, unique ids, single cache key, harness hash
    matches the registry text (when the registry is supplied). Returns problem list."""
    problems = []
    if not records:
        return ["cache file is empty"]
    ids = [r["example_id"] for r in records]
    if len(set(ids)) != len(ids):
        problems.append("duplicate example_ids in cache")
    keys = {(r["dataset"], r["split"], r["model"], r["harness_id"], r["harness_sha256"],
             json.dumps(r["decoding"], sort_keys=True)) for r in records}
    if len(keys) > 1:
        problems.append(f"cache mixes {len(keys)} distinct cache keys: {sorted(map(str, keys))}")
    first = records[0]
    missing = [f for f in REQUIRED_RECORD_FIELDS if any(f not in r for r in records)]
    if missing:
        problems.append(f"records missing required fields: {missing}")
    if registry is not None and first["harness_id"] in registry:
        h = registry[first["harness_id"]]
        if h.sha256 != first["harness_sha256"]:
            problems.append(
                f"harness_sha256 {first['harness_sha256']} does not match registry hash "
                f"{h.sha256} for {first['harness_id']}")
        else:
            import hashlib as _hl
            if _hl.sha256(h.text.encode("utf-8")).hexdigest() != first["harness_sha256"]:
                problems.append("registry text does not hash to the cached harness_sha256")
    schema_versions = {r.get("schema_version") for r in records}
    if schema_versions != {CACHE_SCHEMA_VERSION}:
        problems.append(f"unexpected schema versions: {schema_versions}")
    return problems


def validate_against_dataset(cache_path: str, dataset_name: Optional[str], split: Optional[str],
                             loader=None, recheck_eval: bool = True) -> Dict[str, Any]:
    """Exact alignment between dataset ids and cached ids (requirement 7), plus:
    gold labels match the loader, base_prompt is byte-identical to build_prompt(ex)
    (student-prompt identity), and teacher_correct reproduces loader.evaluate(output, gold).
    `limit` recorded in the records' split must map onto the loader-order prefix."""
    from utils import dataset_dict

    records = read_records(cache_path)
    report: Dict[str, Any] = {"cache_path": cache_path, "n_records": len(records), "problems": []}
    if not records:
        report["problems"].append("cache file is empty")
        return report
    ds_name = dataset_name or records[0]["dataset"]
    split_name = split or records[0]["split"]
    report["dataset"], report["split"] = ds_name, split_name

    if loader is None:
        loader = dataset_dict[ds_name](mode="eval")
    raw = list(loader.raw_dict[split_name])
    triples = loader.load_raw_dataset(return_test=True)[split_name]
    if len(triples) != len(raw):
        report["problems"].append(
            f"load_raw_dataset returned {len(triples)} examples but raw_dict['{split_name}'] "
            f"has {len(raw)}; cannot validate alignment")
        return report
    limit = None
    if records[0].get("cache_meta", {}).get("limit"):
        limit = records[0]["cache_meta"]["limit"]

    expected_ids = list(range(len(raw) if limit is None else min(limit, len(raw))))
    cached_ids = [r["example_id"] for r in records]
    dupes = sorted({i for i in cached_ids if cached_ids.count(i) > 1})
    missing = sorted(set(expected_ids) - set(cached_ids))
    extra = sorted(set(cached_ids) - set(expected_ids))
    if dupes:
        report["problems"].append(f"duplicate cached ids: {dupes[:10]}")
    if missing:
        report["problems"].append(f"missing cached ids (dataset examples without a cache "
                                  f"line): {len(missing)}, e.g. {missing[:10]}")
    if extra:
        report["problems"].append(f"cached ids outside the expected split prefix: {extra[:10]}")

    eval_fn = loader.evaluate
    n_prompt_mismatch, n_label_mismatch, n_correct_mismatch = 0, 0, 0
    for rec in records:
        i = rec["example_id"]
        if not (0 <= i < len(triples)):
            continue
        prompt, _, gold = triples[i]
        if rec["base_prompt"] != prompt:
            n_prompt_mismatch += 1
        if rec["gold_label"] != gold:
            n_label_mismatch += 1
        if recheck_eval:
            if bool(eval_fn(rec["teacher_output"], gold)) != rec["teacher_correct"]:
                n_correct_mismatch += 1
    if n_prompt_mismatch:
        report["problems"].append(f"{n_prompt_mismatch} records whose base_prompt differs "
                                  "from the eval-pipeline prompt (load_raw_dataset)")
    if n_label_mismatch:
        report["problems"].append(f"{n_label_mismatch} gold-label mismatches vs "
                                  "load_raw_dataset labels")
    if n_correct_mismatch:
        report["problems"].append(f"{n_correct_mismatch} records where teacher_correct "
                                  "disagrees with loader.evaluate(output, gold)")
    report["aligned"] = not report["problems"]
    return report


# ---------------------------------------------------------------------------
# Inspection CLI (no GPU, no model load)
# ---------------------------------------------------------------------------

def summarize_cache(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    n_correct = sum(1 for r in records if r["teacher_correct"])
    n_extracted = sum(1 for r in records if r["answer_extracted"])
    methods: Dict[str, int] = {}
    for r in records:
        methods[r["answer_match_method"]] = methods.get(r["answer_match_method"], 0) + 1
    opt_agree = opt_total = 0
    for r in records:
        os_ = r.get("option_scores")
        if os_ and r["answer_extracted"] and os_.get("argmax_label") is not None:
            opt_total += 1
            if str(r["teacher_prediction"]).lower() == str(os_["argmax_label"]).lower():
                opt_agree += 1
    first = records[0]
    return {
        "n_records": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else None,
        "n_extracted": n_extracted,
        "extraction_rate": n_extracted / n if n else None,
        "answer_match_methods": methods,
        "option_argmax_agrees_with_extracted": (f"{opt_agree}/{opt_total}" if opt_total else "n/a"),
        "mean_completion_tokens": (sum(r["completion_token_len"] for r in records) / n) if n else None,
        "cache_key": {k: first[k] for k in ("dataset", "split", "model", "harness_id",
                                            "harness_sha256")},
        "decoding": first["decoding"],
        "dataset_revision": first["dataset_revision"],
        "git_commit": first["git_commit"],
    }


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="Offline summary + internal consistency checks")
    p_ins.add_argument("--cache-file", required=True)
    p_ins.add_argument("--num-examples", type=int, default=2,
                       help="Number of example records to print")
    p_ins.add_argument("--print-records", action="store_true", default=False)

    p_val = sub.add_parser("validate", help="Exact id alignment + eval recheck (loads dataset)")
    p_val.add_argument("--cache-file", required=True)
    p_val.add_argument("--dataset-name", default=None)
    p_val.add_argument("--split", default=None)

    args = parser.parse_args(argv)

    if args.command == "inspect":
        records = read_records(args.cache_file)
        registry = None
        try:
            from harness.harness_text import load_registry
            registry = load_registry()
        except Exception:
            pass
        problems = check_internal_consistency(records, registry)
        summary = summarize_cache(records)
        summary["problems"] = problems
        summary["ok"] = not problems
        _print_json(summary)
        if args.print_records:
            for rec in records[: args.num_examples]:
                _print_json(rec)
        return 0 if not problems else 1

    if args.command == "validate":
        report = validate_against_dataset(args.cache_file, args.dataset_name, args.split)
        _print_json(report)
        return 0 if report.get("aligned") else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
