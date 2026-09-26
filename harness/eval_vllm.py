"""vLLM-backed evaluator — same CLI surface and record conventions as the existing
evaluators, with ONLY the generation backend swapped (HF transformers → vLLM).

INFRASTRUCTURE STAGE (the parity spec): built, NOT adopted. No protocol stage
may use this path; no engines may be mixed within any stage; parity subsets are
diagnostic only. This module changes nothing else in the project: loaders, paired-prompt
construction, leakage assertion, marker layer, the MBPP sandbox, the JSONL schemas, and
the append-only ledger are reused unchanged.

===========================================================================
EVALUATION STACK MAP (read-only; Task-1 survey — no existing file modified)
===========================================================================

Dataloader interface (utils/dataloaders.py, utils/dataloaders_mbpp.py; registry
`utils/utils.py::dataset_dict` — keys: GSM8K, ARC [class ARC_Challenge], BoolQ, MBPP,
SVAMP, ...):
  - `build_prompt(ex)` -> the canonical base prompt H_base (byte-identical to what the
    E0/E1 evaluation pipelines feed the model). BoolQ: passage/question template;
    GSM8K: bare question; ARC: question + (A)-(E) options; MBPP
    (`utils/dataloaders_mbpp.py`): fixed 3-shot `[BEGIN]`/`[DONE]` scaffold ending at
    `[BEGIN]`, all of the problem's test_list asserts shown. Splits are carved with
    fixed seed 42 in each loader `__init__` (MBPP splits are the pre-registered
    training/dev pools + official test).
  - `load_raw_dataset(return_test=True)[split]` -> list of `[prompt, completion, label]`
    triples (prompt == build_prompt output; label feeds `evaluate`).
  - `raw_dict[split]` -> raw examples (this module cross-checks
    `build_prompt(raw_dict[split][i]) == paired[i][0]` per example, the same
    byte-identity guard `harness/prompts.py::build_paired_examples` and
    `harness/eval_arms.py` use).
  - `evaluate(output, label) -> bool` — per-dataset answer extraction/scoring, NEVER
    modified (RESEARCH_PLAN rule 6). BoolQ: `Answer: true/false` regex -> first
    standalone true/false. GSM8K: last `Answer:` number -> last-number fallback.
    ARC: option-letter matching. MBPP: `extract_mbpp_code` + the sandboxed pass@1
    execution below.
  - `get_answer_string(ex)` -> gold completion (training only; not used at eval).

Paired-prompt construction + leakage assertion (harness/prompts.py, reused here):
  - `build_teacher_prompt(H, H_base) = H + "\\n\\n" + H_base` (RESEARCH_PLAN §2.1);
    `H=None` returns H_base unchanged ('base' condition).
  - `PairedExample` separates student (`base_prompt`) vs teacher (`teacher_prompt`)
    inputs; `assert_no_leakage` = harness text ABSENT from the student prompt AND
    `teacher_prompt.endswith(base_prompt)` (task-content identity).
  - Batch-level gates reused verbatim from `harness/eval_arms.py`:
    `assert_no_harness_leak` (student-side; raises on leak) and
    `assert_harness_present` (explicit-arm positive control).

Marker layer (harness/markers.py, reused unchanged): `compute_example_markers(...)`
(verify-action, decomposition steps, restatement, revision event, correction cues,
answer line, arithmetic expressions, option coverage, evidence quote), `summarize_markers`
(the pre-registered rates), MBPP thin normalization inside
`question_and_options_from_prompt` (restatement compares against the task text).
Markers here are computed against `base_prompt` (task content; harness text excluded).

MBPP sandboxed evaluate (utils/dataloaders_mbpp.py, reused unchanged):
`extract_mbpp_code` ([BEGIN]...[DONE] body -> [DONE]-prefix -> fenced block -> failure),
`run_mbpp_sandbox` (fresh temp cwd per example; `unshare -rn` user+network namespace +
in-process socket guard = no network; 10 s hard wall-clock timeout with process-group
SIGKILL; RLIMIT_CPU/AS/FSIZE backstops; stdin=DEVNULL), `evaluate_detailed`
(structured result incl. extraction method and sandbox settings), `sandbox_settings()`
(recorded in every MBPP ledger line). This module calls `evaluate_detailed` for MBPP
and `evaluate` for the other datasets — the loader functions themselves are untouched.

Per-example raw JSONL schema (harness/run_pilot.py / harness/eval_arms.py convention,
extended additively here): example_id (loader index), dataset, split, seed, arm,
arm_kind, harness_id, harness_sha256, model, decoding{...}, prompt, base_prompt,
prompt_token_len, base_prompt_token_len, output, completion_token_len, label, correct,
git_commit, timestamp — PLUS `engine: "vllm"`, `env`, `engine_config{...}`,
`finish_reason`, `generated_tokens`, `markers{...}`, and (MBPP) `sandbox_detail{...}`.
A `.metrics.json` sidecar per run carries accuracy, throughput, VRAM, engine config,
env versions, eos notes, and marker summaries.

Resume-safe caching (the `harness/generate_teacher.py` convention): the output JSONL is
fsync-appended per record; on rerun, example ids already present are skipped (only the
missing ones are generated); a complete file short-circuits the run unless `--force`.

inspect_prompts (harness/inspect_prompts.py): the existing CLI prints paired
base/teacher prompts from `build_paired_examples` (no GPU); used to verify harness text
flows through byte-identically. This module additionally asserts, per batch, that the
harness text is absent from base prompts / present in teacher prompts, and
`harness/parity_report.py` byte-diffs recorded prompts against a fresh
`build_paired_examples` reconstruction.

Ledger (results/ledger.jsonl, append-only): this stage's uniform per-run lines
(engine, env, config, subset, accuracy, agreement stats, throughput, peak VRAM,
wallclock; MBPP sandbox settings) are written by `harness/parity_report.py` so each
parity run appears exactly once; this module writes only its raw JSONL + metrics
sidecar.

Generation call sites in the eval path (survey; none modified):
  1. inference.py:33 `evaluate_model` — steered-checkpoint eval (also train.py
     --do-eval, eval_base.py); base-model arms only are engine-swappable.
  2. harness/error_analysis.py:92 — E1-0 generation.
  3. harness/generate_teacher.py:77 `generate_batch` — teacher caches (E1-C); reused
     by harness/error_analysis_mbpp.py:116 (E1-0-MBPP).
  4. harness/run_pilot.py:86 `generate_batch` — E1-B pilot screening.
  5. harness/eval_arms.py:90 `generate_batch` — E1-Eval unified arm evaluator
     (base/explicit arms swappable; mean_diff/adapter/soft_prompt are hook-based).
  6. harness/soft_prompt.py:146 — SOFT_PROMPT arm (HF-internal prefix mechanism; not
     swappable).
This module adds call site 7: vLLM offline `LLM.generate` over the FULL example batch
(continuous batching; no manual chunking).

===========================================================================
PARITY PROTOCOL (pre-registered in the parity spec BEFORE any
generation; enforced here where the module can):
  - Raw prompts (no chat template), greedy (temperature=0.0), per-dataset v3 caps
    (BoolQ 96 / MBPP 512 / GSM8K 512), seed 42, NO stop strings (eos only).
  - `SamplingParams(temperature=0.0, max_tokens=<cap>)`; `stop=None`.
  - Hard assertion per example: prompt_tokens + max_new_tokens <= max_model_len
    (abort loudly; no silent truncation).
  - Acceptance: |acc_vllm - acc_hf| <= 2.0 points per 100-example dataset
    exact-match agreement reported (target >= 90%,
    itemized, not gated); MBPP extraction-failure rate difference <= 2 points.
  - A FAIL is a code bug (tokenizer/eos/dtype/config): fix and rerun; never loosen
    the tolerance.

Usage (same flag names as the existing evaluators):
    CUDA_VISIBLE_DEVICES=0 python harness/eval_vllm.py \
        --model-name meta-llama/Llama-3.2-1B-Instruct --dataset-name BoolQ \
        --split test --limit 100 --harness-name base --max-new-tokens 96 \
        --seed 42 --out-dir results/eval_vllm_parity
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.append("./")

from transformers import AutoTokenizer

from harness.eval_arms import assert_harness_present, assert_no_harness_leak
from harness.harness_text import load_registry, resolve_harness
from harness.markers import compute_example_markers, summarize_markers
from harness.prompts import build_teacher_prompt
from harness.teacher_cache import E1B_MAX_NEW_TOKENS, git_commit
from utils import dataset_dict

ENGINE = "vllm"
DEFAULT_ENV_NAME = "latentharness"


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _repetition_note(text: str) -> Dict[str, Any]:
    """Diagnostic-only degeneracy note: largest run of identical non-empty lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best = cur = 0
    prev = None
    for ln in lines:
        cur = cur + 1 if ln == prev else 1
        prev = ln
        best = max(best, cur)
    return {"max_identical_line_run": best}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Key of dataset_dict (ARC_Challenge is registered as 'ARC')")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="test")
    parser.add_argument("--limit", type=int, default=None,
                        help="First N examples of the split in loader order (default: all)")
    parser.add_argument("--harness-name", type=str, default="base",
                        help="'base' (no harness) or a registry id (teacher prompt = "
                             "harness text prepended via build_teacher_prompt)")
    parser.add_argument("--harness-text-file", type=str, default=None,
                        help="E3-COMPOSITION (additive): path to a file holding the "
                             "EXACT teacher harness text (UTF-8, no trailing newline) "
                             "for composed prompts that are NOT registry entries "
                             "(e.g. the two-harness EXPLICIT_a_plus_b text built from "
                             "two frozen registry texts). Overrides --harness-name; "
                             "requires --harness-tag. The file's sha256 is recorded.")
    parser.add_argument("--harness-tag", type=str, default=None,
                        help="E3-COMPOSITION (additive): filename/record tag to use "
                             "with --harness-text-file (e.g. 'DECOMPOSE-001+COMPUTE-001')")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Default: the frozen v3 cap for the dataset "
                             f"({E1B_MAX_NEW_TOKENS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/eval_vllm_parity")
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME,
                        help="Recorded in every record/ledger line (bookkeeping only)")
    parser.add_argument("--paper-split", action="store_true",
                        help="E0-v3-ENGINES: evaluate the PAPER's official split "
                             "(BoolQ_Paper/GSM8K_Paper/ARC_Challenge_Paper from "
                             "eval_official.py / eval_arc_official.py) instead of the "
                             "repo loader split — the split source the E0-v3 B0/B1 "
                             "headlines used. Loaders/templates/evaluate unchanged.")
    # ----- engine config (pre-registered defaults; the parity spec) -----
    parser.add_argument("--dtype", type=str, default="auto",
                        help="vLLM dtype (default 'auto' = model-config default)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Hard prompt+cap bound; any violation aborts (no silent "
                             "truncation)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--inject-phases", type=str, choices=["both", "generation"],
                        default="both",
                        help="CLI-parity flag with harness/eval_easysteer.py. THIS engine "
                             "is plain vLLM and applies NO steer vector, so the value is "
                             "recorded in engine_config/records only and has no effect on "
                             "generation (no-op). Use harness/eval_easysteer.py for a "
                             "functional phase switch.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs (default: off, the deployment path)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the output file is complete")
    args = parser.parse_args(argv)
    return args


def fsync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    registry = load_registry()
    if args.harness_text_file:
        if not args.harness_tag:
            raise SystemExit("--harness-text-file requires --harness-tag")
        with open(args.harness_text_file, "rb") as fh:
            _htf_bytes = fh.read()
        harness_text = _htf_bytes.decode("utf-8")
        if harness_text.endswith("\n"):
            raise SystemExit("--harness-text-file must hold the exact text with "
                             "no trailing newline (registry convention)")
        composed_sha256 = hashlib.sha256(_htf_bytes).hexdigest()
        harness = None
    else:
        harness = resolve_harness(args.harness_name, registry) \
            if args.harness_name else None
        harness_text = harness.text if harness else None
        composed_sha256 = None

    max_new_tokens = args.max_new_tokens or E1B_MAX_NEW_TOKENS.get(
        args.dataset_name, E1B_MAX_NEW_TOKENS["GSM8K"])

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cap_tag = f"nmt{max_new_tokens}"
    limit_tag = f"__limit{args.limit}" if args.limit is not None else ""
    paper_tag = "__paper" if args.paper_split else ""
    harness_tag = harness.harness_id if harness else (
        args.harness_tag if args.harness_text_file else "base")
    out_path = os.path.join(
        out_dir,
        f"{args.dataset_name}__{harness_tag}__{args.split}__{args.model_name.replace('/', '--')}"
        f"__{cap_tag}{limit_tag}{paper_tag}.jsonl")
    metrics_path = out_path + ".metrics.json"

    # ----- data: identical selection + byte-identity checks as eval_arms -----
    if args.paper_split:
        from eval_official import BoolQ_Paper, GSM8K_Paper
        from eval_arc_official import ARC_Challenge_Paper
        paper_loaders = {"BoolQ": BoolQ_Paper, "GSM8K": GSM8K_Paper,
                         "ARC": ARC_Challenge_Paper}
        loader = paper_loaders[args.dataset_name](mode="eval")
    else:
        loader = dataset_dict[args.dataset_name](mode="eval")
    paired = loader.load_raw_dataset(return_test=True)[args.split]
    raw_examples = list(loader.raw_dict[args.split])
    n_available = len(paired)
    n = min(args.limit, n_available) if args.limit is not None else n_available
    indices = list(range(n))

    items: List[Dict[str, Any]] = []
    for i in indices:
        base_prompt = loader.build_prompt(raw_examples[i])
        if base_prompt != paired[i][0]:
            raise ValueError(
                f"build_prompt mismatch at example {i}: the base prompt must stay "
                "byte-identical to the eval-pipeline prompt")
        full_prompt = build_teacher_prompt(harness_text, base_prompt)
        items.append({"example_id": i, "base_prompt": base_prompt,
                      "full_prompt": full_prompt, "label": paired[i][2]})

    # ----- tokenizer (bookkeeping + eos notes; vLLM tokenizes internally) -----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    eos_notes: Dict[str, Any] = {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
    }
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(args.model_name)
        eos_notes["generation_config_eos_token_id"] = gc.eos_token_id
    except Exception as exc:  # record-keeping only
        eos_notes["generation_config_eos_token_id"] = f"unavailable: {exc}"

    # ----- length gate: abort loudly instead of silent truncation -----
    prompt_lens = {it["example_id"]: token_len(tokenizer, it["full_prompt"])
                   for it in items}
    over = {i: l for i, l in prompt_lens.items()
            if l + max_new_tokens > args.max_model_len}
    if over:
        raise SystemExit(
            f"max_model_len {args.max_model_len} too small for "
            f"{len(over)} example(s): prompt+cap needed up to "
            f"{max(over.values()) + max_new_tokens} (example "
            f"{max(over, key=lambda k: over[k])}); raise --max-model-len (pre-registered "
            "value: 2048 for the parity model) — never truncate silently")

    # ----- resume: skip already-present example ids (generate_teacher convention) -----
    if args.force and os.path.exists(out_path):
        os.remove(out_path)  # this module's own stage artifact; regenerate from scratch
        done = {}
    else:
        done: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        done[int(rec["example_id"])] = rec
    todo = [it for it in items if it["example_id"] not in done]
    if os.path.exists(out_path) and not todo and not args.force:
        print(f"[eval_vllm] output complete with {len(done)} records; skipping "
              f"(--force to re-run): {out_path}", flush=True)
        return 0
    if done:
        print(f"[eval_vllm] resuming: {len(done)} ids present, {len(todo)} to "
              f"generate", flush=True)

    # ----- engine -----
    from vllm import LLM, SamplingParams
    import torch
    import transformers
    import vllm

    engine_config = {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "dtype_arg": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "env": args.env_name,
        "stop_strings": None,
        "chat_template": False,
        "paper_split": args.paper_split,
        "inject_phases": args.inject_phases,  # recorded only; plain vLLM = no steer vector
        "steering": False,
    }
    dev = torch.cuda.get_device_properties(0)
    engine_config["gpu"] = {"name": dev.name,
                            "total_memory_bytes": dev.total_memory}
    llm = LLM(model=args.model_name, dtype=args.dtype,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              kv_cache_dtype=args.kv_cache_dtype,
              enforce_eager=args.enforce_eager, seed=args.seed)
    try:  # resolved dtype / max_model_len as the engine sees them
        eng = getattr(llm, "llm_engine", None)
        mc = getattr(eng, "model_config", None) or getattr(
            getattr(eng, "vllm_config", None), "model_config", None) \
            if eng is not None else None
        engine_config["dtype_resolved"] = str(getattr(mc, "dtype", None))
        engine_config["max_model_len_resolved"] = getattr(mc, "max_model_len", None)
    except Exception as exc:
        engine_config["dtype_resolved"] = f"unavailable: {exc}"

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    # eos-only termination: `stop` stays None (no stop strings in parity mode)

    # ----- generation: ONE LLM.generate over the full batch (continuous batching) -----
    prompts = [it["full_prompt"] for it in todo]
    if harness_text:
        assert_harness_present(prompts, harness_text, "vllm teacher prompts")
    else:
        assert_no_harness_leak(prompts, harness_text, "vllm base prompts")
    t_gen = time.time()
    outputs = llm.generate(prompts, sampling) if prompts else []
    gen_seconds = time.time() - t_gen
    assert len(outputs) == len(prompts), "vLLM returned a different number of outputs"
    for it, out in zip(todo, outputs):
        assert out.prompt == it["full_prompt"], "vLLM output order/prompt mismatch"

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # ----- records (existing schema + engine/env/engine_config; markers; sandbox) -----
    eval_fn = loader.evaluate
    mbpp = args.dataset_name == "MBPP"
    sandbox_settings = None
    if mbpp:
        from utils.dataloaders_mbpp import sandbox_settings as mbpp_sandbox_settings
        sandbox_settings = mbpp_sandbox_settings()

    n_new = 0
    finish_reasons: Dict[str, int] = {}
    total_gen_tokens = 0
    for it, out in zip(todo, outputs):
        comp = out.outputs[0]
        text = comp.text.strip()
        finish_reasons[comp.finish_reason] = finish_reasons.get(comp.finish_reason, 0) + 1
        gen_tokens = len(comp.token_ids)
        total_gen_tokens += gen_tokens
        label = it["label"]
        rec = {
            "example_id": it["example_id"],
            "dataset": args.dataset_name,
            "split": args.split,
            "seed": args.seed,
            "arm": harness_tag,
            "arm_kind": "explicit" if harness else "base",
            "harness_id": harness.harness_id if harness else (
                args.harness_tag if args.harness_text_file else None),
            "harness_sha256": harness.sha256 if harness else composed_sha256,
            "model": args.model_name,
            "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens,
                         "temperature": 0.0, "batched": True,
                         "engine_continuous_batching": True,
                         "stop_strings": None},
            "engine": ENGINE,
            "env": args.env_name,
            "engine_config": engine_config,
            "prompt": it["full_prompt"],
            "base_prompt": it["base_prompt"],
            "prompt_token_len": prompt_lens[it["example_id"]],
            "base_prompt_token_len": token_len(tokenizer, it["base_prompt"]),
            "output": text,
            "completion_token_len": gen_tokens,
            "finish_reason": comp.finish_reason,
            "generated_tokens": gen_tokens,
            "repetition_note": _repetition_note(text),
            "label": label,
            "correct": bool(eval_fn(text, label)),
            "git_commit": git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if mbpp:
            detail = loader.evaluate_detailed(out.outputs[0].text, label)
            rec["correct"] = bool(detail["all_pass"])
            rec["sandbox_detail"] = {
                "extracted": detail["extracted"],
                "extraction_method": detail["extraction_method"],
                "compile_error": detail["compile_error"],
                "exec_error": detail["exec_error"],
                "n_pass": detail["n_pass"],
                "n_total": detail["n_total"],
                "timed_out": detail["timed_out"],
                "sandbox_settings": detail["sandbox"],
            }
        rec["markers"] = compute_example_markers(
            args.dataset_name, eval_fn, text, label, prompt=it["base_prompt"])
        fsync_append(out_path, json.dumps(rec))
        n_new += 1

    # reload for metrics (resume-safe: metrics cover the full file)
    records: List[Dict[str, Any]] = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    assert len(records) == len(items), \
        f"{len(records)} records vs {len(items)} examples — resume bookkeeping broken"
    records.sort(key=lambda r: int(r["example_id"]))

    acc = sum(r["correct"] for r in records) / len(records)
    cap_hits = sum(1 for r in records if r.get("finish_reason") == "length")
    mean_prompt_tokens = sum(r["base_prompt_token_len"] for r in records) / len(records)
    mean_gen_tokens = sum(r["completion_token_len"] for r in records) / len(records)
    extract_fail = (sum(1 for r in records
                        if not r.get("sandbox_detail", {}).get("extracted", True))
                    / len(records)) if mbpp else None
    degen = sum(1 for r in records
                if r.get("repetition_note", {}).get("max_identical_line_run", 0) >= 3)

    gen_seconds = gen_seconds if todo else max(gen_seconds, 1e-9)
    metrics = {
        "run": os.path.basename(out_path),
        "engine": ENGINE,
        "env": args.env_name,
        "model": args.model_name,
        "dataset": args.dataset_name,
        "split": args.split,
        "n": len(records),
        "subset": f"first {len(records)} in loader order",
        "seed": args.seed,
        "harness_id": harness.harness_id if harness else (
            args.harness_tag if args.harness_text_file else None),
        "harness_sha256": harness.sha256 if harness else composed_sha256,
        "arm_kind": "explicit" if harness else "base",
        "inject_phases": args.inject_phases,
        "max_new_tokens": max_new_tokens,
        "accuracy": acc,
        "mean_base_prompt_tokens": round(mean_prompt_tokens, 2),
        "mean_generated_tokens": round(mean_gen_tokens, 2),
        "total_generated_tokens": total_gen_tokens,
        "wallclock_total_s": round(time.time() - t_start, 1),
        "wallclock_generate_s": round(gen_seconds, 1),
        "examples_per_s": round(len(records) / gen_seconds, 2) if gen_seconds else None,
        "generated_tokens_per_s": round(total_gen_tokens / gen_seconds, 1)
                                  if gen_seconds else None,
        "cap_hit_rate": round(cap_hits / len(records), 4),
        "eos_stop_rate": round(finish_reasons.get("stop", 0) / len(records), 4),
        "finish_reasons": finish_reasons,
        "extraction_failure_rate": round(extract_fail, 4) if extract_fail is not None else None,
        "repetition_degeneracy_rate_ge3lines": round(degen / len(records), 4),
        "peak_vram_allocated_bytes": peak_alloc,
        "peak_vram_reserved_bytes": peak_reserved,
        "engine_config": engine_config,
        "eos_notes": eos_notes,
        "markers_summary": summarize_markers([r["markers"] for r in records]),
        "sandbox_settings": sandbox_settings,
        "output_file": out_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[eval_vllm] DONE {args.dataset_name}/{args.split} engine=vllm: "
          f"acc={acc:.4f} n={len(records)} cap_hit={metrics['cap_hit_rate']:.2f} "
          f"({metrics['wallclock_total_s']}s total, {gen_seconds:.1f}s generate) "
          f"-> {out_path}", flush=True)
    print(f"[eval_vllm] engine: vllm={vllm.__version__} torch={torch.__version__} "
          f"dtype={engine_config.get('dtype_resolved')} "
          f"max_model_len={engine_config.get('max_model_len_resolved')} "
          f"eos={eos_notes.get('eos_token_id')} "
          f"gc_eos={eos_notes.get('generation_config_eos_token_id')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
