"""EasySteer-backed evaluator — the EasySteer sibling of harness/eval_vllm.py,
for the EasySteer parity spike (the parity spec; INFRASTRUCTURE ONLY,
built and measured, NOT adopted; no protocol stage may use this path).

Same CLI surface and record conventions as the existing evaluators, with ONLY
the generation backend swapped (HF transformers -> the EasySteer fork of
vLLM 0.26.0 with residual-stream steer vectors). Everything else in the
project is reused unchanged: loaders (`utils/dataloaders.py` via
`utils.utils.dataset_dict`), paired/base-prompt construction + the
byte-identity guard, the leakage assertion (`harness/eval_arms.py::
assert_no_harness_leak`), the marker layer (`harness/markers.py`), the MBPP
sandbox (unused here — the spike's cells are BoolQ, but the path is kept
loader-agnostic), the per-example raw JSONL schema, and the append-only
ledger.

What is NEW (the generate step only):
  - the engine is the EasySteer fork with `enable_steer_vector=True` and the
    custom `wuas_adapter` algorithm (easysteer_parity_plugin; the
    pre-registered shared algorithm, the parity spec);
  - steering is applied via a SERVER-LEVEL steering_config (a SteeringSpec
    JSON with one VectorSpec: source = the Task-1 sidecar bundle,
    algorithm="wuas_adapter", scale = 0.0 (run iii) or 1.0 (run iv),
    layers = all decoder layers, apply = prompt+generation "all");
  - raw records carry `"engine": "easysteer"`, `"env"`, and a `fork_config`
    block (fork commits, algorithm name, bundle path + sha256, use_silu,
    layers, phases, dtype, graph mode, scale) in addition to the existing
    schema; harness id + sha256 and model id per line as usual.

Resume-safe (the generate_teacher.py convention): fsync-per-record append;
on rerun, already-present example ids are skipped. A `.metrics.json` sidecar
per run carries accuracy, throughput, peak VRAM (nvidia-smi poll — the vLLM
parent's torch counters do not see the EngineCore process), the engine/fork
config, and marker summaries. One uniform parity-schema ledger line
(experiment_id=INFRA-EASYSTEER-PARITY) is appended to the ledger at the end
of a complete run.

Usage (inside the easysteer env):
  python harness/eval_easysteer.py --model-name meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name BoolQ --split test --limit 100 --scale 1.0 \
    --bundle results/easysteer_parity/bundles/wuas_task_adapter_boolq_linear \
    --out-dir results/easysteer_parity --run-tag wuas_adapter
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # FlashInfer sampler JIT is cu13-targeted; greedy decode is argmax, the torch sampler is numerically equivalent (the parity spec)
import easysteer_parity_plugin  # noqa: E402  (parent-process registration)

easysteer_parity_plugin.register()

from transformers import AutoTokenizer  # noqa: E402

from harness.eval_arms import assert_no_harness_leak  # noqa: E402
from harness.markers import compute_example_markers, summarize_markers  # noqa: E402
from harness.teacher_cache import E1B_MAX_NEW_TOKENS, git_commit  # noqa: E402
from utils import dataset_dict  # noqa: E402

ENGINE = "easysteer"
EXPERIMENT_ID = "INFRA-EASYSTEER-PARITY"
DEFAULT_ENV_NAME = "easysteer"


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
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--dataset-name", type=str, required=True,
                   help="Key of dataset_dict (ARC_Challenge is registered as 'ARC')")
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--paper-split", action="store_true",
                   help="E0-v3-ENGINES: evaluate the PAPER's official split "
                        "(BoolQ_Paper/GSM8K_Paper/ARC_Challenge_Paper from "
                        "eval_official.py / eval_arc_official.py) instead of the "
                        "repo loader split. Loaders/templates/evaluate unchanged.")
    p.add_argument("--limit", type=int, default=None,
                   help="First N examples of the split in loader order (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Default: the frozen v3 cap for the dataset")
    p.add_argument("--seed", type=int, default=42)
    # ----- steering (the new part) -----
    p.add_argument("--bundle", type=str, default=None,
                   help="Task-1 sidecar bundle dir (manifest.json + adapter.safetensors); "
                        "exactly one of --bundle / --mean-diff-artifact is required")
    p.add_argument("--scale", type=float, required=True,
                   help="0.0 = EasySteer base run (iii); 1.0 = EasySteer + vector (iv)")
    p.add_argument("--mean-diff-artifact", type=str, default=None,
                   help="E1-Eval-v3 MEAN_DIFF_STEERING (FixedVector): steering-spec JSON "
                        "produced by harness/mean_diff_fixedvec.py convert (the fork's "
                        "built-in `direct` algorithm, in-memory DirectionVector wire "
                        "payload). Mutually exclusive with --bundle.")
    p.add_argument("--inject-phases", type=str, choices=["both", "generation"],
                   default="both",
                   help="Which forward phases the steer vector is applied to "
                        "(EasySteer SelectSpec `apply`; an omitted phase is left "
                        "untouched). 'both' (default) = apply.prompt='all' + "
                        "apply.generation='all' (historical behaviour). 'generation' = "
                        "apply.prompt=None + apply.generation='all' — leaves the prompt "
                        "pass un-steered so a thinking prefix is not perturbed "
                        "(U10 finding, skills2steer/PROJECT_STATE.md §8.9).")
    # ----- engine config (pre-registered defaults, the parity spec) -----
    p.add_argument("--dtype", type=str, default="auto",
                   help="Engine dtype (default 'auto' = model-config default)")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--steer-graph-mode", type=str, default="auto")
    # ----- bookkeeping -----
    p.add_argument("--out-dir", type=str, default="results/easysteer_parity")
    p.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    p.add_argument("--run-tag", type=str, required=True,
                   help="Filename tag distinguishing this run (e.g. wuas_adapter, "
                        "wuas_base, latent_adapter, latent_base)")
    p.add_argument("--config-tag", type=str, default=None,
                   help="Configuration label recorded in ledger/metrics (e.g. "
                        "A_WUAS_TASK_ADAPTER, B_LATENT_DISTILL_ONLY)")
    p.add_argument("--ledger", type=str, default="results/ledger.jsonl")
    p.add_argument("--experiment-id", type=str, default=EXPERIMENT_ID,
                   help="Ledger experiment_id (default: the spike's "
                        "INFRA-EASYSTEER-PARITY; protocol stages override)")
    p.add_argument("--note", type=str, default=None,
                   help="Ledger note override (default: the spike's INFRA note)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    return args


def fsync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VramPoller:
    """nvidia-smi sampling thread (the previous stage's method): the vLLM
    parent's torch.cuda counters read 0 because the engine runs in a separate
    EngineCore process."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                if out:
                    self.peak_bytes = max(self.peak_bytes,
                                          int(float(out[0])) * (1 << 20))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    # U10 (/alfworld_eval_vllm.py): SelectSpec accepts only "all"|None per phase and an
    # omitted phase is left untouched, so generation-only == {"generation": "all"}.
    inject_apply = ({"generation": "all"} if args.inject_phases == "generation"
                    else {"prompt": "all", "generation": "all"})
    inject_phases_recorded = (["generation"] if args.inject_phases == "generation"
                              else ["prompt", "generation"])

    if args.mean_diff_artifact:
        assert args.bundle is None, "--bundle and --mean-diff-artifact are mutually exclusive"
        with open(args.mean_diff_artifact, encoding="utf-8") as f:
            fv_spec = json.load(f)
        manifest = None
        steer_algorithm = "direct"
        use_silu = None
        bundle_sha = None
        md_sha = sha256_file(args.mean_diff_artifact)
    else:
        assert args.bundle is not None, "exactly one of --bundle / --mean-diff-artifact is required"
        fv_spec = None
        steer_algorithm = "wuas_adapter"
        with open(os.path.join(args.bundle, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        bundle_sha = sha256_file(os.path.join(args.bundle, "adapter.safetensors"))
        use_silu = bool(manifest["use_silu"])

    max_new_tokens = args.max_new_tokens or E1B_MAX_NEW_TOKENS.get(
        args.dataset_name, E1B_MAX_NEW_TOKENS["GSM8K"])
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

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
    n = min(args.limit, len(paired)) if args.limit is not None else len(paired)
    items = []
    for i in range(n):
        base_prompt = loader.build_prompt(raw_examples[i])
        if base_prompt != paired[i][0]:
            raise ValueError(
                f"build_prompt mismatch at example {i}: the base prompt must stay "
                "byte-identical to the eval-pipeline prompt")
        items.append({"example_id": i, "base_prompt": base_prompt,
                      "label": paired[i][2]})
    for it in items:  # the leakage gate (student-side), unchanged
        assert_no_harness_leak([it["base_prompt"]], None,
                               context="eval_easysteer base prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    over = [it for it in items
            if token_len(tokenizer, it["base_prompt"]) + max_new_tokens
            > args.max_model_len]
    if over:
        raise SystemExit(
            f"max_model_len {args.max_model_len} too small for {len(over)} "
            f"example(s); raise --max-model-len — never truncate silently")

    cap_tag = f"nmt{max_new_tokens}"
    scale_tag = f"scale{args.scale:g}"
    paper_tag = "__paper" if args.paper_split else ""
    out_path = os.path.join(
        out_dir,
        f"{args.dataset_name}__{args.run_tag}__{args.split}__"
        f"{args.model_name.replace('/', '--')}__{cap_tag}__{scale_tag}{paper_tag}.jsonl")

    # ----- resume: skip already-present example ids -----
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    done: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[int(rec["example_id"])] = rec
    todo = [it for it in items if it["example_id"] not in done]
    if os.path.exists(out_path) and not todo and not args.force:
        print(f"[eval_easysteer] output complete with {len(done)} records; "
              f"skipping (--force to re-run): {out_path}", flush=True)
        todo = []
    elif done:
        print(f"[eval_easysteer] resuming: {len(done)} ids present, "
              f"{len(todo)} to generate: {out_path}", flush=True)

    # ----- server-level steering config (the new generate step) -----
    if fv_spec is not None:
        vector_spec = {
            "data": fv_spec["wire"],
            "algorithm": steer_algorithm,
            "scale": args.scale,
            "normalize": False,
            "layers": list(fv_spec["layers"]),
            "apply": dict(inject_apply),
            "name": f"{args.run_tag}-{scale_tag}",
        }
        fork_config = {
            "algorithm": steer_algorithm,
            "fixedvec_spec": os.path.abspath(args.mean_diff_artifact),
            "fixedvec_spec_sha256": md_sha,
            "mean_diff_source_artifact": fv_spec.get("source_artifact"),
            "mean_diff_source_artifact_sha256": fv_spec.get(
                "source_artifact_sha256"),
            "mean_diff_policy": fv_spec.get("policy"),
            "mean_diff_meta": fv_spec.get("mean_diff_meta", {}),
            "layers": list(fv_spec["layers"]),
            "phases": list(inject_phases_recorded),
            "inject_phases": args.inject_phases,
            "apply": dict(inject_apply),
            "scale": args.scale,
            "normalize": False,
        }
    else:
        # E3-COMPOSITION (additive): merged composition bundles carry
        # `composition` provenance and may omit `layers`/`source_checkpoint`
        # (the rank-2k merged adapter synthesizes those from its two sources).
        comp = manifest.get("composition") or {}
        comp_a = comp.get("component_a") or {}
        comp_b = comp.get("component_b") or {}
        layers = manifest.get("layers")
        if layers is None:
            layers = list(range(int(manifest["n_layers"])))
        src_ckpt = manifest.get("source_checkpoint")
        src_ckpt_sha = manifest.get("source_checkpoint_sha256")
        if comp:
            src_ckpt = src_ckpt or (
                f"e3_merged[{comp.get('kind')}|a={comp_a.get('path')}"
                f"|b={comp_b.get('path')}]")
            src_ckpt_sha = src_ckpt_sha or (
                f"e3_merged[a={comp_a.get('adapter_sha256')}"
                f",b={comp_b.get('adapter_sha256')}]")
        vector_spec = {
            "source": os.path.abspath(args.bundle),
            "algorithm": steer_algorithm,
            "scale": args.scale,
            "layers": list(layers),
            "apply": dict(inject_apply),
            "name": f"{args.run_tag}-{scale_tag}",
        }
        fork_config = {
            "algorithm": steer_algorithm,
            "adapter_bundle": os.path.abspath(args.bundle),
            "adapter_bundle_sha256": bundle_sha,
            "use_silu": use_silu,
            "rank": manifest["rank"],
            "alpha": manifest["alpha"],
            "layers": list(layers),
            "phases": list(inject_phases_recorded),
            "inject_phases": args.inject_phases,
            "apply": dict(inject_apply),
            "scale": args.scale,
            "source_checkpoint": src_ckpt,
            "source_checkpoint_sha256": src_ckpt_sha,
            "fork_commits_json": os.path.join(
                out_dir, "fork_commits.json"),
        }
        if comp:
            fork_config["composition"] = comp
    server_spec = {
        "vectors": [vector_spec],
        "conflict": "priority",
    }

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
        "enforce_eager": args.enforce_eager,
        "steer_graph_mode": args.steer_graph_mode,
        "enable_steer_vector": True,
        "steer_algorithms": [steer_algorithm],
        "server_steering_config": server_spec,
        "seed": args.seed,
        "env": args.env_name,
        "stop_strings": None,
        "chat_template": False,
        "paper_split": args.paper_split,
    }
    dev = torch.cuda.get_device_properties(0)
    engine_config["gpu"] = {"name": dev.name, "total_memory_bytes": dev.total_memory}

    llm = LLM(
        model=args.model_name, dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, seed=args.seed,
        enable_steer_vector=True,
        steer_algorithms=[steer_algorithm],
        steering_config=json.dumps(server_spec),
        steer_graph_mode=args.steer_graph_mode,
    )
    try:
        mc = llm.llm_engine.model_config
        engine_config["dtype_resolved"] = str(mc.dtype)
        engine_config["max_model_len_resolved"] = mc.max_model_len
    except Exception as exc:
        engine_config["dtype_resolved"] = f"unavailable: {exc}"
    try:
        svc = llm.llm_engine.vllm_config.steer_vector_config
        engine_config["steer_vector_dtype_resolved"] = str(svc.adapter_dtype)
        engine_config["graph_mode_resolved"] = svc.graph_mode
    except Exception as exc:
        engine_config["graph_mode_resolved"] = f"unavailable: {exc}"

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    eval_fn = loader.evaluate
    mbpp = args.dataset_name == "MBPP"
    poller_peak = None
    if todo:
        t_file_start = time.time()
        with VramPoller() as poller:
            outputs = llm.generate([it["base_prompt"] for it in todo], sampling)
        poller_peak = poller.peak_bytes
        gen_seconds = time.time() - t_file_start
        total_gen_tokens = 0
        for it, out in zip(todo, outputs):
            comp = out.outputs[0]
            text = comp.text.strip()
            gen_tokens = len(comp.token_ids)
            total_gen_tokens += gen_tokens
            label = it["label"]
            rec = {
                "example_id": it["example_id"],
                "dataset": args.dataset_name,
                "split": args.split,
                "seed": args.seed,
                "arm": f"easysteer_{scale_tag}",
                "arm_kind": ("easysteer_base" if args.scale == 0.0 else "easysteer_fixedvec" if fv_spec is not None else "easysteer_adapter"),
                "harness_id": None,
                "harness_sha256": None,
                "model": args.model_name,
                "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens,
                             "temperature": 0.0, "batched": True,
                             "engine_continuous_batching": True,
                             "stop_strings": None},
                "engine": ENGINE,
                "env": args.env_name,
                "engine_config": engine_config,
                "fork_config": fork_config,
                "prompt": it["base_prompt"],
                "base_prompt": it["base_prompt"],
                "prompt_token_len": token_len(tokenizer, it["base_prompt"]),
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
                    "n_pass": detail["n_pass"],
                    "n_total": detail["n_total"],
                    "timed_out": detail["timed_out"],
                    "sandbox_settings": detail["sandbox"],
                }
            rec["markers"] = compute_example_markers(
                args.dataset_name, eval_fn, text, label, prompt=it["base_prompt"])
            fsync_append(out_path, json.dumps(rec))
    else:
        gen_seconds = None
        total_gen_tokens = 0

    # ----- metrics sidecar over the COMPLETE file -----
    records: List[Dict[str, Any]] = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda r: int(r["example_id"]))
    assert len(records) == len(items), \
        f"{len(records)} records vs {len(items)} examples — resume bookkeeping broken"

    acc = sum(r["correct"] for r in records) / len(records)
    cap_hits = sum(1 for r in records if r.get("finish_reason") == "length")
    finish_reasons: Dict[str, int] = {}
    for r in records:
        fr = str(r.get("finish_reason"))
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1
    degen = sum(1 for r in records
                if r.get("repetition_note", {}).get("max_identical_line_run", 0) >= 3)
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
        "config_tag": args.config_tag,
        "run_tag": args.run_tag,
        "scale": args.scale,
        "inject_phases": args.inject_phases,
        "apply": dict(inject_apply),
        "arm_kind": ("easysteer_base" if args.scale == 0.0 else "easysteer_fixedvec" if fv_spec is not None else "easysteer_adapter"),
        "max_new_tokens": max_new_tokens,
        "accuracy": acc,
        "mean_base_prompt_tokens": round(
            sum(r["base_prompt_token_len"] for r in records) / len(records), 2),
        "mean_generated_tokens": round(
            sum(r["completion_token_len"] for r in records) / len(records), 2),
        "total_generated_tokens": total_gen_tokens,
        "wallclock_total_s": round(time.time() - t_start, 1),
        "wallclock_generate_s": round(gen_seconds, 1) if gen_seconds else None,
        "examples_per_s": round(len(records) / gen_seconds, 2) if gen_seconds else None,
        "generated_tokens_per_s": round(total_gen_tokens / gen_seconds, 1)
                                  if gen_seconds else None,
        "cap_hit_rate": round(cap_hits / len(records), 4),
        "finish_reasons": finish_reasons,
        "repetition_degeneracy_rate_ge3lines": round(degen / len(records), 4),
        "peak_vram_nvidia_smi_bytes": poller_peak,
        "engine_config": engine_config,
        "fork_config": fork_config,
        "markers_summary": summarize_markers([r["markers"] for r in records]),
        "output_file": out_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path + ".metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ----- one uniform parity-schema ledger line per complete run -----
    ledger_line = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "experiment_id": args.experiment_id,
        "stage": "eval_easysteer_parity",
        "engine": ENGINE,
        "env": args.env_name,
        "model": args.model_name,
        "dataset": args.dataset_name,
        "split": args.split,
        "n_eval": len(records),
        "subset": ("full split (paper official)" if (args.limit is None and args.paper_split)
                   else "full split" if args.limit is None
                   else f"first {len(records)} in loader order"),
        "seed": args.seed,
        "harness_id": None,
        "harness_sha256": None,
        "config_tag": args.config_tag,
        "decoding": f"greedy,temperature=0,max_new_tokens={max_new_tokens},stop=none",
        "engine_config": engine_config,
        "fork_config": fork_config,
        "accuracy": acc,
        "throughput": {k: metrics[k] for k in (
            "wallclock_total_s", "wallclock_generate_s", "examples_per_s",
            "generated_tokens_per_s", "total_generated_tokens", "cap_hit_rate")
            if metrics.get(k) is not None},
        "peak_vram": {"nvidia_smi_peak_bytes": metrics["peak_vram_nvidia_smi_bytes"]},
        "paths": {"raw": os.path.abspath(out_path),
                  "metrics": os.path.abspath(out_path + ".metrics.json")},
        "note": args.note if args.note is not None else
                "EasySteer parity spike run (the parity spec; INFRA only, "
                "built and measured, NOT adopted)",
    }
    if args.ledger:
        fsync_append(os.path.join(REPO_ROOT, args.ledger), json.dumps(ledger_line))

    print(f"[eval_easysteer] DONE {args.dataset_name}/{args.split} "
          f"scale={args.scale} tag={args.run_tag} phases={args.inject_phases} "
          f"apply={inject_apply}: acc={acc:.4f} n={len(records)} "
          f"cap_hit={metrics['cap_hit_rate']:.2f} "
          f"({metrics['wallclock_total_s']}s total, "
          f"{metrics['wallclock_generate_s']}s generate) -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
