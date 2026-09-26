# Latent Harness: Activation-Shift Policy Adaptation and Behavior Compilation

Anonymous code release for double-blind review.

This repository contains the minimal code needed to reproduce the main
experiments of the paper:

1. **Route A - task-level policy adaptation with zero weight updates.**
   A post-block low-rank activation shift (the WUAS parameterization) is
   trained on step-level decision pairs produced by HarnessForge rollouts,
   and evaluated with and without harness text on agentic benchmarks
   (ToolHop, SearchQA, API-Bank, TMDB), plus a PEFT LoRA arm as the
   weight-update baseline under matched capacity.
2. **Route B - behavior compilation.** Explicit harness-conditioned teacher
   behavior is distilled into a rank-8 post-block adapter with zero harness
   text at inference, with behavioral markers used for attribution.
3. **B1/B0 protocol reproduction** on BoolQ / ARC-Challenge / GSM8K with the
   official splits of the WUAS paper.

## Repository layout

```
latentharnesssub/
├── train.py                     # Unified training entry (WUAS adapter / fixed vectors / LoRA arm)
├── steerer.py                   # Core intervention: post-block low-rank adapter, fixed vectors
├── custom_trainer.py            # StepwiseTrainer (prefix-wise step loss for distillation)
├── inference.py                 # Greedy evaluation loop shared by all evaluators
├── eval_base.py                 # B0: base-model zero-shot on repo splits
├── eval_official.py             # Official-split eval (BoolQ / GSM8K)
├── eval_arc_official.py         # Official-split eval (ARC-Challenge)
├── reproduce_b1.sh              # One-command B1/B0/smoke reproduction driver
├── setup_env.sh                 # Conda environment bootstrap
├── HARNESS_LIBRARY.md           # Machine-read harness text registry (data; parsed by code)
├── utils/                       # Dataset loaders (BoolQ, ARC, GSM8K, MBPP, HFStepPairs, ...)
├── tina/                        # GRPO reward functions
├── harness/                     # Teacher caching, markers, bundle export, steered-vLLM evals
│   ├── teacher_cache.py         #   teacher rollout cache for distillation targets
│   ├── markers.py               #   behavioral marker counters (attribution)
│   ├── harness_text.py          #   registry loader (reads HARNESS_LIBRARY.md, sha256-checked)
│   ├── prompts.py               #   teacher/student paired prompt construction
│   ├── export_easysteer_bundle.py  # checkpoint -> sidecar bundle for the vLLM plugin
│   ├── eval_easysteer.py        #   steered vLLM serving-side evaluation
│   ├── eval_vllm.py             #   offline vLLM evaluation with markers
│   ├── eval_arms.py             #   leakage guards shared by evaluators
│   ├── mean_diff_fixedvec.py    #   mean-difference steering baseline
│   └── classify_errors_mbpp.py  #   MBPP error taxonomy
├── easysteer_parity_plugin/     # vLLM plugin registering the wuas_adapter steering algorithm
├── scripts/
│   ├── run_harnessforge_infer.py    # launcher for HarnessForge run_infer.py (4B package)
│   └── run_harnessforge_infer8b.py  # launcher for HarnessForge run_infer.py (8B package)
└── tests/                       # Unit tests (registry, markers, MBPP loader, paired prompts)
```

## 1. Environment

```bash
bash setup_env.sh              # creates a conda env named "latentharness"
conda activate latentharness
```

`setup_env.sh` installs torch (CUDA 12.1 wheels), transformers, trl, peft,
datasets, math_verify and other pinned dependencies used by the training and
evaluation scripts. The steered-vLLM evaluation additionally requires an
EasySteer-fork vLLM build whose `steer_vectors` API is present; the plugin in
`easysteer_parity_plugin/` is installed into that environment with
`pip install ./easysteer_parity_plugin`.

## 2. Models and data

Model checkpoints (downloaded automatically from their hub identifiers):

- `meta-llama/Llama-3.2-1B-Instruct` (B0/B1 protocol, behavior screens)
- `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-8B` (main policy experiments)

Datasets: `google/boolq`, `allenai/ai2_arc` (ARC-Challenge), `openai/gsm8k`,
MBPP (loader in `utils/dataloaders_mbpp.py`), all fetched via the `datasets`
library on first use.

HarnessForge is required for the agentic benchmarks (ToolHop, SearchQA,
API-Bank, TMDB) and for generating the step-level decision pairs. Place a
HarnessForge checkout containing `HarnessForge_4B/` (and `HarnessForge_8B/`
for the 8B runs) next to this repository or point the environment variable
`HARNESSFORGE_ROOT` at it.

## 3. Reproducing the main experiments

### 3.1 B1/B0 protocol (BoolQ / ARC / GSM8K)

```bash
# B1: train the rank-8 linear post-block adapter, then eval on the official test split
MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=BoolQ GPU=0 ./reproduce_b1.sh
MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=ARC   GPU=0 ./reproduce_b1.sh
MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=GSM8K GPU=0 ./reproduce_b1.sh

# B0: base-model zero-shot only
MODE=b0 MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=BoolQ ./reproduce_b1.sh

# quick pipeline smoke test (~5 min)
MODE=smoke MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=BoolQ ./reproduce_b1.sh
```

Per-dataset defaults follow the WUAS paper protocol (linear adapters, rank 8,
all layers, block targets, all tokens, per-dataset LR, greedy decoding,
official splits). Predictions are written under `outputs/`.

### 3.2 Route A: policy adaptation on HarnessForge decision pairs

1. **Generate step-level decision pairs** with HarnessForge's rollout tooling
   (`tools/prepare_toolhop_rollout_sft.py` in the HarnessForge 4B tree). This
   produces `<tier>_{action,reasoning,combined}.jsonl` files.
2. **Train the zero-weight adapter:**

   ```bash
   HF_STEP_PAIRS_JSONL=path/to/balanced_combined.jsonl \
   python train.py --model-name Qwen/Qwen3-4B-Instruct-2507 \
       --dataset-name HFStepPairs --linear --lr 1e-3 --epochs 3
   ```

   The loader (`utils/dataloaders_hfstep.py`) renders each decision pair as a
   chat completion and masks loss to the assistant turn, matching the
   supervision seen by the LoRA baseline.
3. **LoRA baseline (same data protocol, matched capacity):**

   ```bash
   HF_STEP_PAIRS_JSONL=path/to/balanced_combined.jsonl \
   python train.py --model-name Qwen/Qwen3-4B-Instruct-2507 \
       --dataset-name HFStepPairs --lora --lr 1e-4 --epochs 3
   ```

   This trains a PEFT LoRA arm (rank 8, all-linear targets, alpha 16),
   mirroring the HarnessForge policy-alignment reference so that LoRA and the
   activation shift differ only in adapter form.
4. **Export the adapter as a serving bundle:**

   ```bash
   python harness/export_easysteer_bundle.py --checkpoint <ckpt_dir> \
       --model-name <model> --dataset-name <dataset> --use-silu true \
       --out-dir bundles/<bundle_name>
   ```

   The bundle (manifest + adapter.safetensors) is consumed by the
   `wuas_adapter` steering algorithm registered by
   `easysteer_parity_plugin/`.
5. **Serve and evaluate.** Start an EasySteer-fork vLLM server with
   `steering_config` pointing at the bundle and the plugin installed, then
   run the HarnessForge benchmark driver:

   ```bash
   python scripts/run_harnessforge_infer.py --benchmark toolhop \
       --harness <harness_module> --model <served_model_tag> [HarnessForge args]
   ```

   For the 8B model use `scripts/run_harnessforge_infer8b.py` instead. The
   launcher injects a bounded JSON-repair guard (fail-closed parsing of
   model-emitted tool calls, enabled with `HF_JSONREPAIR_GUARD=1`) and
   executes HarnessForge's `run_infer.py` unmodified.

### 3.3 Route B: behavior compilation (distillation)

1. **Collect teacher traces** under each harness condition by running the
   teacher model through the paired prompts from `harness/prompts.py`
   (`build_teacher_prompt` over the WUAS loaders); record per-example
   completions and marker fields into a JSONL cache. `harness/teacher_cache.py`
   provides the offline inspection/validation CLI for these caches:

   ```bash
   python harness/teacher_cache.py inspect --cache-file <cache.jsonl>
   ```

   Harness texts and their sha256 hashes are read from `HARNESS_LIBRARY.md`
   (the single source of truth; any wording change is a new entry).
2. **Distill into a rank-8 adapter** with the step-wise loss
   (`custom_trainer.py::StepwiseTrainer`, wired through `train.py`), using the
   cached traces as targets. Three arms are supported by the data protocol:
   distill-only, task-only, and joint; shuffled-harness and mean-difference
   (`harness/mean_diff_fixedvec.py`) serve as controls.
3. **Evaluate with zero harness text** on the EasySteer steered server
   (`harness/eval_easysteer.py`) and score behavioral markers
   (`harness/markers.py`) alongside accuracy to attribute the gain.
   `harness/eval_vllm.py` provides the offline-vLLM cross-engine parity check.

## 4. Tests

```bash
python -m pytest tests/ -q
```

The suite covers the harness registry parser (hash integrity), marker
counters (pre-registered positive/negative controls), the MBPP loader, and
paired prompt construction. No GPU or network is required for the registry
and marker tests; loader tests fetch public datasets on first use.

## 5. Notes

- `HARNESS_LIBRARY.md` is a data registry parsed at runtime
  (`harness/harness_text.py`), not documentation; the tests verify its hashes.
- Training-data protocol matters: the strict-tier, balanced decision-pair
  protocol (`HFStepPairs`) is the configuration reported in the paper.
- All randomized evaluations in the paper use at least 2 independent runs
  (Route A) or 3 seeds with bootstrap confidence intervals (Route B).
