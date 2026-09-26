#!/usr/bin/env bash
# ============================================================================
# reproduce_b1.sh — paper-aligned WUAS B1 baseline for ANY model / dataset / GPU
#
# Trains the WUAS post-block low-rank intervention with the paper's
# protocol (arXiv:2603.00425 Table 1) and evaluates on the paper's official
# test split. Protocol (verified 2026-08-31, see REPRODUCE.md):
#   linear adapters, rank 8, all layers, block targets, all tokens,
#   per-dataset LR (paper Table 9), greedy eval, official splits.
#
# Usage (env-var style; every variable optional):
#   MODEL=meta-llama/Llama-3.2-1B-Instruct DATASET=ARC GPU=0 ./reproduce_b1.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct DATASET=GSM8K GPU=1 EPOCHS=2 ./reproduce_b1.sh
#   MODEL=... DATASET=BoolQ GPU=0 LR=2e-3 MAXLEN=512 EVAL_SPLIT=validation ./reproduce_b1.sh
#
# Other modes:
#   MODE=smoke      -> 6-step train + repo-split eval (pipeline check, ~5 min)
#   MODE=b0         -> base-model zero-shot eval only (no training)
#   MODE=fixedvec   -> fixed steering vectors instead of adapters
#   MODE=repo_eval  -> repo-split eval (--do-eval path) instead of official splits
#
# Per-dataset defaults (paper-aligned; override with LR/MAXLEN/EPOCHS):
#   BoolQ: lr 1e-3, 1 ep, max_len 256 (repo default)   | paper ref 0.862 (ours 0.838)
#   ARC:   lr 1e-3, 1 ep, max_len 256 (repo default)   | paper ref 0.603 (ours 0.603)
#   GSM8K: lr 7.5e-4, 1 ep, max_len 512                | paper ref 0.315 (ours 0.313)
# ============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PY="${PY:-python}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
DATASET="${DATASET:-BoolQ}"          # registry key; ARC-Challenge is "ARC" (see REPRODUCE.md D1)
GPU="${GPU:-0}"
SEED="${SEED:-42}"
MODE="${MODE:-b1}"                   # b1 | smoke | b0 | fixedvec | repo_eval
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# --- per-dataset defaults (paper Table 9 LRs; GSM8K needs max_len 512) ------
case "$DATASET" in
  BoolQ) DEF_LR=1e-3;   DEF_MAXLEN=256 ;;
  ARC)   DEF_LR=1e-3;   DEF_MAXLEN=256 ;;
  GSM8K) DEF_LR=7.5e-4; DEF_MAXLEN=512 ;;
  *)     DEF_LR=9e-4;   DEF_MAXLEN=256 ; echo "NOTE: '$DATASET' has no paper-aligned defaults; using repo defaults" ;;
esac
LR="${LR:-$DEF_LR}"
MAXLEN="${MAXLEN:-$DEF_MAXLEN}"
EPOCHS="${EPOCHS:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE=disabled

MNAME="$(basename "$MODEL")"
LRNAME="$("$PY" -c "print(float('$LR'))")"   # repo names ckpt dirs with float('lr')

echo "=== reproduce_b1: model=$MODEL dataset=$DATASET mode=$MODE gpu=$GPU"
echo "    lr=$LR epochs=$EPOCHS max_len=$MAXLEN seed=$SEED eval_split=$EVAL_SPLIT"

run_train () {  # $1 = extra train args (string, may be empty)
  # shellcheck disable=SC2086
  $PY train.py \
    --model-name "$MODEL" \
    --dataset-name "$DATASET" \
    --linear \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --max-len "$MAXLEN" \
    $1
}

case "$MODE" in
  smoke)
    run_train "--max-steps 6 --do-eval --test-only"
    ;;
  b0)
    echo "=== B0 base-model zero-shot (repo splits)"
    $PY eval_base.py --model-name "$MODEL" --dataset-name "$DATASET" --split "$EVAL_SPLIT" \
      --output-file "outputs/base_${MNAME}/${DATASET}/eval_${EVAL_SPLIT}.json"
    ;;
  fixedvec)
    run_train "--no-adapter --do-eval"
    ;;
  repo_eval)
    run_train "--do-eval"
    ;;
  b1)
    run_train ""
    # train.py (no --do-eval) saves the checkpoint under the repo convention:
    CKPT="${DATASET}_model/${MNAME}_adapter_all_linear/lr${LRNAME}_bs8_ep${EPOCHS}_warmup0.03_wd0.0_rank8"
    OUT="outputs/${DATASET}_paper/${MNAME}_lr${LRNAME}_bs8_ep${EPOCHS}_warmup0.03_wd0.0_rank8/official_${EVAL_SPLIT}.json"
    echo "=== EVAL (official split=$EVAL_SPLIT) ckpt=$CKPT"
    case "$DATASET" in
      ARC) $PY eval_arc_official.py --model-name "$MODEL" --ckpt-path "$CKPT" --split "$EVAL_SPLIT" --output-file "$OUT" ;;
      *)   $PY eval_official.py     --model-name "$MODEL" --dataset-name "$DATASET" --ckpt-path "$CKPT" --split "$EVAL_SPLIT" --output-file "$OUT" ;;
    esac
    echo "=== DONE."
    echo "    Official-split predictions: $OUT"
    echo "    Accuracy is the 'Accuracy =' line above."
    ;;
  *) echo "Unknown MODE=$MODE (b1|smoke|b0|fixedvec|repo_eval)"; exit 1 ;;
esac
