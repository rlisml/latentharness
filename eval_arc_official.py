"""E0 verification: evaluate a saved WUAS checkpoint on the PAPER's official
ARC-Challenge splits (arXiv:2603.00425 Table 7: train/validation/test = official splits),
using the paper's adapter setting (linear).

New E0-verification infrastructure -- NOT a modification of the method.
Reuses the existing ARC_Challenge prompt template, completion builder, and evaluate()
function unchanged; only the split source differs from utils.dataloaders.ARC_Challenge
and only the --split/--ckpt handling differs from inference.py.

Usage:
    python eval_arc_official.py --model-name meta-llama/Llama-3.2-1B \
        --ckpt-path ./ARC_model/Llama-3.2-1B_adapter_all_linear/lr0.001_bs8_ep3_warmup0.03_wd0.0_rank8 \
        --split test
"""

import argparse
import os
import sys

sys.path.append('./')

import torch
from datasets import load_dataset

from inference import evaluate_model
from steerer import AdapterSteerer
from utils.dataloaders import ARC_Challenge, BaseDatasetLoader
from utils.utils import max_new_tokens_per_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer


class ARC_Challenge_Paper(ARC_Challenge):
    """ARC-Challenge with the paper's splits: official train / validation / test.

    Prompt template, answer strings, and evaluate() are inherited unchanged from
    utils.dataloaders.ARC_Challenge.
    """

    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        self.raw_dict = {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="Model load dtype (E0-v3: fp32 default = unchanged; bf16 for 4B/8B models on A30-24GB)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override the registry max_new_tokens (E0-v3 pre-registered caps: BoolQ 96 / ARC 256 / GSM8K 512 / MBPP 512)")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    load_kwargs = {"torch_dtype": __import__("torch").bfloat16} if args.dtype == "bf16" else {}
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    steered_model = AdapterSteerer(
        base_model,
        layers_to_steer="all",
        targets=("block",),
        apply_to="all",
        rank=args.adapter_rank,
        activation=None,  # paper main-results setting: linear adapters
    )

    ckpt_bin = os.path.join(args.ckpt_path, "pytorch_model.bin")
    if os.path.exists(ckpt_bin):
        state_dict = torch.load(ckpt_bin, map_location="cpu")
    else:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(args.ckpt_path, "model.safetensors"))
    missing, unexpected = steered_model.load_state_dict(state_dict, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")

    steered_model.to(device)
    steered_model.eval()

    dataset_obj = ARC_Challenge_Paper(mode="eval")
    datasets = dataset_obj.load_raw_dataset(return_test=True)
    evaluation_dataset = datasets[args.split]
    eval_fn = dataset_obj.evaluate

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            "outputs", "arc_paper", os.path.basename(args.ckpt_path), f"official_{args.split}.json")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    evaluate_model(
        evaluation_dataset, eval_fn, steered_model, tokenizer, output_file,
        max_new_tokens=(args.max_new_tokens if args.max_new_tokens is not None
                        else max_new_tokens_per_dataset.get("ARC", 100)))


if __name__ == "__main__":
    main()
