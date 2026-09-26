"""E0-verify: evaluate a saved WUAS checkpoint on the PAPER's official splits for
BoolQ / GSM8K (arXiv:2603.00425 Table 7), using the paper's adapter setting (linear).

BoolQ:    test = official validation (3,270)   [paper Table 7]
GSM8K:    test = official test (1,319 ≈ paper's 1,320)
Prompt templates, completion builders, and evaluate() are inherited unchanged from
utils.dataloaders; only the split source differs.

Usage:
    python eval_official.py --model-name meta-llama/Llama-3.2-1B-Instruct \
        --dataset-name BoolQ --ckpt-path ./BoolQ_model/... --split test
"""

import argparse
import os
import sys

sys.path.append('./')

import torch
from datasets import load_dataset

from inference import evaluate_model
from steerer import AdapterSteerer
from utils.dataloaders import BoolQ, GSM8K, BaseDatasetLoader
from utils.utils import max_new_tokens_per_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer


class BoolQ_Paper(BoolQ):
    """BoolQ with the paper's splits: train = official train, test = official validation."""

    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("google/boolq")
        self.raw_dict = {
            "train": ds["train"],
            "validation": ds["train"].shuffle(seed=42).train_test_split(test_size=0.2, seed=42)["test"],
            "test": ds["validation"],
        }


class GSM8K_Paper(GSM8K):
    """GSM8K with the paper's splits: train = official train, test = official test."""

    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset("openai/gsm8k", "main")
        self.raw_dict = {
            "train": ds["train"],
            "validation": ds["train"].shuffle(seed=42).train_test_split(test_size=0.2, seed=42)["test"],
            "test": ds["test"],
        }


PAPER_DATASETS = {"BoolQ": BoolQ_Paper, "GSM8K": GSM8K_Paper}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, choices=sorted(PAPER_DATASETS))
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="Model load dtype (E0-v3: fp32 default = unchanged; bf16 for 4B/8B models on A30-24GB)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override the registry max_new_tokens (E0-v3 pre-registered caps: BoolQ 96 / ARC 256 / GSM8K 512 / MBPP 512)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs = {"torch_dtype": torch.bfloat16} if args.dtype == "bf16" else {}
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

    dataset_obj = PAPER_DATASETS[args.dataset_name](mode="eval")
    datasets = dataset_obj.load_raw_dataset(return_test=True)
    evaluation_dataset = datasets[args.split]
    eval_fn = dataset_obj.evaluate

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            "outputs", f"{args.dataset_name}_paper",
            os.path.basename(args.ckpt_path), f"official_{args.split}.json")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    evaluate_model(
        evaluation_dataset, eval_fn, steered_model, tokenizer, output_file,
        max_new_tokens=(args.max_new_tokens if args.max_new_tokens is not None
                        else max_new_tokens_per_dataset.get(args.dataset_name, 100)))


if __name__ == "__main__":
    main()
