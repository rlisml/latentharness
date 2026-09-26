"""E0 baseline B0: zero-shot evaluation of the base (unsteered, untrained) model.

New E0 infrastructure -- NOT part of the method, and NOT a modification
of the intervention. Reuses the existing dataset loaders and the existing
evaluate_model loop from inference.py so that results are directly comparable
with steered runs (same prompts, splits, greedy decoding, JSONL format).

Usage (same conventions as inference.py):
    python eval_base.py --model-name meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name BoolQ --split test --output-file outputs/base/BoolQ/test.json
"""

import argparse
import os
import sys

sys.path.append('./')

from inference import evaluate_model
from utils import dataset_dict, max_new_tokens_per_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Key of dataset_dict (note: ARC-Challenge is registered as 'ARC')")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="Model load dtype (E0-v3: fp32 default = unchanged; bf16 for 4B/8B models on A30-24GB)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override the registry max_new_tokens (E0-v3 pre-registered caps: BoolQ 96 / ARC 256 / GSM8K 512 / MBPP 512)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples in loader order (E0-v3 fallback gate: first 200 of repo validation; diagnostics only)")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    load_kwargs = {"torch_dtype": __import__("torch").bfloat16} if args.dtype == "bf16" else {}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    dataset_obj = dataset_dict[args.dataset_name](mode="eval")
    datasets = dataset_obj.load_raw_dataset(return_test=True)
    evaluation_dataset = datasets[args.split]
    if args.limit is not None:
        evaluation_dataset = evaluation_dataset[: args.limit]
    eval_fn = dataset_obj.evaluate

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            "outputs", "base", args.dataset_name, f"eval_{args.split}.json")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    evaluate_model(
        evaluation_dataset, eval_fn, model, tokenizer, output_file,
        max_new_tokens=(args.max_new_tokens if args.max_new_tokens is not None
                        else max_new_tokens_per_dataset.get(args.dataset_name, 100)))


if __name__ == "__main__":
    main()
