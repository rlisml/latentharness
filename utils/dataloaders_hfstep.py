"""HarnessForge policy-alignment data -> WUAS training loader.

Consumes the step-level decision pairs produced by HarnessForge's
`HarnessForge_4B/tools/prepare_toolhop_rollout_sft.py`:

    {"instruction": <history up to this step>, "input": "", "output": <decision>,
     "meta": {"sample_kind": ..., "quality_tier": ..., "source_item_index": ...,
              "answer_correct": 0/1, "path_score": ..., "step_index": ...}}

Mapping:  prompt = instruction (rendered as a single user turn),  completion = output.
That keeps the SAME supervision signal HarnessForge feeds to its LoRA arm — only the
parameterisation changes (post-block low-rank activation shift instead of LoRA).

Why chat rendering: at inference the model is served over `/v1/chat/completions`, so
training uses the same chat template (`_tokenize_chat_completion`, prefix masked,
loss on the assistant completion only).

Split: by `meta.source_item_index`, deterministically — every 10th source rollout goes
to validation. A whole rollout therefore never contributes to both splits (no leakage),
and no TEST split is used: evaluation happens through HarnessForge's `run_infer.py`,
not through this loader.

Config (env, so no CLI plumbing is needed):
    HF_STEP_PAIRS_JSONL  path to one of prepare_toolhop_rollout_sft.py's
                         <tier>_{action,reasoning,combined}.jsonl
                         (default: balanced_combined)
"""

import json
import os

from datasets import Dataset

from .dataloaders import BaseDatasetLoader

DEFAULT_JSONL = (
    "results/hf_rollout/sft_data/"
    "balanced_combined.jsonl"
)
VAL_EVERY = 10  # every 10th source_item_index -> validation


def _load_pairs(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no step-level decision pairs in {path}")
    return rows


class HFStepPairs(BaseDatasetLoader):
    """HarnessForge step-level decision pairs (training-only loader)."""

    def __init__(self, mode="eval", jsonl_path: str = None):
        super().__init__(mode)
        path = jsonl_path or os.environ.get("HF_STEP_PAIRS_JSONL", DEFAULT_JSONL)
        rows = _load_pairs(path)
        self.jsonl_path = path

        # split by source rollout (meta.source_item_index), deterministic
        indices = sorted({int(r.get("meta", {})
                             .get("source_item_index", i)) for i, r in enumerate(rows)})
        val_items = {ix for pos, ix in enumerate(indices) if pos % VAL_EVERY == 0}

        def _is_val(row):
            return int(row.get("meta", {}).get("source_item_index", -1)) in val_items

        tr = [r for r in rows if not _is_val(r)]
        va = [r for r in rows if _is_val(r)]

        self.raw_dict = {
            "train": Dataset.from_list(tr),
            "validation": Dataset.from_list(va),
            "test": Dataset.from_list(va),  # unused; interface completeness only
        }
        self.n_val_items = len(val_items)

    # --- content builders ---
    def build_prompt(self, ex):
        return ex["instruction"]

    def get_answer_string(self, ex):
        return ex["output"]

    def _messages(self, ex):
        return [{"role": "user", "content": ex["instruction"]}]

    # --- tokenization (chat template + LEFT truncation) ---
    def _tokenize_chat_left_trunc(self, tokenizer, messages, completion, max_len):
        """Render the chat turn, then truncate the PREFIX if too long.

        HarnessForge step-level pairs carry long agent histories (median ~2.3k tokens,
        p90 ~7.4k). Right truncation (the repo default) would cut the supervised
        completion and leave an all-masked sample; cutting the history head instead
        keeps the completion (mean 69 / max 537 tokens) and the most recent context.
        """
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}], tokenize=False)
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        n_prefix = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        n_prefix = min(n_prefix, len(full_ids))

        labels = [(-100 if i < n_prefix else tok) for i, tok in enumerate(full_ids)]

        if self.mode == "eval":
            return {"input_ids": full_ids[:max_len],
                    "attention_mask": [1] * len(full_ids[:max_len]),
                    "labels": full_ids[:max_len]}

        cut = max(0, len(full_ids) - max_len)
        cut = min(cut, n_prefix)  # never truncate into the completion
        input_ids = full_ids[cut:]
        labels = labels[cut:]

        pad_id = tokenizer.pad_token_id
        n_pad = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * n_pad
        labels = labels + [-100] * n_pad
        attention_mask = [1] * (max_len - n_pad) + [0] * n_pad

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

    # --- loader interface ---
    def load_tokenized_dataset(self, tokenizer, max_len: int = 4096,
                               return_test: bool = False):
        def build(ex):
            return self._tokenize_chat_left_trunc(
                tokenizer, self._messages(ex), self.get_answer_string(ex), max_len)
        return self._map_tokenized_splits(build, return_test)

    def load_tokenized_dataset_chat(self, tokenizer, max_len: int = 2048,
                                    return_test: bool = False):
        return self.load_tokenized_dataset(tokenizer, max_len, return_test)

    def load_raw_dataset(self, return_test: bool = False):
        def build(ex):
            label = str(ex.get("meta", {}).get("sample_kind", "step"))
            return self._build_raw_example(self.build_prompt(ex),
                                           self.get_answer_string(ex), label)
        return self._map_raw_splits(build, return_test)

    def load_raw_dataset_chat(self, tokenizer=None, return_test: bool = False):
        return self.load_raw_dataset(return_test)

    def evaluate(self, output, label):
        """Training-only loader: no task metric here (eval goes through run_infer.py)."""
        raise NotImplementedError(
            "HFStepPairs is a training-only loader; evaluate the policy with "
            "HarnessForge's run_infer.py instead.")
