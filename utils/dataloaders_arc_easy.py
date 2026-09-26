"""ARC-Easy loader (E1-Eval-v3 OOD variant; RESEARCH_PLAN §4.6).

Thin subclass of `ARC_Challenge` reusing its template/`evaluate`/tokenization
UNCHANGED (the pre-registered requirement: "new thin subclass reusing the
ARC_Challenge template/evaluate"). Only the split carving differs:

- eval split = the OFFICIAL ARC-Easy test split (full, loader order) — an OOD
  diagnostic split, never used for training or selection.
- train split = the official ARC-Easy train split (unused by this project).

No existing loader, template, or `evaluate` is modified (RESEARCH_PLAN rule 6);
registration is one additive entry in `utils/utils.py::dataset_dict` under the
key "ARC_Easy".
"""

from .dataloaders import ARC_Challenge
from datasets import load_dataset


class ARC_Easy(ARC_Challenge):
    """ARC-Easy = the ARC_Challenge loader pointed at the ARC-Easy config,
    evaluated on the official test split."""

    def __init__(self, mode="eval"):
        super(ARC_Challenge, self).__init__(mode)
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
        self.raw_dict = {
            "train": ds["train"],
            "validation": ds["validation"],
            "test": ds["test"],
        }
