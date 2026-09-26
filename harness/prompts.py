"""Paired prompt construction for Latent Harness Distillation (E1-A).

Roles (RESEARCH_PLAN §2.1; the scientific hard constraint is that the future student
model must NEVER receive the harness text):

- base (student) prompt: byte-identical to the WUAS `build_prompt` output — the
  exact string the E0 evaluation pipeline used. Never contains harness text.
- teacher prompt: `H_behavior + "\\n\\n" + base_prompt` — explicit harness text
  prepended, task content unchanged.

Task content (question, options, gold completion, label, answer format) is identical in
both roles by construction: prompts/labels come from the same loader call, and the gold
completion and per-dataset `evaluate` function are shared. Only the prepended harness
may differ. `PairedExample.base_prompt` vs `.teacher_prompt` makes the two roles
explicit in the API; `assert_no_leakage` is the gate future student runs must pass.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from .harness_text import Harness

# RESEARCH_PLAN §2.1: "{H_behavior}\n\n{H_base(x)}"
TEACHER_SEPARATOR = "\n\n"


def build_teacher_prompt(harness_text: Optional[str], base_prompt: str) -> str:
    """Prepend harness text to a base prompt. `harness_text=None` (base condition)
    returns the base prompt unchanged."""
    if harness_text is None:
        return base_prompt
    return harness_text + TEACHER_SEPARATOR + base_prompt


@dataclass
class PairedExample:
    """One example in both roles, sharing task content, gold completion, and label.

    base_prompt    — STUDENT input; byte-identical to the WUAS eval prompt;
                     must contain no harness text (see `assert_no_leakage`).
    teacher_prompt — TEACHER input; harness text + separator + base prompt.
    completion     — gold task completion (loader `get_answer_string`), the answer
                     format for both roles.
    label          — eval label (loader `load_raw_dataset`), scored by the unchanged
                     per-dataset `evaluate` for both roles.
    """

    example_id: str
    dataset: str
    split: str
    base_prompt: str
    teacher_prompt: str
    completion: str
    label: Any
    harness_id: Optional[str] = None
    harness_sha256: Optional[str] = None
    harness_text: Optional[str] = None

    @property
    def student_prompt(self) -> str:
        """Explicit alias for what the future student model is allowed to see."""
        return self.base_prompt

    @property
    def task_content_identical(self) -> bool:
        return self.teacher_prompt.endswith(self.base_prompt)


def assert_no_leakage(example: PairedExample) -> None:
    """Hard student-side constraint: no harness text may reach the student prompt,
    and task content must be identical across the two roles."""
    if example.harness_text and example.harness_text in example.base_prompt:
        raise AssertionError(
            f"Leakage: harness {example.harness_id} text found in the student prompt "
            f"of {example.example_id}"
        )
    if not example.teacher_prompt.endswith(example.base_prompt):
        raise AssertionError(
            f"Task content differs between teacher and student prompt in {example.example_id}"
        )


def build_paired_examples(
    dataset_name: str,
    loader,
    split: str = "validation",
    harness: Optional[Harness] = None,
    limit: Optional[int] = None,
) -> List[PairedExample]:
    """Build paired examples from a WUAS loader, reusing its pipeline unchanged.

    Prompts, gold completions, and labels come from `loader.load_raw_dataset` (the exact
    values the E0 eval pipeline consumed). Each base prompt is cross-checked against a
    fresh `loader.build_prompt(ex)` call so the student prompt is guaranteed
    byte-identical to the E0 evaluation prompt.
    """
    raw = loader.load_raw_dataset(return_test=True)[split]
    raw_examples = list(loader.raw_dict[split])
    if len(raw) != len(raw_examples):
        raise ValueError(
            f"{dataset_name}: load_raw_dataset returned {len(raw)} examples but "
            f"raw_dict['{split}'] has {len(raw_examples)}; order/length must match"
        )
    paired: List[PairedExample] = []
    for i, ((prompt, completion, label), ex) in enumerate(zip(raw, raw_examples)):
        base_prompt = loader.build_prompt(ex)
        if base_prompt != prompt:
            raise ValueError(
                f"{dataset_name}[{i}]: build_prompt output differs from the eval-pipeline "
                "prompt; paired construction must not alter task content"
            )
        teacher_prompt = build_teacher_prompt(harness.text if harness else None, base_prompt)
        paired.append(
            PairedExample(
                example_id=f"{dataset_name}:{split}:{i}",
                dataset=dataset_name,
                split=split,
                base_prompt=base_prompt,
                teacher_prompt=teacher_prompt,
                completion=completion,
                label=label,
                harness_id=harness.harness_id if harness else None,
                harness_sha256=harness.sha256 if harness else None,
                harness_text=harness.text if harness else None,
            )
        )
        if limit is not None and len(paired) >= limit:
            break
    return paired
