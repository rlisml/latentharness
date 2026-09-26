"""Leakage guards for the evaluators (clean distribution).

The upstream multi-arm HF evaluator (EXPLICIT / MEAN_DIFF / SOFT_PROMPT_TUNING /
WUAS_TASK_ADAPTER arms, plus distillation-era arm infrastructure) is out of scope
for this clean distribution. Only the two shared assertion helpers are kept here,
with bodies extracted verbatim from the upstream `harness/eval_arms.py`, so that
`harness/eval_vllm.py` and `harness/eval_easysteer.py` keep byte-identical
assertion behavior (the student-side leakage gate and the explicit-arm positive
control).
"""

from typing import List, Optional


def assert_no_harness_leak(texts: List[str], harness_text: Optional[str],
                           context: str) -> None:
    if not harness_text:
        return
    for t in texts:
        if harness_text in t:
            at = t.find(harness_text)
            raise AssertionError(
                f"LEAKAGE ({context}): harness text found in a student input at char "
                f"{at}; context: ...{t[max(0, at - 60):at + 40]!r}...")


def assert_harness_present(texts: List[str], harness_text: str, context: str) -> None:
    for t in texts:
        if harness_text not in t:
            raise AssertionError(f"POSITIVE-CONTROL FAILED ({context}): harness text "
                                 "missing from an explicit-arm prompt")
