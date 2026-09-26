"""Paired prompt tests on GSM8K, ARC_Challenge (registry key 'ARC'), and BoolQ.

Verifies, per (dataset, harness) cell:
  - base prompt byte-identical to the WUAS eval prompt (build_prompt output)
  - identical task content / labels / gold completions / answer formatting
  - only the prepended harness text differs between teacher and student prompts
  - leakage: no harness text (full text or any sentence of it) in the student prompt
  - the harness hash used matches the registry hash over that text
  - base condition: teacher prompt == base prompt
"""

import hashlib
import unittest

from harness.harness_text import load_registry, resolve_harness
from harness.prompts import (
    TEACHER_SEPARATOR,
    PairedExample,
    assert_no_leakage,
    build_paired_examples,
)
from utils import dataset_dict

# utils.dataset_dict keys: 'ARC' is ARC_Challenge (README value 'ARC_Challenge'
# crashes — E0_LOG D1). All three required datasets are covered.
DATASETS = ["GSM8K", "ARC", "BoolQ"]
N_EXAMPLES = 5


class TestPairedPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.harnesses = {
            "verify": resolve_harness("verify", cls.registry),
            # full id: the bare short form is ambiguous since the E4 mutants
            # (DECOMPOSE-002/003) joined the registry
            "decompose": resolve_harness("DECOMPOSE-001", cls.registry),
        }
        cls.loaders = {}
        cls.paired = {}
        cls.raw = {}
        for name in DATASETS:
            cls.loaders[name] = dataset_dict[name](mode="eval")
            raw_splits = cls.loaders[name].load_raw_dataset(return_test=True)
            cls.raw[name] = raw_splits["validation"]
            for key, harness in cls.harnesses.items():
                cls.paired[(name, key)] = build_paired_examples(
                    name,
                    cls.loaders[name],
                    split="validation",
                    harness=harness,
                    limit=N_EXAMPLES,
                )
            cls.paired[(name, "base")] = build_paired_examples(
                name, cls.loaders[name], split="validation", harness=None, limit=N_EXAMPLES
            )

    # -- identical task content ------------------------------------------------

    def test_base_prompt_byte_identical_to_eval_pipeline_prompt(self):
        for name in DATASETS:
            for key in ("verify", "decompose", "base"):
                for i, p in enumerate(self.paired[(name, key)]):
                    self.assertEqual(
                        p.base_prompt,
                        self.loaders[name].build_prompt(self.loaders[name].raw_dict["validation"][i]),
                        f"{name}/{key}[{i}]: student prompt is not the eval-pipeline prompt",
                    )

    def test_teacher_prompt_is_harness_plus_identical_task_content(self):
        for name in DATASETS:
            for key in ("verify", "decompose"):
                h = self.harnesses[key]
                for p in self.paired[(name, key)]:
                    self.assertEqual(p.teacher_prompt, h.text + TEACHER_SEPARATOR + p.base_prompt)
                    self.assertTrue(p.task_content_identical)
                    self.assertTrue(p.teacher_prompt.endswith(p.base_prompt))

    def test_only_harness_text_differs(self):
        for name in DATASETS:
            for key in ("verify", "decompose"):
                h = self.harnesses[key]
                for p in self.paired[(name, key)]:
                    prefix = p.teacher_prompt[: -len(p.base_prompt)]
                    self.assertEqual(prefix, h.text + TEACHER_SEPARATOR)

    def test_labels_and_completions_identical_across_roles_and_to_pipeline(self):
        for name in DATASETS:
            for i, p in enumerate(self.paired[(name, "verify")]):
                prompt, completion, label = self.raw[name][i]
                self.assertEqual(p.base_prompt, prompt)
                self.assertEqual(p.completion, completion)
                self.assertEqual(p.label, label)
                d = self.paired[(name, "decompose")][i]
                b = self.paired[(name, "base")][i]
                # one shared label / completion per example across every role
                self.assertEqual((d.label, d.completion), (p.label, p.completion))
                self.assertEqual((b.label, b.completion), (p.label, p.completion))

    def test_answer_formatting_unchanged(self):
        # The gold completion (the answer format both roles are held to) scores
        # correct under the unchanged per-dataset evaluate function.
        for name in DATASETS:
            for key in ("verify", "decompose", "base"):
                for p in self.paired[(name, key)]:
                    self.assertTrue(
                        self.loaders[name].evaluate(p.completion, p.label),
                        f"{name}/{key}[{p.example_id}]: gold completion fails evaluate()",
                    )

    def test_gold_completion_matches_get_answer_string(self):
        for name in DATASETS:
            for i, p in enumerate(self.paired[(name, "verify")]):
                ex = self.loaders[name].raw_dict["validation"][i]
                self.assertEqual(p.completion, self.loaders[name].get_answer_string(ex))

    # -- leakage / role distinction --------------------------------------------

    def test_no_harness_text_in_student_prompt(self):
        for name in DATASETS:
            for key in ("verify", "decompose"):
                h = self.harnesses[key]
                for p in self.paired[(name, key)]:
                    self.assertNotIn(h.text, p.base_prompt)
                    # sentence-level check: no fragment of the harness leaks either
                    for sentence in h.text.split(". "):
                        sentence = sentence.strip()
                        if len(sentence) >= 12:
                            self.assertNotIn(sentence, p.base_prompt,
                                             f"leak in {p.example_id}: {sentence!r}")

    def test_teacher_and_student_inputs_distinguishable(self):
        for name in DATASETS:
            for key in ("verify", "decompose"):
                for p in self.paired[(name, key)]:
                    self.assertNotEqual(p.base_prompt, p.teacher_prompt)
                    self.assertEqual(p.student_prompt, p.base_prompt)
                    assert_no_leakage(p)  # must not raise

    def test_assert_no_leakage_raises_on_contamination(self):
        for name in DATASETS:
            p = self.paired[(name, "verify")][0]
            contaminated = PairedExample(**{**p.__dict__, "base_prompt": p.harness_text + p.base_prompt})
            with self.assertRaises(AssertionError):
                assert_no_leakage(contaminated)
            altered_task = PairedExample(**{**p.__dict__, "teacher_prompt": p.teacher_prompt + " CHANGED"})
            with self.assertRaises(AssertionError):
                assert_no_leakage(altered_task)

    # -- base condition ----------------------------------------------------------

    def test_base_condition_produces_no_harness_text(self):
        for name in DATASETS:
            for p in self.paired[(name, "base")]:
                self.assertIsNone(p.harness_id)
                self.assertIsNone(p.harness_text)
                self.assertEqual(p.teacher_prompt, p.base_prompt)
                self.assertEqual(p.student_prompt, p.teacher_prompt)
                assert_no_leakage(p)

    # -- registry hash integrity --------------------------------------------------

    def test_harness_hash_in_registry_matches_text_used(self):
        for name in DATASETS:
            for key in ("verify", "decompose"):
                h = self.harnesses[key]
                for p in self.paired[(name, key)]:
                    self.assertEqual(p.harness_id, h.harness_id)
                    self.assertEqual(p.harness_sha256, self.registry[h.harness_id].sha256)
                    self.assertEqual(p.harness_text, self.registry[h.harness_id].text)
                    self.assertEqual(
                        hashlib.sha256(p.harness_text.encode("utf-8")).hexdigest(),
                        p.harness_sha256,
                    )


if __name__ == "__main__":
    unittest.main()
