"""MBPP loader, sandbox, and marker-adaptation tests (v3 extension).

Pre-registered positive controls live here (RESEARCH_PLAN §9.3): (a) comment-form
decomposition steps count; (b) restatement fires on a task-text echo; (c) verify-action
patterns fire; (d) option_coverage / evidence_quote are undefined on MBPP. Sandbox
tests execute real subprocesses (one pass, one fail, one syntax error, one timeout with
a short override) — the timeout override exists for test speed only; the pipeline
default is the pre-registered 10 s.

Run: python -m unittest tests.test_mbpp -v   (dataset download required on first use)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataloaders_mbpp import (
    DEV_POOL_N,
    EVAL_TEST_N,
    FEWSHOT_N,
    TRAIN_POOL_N,
    MBPP,
    METHOD_BEGIN_DONE,
    METHOD_BEGIN_OPEN,
    METHOD_DONE_ONLY,
    METHOD_FENCED,
    extract_mbpp_code,
    normalize_code,
    run_mbpp_sandbox,
    unshare_available,
)
from harness.markers import (
    clean_stop_present,
    compute_example_markers,
    decomposition_step_count,
    interface_alignment_present,
    option_coverage,
    passage_from_prompt,
    question_and_options_from_prompt,
    restatement_present,
    single_task_block_present,
    verification_move_count,
)


class TestExtraction(unittest.TestCase):
    def test_begin_done_body(self):
        code, method = extract_mbpp_code("preamble\n[BEGIN]\ndef f():\n    return 1\n[DONE]\nafter")
        self.assertEqual(code, "def f():\n    return 1")
        self.assertEqual(method, METHOD_BEGIN_DONE)

    def test_begin_open_body(self):
        code, method = extract_mbpp_code("[BEGIN]\ndef f():\n    return 1\n")
        self.assertEqual(code, "def f():\n    return 1")
        self.assertEqual(method, METHOD_BEGIN_OPEN)

    def test_fenced_block(self):
        code, method = extract_mbpp_code("Here:\n```python\ndef f():\n    return 1\n```")
        self.assertEqual(code, "def f():\n    return 1")
        self.assertEqual(method, METHOD_FENCED)

    def test_done_only_canonical_continuation(self):
        """Eval pipelines score the continuation only (prompt ends at [BEGIN]); the
        canonical well-formed output has no leading [BEGIN] marker."""
        code, method = extract_mbpp_code("def f():\n    return 1\n[DONE]\nextra rambling")
        self.assertEqual(code, "def f():\n    return 1")
        self.assertEqual(method, METHOD_DONE_ONLY)

    def test_surrounding_fences_stripped_inside_body(self):
        code, method = extract_mbpp_code("[BEGIN]\n```python\nx = 1\n```\n[DONE]")
        self.assertEqual(code, "x = 1")
        self.assertEqual(method, METHOD_BEGIN_DONE)

    def test_no_code_is_none(self):
        for out in ("", "no code here at all", "[BEGIN]\n   \n[DONE]"):
            code, method = extract_mbpp_code(out)
            self.assertIsNone(code)
            self.assertIsNone(method)


class TestSandbox(unittest.TestCase):
    """Real subprocess execution (network isolation verified as available)."""

    @classmethod
    def setUpClass(cls):
        cls.network_ns = unshare_available()

    def test_pass(self):
        res = run_mbpp_sandbox("def add(a, b):\n    return a + b\n",
                               ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
        self.assertTrue(res["all_pass"])
        self.assertEqual(res["n_pass"], 2)
        self.assertFalse(res["timed_out"])
        self.assertIsNone(res["compile_error"])
        self.assertIsNone(res["exec_error"])
        self.assertIn("isolation", res["sandbox"])

    def test_fail_assert(self):
        res = run_mbpp_sandbox("def add(a, b):\n    return a - b\n", ["assert add(1, 2) == 3"])
        self.assertFalse(res["all_pass"])
        self.assertEqual(res["n_pass"], 0)
        self.assertIn("AssertionError", res["asserts"][0]["error"])

    def test_syntax_error(self):
        res = run_mbpp_sandbox("def add(a, b)\n    return a + b\n", ["assert add(1, 2) == 3"])
        self.assertFalse(res["all_pass"])
        self.assertIn("SyntaxError", res["compile_error"])

    def test_runtime_error(self):
        res = run_mbpp_sandbox("def f(x):\n    return x/0\n", ["assert f(1) == 1"])
        self.assertFalse(res["all_pass"])
        self.assertIsNone(res["compile_error"])
        self.assertIn("ZeroDivisionError", res["asserts"][0]["error"])

    def test_timeout_short_override(self):
        res = run_mbpp_sandbox("while True:\n    pass\n", ["assert True"], timeout_s=3)
        self.assertFalse(res["all_pass"])
        self.assertTrue(res["timed_out"])

    def test_network_blocked(self):
        # socket.socket raising -> the body's except branch fires SystemExit(7);
        # if networking worked, the body would reach SystemExit(0) instead.
        body = ("import socket\n"
                "try:\n"
                "    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "    raise SystemExit(0)\n"
                "except Exception as e:\n"
                "    raise SystemExit(7)\n")
        res = run_mbpp_sandbox(body, ["assert True"])
        self.assertFalse(res["all_pass"])
        self.assertIn("SystemExit: 7", str(res.get("exec_error", "")))
        self.assertNotIn("SystemExit: 0", str(res.get("exec_error", "")))
        if self.network_ns:
            self.assertIn("unshare", res["sandbox"]["isolation"])

    def test_setup_code_prepended(self):
        res = run_mbpp_sandbox("def f(x):\n    return x * SCALE\n",
                               ["assert f(2) == 8"], test_setup_code="SCALE = 4")
        self.assertTrue(res["all_pass"])


class TestLoader(unittest.TestCase):
    """Loads the real dataset (network download on first use)."""

    @classmethod
    def setUpClass(cls):
        cls.loader = MBPP(mode="eval")

    def test_split_sizes(self):
        self.assertEqual(len(self.loader.raw_dict["train"]), TRAIN_POOL_N)
        self.assertEqual(len(self.loader.raw_dict["validation"]), DEV_POOL_N)
        self.assertEqual(len(self.loader.raw_dict["test"]), EVAL_TEST_N)
        self.assertEqual(FEWSHOT_N, 3)
        # dev pool = the END of official train, training pool = the rest
        self.assertEqual(self.loader.fewshot_task_ids, [1, 2, 3])

    def test_prompt_format(self):
        ex = self.loader.raw_dict["train"][0]
        prompt = self.loader.build_prompt(ex)
        self.assertIn("You are an expert Python programmer, and here is your task:", prompt)
        self.assertIn("Your code should pass these tests:", prompt)
        # target block ends at [BEGIN]; the 3-shot examples each carry [DONE]
        self.assertTrue(prompt.rstrip("\n").endswith("[BEGIN]"))
        self.assertEqual(prompt.count("[BEGIN]"), 4)
        self.assertEqual(prompt.count("[DONE]"), 3)
        # every test_list assert of the target problem is shown
        for a in ex["test_list"]:
            self.assertIn(a.strip(), prompt)

    def test_label_carries_tests(self):
        triples = self.loader.load_raw_dataset(return_test=True)["train"]
        label = triples[0][2]
        self.assertIsInstance(label, dict)
        self.assertEqual(len(label["test_list"]), 3)
        self.assertIn("test_setup_code", label)

    def test_evaluate_gold_completion_passes(self):
        """Sanity: the reference solution passes its own tests in the sandbox."""
        ex = self.loader.raw_dict["train"][1]
        completion = self.loader.get_answer_string(ex)
        self.assertTrue(completion.endswith("[DONE]"))
        self.assertTrue(self.loader.evaluate(completion, {"test_list": ex["test_list"],
                                                          "test_setup_code": ex["test_setup_code"]}))

    def test_evaluate_extraction_failure_is_incorrect(self):
        ex = self.loader.raw_dict["train"][1]
        label = {"test_list": ex["test_list"], "test_setup_code": ""}
        self.assertFalse(self.loader.evaluate("no code at all", label))
        self.assertFalse(self.loader.evaluate("", label))

    def test_evaluate_wrong_code_fails(self):
        ex = self.loader.raw_dict["train"][1]
        label = {"test_list": ex["test_list"], "test_setup_code": ""}
        self.assertFalse(self.loader.evaluate("[BEGIN]\ndef similar_elements(a, b):\n    return ()\n[DONE]", label))

    def test_paired_prompts_and_leakage(self):
        from harness.harness_text import load_registry
        from harness.prompts import assert_no_leakage, build_paired_examples
        registry = load_registry()
        harness = registry["DECOMPOSE-001"]
        paired = build_paired_examples("MBPP", self.loader, split="train",
                                       harness=harness, limit=2)
        self.assertEqual(len(paired), 2)
        for p in paired:
            assert_no_leakage(p)
            self.assertTrue(p.teacher_prompt.endswith(p.base_prompt))
            self.assertNotIn(harness.text, p.base_prompt)
            self.assertTrue(p.teacher_prompt.startswith(harness.text))


class TestMarkerAdaptation(unittest.TestCase):
    """Pre-registered MBPP marker positive/negative controls (RESEARCH_PLAN §9.3)."""

    PROMPT = ("You are an expert Python programmer, and here is your task: "
              "Write a function to find the similar elements from the given two tuple lists.\n"
              "Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n\n"
              "[BEGIN]\n")

    def test_comment_form_decomposition_steps_count(self):
        out = "[BEGIN]\n# Step 1: intersect the sets\n# Step 2: tuple the result\ndef f():\n    return ()\n[DONE]"
        self.assertEqual(decomposition_step_count(out), 2)

    def test_verify_action_fires(self):
        out = "# re-check the result\nVerification: the intersection is correct.\n[BEGIN]\nx = 1\n[DONE]"
        self.assertGreaterEqual(verification_move_count(out), 2)
        markers = compute_example_markers("MBPP", lambda o, l: True, out, None, prompt=self.PROMPT)
        self.assertTrue(markers["verify_action"])

    def test_restatement_uses_task_text(self):
        question, options = question_and_options_from_prompt("MBPP", self.PROMPT)
        self.assertEqual(question,
                         "Write a function to find the similar elements from the given two tuple lists.")
        self.assertEqual(options, [])
        echo = ("The task: write a function to find the similar elements from the given two "
                "tuple lists.\n[BEGIN]\nx = 1\n[DONE]")
        markers = compute_example_markers("MBPP", lambda o, l: True, echo, None, prompt=self.PROMPT)
        self.assertTrue(markers["restatement_present"])
        markers_none = compute_example_markers(
            "MBPP", lambda o, l: True, "[BEGIN]\nx = 1\n[DONE]", None, prompt=self.PROMPT)
        self.assertFalse(markers_none["restatement_present"])

    def test_option_coverage_and_passage_undefined(self):
        out = "[BEGIN]\nx = 1\n[DONE]"
        markers = compute_example_markers("MBPP", lambda o, l: True, out, None, prompt=self.PROMPT)
        self.assertIsNone(markers["option_coverage"])
        self.assertIsNone(markers["evidence_quote"])
        self.assertIsNone(passage_from_prompt("MBPP", self.PROMPT))
        self.assertEqual(option_coverage(out, []), None)


class TestTestconsultMarkers(unittest.TestCase):
    """TESTCONSULT-001 markers (E1-B-v3, additive): positive/negative controls plus the
    frozen-draw baseline pin (pre-registered BASE rates on the E1-0-MBPP draw:
    interface-alignment 30/193, clean-stop 0/200, single-task-block 19/200)."""

    PROMPT = TestMarkerAdaptation.PROMPT  # asserts call `similar_elements`

    def test_interface_alignment_positive(self):
        out = "def similar_elements(a, b):\n    return set(a) & set(b)\n[DONE]"
        self.assertTrue(interface_alignment_present(out, self.PROMPT))

    def test_interface_alignment_negative_wrong_name(self):
        out = "def plus(a, b):\n    return a + b\n[DONE]"
        self.assertFalse(interface_alignment_present(out, self.PROMPT))

    def test_interface_alignment_unextractable_is_none(self):
        self.assertIsNone(interface_alignment_present("no code here", self.PROMPT))
        self.assertIsNone(interface_alignment_present("code", None))

    def test_clean_stop(self):
        self.assertTrue(clean_stop_present("x = 1\n[DONE]\n"))
        self.assertFalse(clean_stop_present("x = 1\n[DONE]\n\nYou are an expert Python "
                                            "programmer, and here is your task: x"))
        self.assertFalse(clean_stop_present("x = 1"))

    def test_single_task_block(self):
        self.assertTrue(single_task_block_present("def f():\n    return 1\n[DONE]\n"))
        self.assertFalse(single_task_block_present(
            "def f():\n    return 1\n[DONE]\n\nYou are an expert Python programmer, "
            "and here is your task: y\nYour code should pass these tests:\n\nassert g()\n\n[BEGIN]\n"))

    def test_compute_example_markers_wiring(self):
        good = "def similar_elements(a, b):\n    return set(a) & set(b)\n[DONE]"
        m = compute_example_markers("MBPP", lambda o, l: True, good, None, prompt=self.PROMPT)
        self.assertTrue(m["interface_alignment"])
        self.assertTrue(m["clean_stop"])
        self.assertTrue(m["single_task_block"])
        # undefined (None) on every non-MBPP dataset; clean_stop is dataset-branched
        # since OI-28 / E1-B-PERM: the generic Answer-line-anchored variant on
        # no-assert datasets (MBPP's [DONE]-anchored counter unchanged)
        m_b = compute_example_markers("BoolQ", lambda o, l: True, good, None, prompt="Question: q?")
        self.assertIsNone(m_b["interface_alignment"])
        self.assertFalse(m_b["clean_stop"])  # generic variant: no Answer: line in output
        self.assertFalse(m_b["single_answer_line"])
        self.assertIsNone(m_b["single_task_block"])

    def test_frozen_draw_baselines_reproduced(self):
        import json
        import os
        raw = "data/error_analysis/raw/MBPP_train_base.jsonl"
        if not os.path.exists(raw):
            self.skipTest("frozen E1-0-MBPP draw not present")
        recs = [json.loads(l) for l in open(raw, encoding="utf-8") if l.strip()]
        n_ia = n_ext = n_cs = n_stb = 0
        for r in recs:
            ia = interface_alignment_present(r["output"], r["prompt"])
            n_ext += ia is not None
            n_ia += bool(ia)
            n_cs += clean_stop_present(r["output"])
            n_stb += single_task_block_present(r["output"])
        self.assertEqual((n_ia, n_ext), (30, 193))
        self.assertEqual(n_cs, 0)
        self.assertEqual(n_stb, 19)


if __name__ == "__main__":
    unittest.main()
