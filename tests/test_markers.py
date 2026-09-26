"""Marker counter tests (heuristic/regex, RESEARCH_PLAN §2.3).

Covers: verify-action counting, decomposition step counting, restatement presence,
revision detection, error-correction-rate aggregation over initially-wrong cases,
per-example logging, and the leakage rule that marker regexes must NOT fire on bare
H_base prompt echoes (RESEARCH_PLAN §6 rule 7). Prompt-echo checks use synthetic
examples built through the real build_prompt templates, so no dataset download is
needed here.
"""

import json
import tempfile
import unittest
from pathlib import Path

from harness.markers import (
    answer_line_present,
    arithmetic_expression_count,
    clean_stop_generic_present,
    clean_stop_present,
    compute_example_markers,
    count_correction_cues,
    decomposition_step_count,
    evidence_quote_present,
    log_markers,
    option_coverage,
    option_labels_from_prompt,
    passage_from_prompt,
    question_and_options_from_prompt,
    restatement_present,
    revision_event,
    single_answer_line_present,
    summarize_markers,
    verification_move_count,
)
from utils.dataloaders import ARC_Challenge, BoolQ, GSM8K

VERIFY_STYLE_OUTPUT = (
    "She sold 48/2 = 24 clips in May. In total she sold 24 + 48 = 72 clips. "
    "Verification: 72 = 24 + 48, and 48/2 = 24, so the total checks out. Answer: 72"
)
DECOMPOSE_STYLE_OUTPUT = (
    "Step 1: clips sold in May -> 48/2 = 24\n"
    "Step 2: total clips -> 24 + 48 = 72\n"
    "Answer: 72"
)
BASE_STYLE_OUTPUT = "48/2 = 24. 24 + 48 = 72. Answer: 72."
REVISED_OUTPUT = (
    "6 x 7 = 42, so the answer is 42. Verification: re-checking the question, it asks "
    "for 6 x 8, not 6 x 7. 6 x 8 = 48. Answer: 48"
)
CONSISTENT_OUTPUT = (
    "6 x 7 = 42, so the answer is 42. Verification: 6 x 7 = 42, that is correct. "
    "Answer: 42"
)


class TestVerificationMoves(unittest.TestCase):
    def test_verify_style_output_counts_moves(self):
        self.assertGreaterEqual(verification_move_count(VERIFY_STYLE_OUTPUT), 1)

    def test_verification_segment_counted(self):
        self.assertEqual(verification_move_count("The answer is 5. Verification: 2+3 = 5. Answer: 5"), 1)

    def test_recheck_phrases_counted(self):
        self.assertGreaterEqual(verification_move_count("Let me double-check: is the answer consistent with the steps?"), 2)

    def test_plain_output_has_no_moves(self):
        self.assertEqual(verification_move_count(BASE_STYLE_OUTPUT), 0)
        self.assertEqual(verification_move_count(""), 0)

    def test_reverify_counts_once(self):
        # non-overlapping counting: 're-verify' is one move, not two
        self.assertEqual(verification_move_count("I will re-verify the total. Answer: 4"), 1)


class TestDecompositionSteps(unittest.TestCase):
    def test_decompose_style_output_counts_steps(self):
        self.assertEqual(decomposition_step_count(DECOMPOSE_STYLE_OUTPUT), 2)

    def test_mandated_format_counted_case_insensitive(self):
        self.assertEqual(decomposition_step_count("step 1: a -> 1\nSTEP 2: b -> 2"), 2)

    def test_plain_output_has_no_steps(self):
        self.assertEqual(decomposition_step_count(BASE_STYLE_OUTPUT), 0)
        self.assertEqual(decomposition_step_count("First do this, then that. Answer: 3"), 0)


class TestRestatement(unittest.TestCase):
    QUESTION = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."

    def test_question_echo_counts_as_restatement(self):
        out = ("The question says: Natalia sold clips to 48 of her friends in April, and then "
               "she sold half as many clips in May. So 48/2 = 24. Answer: 72")
        self.assertTrue(restatement_present(out, question=self.QUESTION))

    def test_partial_echo_of_fragment_window_counts(self):
        out = "Natalia sold clips to 48 of her friends in april and then. 48/2 = 24. Answer: 72"
        self.assertTrue(restatement_present(out, question=self.QUESTION))

    def test_no_echo_no_restatement(self):
        self.assertFalse(restatement_present(BASE_STYLE_OUTPUT, question=self.QUESTION))

    def test_none_question_returns_none(self):
        self.assertIsNone(restatement_present("anything", question=None))

    def test_option_enumeration_counts_as_restatement(self):
        out = "Option A says ice, Option B says water vapor. Answer: (B)"
        self.assertTrue(restatement_present(out, question="What is water vapor made of?",
                                            options=["ice", "water vapor"]))
        self.assertFalse(restatement_present("Answer: (B)", question="q", options=["ice", "water vapor"]))

    def test_restatement_checked_before_final_answer_only(self):
        q = self.QUESTION
        out = "48/2 = 24, 24 + 48 = 72. Answer: 72. The question mentioned April."
        # question text appears only after the final Answer: -> not restatement
        self.assertFalse(restatement_present(out, question=q))


class TestRevision(unittest.TestCase):
    def test_revised_answer_detected(self):
        self.assertTrue(revision_event(REVISED_OUTPUT))

    def test_consistent_answer_not_flagged(self):
        self.assertFalse(revision_event(CONSISTENT_OUTPUT))

    def test_intermediate_results_not_flagged(self):
        out = ("Step 1: half of 48 -> 24\nStep 2: 24 + 48 -> 72\n"
               "The result of step 2 is 72. Answer: 72")
        self.assertFalse(revision_event(out))

    def test_plain_output_not_flagged(self):
        self.assertFalse(revision_event(BASE_STYLE_OUTPUT))
        self.assertFalse(revision_event(""))

    def test_correction_cues_counted_separately(self):
        # cues alone must not produce a revision event (keyword insertion cannot game it)
        out = "wait, let me think again. 48/2 = 24. Answer: 72"
        self.assertGreaterEqual(count_correction_cues(out), 1)
        self.assertFalse(revision_event(out))

    def test_truncated_final_answer_line_is_not_a_revision(self):
        # cap-truncated output ending in a bare 'Answer:' has no stated final answer;
        # the repeated identical answers before it must not count as a revision
        out = "Answer: A\nsome text\nAnswer: A\nsome text\nAnswer:"
        self.assertFalse(revision_event(out))

    def test_empty_earlier_statement_skipped(self):
        out = "Answer:\nmore text\nAnswer: 72"
        self.assertFalse(revision_event(out))


class TestPromptEchoInert(unittest.TestCase):
    """Marker regexes must not fire on bare H_base prompts (RESEARCH_PLAN §6 rule 7):
    a degenerate model that echoes its prompt must score zero on every marker."""

    SYNTHETIC_EXAMPLES = {
        "GSM8K": (
            GSM8K,
            {"question": "Tom has 3 boxes with 4 pens each. How many pens in total?",
             "answer": "3 boxes with 4 pens each means 3 x 4. #### 12"},
        ),
        "ARC": (
            ARC_Challenge,
            {"question": "Which property of a mineral can be determined by scratching it?",
             "choices": {"text": ["color", "hardness", "mass", "smell"], "label": ["A", "B", "C", "D"]},
             "answerKey": "B"},
        ),
        "BoolQ": (
            BoolQ,
            {"passage": "The town library opened in 1902 and moved to its current building in 1975.",
             "question": "did the town library move in 1975",
             "answer": True},
        ),
    }

    def test_markers_inert_on_base_prompt_echo(self):
        for name, (loader_cls, ex) in self.SYNTHETIC_EXAMPLES.items():
            loader = loader_cls(mode="eval")
            prompt = loader.build_prompt(ex)
            self.assertEqual(verification_move_count(prompt), 0, name)
            self.assertEqual(decomposition_step_count(prompt), 0, name)
            self.assertFalse(revision_event(prompt), name)
            self.assertEqual(count_correction_cues(prompt), 0, name)


class TestQuestionRecovery(unittest.TestCase):
    def test_gsm8k_question_is_whole_prompt(self):
        prompt = GSM8K(mode="eval").build_prompt(
            {"question": "Tom has 3 boxes with 4 pens each. How many pens in total?"})
        q, options = question_and_options_from_prompt("GSM8K", prompt)
        self.assertEqual(q, "Tom has 3 boxes with 4 pens each. How many pens in total?")
        self.assertEqual(options, [])

    def test_arc_question_and_options_recovered(self):
        ex = {"question": "Which property of a mineral can be determined by scratching it?",
              "choices": {"text": ["color", "hardness", "mass", "smell"], "label": ["A", "B", "C", "D"]},
              "answerKey": "B"}
        prompt = ARC_Challenge(mode="eval").build_prompt(ex)
        q, options = question_and_options_from_prompt("ARC", prompt)
        self.assertEqual(q, ex["question"])
        self.assertEqual(options, ex["choices"]["text"])

    def test_boolq_question_recovered(self):
        ex = {"passage": "The town library opened in 1902.",
              "question": "did the town library move in 1975", "answer": True}
        prompt = BoolQ(mode="eval").build_prompt(ex)
        q, options = question_and_options_from_prompt("BoolQ", prompt)
        self.assertEqual(q, ex["question"])
        self.assertEqual(options, [])


COMPUTE_STYLE_OUTPUT = (
    "The total is 48/2 = 24 for May and 24 + 48 = 72 overall. Answer: 72"
)
COMPUTE_ONE_STEP_OUTPUT = "48/2 = 24. Answer: 24"
ELIMINATE_STYLE_OUTPUT = (
    "(A) ice does not fit. (B) hardness fits the question. (C) and (D) do not survive. "
    "Answer: (B) hardness"
)
BOOLQ_PROMPT = BoolQ(mode="eval").build_prompt(
    {"passage": "The town library opened in 1902 and moved to its current building in 1975.",
     "question": "did the town library move in 1975", "answer": True}
)
ARC_PROMPT = ARC_Challenge(mode="eval").build_prompt(
    {"question": "Which property of a mineral can be determined by scratching it?",
     "choices": {"text": ["color", "hardness", "mass", "smell"], "label": ["A", "B", "C", "D"]},
     "answerKey": "B"}
)


class TestAnswerLine(unittest.TestCase):
    def test_answer_marker_detected(self):
        self.assertTrue(answer_line_present(COMPUTE_STYLE_OUTPUT))

    def test_missing_answer_line(self):
        self.assertFalse(answer_line_present("48/2 = 24, so 24 clips in May."))
        self.assertFalse(answer_line_present(""))


class TestArithmeticExpressions(unittest.TestCase):
    def test_two_expressions_trigger_event(self):
        self.assertEqual(arithmetic_expression_count(COMPUTE_STYLE_OUTPUT), 2)
        self.assertTrue(arithmetic_expression_count(COMPUTE_STYLE_OUTPUT) >= 2)

    def test_one_expression_no_event(self):
        self.assertEqual(arithmetic_expression_count(COMPUTE_ONE_STEP_OUTPUT), 1)

    def test_counted_before_final_answer_only(self):
        # an equals-number after the final Answer marker must not count
        self.assertEqual(arithmetic_expression_count("Answer: 72. Check: 24 + 48 = 72"), 0)

    def test_plain_prose_no_event(self):
        self.assertEqual(arithmetic_expression_count("She sold half as many in May. Answer: 24"), 0)


class TestOptionCoverage(unittest.TestCase):
    def test_all_labels_mentioned(self):
        self.assertTrue(option_coverage(ELIMINATE_STYLE_OUTPUT, ["A", "B", "C", "D"]))

    def test_missing_label_not_covered(self):
        partial = "(A) ice does not fit. (B) hardness fits. Answer: (B) hardness"
        self.assertFalse(option_coverage(partial, ["A", "B", "C", "D"]))

    def test_letter_forms_recognized(self):
        out = "A) no. B: no. Option C maybe. D. no. Answer: (C)"
        self.assertTrue(option_coverage(out, ["A", "B", "C", "D"]))

    def test_article_a_is_not_a_mention(self):
        # lowercase standalone 'a' must not count as label A mention
        out = "It was a long shot for B and C. Answer: (B)"
        self.assertFalse(option_coverage(out, ["A", "B", "C"]))

    def test_no_options_returns_none(self):
        self.assertIsNone(option_coverage(ELIMINATE_STYLE_OUTPUT, []))
        self.assertIsNone(option_coverage(ELIMINATE_STYLE_OUTPUT, None))

    def test_labels_recovered_from_arc_prompt(self):
        self.assertEqual(option_labels_from_prompt("ARC", ARC_PROMPT), ["A", "B", "C", "D"])
        self.assertEqual(option_labels_from_prompt("GSM8K", "Some question?\n"), [])

    def test_mention_checked_before_final_answer_only(self):
        out = "(A) no. (B) no. (C) no. (D) no. Answer: (B) hardness. Reconsidering (C) and (D) later."
        self.assertTrue(option_coverage(out, ["A", "B", "C", "D"]))


class TestEvidenceQuote(unittest.TestCase):
    PASSAGE = "The town library opened in 1902 and moved to its current building in 1975."

    def test_verbatim_span_detected(self):
        out = ("The passage states: The town library opened in 1902 and moved to its "
               "current building in 1975. So true. Answer: true.")
        self.assertTrue(evidence_quote_present(out, self.PASSAGE))

    def test_paraphrase_not_detected(self):
        out = "The library moved in 1975 according to the passage. Answer: true."
        self.assertFalse(evidence_quote_present(out, self.PASSAGE))

    def test_quote_after_answer_not_counted(self):
        out = "Answer: true. Quote: The town library opened in 1902 and moved to its current building in 1975."
        self.assertFalse(evidence_quote_present(out, self.PASSAGE))

    def test_no_passage_returns_none(self):
        self.assertIsNone(evidence_quote_present("any output", None))

    def test_passage_recovered_from_boolq_prompt(self):
        self.assertEqual(passage_from_prompt("BoolQ", BOOLQ_PROMPT), self.PASSAGE)
        self.assertIsNone(passage_from_prompt("GSM8K", "question?\n"))


class TestNewMarkersInCompute(unittest.TestCase):
    def test_compute_example_markers_includes_new_fields(self):
        rec = compute_example_markers("ARC", _dummy_evaluate, ELIMINATE_STYLE_OUTPUT, "B",
                                      prompt=ARC_PROMPT)
        self.assertTrue(rec["option_coverage"])
        self.assertFalse(rec["evidence_quote"])  # ARC prompt has no passage -> defined, False
        rec_gsm = compute_example_markers("GSM8K", _dummy_evaluate, COMPUTE_STYLE_OUTPUT, "72",
                                          prompt="Q?\n")
        self.assertTrue(rec_gsm["answer_line_present"])
        self.assertEqual(rec_gsm["arithmetic_expression_count"], 2)
        self.assertTrue(rec_gsm["arithmetic_expression_event"])
        self.assertIsNone(rec_gsm["option_coverage"])  # no options in GSM8K prompt
        self.assertIsNone(rec_gsm["evidence_quote"])

    def test_summary_includes_new_rates(self):
        recs = [
            compute_example_markers("GSM8K", _dummy_evaluate, COMPUTE_STYLE_OUTPUT, "72", prompt="Q?\n"),
            compute_example_markers("GSM8K", _dummy_evaluate, COMPUTE_ONE_STEP_OUTPUT, "24", prompt="Q?\n"),
        ]
        s = summarize_markers(recs)
        self.assertAlmostEqual(s["answer_line_rate"], 1.0)
        self.assertAlmostEqual(s["arithmetic_expression_rate"], 0.5)
        self.assertAlmostEqual(s["mean_arithmetic_expressions"], 1.5)
        self.assertIsNone(s["option_coverage_rate"])  # undefined on GSM8K
        self.assertIsNone(s["evidence_quote_rate"])


class TestSingletaskMarkers(unittest.TestCase):
    """SINGLETASK-001/002 positive controls (E1-B-PERM, OI-28): stop-clause behavior,
    invented-options behavior, multi-letter collision behavior. The generic
    clean_stop variant applies to no-assert datasets; the MBPP [DONE]-anchored
    counter is unchanged (checked below)."""

    ST_TERMINAL = "She sold 48/2 = 24 clips in May. 24 + 48 = 72.\nAnswer: 72"
    ST_TERMINAL_TRAILING_NL = "work shown\nAnswer: 72\n"
    ST_POST_COMMIT_CONT = "work shown\nAnswer: 72\nLet me also solve the next one. Q: ...\nAnswer: 8"
    ST_NO_ANSWER = "48/2 = 24. So 24 clips in May."
    # BoolQ@M3 fabricated-option invasion: committed answer, then an invented
    # A)/B) option list continues after the terminal line
    ST_INVENTED_OPTIONS = "The passage says so.\nAnswer: true\nA) True\nB) False\nAnswer: A) True"
    # ARC@M1 multi-letter collision: option list restated, answer committed twice
    ST_MULTI_LETTER = ("(A) ice  (B) hardness  (C) mass  (D) smell\n"
                       "Answer: (B)\nWait, the options again: A, B, C, D. Answer: (A)")

    def test_stop_clause_terminal_line_is_clean(self):
        self.assertTrue(clean_stop_generic_present(self.ST_TERMINAL))
        self.assertTrue(clean_stop_generic_present(self.ST_TERMINAL_TRAILING_NL))

    def test_stop_clause_post_commit_continuation_not_clean(self):
        self.assertFalse(clean_stop_generic_present(self.ST_POST_COMMIT_CONT))

    def test_stop_clause_needs_an_answer_line(self):
        self.assertFalse(clean_stop_generic_present(self.ST_NO_ANSWER))
        self.assertFalse(clean_stop_generic_present(""))

    def test_single_answer_line(self):
        self.assertTrue(single_answer_line_present(self.ST_TERMINAL))
        self.assertFalse(single_answer_line_present(self.ST_NO_ANSWER))
        self.assertFalse(single_answer_line_present(""))
        # more than one Answer: match = the collision family
        self.assertFalse(single_answer_line_present(self.ST_POST_COMMIT_CONT))
        self.assertFalse(single_answer_line_present(self.ST_MULTI_LETTER))

    def test_invented_options_behavior_detected(self):
        # fabricated option list after the commitment: terminal line is no longer
        # clean, and the extra "Answer:" line breaks single-answer-line
        self.assertFalse(clean_stop_generic_present(self.ST_INVENTED_OPTIONS))
        self.assertFalse(single_answer_line_present(self.ST_INVENTED_OPTIONS))

    def test_multi_letter_collision_behavior_detected(self):
        self.assertFalse(single_answer_line_present(self.ST_MULTI_LETTER))
        # first-match anchoring: content (the re-stated option list + re-commitment)
        # after the FIRST Answer line is a post-commit continuation -> not clean
        self.assertFalse(clean_stop_generic_present(self.ST_MULTI_LETTER))

    def test_compute_example_markers_dataset_branch(self):
        rec_gsm = compute_example_markers("GSM8K", _dummy_evaluate, self.ST_TERMINAL,
                                          "72", prompt="Q?\n")
        self.assertTrue(rec_gsm["clean_stop"])          # generic variant on GSM8K
        self.assertTrue(rec_gsm["single_answer_line"])
        rec_cont = compute_example_markers("GSM8K", _dummy_evaluate,
                                           self.ST_POST_COMMIT_CONT, "72", prompt="Q?\n")
        self.assertFalse(rec_cont["clean_stop"])
        self.assertFalse(rec_cont["single_answer_line"])
        # MBPP branch unchanged: [DONE]-anchored clean_stop
        rec_mbpp_done = compute_example_markers("MBPP", _dummy_evaluate,
                                                "def f():\n    return 1\n[DONE]", "1")
        self.assertTrue(rec_mbpp_done["clean_stop"])
        rec_mbpp_cont = compute_example_markers("MBPP", _dummy_evaluate,
                                                "def f():\n    return 1\n[DONE]\nnext task",
                                                "1")
        self.assertFalse(rec_mbpp_cont["clean_stop"])

    def test_summary_includes_single_answer_line_rate(self):
        recs = [
            # terminal line only: clean AND single
            compute_example_markers("GSM8K", _dummy_evaluate, self.ST_TERMINAL, "72", prompt="Q?\n"),
            # one commitment, then trailing text on a later line: not clean, still single
            compute_example_markers("GSM8K", _dummy_evaluate, "Answer: (A)\nextra text", "A", prompt="Q?\n"),
            # multi-commitment collision: neither
            compute_example_markers("GSM8K", _dummy_evaluate, self.ST_MULTI_LETTER, "72", prompt="Q?\n"),
        ]
        s = summarize_markers(recs)
        self.assertAlmostEqual(s["clean_stop_rate"], 1 / 3)
        self.assertAlmostEqual(s["single_answer_line_rate"], 2 / 3)
        s_empty = summarize_markers([])
        self.assertIsNone(s_empty["single_answer_line_rate"])

    def test_mbpp_done_anchor_unchanged(self):
        # the pre-registered TESTCONSULT-001 counter is untouched
        self.assertTrue(clean_stop_present("code\n[DONE]"))
        self.assertFalse(clean_stop_present("code\n[DONE]\nextra"))


def _dummy_evaluate(output, label):
    return str(output).strip() == str(label).strip()


class TestAggregationAndLogging(unittest.TestCase):
    QUESTION = "Tom has 3 boxes with 4 pens each"

    def records(self):
        mk = compute_example_markers
        return [
            # base wrong + revised -> counts toward error correction
            mk("GSM8K", _dummy_evaluate, REVISED_OUTPUT, "48", base_output="bad", question=self.QUESTION),
            # base wrong + not revised
            mk("GSM8K", _dummy_evaluate, CONSISTENT_OUTPUT, "48", base_output="bad", question=self.QUESTION),
            # base correct -> excluded from the error-correction denominator
            mk("GSM8K", _dummy_evaluate, BASE_STYLE_OUTPUT, "72", base_output="72", question=self.QUESTION),
        ]

    def test_summary_rates(self):
        s = summarize_markers(self.records())
        self.assertEqual(s["n_examples"], 3)
        self.assertAlmostEqual(s["verify_action_rate"], 2 / 3)
        self.assertAlmostEqual(s["restatement_rate"], 0.0)
        self.assertEqual(s["n_initially_wrong"], 2)
        self.assertAlmostEqual(s["error_correction_rate"], 0.5)
        self.assertAlmostEqual(s["mean_decomposition_steps"], 0.0)

    def test_empty_records(self):
        s = summarize_markers([])
        self.assertEqual(s["n_examples"], 0)
        self.assertIsNone(s["error_correction_rate"])

    def test_no_initially_wrong_gives_none_rate(self):
        recs = [compute_example_markers("GSM8K", _dummy_evaluate, "a", "a", base_output="a")]
        s = summarize_markers(recs)
        self.assertEqual(s["n_initially_wrong"], 0)
        self.assertIsNone(s["error_correction_rate"])

    def test_per_example_jsonl_log(self):
        records = self.records()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "markers.jsonl"
            log_markers(records, path)
            lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0]["verification_move_count"], verification_move_count(REVISED_OUTPUT))
        self.assertIn("summary", lines[-1])
        self.assertEqual(lines[-1]["summary"]["n_examples"], 3)

    def test_compute_with_prompt_derives_question(self):
        prompt = GSM8K(mode="eval").build_prompt(
            {"question": "Tom has 3 boxes with 4 pens each. How many pens in total?"})
        out = ("The question asks: Tom has 3 boxes with 4 pens each. How many pens in total? "
               "3 x 4 = 12. Answer: 12")
        rec = compute_example_markers("GSM8K", _dummy_evaluate, out, 12, prompt=prompt)
        self.assertTrue(rec["restatement_present"])


if __name__ == "__main__":
    unittest.main()
