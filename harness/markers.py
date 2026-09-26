"""Behavioral marker counters — first version, heuristic/regex (E1-A).

Pre-registered definitions: RESEARCH_PLAN §2.3. These are cheap, deterministic,
keyword-level heuristics: they index *surface compliance* with a harness, not reasoning
quality, and can be gamed by superficial keyword insertion. Per OI-7 they must reach
>= 80% agreement with human labels on a 30-example spot check (E1-B) before being used
for screening decisions. They never replace accuracy reporting.

Markers, computed per example on raw model outputs:

- verification_move_count / verify_action — explicit verification acts: the mandated
  `Verification:` segment, re-check phrases, and answer-vs-steps consistency checks.
- decomposition_step_count — number of enumerated `Step k:` intermediate steps.
- restatement_present — whether the question (or, for multiple choice, the answer
  options) is restated before the final answer.
- revision_event / error_correction_rate — whether a stated answer visibly differs from
  an earlier stated answer on the way to the final one (detected-and-corrected
  intermediate result). The *rate* is conditioned on initially-wrong cases: examples
  where the base model's output on the same example is scored incorrect.

E1-0 pre-registered markers (HARNESS_LIBRARY.md predicted-marker fields; implemented in
the E1-A remainder before the E1-B pilot):

- answer_line_present — the output reaches a final `Answer:` line (CONCISE-001).
- arithmetic_expression_count / arithmetic_expression_event — number of
  `<expr> = <number>` evaluation steps before the final `Answer:` marker; the event is
  >= 2 such steps (COMPUTE-001).
- option_coverage — every listed option label is mentioned before the final `Answer:`
  marker; None when the prompt lists no options (ELIMINATE-001).
- evidence_quote — a verbatim >= 8-word span of the prompt passage appears before the
  final `Answer:` marker; None when the prompt has no passage (QUOTE-001).

All counters are logged per example; `summarize_markers` aggregates rates.

TESTCONSULT-001 markers (E1-B-v3, additive; pre-registered in HARNESS_LIBRARY.md and
`E1_0_V3_LIBRARY_REPORT.md` before any v3 generation, with BASE baselines computed on
the frozen E1-0-MBPP draw — interface-alignment 30/193 = 0.1554, clean-stop 0/200,
single-task-block 19/200 = 0.095). MBPP-only; None on every other dataset:

- interface_alignment — >= 1 function/class name defined in the EXTRACTED code (the
  pre-registered MBPP extraction, `utils/dataloaders_mbpp.extract_mbpp_code`) appears
  among the names called by the prompt-visible test_list asserts — the exact negation
  of the E-COMMIT-CODE trigger (`harness/classify_errors_mbpp.py` logic reused).
  Undefined (None) when no code is extractable.
- clean_stop — nothing but whitespace after the first `[DONE]` in the output.
- single_task_block — the output opens NO additional `[BEGIN]` task block (the
  baseline-anchored counter for the registry's "exactly one task block" mechanism
  marker; see the function docstring).

SINGLETASK-001/002 markers (E1-B-PERM, additive; OI-28 pre-registered in
`results/e1b_perm/PREREGISTRATION.md` before any generation, with BASE baselines
pinned on the frozen E1-0-PERM raws — TESTCONSULT-001 precedent). The existing
TESTCONSULT-001 counters above are UNCHANGED:

- clean_stop (dataset-branched) — on MBPP it stays the [DONE]-anchored counter
  (baseline 0/200); on every no-assert dataset (GSM8K/ARC/BoolQ) it is the generic
  Answer-line-anchored variant: nothing but whitespace after the LINE holding the
  FIRST `Answer:` marker (`clean_stop_generic_present`; the first `Answer:` line is
  the terminal commitment — see the function docstring for the first-vs-last-match
  operationalization note). False when no `Answer:` marker is present.
- single_answer_line — EXACTLY one `Answer:`-pattern match in the whole output
  (the mandated single terminal answer line; more than one match = the
  invented-Q/A / option-restatement collision family SINGLETASK-002 targets).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- verification moves --------------------------------------------------------

# Each pattern is one kind of verification act; matches are counted
# non-overlapping, so "re-verify" counts once, not twice.
_VERIFY_MOVE_PATTERNS = [
    r"verification\s*:",                     # the VERIFY-001-mandated segment
    r"\b(?:re-?verif\w*|re-?check\w*|re-?evaluat\w*|double[-\s]?check\w*|cross[-\s]?check\w*)",
    r"\bverify\b",
    r"\bconfirm(?:ing|ed|s)?\b",
    r"\bcheck(?:ing)?\s+(?:that|the|my|this|whether)\b",
    r"\bconsistent\s+with\b",                # answer-vs-steps consistency language
    r"\b(?:is|does)\s+(?:this|that|the)\s+(?:answer|result)\b",
    r"\bsanity\s+check\b",
]
_VERIFY_MOVE_RE = re.compile("|".join(f"(?:{p})" for p in _VERIFY_MOVE_PATTERNS), re.IGNORECASE)

# --- decomposition steps -------------------------------------------------------

_STEP_RE = re.compile(r"\bstep\s*(\d+)\s*:", re.IGNORECASE)  # DECOMPOSE-001 'Step k:' format

# --- restatement ---------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_ANSWER_MARKER_RE = re.compile(r"answer\s*:", re.IGNORECASE)


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


# --- revision detection --------------------------------------------------------

# Statements that propose an answer ("the answer is 42", "proposed answer: 42",
# "Answer: 42"). Bare intermediate results ("the result of step 1 is 24") do NOT match,
# so enumerated sub-steps are not mistaken for revised answers.
_ANSWER_STATEMENT_RE = re.compile(
    r"\b(?:proposed\s+answer|final\s+answer|(?:the\s+)?answer)\s*(?:is|:)[ \t]*(?P<val>[^\n]{1,60})",
    re.IGNORECASE,
)
_CORRECTION_CUE_RE = re.compile(
    r"\b(?:wait|actually|oops|scratch\s+that|incorrect|not\s+correct|correction|revised?|"
    r"revising|corrected|recalculat\w*|made\s+an\s+error|i\s+err(?:ed|oneously)?)\b",
    re.IGNORECASE,
)


def verification_move_count(output: str) -> int:
    """Number of explicit verification acts in the output."""
    if not output:
        return 0
    return len(_VERIFY_MOVE_RE.findall(output))


def decomposition_step_count(output: str) -> int:
    """Number of enumerated `Step k:` intermediate steps in the output."""
    if not output:
        return 0
    return len(_STEP_RE.findall(output))


def restatement_present(
    output: str,
    question: Optional[str] = None,
    options: Optional[List[str]] = None,
) -> Optional[bool]:
    """Whether the question (or any multiple-choice option) is restated before the
    final answer. Returns None when no question text is available to compare against.

    Heuristic: an exact (normalized) 12-word fragment of the question, or a normalized
    option text of >= 8 characters, must appear in the output prefix preceding the last
    'Answer:' marker. Paraphrased restatements are not detected — known limitation,
    to be validated in the E1-B human spot check.
    """
    if question is None:
        return None
    q_words = _normalize(question).split()
    if not q_words:
        return None
    fragment = " ".join(q_words[:12]) if len(q_words) > 12 else " ".join(q_words)
    text = output
    answer_markers = list(_ANSWER_MARKER_RE.finditer(output))
    if answer_markers:
        text = output[: answer_markers[-1].start()]
    norm_out = _normalize(text)
    if fragment and fragment in norm_out:
        return True
    for opt in options or []:
        norm_opt = _normalize(opt)
        if len(norm_opt) >= 8 and norm_opt in norm_out:
            return True
    return False


def _normalize_value(text: str) -> str:
    s = text.strip().lower().rstrip(".").strip()
    s = s.replace(",", "").replace("$", "").replace("(", " ").replace(")", " ")
    s = re.sub(r"[\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # A leading number stands for the value even when followed by units/words
    # ("48 clips" vs "48"); strip currency/commas already handled above.
    m = re.match(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return str(float(m.group(0)))
        except ValueError:
            pass
    return s


def revision_event(output: str) -> bool:
    """Whether a stated answer was visibly revised on the way to the final one:
    some stated answer earlier in the output differs from the final answer
    (RESEARCH_PLAN §2.3, 'detected-and-corrected intermediate result').

    Explicit correction cues ('wait', 'actually', 'corrected', ...) are counted
    separately (`correction_cue_count`) but do not by themselves set the event —
    a differing earlier answer statement is required, so keyword insertion alone
    cannot produce a positive revision_event.

    Edge case (found in the E1-B marker spot check, OI-7): outputs truncated right
    after an `Answer:` marker end with an empty final statement; an empty value is
    not a stated answer, so such outputs count as having no final answer (event
    False) and empty-valued earlier statements are skipped as well.
    """
    if not output:
        return False
    final_matches = list(re.finditer(r"(?i)answer\s*:[ \t]*([^\n]{0,60})", output))
    if not final_matches:
        return False
    final_value = _normalize_value(final_matches[-1].group(1))
    if not final_value:
        return False
    for m in _ANSWER_STATEMENT_RE.finditer(output):
        if m.start() >= final_matches[-1].start():
            break
        val = _normalize_value(m.group("val"))
        if val and val != final_value:
            return True
    return False


def count_correction_cues(output: str) -> int:
    if not output:
        return 0
    return len(_CORRECTION_CUE_RE.findall(output))


# --- E1-0 pre-registered markers -----------------------------------------------

def _text_before_last_answer(output: str) -> str:
    """Output prefix preceding the final `Answer:` marker (whole output if absent)."""
    if not output:
        return ""
    answer_markers = list(_ANSWER_MARKER_RE.finditer(output))
    return output[: answer_markers[-1].start()] if answer_markers else output


def answer_line_present(output: str) -> bool:
    """CONCISE-001 marker: the output reaches a final `Answer:` line."""
    return bool(_ANSWER_MARKER_RE.search(output or ""))


# `<expr> = <number>`: an equals sign followed by a numeric result ("48/2 = 24").
_ARITH_EQ_RE = re.compile(r"=\s*[-+]?\d")


def arithmetic_expression_count(output: str) -> int:
    """COMPUTE-001 marker: number of `<expr> = <number>` evaluation steps before the
    final `Answer:` marker. The rate event is >= 2 such steps (registry definition)."""
    if not output:
        return 0
    return len(_ARITH_EQ_RE.findall(_text_before_last_answer(output)))


def option_labels_from_prompt(dataset_name: str, prompt: str) -> List[str]:
    """Answer-option labels listed in an eval prompt (['A', 'B', ...] for ARC-style
    option blocks); empty list when the prompt lists no options."""
    if dataset_name in ("ARC", "AQuA"):
        return [m.group(1).upper() for m in re.finditer(r"(?m)^\(([A-E1-5])\)\s", prompt)]
    return []


def passage_from_prompt(dataset_name: str, prompt: str) -> Optional[str]:
    """Passage text of a passage-grounded eval prompt (BoolQ); None otherwise."""
    if dataset_name == "BoolQ":
        m = re.search(r"(?s)Passage:\s*(.*?)\nQuestion:", prompt)
        return m.group(1).strip() if m else None
    return None


def _label_mention_re(label: str) -> "re.Pattern":
    # Mention forms seen in option-walk outputs: "(A)" / "(a)", "A)" / "A:" / "A." /
    # "A -" (uppercase letter at a token boundary), "Option A" (any case).
    # The leading (?i) already makes the whole pattern case-insensitive; a second
    # mid-pattern (?i) was redundant on py3.10 and a hard re.error on py3.11+
    # ("global flags not at the start") — hit by the easysteer env (py3.12), fixed
    # 2026-09-03 E0-v3-ENGINES (semantics unchanged on every version).
    return re.compile(
        rf"(?i)\(\s*{label}\s*\)|(?<![A-Za-z]){label}\s*[\):.\-]|\boption\s+{label}\b"
    )


def option_coverage(output: str, labels: Optional[List[str]]) -> Optional[bool]:
    """ELIMINATE-001 marker: every listed option label is mentioned before the final
    `Answer:` marker. None when the prompt lists no options (marker undefined)."""
    if not labels:
        return None
    text = _text_before_last_answer(output or "")
    return all(_label_mention_re(label).search(text) for label in labels)


def evidence_quote_present(output: str, passage: Optional[str]) -> Optional[bool]:
    """QUOTE-001 marker: a verbatim >= 8-word passage span appears before the final
    `Answer:` marker. None when the prompt has no passage (marker undefined)."""
    if passage is None:
        return None
    out_words = _normalize(_text_before_last_answer(output or "")).split()
    passage_words = _normalize(passage).split()
    if len(out_words) < 8 or len(passage_words) < 8:
        return False
    grams = {" ".join(passage_words[i:i + 8]) for i in range(len(passage_words) - 7)}
    return any(" ".join(out_words[i:i + 8]) in grams for i in range(len(out_words) - 7))


# --- TESTCONSULT-001 markers (MBPP-only; see module docstring) ------------------

_MBBP_TESTS_HEADER_RE = re.compile(r"Your code should pass these tests:", re.DOTALL)


def mbpp_target_asserts_from_prompt(prompt: str) -> List[str]:
    """The target task's test_list asserts shown in an MBPP base prompt (the block
    after the LAST `Your code should pass these tests:` header — the 3-shot examples
    carry their own earlier test blocks; the prompt ends at the target `[BEGIN]`)."""
    if not prompt:
        return []
    headers = list(_MBBP_TESTS_HEADER_RE.finditer(prompt))
    if not headers:
        return []
    tail = prompt[headers[-1].end():]
    tail = tail.split("[BEGIN]")[0]
    return [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("assert ")]


def clean_stop_present(output: str) -> bool:
    """TESTCONSULT-001 mechanism marker: nothing but whitespace after the first
    `[DONE]` (pre-registered definition; BASE rate 0/200 on the E1-0-MBPP draw)."""
    if not output:
        return False
    idx = output.find("[DONE]")
    if idx < 0:
        return False
    return output[idx + len("[DONE]"):].strip() == ""


def single_task_block_present(output: str) -> bool:
    """TESTCONSULT-001 mechanism marker: the output opens NO additional task block,
    i.e. contains zero `[BEGIN]` scaffold openings (the prompt ends at the target
    `[BEGIN]`, so any further one is a self-invented continuation task).

    Operationalization note (E1-B-v3): the registry/report paraphrase is "exactly one
    `You are an expert Python programmer` block in the output" (BASE 19/200 = 0.095 on
    the frozen E1-0-MBPP draw); the header-count==1 reading gives 12/200, while the
    zero-additional-`[BEGIN]` reading reproduces the pre-registered 19/200 exactly, so
    the latter is the implemented counter (baseline-anchored)."""
    if not output:
        return False
    return output.count("[BEGIN]") == 0


def interface_alignment_present(output: str, prompt: Optional[str]) -> Optional[bool]:
    """TESTCONSULT-001 primary marker: >= 1 name defined in the EXTRACTED code appears
    among the names called by the prompt-visible test_list asserts (the exact negation
    of the E-COMMIT-CODE trigger). None when no code is extractable or no prompt is
    available (marker undefined, not a failure)."""
    if output is None or prompt is None:
        return None
    # Lazy imports: keeps `harness.markers` importable without the heavy dataset
    # stack (the trigger logic is reused, not re-implemented — registry rule).
    from utils.dataloaders_mbpp import extract_mbpp_code
    from harness.classify_errors_mbpp import called_names_in_asserts, defined_names
    code, _method = extract_mbpp_code(output)
    if code is None:
        return None
    asserts = mbpp_target_asserts_from_prompt(prompt)
    called = called_names_in_asserts(asserts)
    return bool(defined_names(code) & called)


# --- SINGLETASK-001/002 markers (no-assert cells; see module docstring) ----------

def clean_stop_generic_present(output: str) -> bool:
    """SINGLETASK-001 mechanism marker (generic Answer-line-anchored variant for
    no-assert cells): nothing but whitespace after the LINE that holds the FIRST
    `Answer:`-pattern match — the first `Answer:` line IS the terminal commitment,
    so any later content (echoed working, invented Q/A blocks, option lists,
    re-commitments) is the post-commit continuation the harness targets.

    First-match anchoring (not last-match) is the no-assert analog of
    TESTCONSULT-001's first-`[DONE]` counter: the target pathology on the frozen
    E1-0-PERM raws ends its ramble WITH a re-committed answer, which last-match
    anchoring would score clean. False when no `Answer:` marker is present (nothing
    to terminate after)."""
    if not output:
        return False
    m = _ANSWER_MARKER_RE.search(output)
    if not m:
        return False
    line_end = output.find("\n", m.end())
    tail = output[line_end:] if line_end >= 0 else ""
    return tail.strip() == ""


def single_answer_line_present(output: str) -> bool:
    """SINGLETASK-002 mechanism marker: EXACTLY one `Answer:`-pattern match in the
    whole output (any case). Zero matches = no terminal answer line; more than one =
    the invented-options / repeated-answer collision family the harness targets."""
    if not output:
        return False
    return len(_ANSWER_MARKER_RE.findall(output)) == 1


# --- question recovery from eval prompts (post-hoc marker passes) ---------------

def question_and_options_from_prompt(dataset_name: str, prompt: str):
    """Best-effort recovery of the question (and answer options) from an eval prompt,
    so markers can run post-hoc over existing eval JSONL ({prompt, output, label})
    without re-deriving dataset fields. Read-only w.r.t. the prompt templates.
    """
    if dataset_name in ("GSM8K", "SVAMP", "MAWPS", "ASDiv"):
        return prompt.strip(), []
    if dataset_name == "ARC":
        m = re.search(r"(?s)Question:\s*(.*?)\n\(", prompt)
        if not m:
            m = re.search(r"(?m)^Question:\s*(.*)$", prompt)
        question = m.group(1).strip() if m else prompt.strip()
        options = [mm.group(2).strip() for mm in re.finditer(r"(?m)^\(([A-E1-5])\)\s*(.+)$", prompt)]
        return question, options
    if dataset_name == "BoolQ":
        matches = re.findall(r"(?m)^Question:\s*(.*)$", prompt)
        return (matches[-1].strip() if matches else prompt.strip()), []
    if dataset_name == "MBPP":
        # v3 thin normalization (RESEARCH_PLAN §9.3): the restatement "question" is
        # the task text, not the whole 3-shot prompt (tests + [BEGIN] scaffold would
        # nearly never match a restatement check).
        m = re.search(r"here is your task:\s*(.*?)\nYour code should pass these tests:",
                      prompt, re.DOTALL)
        return (m.group(1).strip() if m else prompt.strip()), []
    return prompt.strip(), []  # unknown dataset: treat whole prompt as context


# --- per-example records and aggregation ----------------------------------------

def compute_example_markers(
    dataset_name: str,
    evaluate_fn,
    output: str,
    label: Any,
    prompt: Optional[str] = None,
    question: Optional[str] = None,
    options: Optional[List[str]] = None,
    base_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Marker record for one model output. Pass `base_output` (the BASE arm's output on
    the same example) to enable the initially-wrong conditioning used by the
    error-correction rate; pass `prompt` (or `question`/`options`) to enable the
    restatement check."""
    if question is None and prompt is not None:
        question, derived_options = question_and_options_from_prompt(dataset_name, prompt)
        options = options if options is not None else derived_options
    vcount = verification_move_count(output)
    base_correct = bool(evaluate_fn(base_output, label)) if base_output is not None else None
    labels = option_labels_from_prompt(dataset_name, prompt) if prompt is not None else []
    passage = passage_from_prompt(dataset_name, prompt) if prompt is not None else None
    arith_count = arithmetic_expression_count(output)
    is_mbpp = dataset_name == "MBPP"
    return {
        "dataset": dataset_name,
        "verification_move_count": vcount,
        "verify_action": vcount > 0,
        "decomposition_step_count": decomposition_step_count(output),
        "restatement_present": restatement_present(output, question=question, options=options),
        "revision_event": revision_event(output),
        "correction_cue_count": count_correction_cues(output),
        "answer_line_present": answer_line_present(output),
        "arithmetic_expression_count": arith_count,
        "arithmetic_expression_event": arith_count >= 2,
        "option_coverage": option_coverage(output, labels),
        "evidence_quote": evidence_quote_present(output, passage),
        "interface_alignment": (interface_alignment_present(output, prompt)
                                if is_mbpp else None),
        # clean_stop is dataset-branched (additive, OI-28): MBPP keeps the
        # pre-registered [DONE]-anchored counter unchanged; no-assert datasets get
        # the generic Answer-line-anchored variant.
        "clean_stop": (clean_stop_present(output) if is_mbpp
                       else clean_stop_generic_present(output)),
        "single_answer_line": single_answer_line_present(output),
        "single_task_block": single_task_block_present(output) if is_mbpp else None,
        "base_correct": base_correct,
        "initially_wrong": (not base_correct) if base_correct is not None else None,
    }


def summarize_markers(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate marker records into the four pre-registered rates.

    error_correction_rate is the fraction of initially-wrong examples (base output
    incorrect on the same example) whose harness output shows a revision event;
    None when the record set contains no initially-wrong examples.
    """
    n = len(records)
    if n == 0:
        return {
            "n_examples": 0,
            "verify_action_rate": None,
            "mean_verification_moves": None,
            "mean_decomposition_steps": None,
            "restatement_rate": None,
            "n_initially_wrong": 0,
            "error_correction_rate": None,
            "answer_line_rate": None,
            "arithmetic_expression_rate": None,
            "mean_arithmetic_expressions": None,
            "option_coverage_rate": None,
            "evidence_quote_rate": None,
            "interface_alignment_rate": None,
            "clean_stop_rate": None,
            "single_answer_line_rate": None,
            "single_task_block_rate": None,
        }

    def _rate(key: str) -> Optional[float]:
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    wrong = [r for r in records if r.get("initially_wrong")]
    revised = [r for r in wrong if r.get("revision_event")]
    return {
        "n_examples": n,
        "verify_action_rate": _rate("verify_action"),
        "mean_verification_moves": sum(r["verification_move_count"] for r in records) / n,
        "mean_decomposition_steps": sum(r["decomposition_step_count"] for r in records) / n,
        "restatement_rate": _rate("restatement_present"),
        "n_initially_wrong": len(wrong),
        "error_correction_rate": (len(revised) / len(wrong)) if wrong else None,
        "answer_line_rate": _rate("answer_line_present"),
        "arithmetic_expression_rate": _rate("arithmetic_expression_event"),
        "mean_arithmetic_expressions": sum(r["arithmetic_expression_count"] for r in records) / n,
        "option_coverage_rate": _rate("option_coverage"),
        "evidence_quote_rate": _rate("evidence_quote"),
        "interface_alignment_rate": _rate("interface_alignment"),
        "clean_stop_rate": _rate("clean_stop"),
        "single_answer_line_rate": _rate("single_answer_line"),
        "single_task_block_rate": _rate("single_task_block"),
    }


def log_markers(records: List[Dict[str, Any]], path) -> None:
    """Per-example marker log (raw JSONL, kept for marker audits), one line per
    example, with a final {'summary': ...} line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.write(json.dumps({"summary": summarize_markers(records)}) + "\n")
