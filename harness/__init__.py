"""harness — Latent Harness explicit-infrastructure package (E1-A).

Paired prompt construction (base/student vs teacher/harness), the harness text
registry loader (texts live in HARNESS_LIBRARY.md, never in code), behavioral
marker counters, and the paired-prompt inspection CLI.

Not in this package (explicitly deferred): teacher caching, distillation loss,
latent harness training, evolution.
"""

from .harness_text import (
    BASE_HARNESS_NAME,
    REGISTRY_PATH,
    Harness,
    add_harness_argument,
    harness_from_args,
    load_registry,
    parse_registry,
    resolve_harness,
)
from .markers import (
    compute_example_markers,
    count_correction_cues,
    decomposition_step_count,
    log_markers,
    question_and_options_from_prompt,
    restatement_present,
    revision_event,
    summarize_markers,
    verification_move_count,
)
from .prompts import (
    TEACHER_SEPARATOR,
    PairedExample,
    assert_no_leakage,
    build_paired_examples,
    build_teacher_prompt,
)

__all__ = [
    "BASE_HARNESS_NAME",
    "REGISTRY_PATH",
    "Harness",
    "add_harness_argument",
    "harness_from_args",
    "load_registry",
    "parse_registry",
    "resolve_harness",
    "TEACHER_SEPARATOR",
    "PairedExample",
    "assert_no_leakage",
    "build_paired_examples",
    "build_teacher_prompt",
    "compute_example_markers",
    "count_correction_cues",
    "decomposition_step_count",
    "log_markers",
    "question_and_options_from_prompt",
    "restatement_present",
    "revision_event",
    "summarize_markers",
    "verification_move_count",
]
