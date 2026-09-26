"""MBPP loader (v3 multi-model extension) — fourth dataset, code generation.

Pre-registration: RESEARCH_PLAN.md §9.2 (v3 multi-model extension). This module is
additive: no existing loader, template, split, or `evaluate` function was modified.
Registration = the dict entries in `utils/utils.py` (import line, `dataset_dict`,
`epochs_per_dataset`, `max_new_tokens_per_dataset`).

Dataset identity and split convention [Pre-registered, verified at load 2026-09-01]:

- HF dataset id ``mbpp`` (google-research-datasets/mbpp), config ``full`` (fixed).
- Verified split sizes of config ``full``: train 374, test 500, validation 90,
  prompt 10.
- **Evaluation split = the official test split (500); untouched until E1-Eval.**
- **Dev pool = the LAST 100 examples of the official train split** (rows 274-373,
  dataset order, no shuffle) — the screening/selection split; used in full (n=100)
  for every seed. **Training pool = the remaining FIRST 274 train examples**
  (rows 0-273). The HF ``validation`` split (90) is not used anywhere.
- **Few-shot source = the ``prompt`` split (10 problems); the base prompt is fixed
  3-shot using the first 3 in dataset order (task_ids 1, 2, 3).**
- All four splits carry (task_id, text, code, test_list, test_setup_code,
  challenge_test_list). On train, ``test_list`` has exactly 3 asserts per problem;
  ``test_setup_code`` is non-empty on 1/374 train rows; ``challenge_test_list`` is
  empty on all train rows. Evaluation uses **test_list only** (challenge_test_list
  excluded — pre-registered); ``test_setup_code`` is prepended in the sandbox when
  non-empty.

Base prompt [Pre-registered]: fixed 3-shot format, identical for every arm:

    You are an expert Python programmer, and here is your task: {text}
    Your code should pass these tests:

    {all of the problem's test_list asserts, one per line}

    [BEGIN]
    {code}            # few-shot examples show the reference code here
    [DONE]

The 3-shot block is ``\\n\\n``-separated; the target block ends at ``[BEGIN]`` and the
gold completion (``get_answer_string``) is the normalized reference code plus
``[DONE]``. Reference code is normalized (``\\r\\n``→``\\n``, tabs→4 spaces, trailing
whitespace stripped). Only the harness text may differ between teacher and student
prompts (the usual paired-prompt construction + leakage assertion apply unchanged).

Answer extraction [Pre-registered; refined 2026-09-01 BEFORE any generation — eval
pipelines score the continuation only, and the base prompt already ends at
``[BEGIN]``, so the canonical output is ``code…[DONE]`` without a leading marker]:
(1) if the output contains ``[BEGIN]``: the body between ``[BEGIN]`` and the first
``[DONE]`` after it (body-to-end when ``[DONE]`` is missing); (2) else if the output
contains ``[DONE]``: everything before the first ``[DONE]`` (the generation continues
the prompt's ``[BEGIN]`` scaffold); (3) else the first fenced code block, fences
stripped; (4) else extraction failure (counted incorrect). Surrounding markdown fences
inside an extracted body are stripped. Only the harness text may differ between
teacher and student prompts (the usual paired-prompt construction + leakage assertion
apply unchanged).

Evaluation sandbox [Pre-registered; settings recorded in every MBPP ledger line]:
``evaluate`` executes the extracted code against the problem's own test_list
assertions in a **subprocess**:

- fresh temp working directory per example (cwd, HOME, TMPDIR all inside it);
- **no network access** — primary layer: ``unshare -rn`` (user + network namespace;
  probed lazily once per process, no root required); defense in depth: an in-process
  socket guard (``socket.socket`` / ``create_connection`` / ``getaddrinfo`` /
  ``gethostbyname`` disabled) installed in the runner BEFORE the solution executes;
- hard per-example timeout **10 s** wall clock (on timeout the whole process group is
  SIGKILLed); resource backstops RLIMIT_CPU = 10 s, RLIMIT_AS = 2 GiB,
  RLIMIT_FSIZE = 16 MiB; stdin = DEVNULL; stdout/stderr redirected to files inside the
  temp dir (suppressed, discarded with the dir); the structured result comes back via
  a small ``result.json`` (PIPE deadlocks and unbounded output are impossible);
- pass@1 = **all** test_list asserts pass. Any exception, timeout, or extraction
  failure counts as incorrect.

Confinement documentation (untrusted generated code): the code is model-generated
diagnostic output, not adversarial input, but it is still confined: (1) OS-level
network isolation via the ``unshare -rn`` user+network namespace (no network
interfaces exist inside); (2) the in-process socket guard blocks socket creation even
if the namespace layer were unavailable; (3) the process runs in a fresh temp working
directory with HOME/TMPDIR redirected there and a minimal environment (no HF/API
tokens), so it cannot read credentials or the repo through the environment; (4)
RLIMIT_CPU/AS/FSIZE bound CPU, memory, and file writes; (5) the 10 s wall-clock
timeout kills the whole process group (children included); (6) the temp dir is deleted
afterwards. Residual limitation (documented honestly): the process inherits the
invoking user's filesystem read permissions — there is no OS-level filesystem sandbox
on this host without root; the dataset's test_list asserts and the model's own output
are the only code ever executed.
"""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap

from .dataloaders import BaseDatasetLoader
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Pre-registered constants (RESEARCH_PLAN §9.2)
# ---------------------------------------------------------------------------

HF_DATASET_ID = "mbpp"
HF_CONFIG = "full"

TRAIN_POOL_N = 274          # first 274 train rows
DEV_POOL_N = 100            # last 100 train rows (rows 274-373)
EVAL_TEST_N = 500           # official test split, untouched until E1-Eval
FEWSHOT_N = 3               # first 3 problems of the `prompt` split (task_ids 1, 2, 3)

SANDBOX_TIMEOUT_S = 10      # hard per-example wall-clock timeout [Pre-registered]
SANDBOX_MEM_BYTES = 2 << 30     # RLIMIT_AS backstop
SANDBOX_FSIZE_BYTES = 16 << 20  # RLIMIT_FSIZE backstop

# Extraction methods reported by extract_mbpp_code / the answer-position mirror.
METHOD_BEGIN_DONE = "code_begin_done"   # [BEGIN] ... [DONE] body
METHOD_BEGIN_OPEN = "code_begin_open"   # [BEGIN] body to end of output (no [DONE])
METHOD_DONE_ONLY = "code_done_only"     # implicit [BEGIN] (prompt ended at [BEGIN]):
                                        # output up to the first [DONE]
METHOD_FENCED = "code_fenced"           # first fenced code block


def normalize_code(code: str) -> str:
    """Cosmetic normalization applied to reference code shown in prompts/completions:
    CRLF → LF, tabs → 4 spaces, trailing whitespace stripped per line, blank edges
    trimmed. Model-generated code is NOT normalized before execution."""
    code = code.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    code = "\n".join(ln.rstrip() for ln in code.split("\n"))
    return code.strip("\n")


_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


def _strip_surrounding_fences(code: str) -> str:
    """If the whole body is one fenced block, strip the fences (task instruction:
    'strip markdown fences'). Inner prose/other content is left untouched."""
    m = re.fullmatch(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n?```", code, re.DOTALL)
    return m.group(1).strip() if m else code


def extract_mbpp_code(output: str):
    """[Pre-registered extraction, refined 2026-09-01 before any generation.]
    Returns ``(code, method)``; ``(None, None)`` when nothing is extractable.

    (1) ``[BEGIN]`` present: body between ``[BEGIN]`` and the first ``[DONE]`` after
    it (body-to-end when ``[DONE]`` is missing). (2) else ``[DONE]`` present: output
    up to the first ``[DONE]`` — the generation continues the prompt's ``[BEGIN]``
    scaffold, so the canonical well-formed output has no leading marker. (3) else the
    first fenced code block. Surrounding fences in an extracted body are stripped.
    Single source of truth for every MBPP consumer (loader evaluate, teacher-cache
    answer-position mirror, error analysis)."""
    if not output:
        return None, None
    m_begin = re.search(r"\[BEGIN\]", output)
    if m_begin:
        rest = output[m_begin.end():]
        m_done = re.search(r"\[DONE\]", rest)
        body = rest[: m_done.start()] if m_done else rest
        code = _strip_surrounding_fences(body.strip())
        if code:
            return code, (METHOD_BEGIN_DONE if m_done else METHOD_BEGIN_OPEN)
        return None, None
    m_done = re.search(r"\[DONE\]", output)
    if m_done:
        code = _strip_surrounding_fences(output[: m_done.start()].strip())
        if code:
            return code, METHOD_DONE_ONLY
        return None, None
    m_fence = _FENCE_RE.search(output)
    if m_fence:
        code = m_fence.group(1).strip()
        if code:
            return code, METHOD_FENCED
    return None, None


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

# The runner is written into the fresh temp dir per evaluation. __LIMITS__ is
# substituted by the parent. The socket guard is installed BEFORE the solution
# executes; rlimits are applied first so even runner startup is bounded.
_RUNNER_TEMPLATE = textwrap.dedent(
    """
    import json, resource, socket, sys

    class _NetworkDisabled(Exception):
        pass

    def _blocked(*_a, **_k):
        raise _NetworkDisabled("network access is disabled in the MBPP sandbox")

    socket.socket = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    socket.gethostbyname = _blocked

    limits = json.loads(r'''__LIMITS__''')
    resource.setrlimit(resource.RLIMIT_AS, (limits["mem"], limits["mem"]))
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu"], limits["cpu"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["fsize"], limits["fsize"]))

    result = {"compile_error": None, "exec_error": None, "asserts": [],
              "n_pass": 0, "n_total": 0}
    ns = {"__name__": "solution"}
    try:
        task = json.load(open("task.json"))
        solution = open("solution.py").read()
    except BaseException as e:  # unreachable in practice (parent wrote the files)
        result["exec_error"] = f"{type(e).__name__}: {e}"
    else:
        try:
            if task.get("test_setup_code", "").strip():
                exec(compile(task["test_setup_code"], "test_setup.py", "exec"), ns)
            compile(solution, "solution.py", "exec")
        except SyntaxError as e:
            result["compile_error"] = f"{type(e).__name__}: {e.msg} (line {e.lineno})"
        except BaseException as e:
            result["exec_error"] = f"{type(e).__name__}: {e}"
        else:
            try:
                if task.get("test_setup_code", "").strip():
                    exec(compile(task["test_setup_code"], "test_setup.py", "exec"), ns)
                exec(compile(solution, "solution.py", "exec"), ns)
            except BaseException as e:
                result["exec_error"] = f"{type(e).__name__}: {e}"
            else:
                for i, a in enumerate(task["asserts"]):
                    try:
                        exec(compile(a, f"<test {i}>", "exec"), ns)
                        result["asserts"].append({"i": i, "ok": True, "error": None})
                    except AssertionError as e:
                        result["asserts"].append(
                            {"i": i, "ok": False, "error": f"AssertionError: {e}"})
                    except BaseException as e:
                        result["asserts"].append(
                            {"i": i, "ok": False, "error": f"{type(e).__name__}: {e}"})
                result["n_total"] = len(task["asserts"])
                result["n_pass"] = sum(1 for r in result["asserts"] if r["ok"])
    with open("result.json", "w") as fh:
        json.dump(result, fh)
    """
).strip() + "\n"

# Cached lazily (no subprocess at import time).
_UNSHARE_AVAILABLE = None

_UNSHARE_PROBE_TIMEOUT_S = 60


def unshare_available() -> bool:
    """Probe (once per process) whether `unshare -rn` works on this host (user +
    network namespace without root). The primary network-isolation layer."""
    global _UNSHARE_AVAILABLE
    if _UNSHARE_AVAILABLE is None:
        try:
            proc = subprocess.run(
                ["unshare", "-rn", sys.executable, "-c", "pass"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=_UNSHARE_PROBE_TIMEOUT_S)
            _UNSHARE_AVAILABLE = proc.returncode == 0
        except Exception:
            _UNSHARE_AVAILABLE = False
    return _UNSHARE_AVAILABLE


def sandbox_settings(timeout_s: int = SANDBOX_TIMEOUT_S):
    """The sandbox settings recorded in every MBPP ledger line / detailed result."""
    return {
        "timeout_s": timeout_s,
        "timeout_enforcement": "wall-clock; on expiry the process group is SIGKILLed "
                               "(RLIMIT_CPU backstop)",
        "interpreter": sys.executable,
        "isolation": ("unshare -rn (user+network namespace) + in-process socket guard"
                      if unshare_available() else
                      "in-process socket guard (unshare unavailable on this host)"),
        "cwd": "fresh temp dir per example (HOME/TMPDIR redirected into it)",
        "env": "minimal (PATH, HOME=tmp, TMPDIR=tmp, PYTHONDONTWRITEBYTECODE=1, LANG)",
        "rlimits": {"RLIMIT_AS": SANDBOX_MEM_BYTES, "RLIMIT_CPU": timeout_s,
                    "RLIMIT_FSIZE": SANDBOX_FSIZE_BYTES},
        "stdin": "DEVNULL",
        "stdout_stderr": "suppressed (temp files inside the sandbox dir, discarded)",
        "pass_rule": "pass@1 = all test_list asserts pass; exception/timeout/"
                     "extraction failure = incorrect",
    }


def run_mbpp_sandbox(code: str, test_list, test_setup_code: str = "",
                     timeout_s: int = SANDBOX_TIMEOUT_S) -> dict:
    """Execute extracted code against the problem's own test_list asserts in the
    subprocess sandbox. Returns a structured dict (never raises):
    {compile_error, exec_error, asserts, n_pass, n_total, all_pass, timed_out,
     sandbox_error, returncode, elapsed_s, sandbox}."""
    import time

    settings = sandbox_settings(timeout_s)
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="mbpp_sandbox_")
    try:
        with open(os.path.join(tmp, "solution.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(os.path.join(tmp, "task.json"), "w", encoding="utf-8") as fh:
            json.dump({"test_setup_code": test_setup_code or "",
                       "asserts": list(test_list)}, fh)
        runner_src = _RUNNER_TEMPLATE.replace(
            "__LIMITS__",
            json.dumps({"mem": SANDBOX_MEM_BYTES, "cpu": timeout_s,
                        "fsize": SANDBOX_FSIZE_BYTES}))
        with open(os.path.join(tmp, "mbpp_runner.py"), "w", encoding="utf-8") as fh:
            fh.write(runner_src)

        cmd = ([ "unshare", "-rn"] if unshare_available() else []) + \
            [sys.executable, "mbpp_runner.py"]
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp,
               "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        out_fh = open(os.path.join(tmp, "stdout.log"), "wb")
        err_fh = open(os.path.join(tmp, "stderr.log"), "wb")
        try:
            proc = subprocess.Popen(
                cmd, cwd=tmp, stdin=subprocess.DEVNULL, stdout=out_fh, stderr=err_fh,
                start_new_session=True, env=env)
            timed_out = False
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
        finally:
            out_fh.close()
            err_fh.close()

        result_path = os.path.join(tmp, "result.json")
        sandbox_error = None
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as fh:
                result = json.load(fh)
            # SIGXCPU death (rc -24) means the CPU backstop fired: a hang by any
            # other name. rc -9 with the parent timer expired is the wall timeout.
            if timed_out:
                result["timed_out"] = True
            elif proc.returncode == -24 and result.get("n_total", 0) == 0 \
                    and not result.get("compile_error") and not result.get("exec_error"):
                result["timed_out"] = True
            else:
                result["timed_out"] = False
        else:
            result = {"compile_error": None, "exec_error": None, "asserts": [],
                      "n_pass": 0, "n_total": 0}
            if timed_out:
                result["timed_out"] = True
            else:
                result["timed_out"] = False
                result["sandbox_error"] = (
                    f"no result produced (returncode {proc.returncode})")
        result["returncode"] = proc.returncode
        result["sandbox_error"] = result.get("sandbox_error", sandbox_error)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    result["all_pass"] = bool(
        not result.get("timed_out") and not result.get("compile_error")
        and not result.get("exec_error") and not result.get("sandbox_error")
        and result.get("n_total", 0) > 0
        and result.get("n_pass") == result.get("n_total"))
    result["sandbox"] = settings
    result["elapsed_s"] = round(time.time() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Prompt formatting (frozen template; RESEARCH_PLAN §9.2)
# ---------------------------------------------------------------------------

_PROMPT_HEADER = "You are an expert Python programmer, and here is your task: "
_TESTS_LINE = "Your code should pass these tests:"
_BEGIN_MARKER = "[BEGIN]"
_DONE_MARKER = "[DONE]"
_EXAMPLE_SEPARATOR = "\n\n"


def _format_block(text: str, test_list, code=None) -> str:
    """One prompt block: header + tests + [BEGIN] (+ code + [DONE] when a reference
    solution is given, i.e. for few-shot examples and gold completions)."""
    lines = [
        _PROMPT_HEADER + text.strip(),
        _TESTS_LINE,
        "",
        "\n".join(t.strip() for t in test_list),
        "",
        _BEGIN_MARKER,
    ]
    if code is not None:
        lines.append(normalize_code(code))
        lines.append(_DONE_MARKER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class MBPP(BaseDatasetLoader):
    """MBPP (config `full`) following the BaseDatasetLoader interface.

    Split sizes verified at load time (asserted; see module docstring for the full
    pre-registration): train pool 274 (first 274 of official train), dev pool 100
    (last 100 of official train), test 500 (official test, untouched until E1-Eval),
    few-shot = first 3 `prompt`-split problems (task_ids 1, 2, 3). The HF
    `validation` split (90) is unused. `evaluate` runs the sandboxed test_list
    execution described in the module docstring (10 s, no network, pass@1).
    """

    def __init__(self, mode="eval"):
        super().__init__(mode)
        ds = load_dataset(HF_DATASET_ID, HF_CONFIG)
        # Load-time verification of the pre-registered split sizes (loud failure if
        # the upstream dataset changes; sizes recorded in the module docstring).
        assert len(ds["train"]) == TRAIN_POOL_N + DEV_POOL_N, (
            f"mbpp full train split has {len(ds['train'])} rows; expected "
            f"{TRAIN_POOL_N + DEV_POOL_N} (training pool {TRAIN_POOL_N} + dev pool "
            f"{DEV_POOL_N}) — the pre-registered split convention no longer holds")
        assert len(ds["test"]) == EVAL_TEST_N, (
            f"mbpp full test split has {len(ds['test'])} rows; expected {EVAL_TEST_N}")
        assert len(ds["prompt"]) >= FEWSHOT_N
        # The HF `validation` split (90) is intentionally unused (dev pool replaces
        # it); `test` is passed through unchanged.
        self.raw_dict = {
            "train": ds["train"].select(range(TRAIN_POOL_N)),
            "validation": ds["train"].select(range(TRAIN_POOL_N, TRAIN_POOL_N + DEV_POOL_N)),
            "test": ds["test"],
        }
        fewshot_rows = [ds["prompt"][i] for i in range(FEWSHOT_N)]
        self.fewshot_task_ids = [r["task_id"] for r in fewshot_rows]
        self._fewshot_block = _EXAMPLE_SEPARATOR.join(
            _format_block(r["text"], r["test_list"], code=r["code"])
            for r in fewshot_rows)

    # --- prompt / completion ------------------------------------------------

    def build_prompt(self, ex):
        """The canonical H_base for MBPP: fixed 3-shot block + the target block
        (header + the problem's own test_list asserts + `[BEGIN]`). Identical for
        every arm."""
        return self._fewshot_block + _EXAMPLE_SEPARATOR + \
            _format_block(ex["text"], ex["test_list"])

    def get_answer_string(self, ex):
        """Gold completion in the same scaffold: normalized reference code + [DONE]."""
        return normalize_code(ex["code"]) + "\n" + _DONE_MARKER

    # --- evaluation ----------------------------------------------------------

    @staticmethod
    def _tests_from_label(label):
        if isinstance(label, dict):
            return list(label.get("test_list", [])), label.get("test_setup_code", "") or ""
        if isinstance(label, str):
            obj = json.loads(label)
            return list(obj.get("test_list", [])), obj.get("test_setup_code", "") or ""
        raise ValueError(f"MBPP label must be a dict/JSON string with test_list; "
                         f"got {type(label)}")

    def evaluate_detailed(self, output, label, timeout_s=SANDBOX_TIMEOUT_S):
        """Sandbox result with full failure structure (used by the E1-0-MBPP error
        analysis). `evaluate` below is the boolean interface required by the
        BaseDatasetLoader contract."""
        code, method = extract_mbpp_code(output)
        test_list, setup = self._tests_from_label(label)
        detail = {"extracted": code is not None, "extraction_method": method,
                  "code": code, "sandbox": sandbox_settings(timeout_s)}
        if code is None:
            detail.update({"compile_error": None, "exec_error": None, "asserts": [],
                           "n_pass": 0, "n_total": len(test_list), "all_pass": False,
                           "timed_out": False, "returncode": None, "elapsed_s": 0.0})
            return detail
        result = run_mbpp_sandbox(code, test_list, test_setup_code=setup,
                                  timeout_s=timeout_s)
        result.update({"extracted": True, "extraction_method": method, "code": code})
        return result

    def evaluate(self, output, label) -> bool:
        """pass@1: execute the extracted code against the problem's own test_list
        asserts in the subprocess sandbox (10 s hard timeout, no network, fresh temp
        cwd). Any exception, timeout, or extraction failure counts as incorrect."""
        return bool(self.evaluate_detailed(output, label)["all_pass"])

    # --- dataset plumbing (interface-complete; mirrors the existing loaders) --

    def load_tokenized_dataset(self, tokenizer, max_len=1280, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            return self._tokenize_prompt_completion(tokenizer, prompt, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)

    def load_raw_dataset(self, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            label = {"test_list": list(ex["test_list"]),
                     "test_setup_code": ex.get("test_setup_code", "") or ""}
            return self._build_raw_example(prompt, completion, label)
        return self._map_raw_splits(build_raw_features, return_test)

    def load_tokenized_dataset_chat(self, tokenizer, max_len=1280, return_test=False):
        def build_tokenized_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            base_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            return self._tokenize_chat_completion(tokenizer, base_msgs, completion, max_len)
        return self._map_tokenized_splits(build_tokenized_features, return_test)

    def load_raw_dataset_chat(self, tokenizer, return_test=False):
        def build_raw_features(ex):
            prompt = self.build_prompt(ex)
            completion = self.get_answer_string(ex)
            base_msgs = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            prefix = tokenizer.apply_chat_template(
                base_msgs, tokenize=False, add_generation_prompt=True)
            label = {"test_list": list(ex["test_list"]),
                     "test_setup_code": ex.get("test_setup_code", "") or ""}
            return [prefix, completion, label]
        return self._map_raw_chat_splits(build_raw_features, return_test)
