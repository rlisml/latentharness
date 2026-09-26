"""Launcher for HarnessForge's `run_infer.py` in the WUAS experiments.

Why this exists
---------------
HarnessForge_4B's release omits `runtime/taubench/` (it IS present in
HarnessForge_8B), so `run_infer.py` dies at import time:

    from runtime.taubench.evaluation import (
        evaluate_taubench_item, summarize_taubench_results, write_taubench_metrics)

This launcher resolves that BEFORE executing run_infer.py, without touching the
HarnessForge tree:
  1. if `HarnessForge_8B/runtime/taubench/` exists -> load the real module from it;
  2. otherwise -> register a stub whose functions raise RuntimeError when called
     (harmless: we only run ToolHop/HotQA/TMDB/API-Bank, never TauBench).

Everything else is passed through verbatim, so this behaves exactly like
`python run_infer.py <args>` executed from the HarnessForge_4B directory.

Usage
-----
    <hf4b python> scripts/run_harnessforge_infer.py --benchmark toolhop --infile test ...
"""

import importlib
import importlib.util
import os
import runpy
import sys
import types

HF_ROOT = os.environ.get("HARNESSFORGE_ROOT",
                         "./HarnessForge")
HF_4B = os.environ.get("HARNESSFORGE_4B", os.path.join(HF_ROOT, "HarnessForge_4B"))
HF_8B = os.environ.get("HARNESSFORGE_8B", os.path.join(HF_ROOT, "HarnessForge_8B"))
REQUIRED = ("evaluate_taubench_item", "summarize_taubench_results",
            "write_taubench_metrics")


def _load_from_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_taubench_stub(reason: str):
    pkg = types.ModuleType("runtime.taubench")
    ev = types.ModuleType("runtime.taubench.evaluation")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(f"runtime.taubench is unavailable ({reason})")

    for fn in REQUIRED:
        setattr(ev, fn, _unavailable)
    pkg.evaluation = ev
    return pkg, ev


def install_taubench(runtime_pkg):
    """Register runtime.taubench (+ .evaluation) pointing at the 8B copy or a stub."""
    src = os.path.join(HF_8B, "runtime", "taubench")
    if os.path.isfile(os.path.join(src, "evaluation.py")):
        pkg = types.ModuleType("runtime.taubench")
        pkg.__path__ = [src]
        sys.modules["runtime.taubench"] = pkg
        rt = _load_from_file("runtime.taubench.runtime",
                             os.path.join(src, "runtime.py"))
        ev = _load_from_file("runtime.taubench.evaluation",
                            os.path.join(src, "evaluation.py"))
        pkg.runtime, pkg.evaluation = rt, ev
        # re-export public names declared by the package __init__
        init_src = os.path.join(src, "__init__.py")
        if os.path.isfile(init_src):
            for name in getattr(pkg, "__all__", []):
                if hasattr(rt, name):
                    setattr(pkg, name, getattr(rt, name))
        origin = f"loaded from {src}"
    else:
        pkg, ev = _install_taubench_stub(f"no copy at {src}")
        origin = "stub"
    runtime_pkg.taubench = pkg
    return origin


def _linear_json_repair(s: str) -> str:
    """Single-pass, O(n) repair for the malformations HarnessForge actually sees
    (unclosed brackets/quotes, trailing commas, single quotes, Python literals,
    markdown fences). Never backtracks, so it cannot blow up the way
    `json_repair.repair_json` does on adversarial input."""
    out = []
    stack = []
    in_str = False
    quote = '"'
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\" and i + 1 < n:
                out.append(c)
                out.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                out.append('"')
                in_str = False
                i += 1
                continue
            if c == '"' and quote == "'":
                out.append('\\"')
                i += 1
                continue
            if c == "\n":
                # raw newline in a string literal: close it instead of failing
                out.append('"')
                in_str = False
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if c == '"':
            out.append('"')
            quote = '"'
            in_str = True
            i += 1
            continue
        if c == "'":
            out.append('"')
            quote = "'"
            in_str = True
            i += 1
            continue
        if c in "“”":
            out.append('"')
            quote = '"'
            in_str = True
            i += 1
            continue
        if c in "‘’":
            out.append('"')
            quote = "'"
            in_str = True
            i += 1
            continue
        if c in "{[":
            stack.append(c)
            out.append(c)
            i += 1
            continue
        if c in "}]":
            if stack:
                stack.pop()
            out.append(c)
            i += 1
            continue
        if c == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                i += 1  # trailing comma
                continue
            out.append(c)
            i += 1
            continue
        if c.isalpha():
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            word = s[i:j]
            out.append({"True": "true", "False": "false",
                        "None": "null"}.get(word, word))
            i = j
            continue
        out.append(c)
        i += 1
    if in_str:
        out.append('"')
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return "".join(out)


def install_jsonrepair_guard():
    """Bounded JSON parsing for model-emitted tool-call / answer strings.

    `Agents/agents.py` (and react_agent) parse model output with
    `json_repair.loads`, which can blow up combinatorially on malformed input
    and spin forever (observed hangs under the `both` phase and
    LoRA delta >= 0.5 with payloads of only ~1.6k chars). Length truncation does
    NOT help, so instead we replace `json_repair.loads` with a bounded pipeline:

      1. strict `json.loads` — identical behaviour on well-formed JSON;
      2. on failure, `_linear_json_repair` (O(n) heuristic) + `json.loads` again;
      3. still unparsable -> fail-closed `{}` (call sites skip items with no
         valid tool call / answer, so the step degrades instead of hanging).

    `json_repair` itself is never invoked, so no combinatorial path remains.
    Enabled with HF_JSONREPAIR_GUARD=1; HF_JSONREPAIR_LOG records every
    non-trivial repair for post-hoc inspection.
    """
    if os.environ.get("HF_JSONREPAIR_GUARD") != "1":
        return "off"
    import json
    import json_repair

    log_path = os.environ.get("HF_JSONREPAIR_LOG", "/tmp/hf_jsonrepair.log")
    log_min = int(os.environ.get("HF_JSONREPAIR_LOG_MIN", "0"))

    def bounded_loads(s=None, *args, **kwargs):
        if not isinstance(s, str):
            return {} if s is None else s
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            repaired = _linear_json_repair(s)
            return json.loads(repaired)
        except Exception:
            pass
        if len(s) >= log_min:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"event": "fail_closed", "len": len(s),
                                         "head": s[:600], "tail": s[-600:]}) + "\n")
            except Exception:
                pass
        return {}

    json_repair.loads = bounded_loads
    return f"bounded (log={log_path}, min={log_min})"


def main() -> int:
    if not os.path.isfile(os.path.join(HF_4B, "run_infer.py")):
        raise SystemExit(f"run_infer.py not found under {HF_4B}")

    os.chdir(HF_4B)
    sys.path.insert(0, HF_4B)

    runtime_pkg = importlib.import_module("runtime")
    origin = install_taubench(runtime_pkg)
    print(f"[run_harnessforge_infer] runtime.taubench: {origin}", flush=True)
    print(f"[run_harnessforge_infer] json_repair guard: {install_jsonrepair_guard()}",
          flush=True)

    sys.argv = ["run_infer.py"] + sys.argv[1:]
    runpy.run_path(os.path.join(HF_4B, "run_infer.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
