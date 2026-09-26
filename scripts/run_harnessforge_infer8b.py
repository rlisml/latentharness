"""Launcher for HarnessForge_8B's `run_infer.py` (official 8B harness protocol).

Same role as `run_harnessforge_infer.py` but executes from the **HarnessForge_8B**
package instead of HarnessForge_4B, so `--harness_package evolved_pairs.harness_factory`
resolves the 8B `rounds/round_03_0X/*` bundles listed in
`HarnessForge_8B/evolved_pairs/pairs.yaml`.

Two things are injected before runpy, without touching the HarnessForge tree:

  1. the bounded `json_repair.loads` guard (HF_JSONREPAIR_GUARD=1) — identical to the
     4B launcher, so the protocol stays byte-comparable with the old runs;
  2. `/v1/steering` reachability is not affected (that lives in the vLLM server).

Usage
-----
    <hf4b python> scripts/run_harnessforge_infer8b.py --benchmark toolhop \
        --harness_package evolved_pairs.harness_factory \
        --harness rounds.round_03_01.harness_round03_01_5 ...
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HF_ROOT = os.environ.get("HARNESSFORGE_ROOT", "./HarnessForge")
HF_8B = os.environ.get("HARNESSFORGE_8B", os.path.join(HF_ROOT, "HarnessForge_8B"))


def main() -> int:
    if not os.path.isfile(os.path.join(HF_8B, "run_infer.py")):
        raise SystemExit(f"run_infer.py not found under {HF_8B}")

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import run_harnessforge_infer as guard_mod  # noqa: E402  (same dir, 4B launcher)

    os.chdir(HF_8B)
    sys.path.insert(0, HF_8B)

    print(f"[run_harnessforge_infer8b] root: {HF_8B}", flush=True)
    print(f"[run_harnessforge_infer8b] json_repair guard: "
          f"{guard_mod.install_jsonrepair_guard()}", flush=True)

    sys.argv = ["run_infer.py"] + sys.argv[1:]
    runpy.run_path(os.path.join(HF_8B, "run_infer.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
