"""MEAN_DIFF_STEERING → EasySteer FixedVector spec (E1-Eval-v3).

Converts a frozen mean-diff artifact (`harness/mean_diff.py` output, .pt) into an
EasySteer steering spec using the FORK'S BUILT-IN FixedVector implementation: the
`direct` algorithm (h' = h + scale·v) applied via an in-memory
`DirectionVector` payload (wire dict, base64 tensors) — NO GGUF, NO env
modification, NO new packages.

The vectors are applied EXACTLY as frozen: the pre-registered all-positions
primary policy (`vectors` in the artifact), scale 1.0, unnormalized, at every
decoder layer, on prompt+generation tokens.

Subcommands:
  convert  artifact .pt -> steering-spec JSON (wire dict + provenance)
  validate spec JSON vs the source .pt: materialize via the engine's own
           `materialize()` and require exact per-layer equality (fp32 bit-level,
           tolerance 0.0 on the wire round-trip).

Usage (inside the easysteer env; CPU-only):
  python harness/mean_diff_fixedvec.py convert \
    --artifact results/mean_diff/BoolQ__CONCISE-001__train__<MODEL>__n128.pt \
    --out results/e1_eval_v3/fixedvec/BoolQ__CONCISE-001__<MODEL>.fixedvec.json
  python harness/mean_diff_fixedvec.py validate --spec <same json> \
    --artifact <same pt>
"""

import argparse
import base64
import hashlib
import json
import os
import sys

import torch

from vllm.steer_vectors.payloads import DirectionVector, materialize


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vectors(artifact: str, policy: str):
    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    suffix = "" if policy == "primary" else "__last_token"
    keys = [k for k in blob["vectors"] if k.endswith("_block" + suffix)]
    layers = {}
    for k in keys:
        li = int(k.split("_")[0][len("layer"):])
        layers[li] = blob["vectors"][k].float().contiguous()
    assert layers, f"no layer vectors for policy {policy!r} in {artifact}"
    return blob, dict(sorted(layers.items()))


def cmd_convert(args):
    blob, layers = load_vectors(args.artifact, args.policy)
    dv = DirectionVector(layers)          # the fork's fixed-vector payload
    wire = dv.to_wire()
    # base64 the tensor data so the spec travels as JSON (engine accepts both)
    for name, entry in wire["tensors"].items():
        raw = entry["data"]
        if isinstance(raw, (bytes, bytearray)):
            entry["data"] = base64.b64encode(bytes(raw)).decode("ascii")
    spec = {
        "schema": "e1_eval_v3.fixedvec/1",
        "algorithm": "direct",
        "scale": args.scale,
        "normalize": False,
        "policy": args.policy,
        "source_artifact": os.path.abspath(args.artifact),
        "source_artifact_sha256": sha256_file(args.artifact),
        "mean_diff_meta": {
            k: blob[k] for k in ("model", "dataset", "harness_id", "n_examples",
                                 "position_policy_primary", "hidden_size")
            if k in blob
        },
        "layers": sorted(layers.keys()),
        "wire": wire,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    print(f"[fixedvec] wrote {args.out}: {len(layers)} layers, "
          f"hidden={next(iter(layers.values())).shape[0]}, policy={args.policy}")


def cmd_validate(args):
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    _, layers = load_vectors(args.artifact, spec["policy"])
    payload = dict(spec["wire"])
    mat = materialize(payload, device="cpu", dtype=torch.float32,
                      target_layers=None)
    assert set(mat.keys()) == set(layers.keys()), "layer keys diverged"
    worst = 0.0
    for li, vec in layers.items():
        got = mat[li]
        worst = max(worst, float((got - vec).abs().max()))
    assert worst == 0.0, f"wire round-trip not exact: max abs diff {worst}"
    print(f"[fixedvec] validate OK: {len(layers)} layers exact (max abs diff 0.0)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--artifact", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--policy", choices=["primary", "last_token"],
                   default="primary")
    c.add_argument("--scale", type=float, default=1.0)
    v = sub.add_parser("validate")
    v.add_argument("--spec", required=True)
    v.add_argument("--artifact", required=True)
    args = p.parse_args()
    dict(convert=cmd_convert, validate=cmd_validate)[args.cmd](args)


if __name__ == "__main__":
    main()
