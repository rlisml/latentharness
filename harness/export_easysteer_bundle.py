"""Export a trained WUAS post-block low-rank adapter into a sidecar
bundle for the EasySteer parity spike (the parity spec, Task 1).

INFRASTRUCTURE ONLY. Read-only w.r.t. the source checkpoints; writes a NEW
bundle directory:

    <out-dir>/
      manifest.json       schema, provenance, rank/alpha/use_silu/layers, key map
      adapter.safetensors per-layer `layer{i}.down` / `layer{i}.up` (fp32,
                          exact copies of the checkpoint tensors)

and performs the pre-registered load-back unit checks (both must pass):
  1. tensor level  — every reloaded tensor vs its source checkpoint tensor:
     required max abs diff == 0.0 (exact copies);
  2. functional    — delta h = up(act(down(h))) rebuilt from the reloaded
     bundle vs the source `steerer.LowRankAdapter` module on fixed seeded
     random inputs: required max abs diff <= 1e-6 (fp32).

The bundle's per-layer payload (down, up, use_silu, alpha) is what the
`wuas_adapter` plugin's load_from_path consumes (Task 2); the plugin is
self-contained — the silu flag travels inside the manifest and inside every
layer payload dict.

Usage (run in any Python env with torch):
  python harness/export_easysteer_bundle.py \
    --checkpoint BoolQ_model/Llama-3.2-1B-Instruct_adapter_all_linear/lr0.001_bs8_ep1_warmup0.03_wd0.0_rank8/pytorch_model.bin \
    --use-silu false --model-name meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name BoolQ --out-dir results/easysteer_parity/bundles/wuas_task_adapter_boolq_linear
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from steerer import LowRankAdapter  # noqa: E402  (the frozen intervention)

ADAPTER_KEY_RE = re.compile(r"^adapters\.layer(\d+)_block\.(down|up)\.weight$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_adapters(state_dict):
    """Extract {(layer_idx): {'down': t, 'up': t}} from a train.py-style state
    dict (full model save or adapter-only; both use adapters.layer{i}_block.*)."""
    layers = {}
    for key, tensor in state_dict.items():
        m = ADAPTER_KEY_RE.match(key)
        if not m:
            continue
        idx, which = int(m.group(1)), m.group(2)
        layers.setdefault(idx, {})[which] = tensor.detach().cpu().float().contiguous().clone()
    if not layers:
        raise ValueError(
            f"no adapters.layer*_block.{{down,up}}.weight keys found "
            f"({len(state_dict)} state-dict keys scanned)"
        )
    for idx, pair in layers.items():
        if set(pair) != {"down", "up"}:
            raise ValueError(f"layer {idx} is missing down/up: {sorted(pair)}")
    return layers


def silu(x):
    return x * torch.sigmoid(x)


def adapter_delta(pair, use_silu, alpha, h):
    r = h @ pair["down"].T          # nn.Linear(2048->8): h @ W_down.T
    if use_silu:
        r = silu(r)
    d = r @ pair["up"].T            # nn.Linear(8->2048): x @ W_up.T
    if alpha != 1.0:
        d = alpha * d
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--use-silu", required=True, choices=["true", "false"])
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--harness-id", default=None,
                   help="registry id when the adapter was distilled from a harness (config b)")
    p.add_argument("--harness-sha256", default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    use_silu = args.use_silu == "true"

    from safetensors.torch import load_file, save_file

    src_sha = sha256_file(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layers = extract_adapters(state)
    layer_ids = sorted(layers)
    n_layers = len(layer_ids)
    if layer_ids != list(range(n_layers)):
        raise ValueError(f"non-contiguous layer ids: {layer_ids}")
    rank, hidden = layers[layer_ids[0]]["down"].shape
    for idx in layer_ids:
        assert layers[idx]["down"].shape == (rank, hidden), idx
        assert layers[idx]["up"].shape == (hidden, rank), idx

    # --- write the bundle ---------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tensor_map = {}
    for idx in layer_ids:
        tensor_map[f"layer{idx}.down"] = layers[idx]["down"]
        tensor_map[f"layer{idx}.up"] = layers[idx]["up"]
    st_path = os.path.join(args.out_dir, "adapter.safetensors")
    save_file({k: v.contiguous() for k, v in tensor_map.items()}, st_path)

    manifest = {
        "schema_version": 1,
        "kind": "wuas_lowrank_adapter",
        "rank": rank,
        "alpha": args.alpha,
        "use_silu": use_silu,
        "n_layers": n_layers,
        "layers": layer_ids,
        "hidden_size": hidden,
        "storage_dtype": "float32",
        "tensor_layout": "down: (rank, hidden) as in nn.Linear(2048->8); "
                         "up: (hidden, rank) as in nn.Linear(8->2048); "
                         "delta = alpha * up(silu(down @ h)) with silu skipped when "
                         "use_silu=false; plugin mirrors steerer.py::LowRankAdapter",
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "harness_id": args.harness_id,
        "harness_sha256": args.harness_sha256,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": src_sha,
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_by": "harness/export_easysteer_bundle.py",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- pre-registered load-back unit checks -------------------------------
    reloaded = load_file(st_path)
    tensor_max_diff = 0.0
    for idx in layer_ids:
        for which in ("down", "up"):
            src = layers[idx][which]
            back = reloaded[f"layer{idx}.{which}"]
            assert back.shape == src.shape and back.dtype == src.dtype
            tensor_max_diff = max(tensor_max_diff, (back - src).abs().max().item())

    g = torch.Generator().manual_seed(args.seed)
    h = torch.randn(64, hidden, generator=g, dtype=torch.float32)
    functional_max_diff = 0.0
    for idx in layer_ids:
        module = LowRankAdapter(
            hidden_size=hidden,
            rank=rank,
            alpha=args.alpha,
            activation="silu" if use_silu else None,
            dtype=torch.float32,
        )
        module.eval()
        with torch.no_grad():
            module.down.weight.copy_(layers[idx]["down"])
            module.up.weight.copy_(layers[idx]["up"])
            ref = module(h)                      # steerer.py::LowRankAdapter math
        back_pair = {
            "down": reloaded[f"layer{idx}.down"],
            "up": reloaded[f"layer{idx}.up"],
        }
        ours = adapter_delta(back_pair, use_silu, args.alpha, h)
        functional_max_diff = max(functional_max_diff, (ours - ref).abs().max().item())

    report = {
        "bundle_dir": os.path.abspath(args.out_dir),
        "bundle_safetensors": st_path,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": src_sha,
        "use_silu": use_silu,
        "rank": rank,
        "alpha": args.alpha,
        "n_layers": n_layers,
        "hidden_size": hidden,
        "loadback_tensor_max_abs_diff": tensor_max_diff,
        "loadback_tensor_check_pass": tensor_max_diff == 0.0,
        "loadback_functional_max_abs_diff": functional_max_diff,
        "loadback_functional_check_pass": functional_max_diff <= 1e-6,
        "seed": args.seed,
    }
    assert report["loadback_tensor_check_pass"], \
        f"tensor load-back check FAILED: max abs diff {tensor_max_diff}"
    assert report["loadback_functional_check_pass"], \
        f"functional load-back check FAILED: max abs diff {functional_max_diff}"
    with open(os.path.join(args.out_dir, "loadback_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
