import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from utils.profiling import measure_latency_ms, estimate_flops_tiny_audiocnn


def load_first_test_batch(segments_csv, features_root, batch_size=16, num_workers=2):
    """
    Build loaders and return the first batch from the TEST loader.
    This batch is used to measure latency.
    """
    dl_tr, dl_va, dl_te, label2id = make_loaders(
        segments_csv,
        features_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    x, y = next(iter(dl_te))
    return x, y, label2id


def infer_feature_shape(segments_csv, features_root):
    """
    Load one feature file from the TEST split to infer (n_mels, frames).
    """
    import pandas as pd

    seg = pd.read_csv(segments_csv)
    test_rows = seg[seg["split"] == "test"]
    if test_rows.empty:
        raise SystemExit("No TEST rows found in segments.csv")

    feat_rel = test_rows.iloc[0]["feat_relpath"]
    feat_path = os.path.join(features_root, feat_rel)
    S = np.load(feat_path)  # (n_mels, T)
    n_mels, frames = S.shape
    return int(n_mels), int(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to a single run directory, e.g. runs_v0/20251113_142831")
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--variant", default="V0",
                    help="Variant name label to store (e.g., V0, V1, V2).")
    ap.add_argument("--device", default="auto",
                    help="cpu | cuda | auto (default: auto picks cuda if available)")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Batch size used for latency measurement.")
    ap.add_argument("--n_warm", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # --- Load first test batch ---
    x, y, label2id = load_first_test_batch(
        args.segments_csv,
        args.features_root,
        batch_size=args.batch_size,
        num_workers=2,
    )
    batch_size = x.size(0)
    num_classes = len(label2id)

    # --- Build and load model ---
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    backbone = TinyAudioCNN()
    model = ExitNet(backbone, tap_dims=(16, 32), final_dim=64, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # --- Measure latency for "full" forward (all exits) ---
    latency_full_ms = measure_latency_ms(
        model,
        x,  # measure_latency_ms moves to device internally
        n_warm=args.n_warm,
        n_iter=args.n_iter,
        device=device,
    )

    # --- Estimate FLOPs per exit to approximate per-exit latency ---
    n_mels, frames = infer_feature_shape(args.segments_csv, args.features_root)
    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=num_classes,
    )
    fl1 = float(flops["exit1"])
    fl2 = float(flops["exit2"])
    fl3 = float(flops["exit3"])  # full backbone + final head

    # Approximate per-exit latency by scaling with FLOPs ratio
    lat_exit3_ms = float(latency_full_ms)
    lat_exit1_ms = lat_exit3_ms * (fl1 / fl3)
    lat_exit2_ms = lat_exit3_ms * (fl2 / fl3)

    # --- Load summary.json (for MFLOPs + compute saving) if available ---
    summary_path = run_dir / "summary.json"
    expected_mflops = None
    full_mflops = None
    compute_saving_pct = None

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        policy = summary.get("policy_summary", {})
        expected_mflops = policy.get("expected_mflops", None)
        full_mflops = policy.get("full_mflops", None)
        compute_saving_pct = policy.get("compute_saving_pct", None)

    # --- Build profiling dict ---
    profiling = {
        "variant": args.variant,
        "run_id": run_dir.name,
        "device": device,
        "batch_size": batch_size,
        "latency_ms": {
            "exit1": lat_exit1_ms,
            "exit2": lat_exit2_ms,
            "exit3": lat_exit3_ms,
        },
        "flops": flops,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": compute_saving_pct,
    }

    # Save per-run profiling.json
    out_json = run_dir / "profiling.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profiling, f, indent=2)
    print(f"[profile_latency] Saved {out_json}")

    # --- Append row to analysis/on_device_summary.csv ---
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    csv_path = analysis_dir / "on_device_summary.csv"

    row = {
        "variant": profiling["variant"],
        "run_id": profiling["run_id"],
        "device": profiling["device"],
        "batch_size": profiling["batch_size"],
        "lat_exit1_ms": profiling["latency_ms"]["exit1"],
        "lat_exit2_ms": profiling["latency_ms"]["exit2"],
        "lat_exit3_ms": profiling["latency_ms"]["exit3"],
        "exit1_flops": fl1,
        "exit2_flops": fl2,
        "exit3_flops": fl3,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": compute_saving_pct,
    }

    write_header = not csv_path.exists()
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[profile_latency] Appended row to {csv_path}")


if __name__ == "__main__":
    main()
