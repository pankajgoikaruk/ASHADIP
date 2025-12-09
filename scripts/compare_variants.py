import os
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_summary_files(root: Path):
    """
    Recursively find all summary.json files under directories whose name starts with 'runs'.
    For example:
      runs/20251113_142831/summary.json
      runs_v1/20251120_101500/summary.json
      runs_v2/20251125_093000/summary.json
    """
    summaries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        # Only consider directories like 'runs', 'runs_v1', 'runs_v2', ...
        parts = dirpath.parts
        if "summary.json" in filenames:
            # dirpath is .../<run_id>, its parent is runs / runs_v1 / runs_v2 / ...
            run_dir = dirpath
            runs_root = run_dir.parent
            if runs_root.name.startswith("runs"):
                summaries.append(run_dir / "summary.json")
    return summaries


def parse_variant_from_path(summary_path: Path):
    """
    Infer the variant name from the folder structure.

    Example paths:
      ASHADIP/runs/20251113_142831/summary.json   -> 'v0'  (baseline)
      ASHADIP/runs_v1/20251120_101500/summary.json -> 'v1'
      ASHADIP/runs_v2/20251125_093000/summary.json -> 'v2'
    """
    run_dir = summary_path.parent          # .../<run_id>
    runs_root = run_dir.parent             # .../runs or .../runs_v1

    name = runs_root.name
    if name == "runs":
        return "v0"
    if name.startswith("runs_v"):
        return name.replace("runs_", "")   # 'runs_v1' -> 'v1'
    # fallback
    return name


def load_summary(summary_path: Path):
    """
    Load one summary.json and extract key metrics for comparison.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        summ = json.load(f)

    run_id = summ.get("run_id", summary_path.parent.name)
    policy = summ.get("policy_summary", {})

    tau = policy.get("tau", None)
    temps = policy.get("temperatures", [None, None, None])
    exit_mix = policy.get("exit_mix", {})
    e1 = exit_mix.get("e1", None)
    e2 = exit_mix.get("e2", None)
    e3 = exit_mix.get("e3", None)

    row = {
        "run_id": run_id,
        "variant": parse_variant_from_path(summary_path),
        "tau": tau,
        "temp_e1": temps[0] if len(temps) > 0 else None,
        "temp_e2": temps[1] if len(temps) > 1 else None,
        "temp_e3": temps[2] if len(temps) > 2 else None,
        "policy_test_acc": policy.get("policy_test_acc", None),
        "exit_e1": e1,
        "exit_e2": e2,
        "exit_e3": e3,
        "expected_mflops": policy.get("expected_mflops", None),
        "full_mflops": policy.get("full_mflops", None),
        "compute_saving_pct": policy.get("compute_saving_pct", None),
        "ece_policy": (policy.get("policy_calibration") or {}).get("ece", None),
        "n_mels": policy.get("n_mels", None),
        "frames": policy.get("frames", None),
        "num_classes": policy.get("num_classes", None),
    }
    return row


def make_plots(df: pd.DataFrame, out_dir: Path):
    """
    Make a couple of basic comparison plots:
      1) Accuracy vs compute_saving_pct (scatter, coloured by variant)
      2) Exit mix per run (stacked bar)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accuracy vs compute saving
    plt.figure()
    for variant, df_v in df.groupby("variant"):
        plt.scatter(
            df_v["compute_saving_pct"],
            df_v["policy_test_acc"],
            label=variant,
            s=40,
        )
        for _, row in df_v.iterrows():
            plt.text(
                row["compute_saving_pct"],
                row["policy_test_acc"],
                row["run_id"],
                fontsize=7,
                ha="left",
                va="bottom",
            )
    plt.xlabel("Compute saving (%)")
    plt.ylabel("Policy test accuracy")
    plt.title("Accuracy vs compute saving (by variant)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_compute_saving.png", dpi=150)
    plt.close()

    # 2) Exit mix stacked bar per run
    plt.figure(figsize=(10, 4))
    df_plot = df.reset_index(drop=True)
    x = range(len(df_plot))
    e1 = df_plot["exit_e1"]
    e2 = df_plot["exit_e2"]
    e3 = df_plot["exit_e3"]

    plt.bar(x, e1, label="exit1")
    plt.bar(x, e2, bottom=e1, label="exit2")
    plt.bar(x, e3, bottom=e1 + e2, label="exit3")
    plt.xticks(x, df_plot["run_id"], rotation=45, ha="right")
    plt.ylabel("Fraction of samples")
    plt.title("Exit mix per run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exit_mix_per_run.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Project root to search for runs, runs_v1, runs_v2, ... (default: current dir).",
    )
    ap.add_argument(
        "--out_csv",
        default="analysis/all_runs_summary.csv",
        help="Path to save aggregated CSV.",
    )
    ap.add_argument(
        "--out_dir",
        default="analysis/plots",
        help="Directory to save comparison plots.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    summaries = find_summary_files(root)
    if not summaries:
        raise SystemExit(f"No summary.json files found under {root}")

    rows = [load_summary(p) for p in summaries]
    df = pd.DataFrame(rows).sort_values(["variant", "run_id"])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV to {out_csv}")

    make_plots(df, Path(args.out_dir))
    print(f"Saved comparison plots under {args.out_dir}")


if __name__ == "__main__":
    main()
