import argparse
from pathlib import Path

import pandas as pd


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table summarising average on-device latency per variant.

    Expected columns in df (after grouping):
      - n_runs
      - lat_exit1_ms_mean
      - lat_exit2_ms_mean
      - lat_exit3_ms_mean
      - compute_saving_pct_mean
    Index: variant, device
    """
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Average on-device latency per ASHADIP variant (per-exit).}")
    lines.append(r"  \label{tab:on_device_performance}")
    lines.append(r"  \begin{tabular}{llrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Variant & Device & Runs & Exit1 (ms) & Exit2 (ms) & Exit3 (ms) \\"
    )
    lines.append(r"    \midrule")

    for (variant, device), row in df.iterrows():
        n_runs = int(row.get("n_runs", 1))

        def fmt_ms(x):
            if pd.isna(x):
                return "--"
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "--"

        e1 = fmt_ms(row.get("lat_exit1_ms_mean"))
        e2 = fmt_ms(row.get("lat_exit2_ms_mean"))
        e3 = fmt_ms(row.get("lat_exit3_ms_mean"))

        lines.append(
            rf"    {variant} & {device} & {n_runs} & {e1} & {e2} & {e3} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default="analysis/on_device_summary.csv",
        help="CSV produced by profile_latency.py (one row per run).",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/on_device_performance_table.tex",
        help="Output LaTeX file for averaged on-device performance.",
    )
    # NEW: device filter
    ap.add_argument(
        "--device_filter",
        choices=["all", "cpu", "cuda"],
        default="all",
        help="Filter rows by device before averaging (default: all).",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"On-device summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(summary_path)
    if df_runs.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    required_cols = {"variant", "device", "lat_exit1_ms", "lat_exit2_ms", "lat_exit3_ms"}
    missing = required_cols - set(df_runs.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    # NEW: apply device filter if requested
    if args.device_filter != "all":
        df_runs = df_runs[df_runs["device"] == args.device_filter]
        if df_runs.empty:
            raise SystemExit(
                f"No rows left after filtering for device='{args.device_filter}'."
            )

    # Group by variant + device
    grouped = df_runs.groupby(["variant", "device"])

    agg_df = grouped.agg(
        n_runs=("run_id", "count"),
        lat_exit1_ms_mean=("lat_exit1_ms", "mean"),
        lat_exit2_ms_mean=("lat_exit2_ms", "mean"),
        lat_exit3_ms_mean=("lat_exit3_ms", "mean"),
        compute_saving_pct_mean=("compute_saving_pct", "mean"),
    )

    # ALSO save a CSV backup next to the .tex file
    out_csv_path = out_tex_path.with_suffix(".csv")
    agg_df.to_csv(out_csv_path, index=True)
    print(f"[ondevice_to_latex] Wrote CSV backup to {out_csv_path}")

    table_str = make_latex_table(agg_df)

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"[ondevice_to_latex] Wrote LaTeX table to {out_tex_path}")

if __name__ == "__main__":
    main()
