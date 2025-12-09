import argparse
from pathlib import Path

import pandas as pd


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table summarising the AVERAGE performance per variant.

    Input df is expected to be the output of groupby('variant').agg(...),
    with columns such as:
      - n_runs
      - policy_test_acc_mean, policy_test_acc_std
      - compute_saving_pct_mean
      - exit_e1_mean, exit_e2_mean, exit_e3_mean
      - expected_mflops_mean, full_mflops_mean
    """

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Average early-exit performance per ASHADIP variant.}")
    lines.append(r"  \label{tab:ashadip_variants_avg}")
    lines.append(r"  \begin{tabular}{lrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Variant & Runs & Acc$_{\text{policy}}$ (\%) & Save (\%) & Exit1 (\%) & Exit2 (\%) & Exit3 (\%) \\"
    )
    lines.append(r"    \midrule")

    for _, row in df.iterrows():
        variant = row.name
        n_runs = int(row.get("n_runs", 1))

        def fmt_pct_frac(x):
            if pd.isna(x):
                return "--"
            try:
                return f"{float(x) * 100:.1f}"
            except Exception:
                return "--"

        def fmt_pct(x):
            if pd.isna(x):
                return "--"
            try:
                return f"{float(x):.1f}"
            except Exception:
                return "--"

        acc_mean = fmt_pct_frac(row.get("policy_test_acc_mean"))
        save_mean = fmt_pct(row.get("compute_saving_pct_mean"))
        e1_mean = fmt_pct_frac(row.get("exit_e1_mean"))
        e2_mean = fmt_pct_frac(row.get("exit_e2_mean"))
        e3_mean = fmt_pct_frac(row.get("exit_e3_mean"))

        lines.append(
            rf"    {variant} & {n_runs} & {acc_mean} & {save_mean} & {e1_mean} & {e2_mean} & {e3_mean} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default="analysis/all_runs_summary.csv",
        help="Per-run summary CSV produced by compare_variants.py",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/variants_avg_summary_table.tex",
        help="Output LaTeX table (average per variant).",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(summary_path)

    if df_runs.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    if "variant" not in df_runs.columns:
        raise SystemExit(
            "Column 'variant' not found in all_runs_summary.csv. "
            "Make sure compare_variants.py writes a 'variant' column."
        )

    # Group by variant and compute averages
    grouped = df_runs.groupby("variant")

    agg_df = grouped.agg(
        n_runs=("run_id", "count"),
        policy_test_acc_mean=("policy_test_acc", "mean"),
        compute_saving_pct_mean=("compute_saving_pct", "mean"),
        exit_e1_mean=("exit_e1", "mean"),
        exit_e2_mean=("exit_e2", "mean"),
        exit_e3_mean=("exit_e3", "mean"),
        expected_mflops_mean=("expected_mflops", "mean"),
        full_mflops_mean=("full_mflops", "mean"),
    )

    # (Optional) you could also compute std devs and add to the table later.

    table_str = make_latex_table(agg_df)

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print(f"[variants_avg_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()
