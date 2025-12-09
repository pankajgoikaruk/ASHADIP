import argparse
from pathlib import Path

import pandas as pd


def make_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table summarising multiple runs/variants.

    Expected columns in df (if present):
      - variant
      - run_id
      - tau
      - policy_test_acc
      - compute_saving_pct
      - exit_e1, exit_e2, exit_e3
      - expected_mflops
      - full_mflops
      - ece_policy

    Any missing column will just be shown as '--'.

    We also compute:
      - avg_exit_depth = 1*exit_e1 + 2*exit_e2 + 3*exit_e3
        (assuming exit_e* are fractions in [0,1]).
    """

    # Sort (if available): by variant, then by policy_test_acc descending
    sort_cols = []
    if "variant" in df.columns:
        sort_cols.append("variant")
    if "policy_test_acc" in df.columns:
        sort_cols.append("policy_test_acc")

    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df = df.sort_values(sort_cols, ascending=ascending)

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Policy accuracy, compute saving, and exit behaviour across ASHADIP variants.}"
    )
    lines.append(r"  \label{tab:ashadip_variants_summary}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    # 12 columns: 2 left-aligned + 10 right-aligned
    lines.append(r"  \begin{tabular}{llrrrrrrrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Variant & Run ID & $\tau$ & Acc$_{\text{policy}}$ & Save~(\%) & Avg depth & "
        r"Exit1~(\%) & Exit2~(\%) & Exit3~(\%) & Exp.~MFLOPs & Full.~MFLOPs & ECE$_{\text{policy}}$ \\"
    )
    lines.append(r"    \midrule")

    def get(row, key, default="--"):
        return row[key] if key in row and pd.notna(row[key]) else default

    def fmt_pct_frac(x):
        """fraction [0,1] -> percentage with 1 decimal"""
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x) * 100.0:.1f}"
        except Exception:
            return "--"

    def fmt_pct(x):
        """value already in % -> 1 decimal"""
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "--"

    def fmt_float(x, ndigits=1):
        if x is None or x == "--":
            return "--"
        try:
            fmt = f"{{:.{ndigits}f}}"
            return fmt.format(float(x))
        except Exception:
            return "--"

    def fmt_tau(x):
        """tau (threshold) as ~0.92"""
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "--"

    def fmt_ece(x):
        """ECE typically small; show 3 decimals"""
        if x is None or x == "--":
            return "--"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "--"

    for _, row in df.iterrows():
        variant   = get(row, "variant")
        run_id    = get(row, "run_id")

        tau       = get(row, "tau", None)
        acc       = get(row, "policy_test_acc", None)
        save      = get(row, "compute_saving_pct", None)
        e1        = get(row, "exit_e1", None)
        e2        = get(row, "exit_e2", None)
        e3        = get(row, "exit_e3", None)
        exp_flops = get(row, "expected_mflops", None)
        full_flops= get(row, "full_mflops", None)
        ece_pol   = get(row, "ece_policy", None)

        # Compute average exit depth if possible: 1*e1 + 2*e2 + 3*e3
        avg_depth = None
        try:
            if (e1 not in (None, "--")) and (e2 not in (None, "--")) and (e3 not in (None, "--")):
                e1_f = float(e1)
                e2_f = float(e2)
                e3_f = float(e3)
                avg_depth = 1.0 * e1_f + 2.0 * e2_f + 3.0 * e3_f
        except Exception:
            avg_depth = None

        # Format numeric values
        tau_str      = fmt_tau(tau)
        acc_str      = fmt_pct_frac(acc)      # policy_test_acc in [0,1]
        save_str     = fmt_pct(save)          # compute_saving_pct already in %
        avg_depth_str= fmt_float(avg_depth, ndigits=2)
        e1_str       = fmt_pct_frac(e1)       # fractions -> %
        e2_str       = fmt_pct_frac(e2)
        e3_str       = fmt_pct_frac(e3)
        exp_str      = fmt_float(exp_flops, ndigits=1)
        full_str     = fmt_float(full_flops, ndigits=1)
        ece_str      = fmt_ece(ece_pol)

        lines.append(
            rf"    {variant} & {run_id} & {tau_str} & {acc_str} & {save_str} & {avg_depth_str} & "
            rf"{e1_str} & {e2_str} & {e3_str} & {exp_str} & {full_str} & {ece_str} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }")  # end resizebox
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        default="analysis/all_runs_summary.csv",
        help="Path to all_runs_summary.csv produced by compare_variants.py",
    )
    ap.add_argument(
        "--out_tex",
        default="analysis/tables/variants_avg_summary_table.tex",
        help="Output LaTeX file for the variants summary table.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    if df.empty:
        raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")

    table_str = make_latex_table(df)

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print(f"[variants_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()






# import argparse
# from pathlib import Path
#
# import pandas as pd
#
#
# def make_latex_table(df: pd.DataFrame, run_label: str = "ASHADIP variants summary") -> str:
#     """
#     Build a LaTeX table summarising multiple runs/variants.
#
#     Expected columns in df (if present):
#       - variant
#       - run_id
#       - policy_test_acc
#       - compute_saving_pct
#       - exit_e1, exit_e2, exit_e3
#       - expected_mflops
#       - full_mflops
#
#     Any missing column will just be shown as '--'.
#     """
#
#     # Sort (if available): by variant, then by policy_test_acc descending
#     sort_cols = []
#     if "variant" in df.columns:
#         sort_cols.append("variant")
#     if "policy_test_acc" in df.columns:
#         sort_cols.append("policy_test_acc")
#
#     if sort_cols:
#         ascending = [True] + [False] * (len(sort_cols) - 1)
#         df = df.sort_values(sort_cols, ascending=ascending)
#
#     lines = []
#     lines.append(r"\begin{table*}[ht]")
#     lines.append(r"  \centering")
#     lines.append(r"  \caption{Policy accuracy vs Compute saving / Exit mix across ASHADIP variants.}")
#     lines.append(r"  \label{tab:ashadip_variants_summary}")
#     lines.append(r"  \resizebox{\textwidth}{!}{%")
#     lines.append(r"  \begin{tabular}{llrrrrrrr}")
#     lines.append(r"    \toprule")
#     lines.append(
#         r"    Variant & Run ID & Acc$_{\text{policy}}$ & Save~(\%) & Exit1~(\%) & Exit2~(\%) & Exit3~(\%) & Exp.~MFLOPs & Full~MFLOPs \\"
#     )
#     lines.append(r"    \midrule")
#
#     def get(row, key, default="--"):
#         return row[key] if key in row and pd.notna(row[key]) else default
#
#     for _, row in df.iterrows():
#         variant   = get(row, "variant")
#         run_id    = get(row, "run_id")
#
#         acc       = get(row, "policy_test_acc", None)
#         save      = get(row, "compute_saving_pct", None)
#         e1        = get(row, "exit_e1", None)
#         e2        = get(row, "exit_e2", None)
#         e3        = get(row, "exit_e3", None)
#         exp_flops = get(row, "expected_mflops", None)
#         full_flops= get(row, "full_mflops", None)
#
#         # Format numeric values if present
#         def fmt_pct_frac(x):
#             if x is None or x == "--":
#                 return "--"
#             try:
#                 return f"{float(x)*100:.1f}"
#             except Exception:
#                 return "--"
#
#         def fmt_pct(x):
#             if x is None or x == "--":
#                 return "--"
#             try:
#                 return f"{float(x):.1f}"
#             except Exception:
#                 return "--"
#
#         def fmt_float(x):
#             if x is None or x == "--":
#                 return "--"
#             try:
#                 return f"{float(x):.1f}"
#             except Exception:
#                 return "--"
#
#         acc_str   = fmt_pct_frac(acc)    # policy_test_acc in [0,1]
#         save_str  = fmt_pct(save)        # compute_saving_pct already in %
#         e1_str    = fmt_pct_frac(e1)     # fractions -> %
#         e2_str    = fmt_pct_frac(e2)
#         e3_str    = fmt_pct_frac(e3)
#         exp_str   = fmt_float(exp_flops)
#         full_str  = fmt_float(full_flops)
#
#         lines.append(
#             rf"    {variant} & {run_id} & {acc_str} & {save_str} & {e1_str} & {e2_str} & {e3_str} & {exp_str} & {full_str} \\"
#         )
#
#     lines.append(r"    \bottomrule")
#     lines.append(r"  \end{tabular}")
#     lines.append(r"  }")  # end resizebox
#     lines.append(r"\end{table*}")
#
#     return "\n".join(lines)
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--summary_csv",
#         default="analysis/all_runs_summary.csv",
#         help="Path to all_runs_summary.csv produced by compare_variants.py",
#     )
#     ap.add_argument(
#         "--out_tex",
#         default="analysis/tables/variants_summary_table.tex",
#         help="Output LaTeX file for the variants summary table.",
#     )
#     args = ap.parse_args()
#
#     summary_path = Path(args.summary_csv)
#     if not summary_path.exists():
#         raise SystemExit(f"Summary CSV not found: {summary_path}")
#
#     out_tex_path = Path(args.out_tex)
#     out_tex_path.parent.mkdir(parents=True, exist_ok=True)
#
#     df = pd.read_csv(summary_path)
#
#     if df.empty:
#         raise SystemExit(f"No rows in {summary_path}; nothing to summarise.")
#
#     # ---- NEW: save a sorted CSV backup next to the .tex ----
#     # Use the same sorting logic as in make_latex_table
#     table_df = df.copy()
#     sort_cols = []
#     if "variant" in table_df.columns:
#         sort_cols.append("variant")
#     if "policy_test_acc" in table_df.columns:
#         sort_cols.append("policy_test_acc")
#     if sort_cols:
#         ascending = [True] + [False] * (len(sort_cols) - 1)
#         table_df = table_df.sort_values(sort_cols, ascending=ascending)
#
#     out_csv_path = out_tex_path.with_suffix(".csv")
#     table_df.to_csv(out_csv_path, index=False)
#     print(f"[variants_to_latex] Wrote CSV backup to {out_csv_path}")
#
#     # Build LaTeX from the same sorted DataFrame
#     table_str = make_latex_table(table_df)
#
#     with open(out_tex_path, "w", encoding="utf-8") as f:
#         f.write(table_str + "\n")
#
#     print(f"[variants_to_latex] Wrote LaTeX table to {out_tex_path}")
#
#
# if __name__ == "__main__":
#     main()
#
