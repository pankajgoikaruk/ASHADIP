import json
import argparse
from pathlib import Path

import pandas as pd  # NEW: for CSV backup


def load_analysis_json(path):
    """Load analysis_run.json and return its parsed content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_latex_table(classification_per_exit, run_label="ASHADIP_V0 run", label_names=None):
    """
    Build a LaTeX table string from the 'classification_per_exit' block inside analysis_run.json.

    classification_per_exit is expected to have the structure produced by
    sklearn.classification_report(output_dict=True), e.g.:

      {
        "exit1": {
          "0": {"precision": ..., "recall": ..., "f1-score": ..., "support": ...},
          "1": {...},
          "accuracy": 0.96,
          "macro avg": {...},
          "weighted avg": {...}
        },
        "exit2": {...},
        "exit3": {...}
      }

    label_names (optional) is a list like ["female", "male", ...] in index order 0..C-1.
    """

    lines = []

    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{Classification metrics per exit for {run_label}.}}")
    lines.append(r"  \label{tab:" + run_label.replace(" ", "_").lower() + r"_cls}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Class / summary & Precision & Recall & F1-score & Support \\")
    lines.append(r"    \midrule")

    # Helper to format one exit block
    def add_exit_block(exit_name, exit_dict):
        # exit_name: "exit1", "exit2", ...
        # exit_dict: classification_report dict for that exit

        # Header row for this exit
        lines.append(r"    \multicolumn{5}{c}{" + exit_name.capitalize() + r"} \\")
        lines.append(r"    \midrule")

        # Identify class keys vs aggregate keys
        aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
        class_keys = [k for k in exit_dict.keys() if k not in aggregate_keys]

        # Sort class keys so rows are deterministic
        try:
            # if keys are numeric strings ("0","1",...) sort by int
            class_keys = sorted(class_keys, key=lambda x: int(x))
        except ValueError:
            # otherwise sort lexicographically
            class_keys = sorted(class_keys)

        # Per-class rows
        for cls in class_keys:
            stats = exit_dict[cls]
            prec = stats.get("precision", 0.0)
            rec = stats.get("recall", 0.0)
            f1 = stats.get("f1-score", 0.0)
            sup = stats.get("support", 0)

            # Map class index to human-readable label if label_names is provided
            if label_names is not None:
                try:
                    cls_idx = int(cls)
                    cls_name = label_names[cls_idx]
                except (ValueError, IndexError):
                    cls_name = str(cls)
            else:
                cls_name = str(cls)

            lines.append(
                rf"    {cls_name} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {int(sup)} \\"
            )

        # Accuracy row (single scalar)
        acc = exit_dict.get("accuracy", None)
        if acc is not None:
            lines.append(r"    \midrule")
            lines.append(rf"    accuracy & \multicolumn{{3}}{{r}}{{{acc:.3f}}} & -- \\")

        # Macro and weighted averages
        for agg_key in ["macro avg", "weighted avg"]:
            if agg_key in exit_dict:
                stats = exit_dict[agg_key]
                prec = stats.get("precision", 0.0)
                rec = stats.get("recall", 0.0)
                f1 = stats.get("f1-score", 0.0)
                sup = stats.get("support", 0)
                lines.append(
                    rf"    {agg_key} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {int(sup)} \\"
                )

        # Separate exits
        lines.append(r"    \midrule")

    # We expect exits like "exit1", "exit2", "exit3"
    for exit_name in sorted(classification_per_exit.keys()):
        add_exit_block(exit_name, classification_per_exit[exit_name])

    # Replace the last \midrule with \bottomrule (only if we actually added something)
    for i in range(len(lines) - 1, -1, -1):
        if r"\midrule" in lines[i]:
            lines[i] = lines[i].replace(r"\midrule", r"\bottomrule")
            break

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis_json",
        required=True,
        help="Path to analysis_run.json for a single run.",
    )
    ap.add_argument(
        "--out_tex",
        required=True,
        help="Output .tex file to write the table into.",
    )
    ap.add_argument(
        "--run_label",
        default="ASHADIP_V0 run",
        help="Label used in the table caption and label (e.g. 'V0 baseline').",
    )
    args = ap.parse_args()

    analysis_path = Path(args.analysis_json)
    out_tex_path = Path(args.out_tex)
    out_tex_path.parent.mkdir(parents=True, exist_ok=True)

    analysis = load_analysis_json(analysis_path)
    classification_per_exit = analysis.get("classification_per_exit", None)
    label_names = analysis.get("label_names")

    if classification_per_exit is None:
        raise SystemExit(
            f"No 'classification_per_exit' found in {analysis_path}. "
            f"Did you generate analysis_run.json with analyse_run.py?"
        )

    # -------- CSV backup: flatten per-exit metrics into a DataFrame --------
    rows = []
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}

    for exit_name, exit_dict in classification_per_exit.items():
        for key, stats in exit_dict.items():
            # Accuracy is a scalar, not a dict
            if key == "accuracy":
                rows.append(
                    {
                        "exit": exit_name,
                        "row_type": "summary",
                        "name": "accuracy",
                        "precision": None,
                        "recall": None,
                        "f1_score": None,
                        "support": None,
                        "accuracy": float(stats),
                    }
                )
                continue

            # Everything else (classes, macro avg, weighted avg) is a dict
            if not isinstance(stats, dict):
                continue

            row_type = "summary" if key in aggregate_keys else "class"

            # Map key to human-readable label for class rows
            if (row_type == "class") and (label_names is not None):
                try:
                    cls_idx = int(key)
                    name = label_names[cls_idx]
                except (ValueError, IndexError):
                    name = str(key)
            else:
                name = str(key)

            rows.append(
                {
                    "exit": exit_name,
                    "row_type": row_type,
                    "name": name,
                    "precision": stats.get("precision"),
                    "recall": stats.get("recall"),
                    "f1_score": stats.get("f1-score"),
                    "support": stats.get("support"),
                    "accuracy": None,
                }
            )

    df_csv = pd.DataFrame(rows)
    out_csv_path = out_tex_path.with_suffix(".csv")
    df_csv.to_csv(out_csv_path, index=False)
    print(f"[analysis_to_latex] Wrote CSV backup to {out_csv_path}")

    # -------- LaTeX table --------
    table_str = make_latex_table(
        classification_per_exit,
        run_label=args.run_label,
        label_names=label_names,
    )

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print(f"[analysis_to_latex] Wrote LaTeX table to {out_tex_path}")


if __name__ == "__main__":
    main()
