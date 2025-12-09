import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


def load_json_safepath(path, default=None):
    """
    Small helper: load JSON if it exists, otherwise return a default value.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def plot_training_curves(metrics_path: Path, plots_dir: Path):
    """
    Plot training loss and validation accuracy per exit over epochs
    using the content of metrics.json produced by training/train.py.
    """
    metrics = load_json_safepath(metrics_path, {})
    if not metrics:
        print(f"[analyse_run] No metrics.json found at {metrics_path}, skipping training curves.")
        return

    train_hist = metrics.get("train", [])
    val_hist = metrics.get("val", [])

    if not train_hist or not val_hist:
        print("[analyse_run] metrics.json has no train/val history, skipping training curves.")
        return

    # --- Train loss ---
    epochs = [e["epoch"] for e in train_hist]
    losses = [e["loss"] for e in train_hist]

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs epoch")
    plt.grid(True)
    out = plots_dir / "train_loss.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")

    # --- Validation accuracy per exit ---
    # Each val entry has acc = [acc_exit1, acc_exit2, acc_exit3]
    epochs_val = [e["epoch"] for e in val_hist]
    acc_exit1 = [e["acc"][0] for e in val_hist]
    acc_exit2 = [e["acc"][1] for e in val_hist]
    acc_exit3 = [e["acc"][2] for e in val_hist]

    plt.figure()
    plt.plot(epochs_val, acc_exit1, marker="o", label="exit1")
    plt.plot(epochs_val, acc_exit2, marker="o", label="exit2")
    plt.plot(epochs_val, acc_exit3, marker="o", label="exit3")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy per exit vs epoch")
    plt.legend()
    plt.grid(True)
    out = plots_dir / "val_acc_exits.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyse_run] Saved {out}")


@torch.no_grad()
def collect_test_predictions(run_dir: Path,
                             segments_csv: Path,
                             features_root: Path,
                             device: str = None):
    """
    Reload the trained ExitNet model from ckpt/best.pt and run it on the test set.

    Returns:
      y_true       : (N,) numpy array of ground-truth labels
      y_pred_exits : list of length 3, each (N,) array of predicted labels for that exit
      y_prob_exits : list of length 3, each (N, C) array of softmax probabilities for that exit
      label2id     : mapping label string -> int
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data loaders ---
    dl_tr, dl_va, dl_te, label2id = make_loaders(
        str(segments_csv),
        str(features_root),
        batch_size=64,
        num_workers=4
    )
    num_classes = len(label2id)

    # --- Model ---
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint at {ckpt_path}")

    backbone = TinyAudioCNN()
    model = ExitNet(backbone, tap_dims=(16, 32), final_dim=64, num_classes=num_classes).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Run through test set ---
    y_true_list = []
    y_pred_exits = [[], [], []]   # one list per exit
    y_prob_exits = [[], [], []]   # one list per exit

    for x, y in dl_te:
        x = x.to(device)
        y_true_list.extend(y.numpy().tolist())
        logits_list = model(x)
        probs_list = [softmax(lg, dim=1).cpu().numpy() for lg in logits_list]

        # Append predictions & probs per exit
        for k in range(3):
            preds = np.argmax(probs_list[k], axis=1)
            y_pred_exits[k].extend(preds.tolist())
            y_prob_exits[k].append(probs_list[k])

    y_true = np.array(y_true_list, dtype=np.int64)
    # Concatenate prob chunks per exit
    y_prob_exits = [np.concatenate(chunks, axis=0) if len(chunks) > 0 else None
                    for chunks in y_prob_exits]
    y_pred_exits = [np.array(preds, dtype=np.int64) for preds in y_pred_exits]

    return y_true, y_pred_exits, y_prob_exits, label2id


def compute_and_plot_confusion_matrices(y_true,
                                        y_pred_exits,
                                        label2id,
                                        plots_dir: Path,
                                        out_json: Path):
    """
    Compute confusion matrices (counts + row-normalised) for each exit,
    save them as images and a JSON summary.
    """
    id2label = {v: k for k, v in label2id.items()}
    labels_sorted = [id2label[i] for i in range(len(id2label))]

    cm_info = {}

    for exit_idx, y_pred in enumerate(y_pred_exits, start=1):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(id2label))))
        # Row-normalised confusion matrix (each row sums to 1)
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = cm.astype(float) / np.maximum(row_sums, 1e-12)

        # Save plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm_norm, interpolation="nearest")
        plt.title(f"Confusion matrix – exit{exit_idx}")
        plt.colorbar()
        tick_marks = np.arange(len(labels_sorted))
        plt.xticks(tick_marks, labels_sorted, rotation=45, ha="right")
        plt.yticks(tick_marks, labels_sorted)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # Annotate cells with values
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        out_png = plots_dir / f"cm_exit{exit_idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[analyse_run] Saved {out_png}")

        cm_info[f"exit{exit_idx}"] = {
            "labels": labels_sorted,
            "counts": cm.tolist(),
            "row_normalised": cm_norm.tolist(),
        }

    # Save JSON summary for confusion matrices
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cm_info, f, indent=2)
    print(f"[analyse_run] Saved confusion matrices JSON -> {out_json}")
    return cm_info


def compute_and_plot_roc(y_true,
                         y_prob_exits,
                         plots_dir: Path,
                         out_json: Path):
    """
    Compute ROC curves and AUC per exit for binary classification (num_classes == 2).
    If num_classes != 2, this function simply returns {} and does nothing.

    We treat class '1' as the "positive" class by convention.
    """
    if y_prob_exits[0] is None:
        print("[analyse_run] No probabilities for exit1, skipping ROC/AUC.")
        return {}

    num_classes = y_prob_exits[0].shape[1]
    if num_classes != 2:
        print(f"[analyse_run] ROC/AUC currently implemented only for binary tasks, "
              f"but got num_classes={num_classes}. Skipping ROC/AUC.")
        return {}

    roc_info = {}
    y_true_bin = (y_true == 1).astype(int)  # treat label '1' as positive

    for exit_idx, probs in enumerate(y_prob_exits, start=1):
        if probs is None:
            continue
        # positive-class probability
        y_score = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"exit{exit_idx} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curve – exit{exit_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_png = plots_dir / f"roc_exit{exit_idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[analyse_run] Saved {out_png}")

        roc_info[f"exit{exit_idx}"] = {
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    # Save AUC + ROC data to JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(roc_info, f, indent=2)
    print(f"[analyse_run] Saved ROC/AUC JSON -> {out_json}")
    return roc_info


def build_analysis_summary(run_dir: Path,
                           cm_info: dict,
                           roc_info: dict,
                           label_names):
    """
    Aggregate key metrics into one 'analysis_run.json' file:

    - Per-exit classification metrics (precision/recall/F1) from report.json
    - Policy-level metrics from summary.json (accuracy, exit mix, compute saving, ECE)
    - Confusion matrix & AUC info (if available)
    - Label names (id -> human-readable label)
    """
    report = load_json_safepath(run_dir / "report.json", {})
    summary = load_json_safepath(run_dir / "summary.json", {})

    out = {
        "run_id": summary.get("run_id", run_dir.name),
        "classification_per_exit": report,
        "policy_summary": summary.get("policy_summary", {}),
        "confusion_matrices": cm_info,
        "roc_auc": roc_info,
        "label_names": label_names,
    }

    out_path = run_dir / "analysis_run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[analyse_run] Saved consolidated analysis JSON -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to a single run directory, e.g. runs/20251113_142831")
    ap.add_argument("--segments_csv", default="data_cache/segments.csv",
                    help="Path to segments.csv used for this run")
    ap.add_argument("--features_root", default="data_cache/features",
                    help="Root directory for .npy features used for this run")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1) Training curves from metrics.json
    metrics_path = run_dir / "metrics.json"
    plot_training_curves(metrics_path, plots_dir)

    # 2) Test-set predictions, confusion matrices, ROC
    y_true, y_pred_exits, y_prob_exits, label2id = collect_test_predictions(
        run_dir,
        Path(args.segments_csv),
        Path(args.features_root),
    )

    # Build label_names (id -> human-readable label) in index order 0..C-1
    id2label = {v: k for k, v in label2id.items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    cm_json_path = run_dir / "confusion_matrices.json"
    cm_info = compute_and_plot_confusion_matrices(
        y_true,
        y_pred_exits,
        label2id,
        plots_dir,
        cm_json_path,
    )

    roc_json_path = run_dir / "roc_curves.json"
    roc_info = compute_and_plot_roc(
        y_true,
        y_prob_exits,
        plots_dir,
        roc_json_path,
    )

    # 3) Consolidated analysis JSON (now with label_names)
    build_analysis_summary(run_dir, cm_info, roc_info, label_names)


if __name__ == "__main__":
    main()
