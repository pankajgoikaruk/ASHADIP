import os, json, argparse, csv, time
import numpy as np
import torch
from torch.nn.functional import softmax, cross_entropy
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from utils.profiling import estimate_flops_tiny_audiocnn
import matplotlib.pyplot as plt

def load_json_safepath(path, default=None):
    try:
        with open(path, "r") as f: return json.load(f)
    except FileNotFoundError:
        return default

def ece_score(conf, corr, n_bins=15):
    """conf: (N,) confidences; corr: (N,) correctness (0/1)."""
    conf = np.asarray(conf); corr = np.asarray(corr).astype(np.float32)
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    bin_summ = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        sel = (conf >= lo) & (conf < hi) if b < n_bins-1 else (conf >= lo) & (conf <= hi)
        if not np.any(sel):
            bin_summ.append({"bin":[float(lo), float(hi)], "count":0, "acc":None, "conf":None})
            continue
        p = sel.mean()
        acc = corr[sel].mean()
        cbar = conf[sel].mean()
        ece += p * abs(acc - cbar)
        bin_summ.append({"bin":[float(lo), float(hi)], "count":int(sel.sum()), "acc":float(acc), "conf":float(cbar)})
    return float(ece), bin_summ

def plot_hist_and_reliability(run_dir, key, conf, corr, n_bins=15):
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    # Histogram
    plt.figure()
    plt.hist(conf, bins=30, range=(0,1))
    plt.xlabel("Confidence"); plt.ylabel("Count"); plt.title(f"Confidence Histogram – {key}")
    p_hist = os.path.join(run_dir, "plots", f"{key}_conf_hist.png"); plt.savefig(p_hist, bbox_inches="tight"); plt.close()
    # Reliability
    ece, bins = ece_score(conf, corr, n_bins=n_bins)
    xs, accs, confs = [], [], []
    for b in bins:
        if b["acc"] is not None:
            xs.append(np.mean(b["bin"])); accs.append(b["acc"]); confs.append(b["conf"])
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence (bin mean)"); plt.ylabel("Accuracy (bin)"); plt.title(f"Reliability – {key} (ECE={ece:.3f})")
    p_rel = os.path.join(run_dir, "plots", f"{key}_reliability.png"); plt.savefig(p_rel, bbox_inches="tight"); plt.close()
    return {"ece": ece, "bins": bins, "hist_path": p_hist, "reliability_path": p_rel}


def plot_conf_vs_correct(run_dir, key, conf, corr, jitter=0.02):
    """Scatter: confidence (x) vs correctness (y∈{0,1})."""
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    import numpy as np, matplotlib.pyplot as plt
    y = np.asarray(corr).astype(float)
    x = np.asarray(conf).astype(float)
    # jitter correctness for visibility
    yj = y + (np.random.rand(*y.shape)-0.5)*jitter
    plt.figure()
    plt.scatter(x, yj, s=8, alpha=0.5)
    plt.yticks([0,1], ["wrong","correct"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Confidence"); plt.ylabel("Correctness")
    plt.title(f"Confidence vs Correctness – {key}")
    p_sc = os.path.join(run_dir, "plots", f"{key}_conf_vs_correct.png")
    plt.savefig(p_sc, bbox_inches="tight"); plt.close()
    return p_sc


@torch.no_grad()
def collect_exit_logits_on_split(model, dl, device):
    """Returns dict per-exit: probs (N,C), y (N,), conf (N,), corr (N,) for using that exit directly."""
    all = []
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        lg = model(x)
        pr = [softmax(l, dim=1) for l in lg]
        all.append(( [p.cpu() for p in pr], y.cpu()))
    probs = [torch.cat([a[0][k] for a in all], 0).numpy() for k in range(3)]
    ytrue = torch.cat([a[1] for a in all], 0).numpy()
    out = {}
    for k in range(3):
        conf = probs[k].max(axis=1)
        pred = probs[k].argmax(axis=1)
        corr = (pred == ytrue).astype(np.float32)
        out[f"exit{k+1}"] = {"probs": probs[k], "y": ytrue, "conf": conf, "corr": corr}
    return out

@torch.no_grad()
def policy_eval(run_dir, segments_csv, features_root, save_plots=True):
    # thresholds & temps
    th = load_json_safepath(os.path.join(run_dir, "thresholds.json"), {"tau": 0.95})
    tau = float(th["tau"])
    temps = load_json_safepath(os.path.join(run_dir, "temperature.json"), {"temperatures":[1.0,1.0,1.0]})["temperatures"]
    temps = [max(float(t), 0.5) for t in temps]  # clamp small Ts

    # data & model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr, dl_va, dl_te, label2id = make_loaders(segments_csv, features_root, batch_size=64, num_workers=2)
    num_classes = len(label2id)
    model = ExitNet(TinyAudioCNN(), (16,32), 64, num_classes).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(run_dir,"ckpt","best.pt"), map_location=device))

    # Evaluate greedy early-exit on TEST
    n=0; correct=0; exits=[]; confs=[]; corrs=[]
    def scale(lg,t): return lg/max(t,1e-3)

    for x,y in dl_te:
        x,y = x.to(device), y.to(device)
        logits = [scale(l,temps[i]) for i,l in enumerate(model(x))]
        probs = [softmax(l, dim=1) for l in logits]
        for i in range(x.size(0)):
            taken = 2
            for k in (0,1,2):
                if float(probs[k][i].max()) >= tau: taken=k; break
            p = probs[taken][i]
            pred = int(torch.argmax(p))
            conf = float(torch.max(p))
            corr = float(pred == int(y[i]))
            correct += int(corr); exits.append(taken+1); n += 1
            confs.append(conf); corrs.append(corr)

    p1 = exits.count(1)/n; p2 = exits.count(2)/n; p3 = exits.count(3)/n
    policy_acc = correct/n

    # feature shape for FLOPs
    import pandas as pd, numpy as np
    seg = pd.read_csv(segments_csv)
    test_row = seg[seg["split"]=="test"].iloc[0]
    feat_path = os.path.join(features_root, test_row["feat_relpath"])
    n_mels, frames = np.load(feat_path).shape

    # FLOPs
    fl = estimate_flops_tiny_audiocnn(n_mels=int(n_mels), frames=int(frames), num_classes=num_classes)
    full_mflops = fl["exit3"]/1e6
    expected_mflops = (p1*fl["exit1"] + p2*fl["exit2"] + p3*fl["exit3"])/1e6
    saving_pct = 100.0*(1.0 - expected_mflops/full_mflops)

    # ECE & plots for policy decisions
    policy_calib = {"ece": None, "bins": None, "hist_path": None, "reliability_path": None, "scatter_path": None}
    if save_plots:
        policy_calib = plot_hist_and_reliability(run_dir, "policy_test", np.array(confs), np.array(corrs), n_bins=15)
        policy_scatter = plot_conf_vs_correct(run_dir, "policy_test", np.array(confs), np.array(corrs))
        policy_calib["scatter_path"] = policy_scatter

    # Also per-exit calibration on TEST (no early exit; just each head)
    device2 = device
    _, _, dl_te2, _ = make_loaders(segments_csv, features_root, batch_size=128, num_workers=2)
    per_exit = collect_exit_logits_on_split(model, dl_te2, device2)
    per_exit_calib = {}
    for k in (1,2,3):
        conf = per_exit[f"exit{k}"]["conf"]; corr = per_exit[f"exit{k}"]["corr"]
        per_exit_calib[f"exit{k}"] = plot_hist_and_reliability(run_dir, f"exit{k}_test", conf, corr, n_bins=15)
        # ▼ add the scatter plot per-exit
        sc_path = plot_conf_vs_correct(run_dir, f"exit{k}_test", conf, corr)
        per_exit_calib[f"exit{k}"]["scatter_path"] = sc_path

    return {
        "tau": tau,
        "temperatures": temps,
        "exit_mix": {"e1": p1, "e2": p2, "e3": p3},
        "policy_test_acc": policy_acc,
        "n_mels": int(n_mels), "frames": int(frames), "num_classes": num_classes,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": saving_pct,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "policy_calibration": policy_calib,
        "per_exit_calibration": per_exit_calib
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--report_name", default="summary.json")
    ap.add_argument("--experiments_csv", default="runs/experiments.csv")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    # base artifacts
    metrics = load_json_safepath(os.path.join(args.run_dir, "metrics.json"), {})
    report  = load_json_safepath(os.path.join(args.run_dir, "report.json"),  {})
    calib   = load_json_safepath(os.path.join(args.run_dir, "temperature.json"), {})
    thres   = load_json_safepath(os.path.join(args.run_dir, "thresholds.json"), {})

    # policy stats + FLOPs + calibration (with plots)
    policy = policy_eval(args.run_dir, args.segments_csv, args.features_root, save_plots=not args.no_plots)

    # summary
    run_id = os.path.basename(args.run_dir.rstrip("\\/"))
    summary = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_tail": metrics.get("val", [-1])[-1] if metrics else None,
        "val_curve": metrics.get("val", []),
        "test_report": report,
        "temperature": calib,
        "thresholds": thres,
        "policy_summary": policy,
    }

    out_json = os.path.join(args.run_dir, args.report_name)
    with open(out_json, "w") as f: json.dump(summary, f, indent=2)
    print("Saved", out_json)

    # experiments.csv (append)
    os.makedirs(os.path.dirname(args.experiments_csv), exist_ok=True)
    row = {
        "run_id": run_id,
        "tau": policy["tau"],
        "temp_e1": policy["temperatures"][0],
        "temp_e2": policy["temperatures"][1],
        "temp_e3": policy["temperatures"][2],
        "test_acc_policy": policy["policy_test_acc"],
        "exit_e1": policy["exit_mix"]["e1"],
        "exit_e2": policy["exit_mix"]["e2"],
        "exit_e3": policy["exit_mix"]["e3"],
        "expected_mflops": policy["expected_mflops"],
        "full_mflops": policy["full_mflops"],
        "compute_saving_pct": policy["compute_saving_pct"],
        "n_mels": policy["n_mels"],
        "frames": policy["frames"],
        "num_classes": policy["num_classes"],
        "torch_version": policy["torch_version"],
        "cuda": policy["cuda_available"],
        "ece_policy": policy["policy_calibration"]["ece"],
        "ece_exit1": policy["per_exit_calibration"]["exit1"]["ece"],
        "ece_exit2": policy["per_exit_calibration"]["exit2"]["ece"],
        "ece_exit3": policy["per_exit_calibration"]["exit3"]["ece"],
    }
    write_header = not os.path.exists(args.experiments_csv)
    with open(args.experiments_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print("Logged row to", args.experiments_csv)

if __name__ == "__main__":
    main()
