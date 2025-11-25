import os, json, argparse, torch
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet

@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None, temps=None):
    """Collect logits once; optionally apply temperature scaling per-exit."""
    model.eval()
    all_logits = [[], [], []]; all_y = []
    seen = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lg = model(x)  # list of 3 tensors
        if temps is not None:
            lg = [l / max(float(temps[i]), 1e-3) for i, l in enumerate(lg)]
        for k in range(3): all_logits[k].append(lg[k].cpu())
        all_y.append(y.cpu())
        seen += x.size(0)
        if max_samples and seen >= max_samples: break
    all_logits = [torch.cat(L, 0) for L in all_logits]
    all_y = torch.cat(all_y, 0)
    return all_logits, all_y

def eval_policy_for_tau(logits_list, y_true, tau):
    """Greedy early-exit: stop at first exit where max prob >= tau."""
    probs = [softmax(l, dim=1) for l in logits_list]
    y_hat = []
    for i in range(y_true.numel()):
        pred = None
        for k in (0, 1, 2):
            if float(probs[k][i].max()) >= tau:
                pred = int(torch.argmax(probs[k][i]))
                break
        if pred is None:
            pred = int(torch.argmax(probs[-1][i]))
        y_hat.append(pred)
    f1 = f1_score(y_true.numpy(), y_hat, average="macro")
    acc = (y_true.numpy() == y_hat).mean()
    return f1, acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--tau", nargs="+", type=float,
                    default=[0.70,0.75,0.80,0.85,0.90,0.92,0.95])
    ap.add_argument("--max_samples", type=int, default=0,
                    help="limit val samples to speed up (0=all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load validation loader (we optimize thresholds on VAL)
    _, dl_val, _, _ = make_loaders(args.segments_csv, args.features_root, batch_size=128, num_workers=2)

    # 2) Load model
    model = ExitNet(TinyAudioCNN(), (16, 32), 64, 2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "ckpt", "best.pt"), map_location=device))

    # 3) Load temperatures if available (and clamp small T)
    temps = None
    tpath = os.path.join(args.run_dir, "temperature.json")
    if os.path.exists(tpath):
        temps = json.load(open(tpath))["temperatures"]
        # optional safety clamp
        temps = [max(float(t), 0.5) for t in temps]
        print("Using temperatures:", temps)
    else:
        print("No temperature.json found; using raw logits.")

    # 4) Precompute logits once
    print("Collecting validation logits...")
    logits_val, y_val = collect_val_logits(model, dl_val, device,
                                           max_samples=(args.max_samples or None),
                                           temps=temps)
    print(f"Collected {y_val.numel()} validation samples.")

    # 5) Sweep tau grid
    best = None
    print("Sweeping tau grid...")
    for tau in args.tau:
        f1, acc = eval_policy_for_tau(logits_val, y_val, tau)
        print(f"  tau={tau:.3f} -> macroF1={f1:.4f}, acc={acc:.4f}")
        if best is None or f1 > best["f1"]:
            best = {"tau": float(tau), "f1": float(f1), "acc": float(acc)}

    # 6) Save best thresholds
    outpath = os.path.join(args.run_dir, "thresholds.json")
    with open(outpath, "w") as f:
        json.dump(best, f, indent=2)
    print("Saved thresholds.json:", best)

if __name__ == "__main__":
    main()
