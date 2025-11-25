import os, json, argparse, torch
from torch.nn.functional import softmax
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet

def main(run_dir, segments_csv, features_root):
    tau = json.load(open(os.path.join(run_dir, "thresholds.json")))["tau"]
    temps = json.load(open(os.path.join(run_dir, "temperature.json")))["temperatures"]
    temps = [max(float(t), 0.5) for t in temps]  # stability

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExitNet(TinyAudioCNN(), (16,32), 64, 2).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt", "best.pt"), map_location=device))
    model.eval()

    def scale(logits, t): return logits / max(t, 1e-3)

    _, _, dl_te, _ = make_loaders(segments_csv, features_root, batch_size=64, num_workers=2)

    n = 0; correct = 0; exits = []
    with torch.no_grad():
        for x, y in dl_te:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = [scale(lg, temps[i]) for i, lg in enumerate(logits)]
            probs = [softmax(lg, dim=1) for lg in logits]
            for i in range(x.size(0)):
                taken = 2
                for k in (0,1,2):
                    if float(probs[k][i].max()) >= tau:
                        taken = k; break
                pred = int(torch.argmax(probs[taken][i]))
                correct += int(pred == int(y[i])); exits.append(taken+1); n += 1

    from statistics import mean
    print(f"Policy test accuracy: {correct/n:.4f}")
    print(f"Avg exit depth: {mean(exits):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    args = ap.parse_args()
    main(args.run_dir, args.segments_csv, args.features_root)
