import os, json, torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, softmax
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


class TempScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Log-parameterization so T is strictly positive (T = exp(log_t))
        self.log_t = torch.nn.Parameter(torch.zeros(1))  # init T=1.0
    def forward(self, logits):
        T = torch.exp(self.log_t)
        return logits / T.clamp(min=1e-3)


@torch.no_grad()
def collect_val_logits(model, dl, device, max_samples=None):
    model.eval()
    all_logits = [[],[],[]]; all_y = []
    seen = 0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        lg = model(x)
        for k in range(3): all_logits[k].append(lg[k].cpu())
        all_y.append(y.cpu())
        seen += x.size(0)
        if max_samples and seen >= max_samples: break
    all_logits = [torch.cat(L,0) for L in all_logits]
    all_y = torch.cat(all_y,0)
    return all_logits, all_y


def fit_temperature_for_exit(logits, y, max_iter=60, verbose=True):
    device = logits.device
    ts = TempScale().to(device)
    opt = torch.optim.LBFGS(ts.parameters(), lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cross_entropy(ts(logits), y)
        loss.backward()
        return loss

    last = None
    for i in range(max_iter):
        loss = opt.step(closure)
        cur = float(loss.detach().cpu())
        T_val = float(torch.exp(ts.log_t).detach().cpu())
        if verbose and (last is None or abs(cur - last) > 1e-5):
            print(f"  iter {i+1:02d}: loss={cur:.6f}, T={T_val:.4f}")
        last = cur
    # Return the positive temperature
    return float(torch.exp(ts.log_t).detach().cpu())

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--segments_csv', default='data_cache/segments.csv')
    ap.add_argument('--features_root', default='data_cache/features')
    ap.add_argument('--max_samples', type=int, default=0, help='limit val samples (0=all)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # val loader only
    _, dl_va, _, _ = make_loaders(args.segments_csv, args.features_root, batch_size=128, num_workers=2)

    # load model
    model = ExitNet(TinyAudioCNN(), (16,32), 64, 2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir,'ckpt','best.pt'), map_location=device))

    # collect logits once
    print("Collecting validation logits...")
    logits, y = collect_val_logits(model, dl_va, device, max_samples=(args.max_samples or None))
    print(f"Collected {y.numel()} samples.")

    # fit temperature per exit
    temps = []
    for k in range(3):
        print(f"Fitting temperature for exit {k+1}...")
        t = fit_temperature_for_exit(logits[k].to(device), y.to(device), max_iter=40, verbose=True)
        temps.append(t)

    out = {'temperatures': temps}
    with open(os.path.join(args.run_dir,'temperature.json'),'w') as f:
        json.dump(out, f, indent=2)
    print("Saved temperature.json:", out)