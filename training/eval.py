import os, json, numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix
from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet


@torch.no_grad()
def main(run_dir, segments_csv, features_root, num_classes=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl_tr, dl_va, dl_te, label2id = make_loaders(segments_csv, features_root, 64, 4)
    model = ExitNet(TinyAudioCNN(), tap_dims=(16,32), final_dim=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir,'ckpt','best.pt'), map_location=device))
    model.eval()
    y_true, y_pred = [], [[] for _ in range(3)]
    for x,y in dl_te:
        x = x.to(device)
        logits = model(x)
        for k in range(3):
            y_pred[k].extend(torch.argmax(logits[k],1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
    reports = {f'exit{k+1}': classification_report(y_true, y_pred[k], output_dict=True) for k in range(3)}
    with open(os.path.join(run_dir,'report.json'),'w') as f:
        json.dump(reports, f, indent=2)
    print('Saved report.json')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--segments_csv', default='data_cache/segments.csv')
    ap.add_argument('--features_root', default='data_cache/features')
    args = ap.parse_args()
    main(args.run_dir, args.segments_csv, args.features_root)