import os, argparse
import numpy as np, pandas as pd, soundfile as sf
from pathlib import Path
from data.transforms_audio import to_logmel, cmvn_feat


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data_cache')
    ap.add_argument('--n_mels', type=int, default=64)
    ap.add_argument('--n_fft', type=int, default=1024)
    ap.add_argument('--win_ms', type=int, default=25)
    ap.add_argument('--hop_ms', type=int, default=10)
    ap.add_argument('--cmvn', action='store_true')
    args = ap.parse_args()


    cache = Path(args.cache)
    seg = pd.read_csv(cache/'segments.csv')
    feat_root = cache/'features'
    feat_root.mkdir(parents=True, exist_ok=True)


    feats = []
    for i,row in seg.iterrows():
        wav = cache/'clean'/row['wav_relpath']
        y, sr = sf.read(wav)
        start = int(row['start']*sr)
        dur = int(row['duration']*sr)
        clip = y[start:start+dur]
        S = to_logmel(clip, sr, args.n_mels, args.n_fft, args.win_ms, args.hop_ms)
        if args.cmvn:
            S = cmvn_feat(S)
        rel = row['wav_relpath']
        out_rel = rel.replace('.wav','.npy')
        out_path = feat_root/out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, S)
        feats.append(out_rel)
    seg['feat_relpath'] = feats
    seg.to_csv(cache/'segments.csv', index=False)
    print('Saved features to', feat_root)