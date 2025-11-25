import os, argparse, warnings
import pandas as pd, numpy as np, soundfile as sf, librosa
from pathlib import Path
from data.transforms_audio import bandpass

def rms_dbfs(y):
    if y.size == 0: return -120.0
    return 20*np.log10(np.sqrt(np.mean(y**2))+1e-9)


def safe_read_audio(path, dtype='float32'):
    """Read audio if it's a valid PCM/WAV. Returns (y, sr) or (None, None) if unreadable."""
    try:
        y, sr = sf.read(path, dtype=dtype)
        return y, sr
    except Exception as e:
        warnings.warn(f"Skipping unreadable file: {path} ({e})")
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data')
    ap.add_argument('--cache', default='data_cache')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--segment_sec', type=float, default=1.0)
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--silence_dbfs', type=float, default=-40)
    ap.add_argument('--bandpass', nargs=2, type=float, default=[100,3000])
    args = ap.parse_args()


    root = Path(args.root); cache = Path(args.cache)
    (cache/ 'clean').mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for label in ['male','female']:
        # Ignore MacOS resource-fork files like '._*.wav' and non-wav files
        files = [p for p in (root/label).rglob('*.wav') if not p.name.startswith('._')]
        for wav in sorted(files):
            y, sr = safe_read_audio(wav, dtype='float32')
            if y is None:
                skipped.append(str(wav)); continue
            if y.ndim>1: y = y.mean(axis=1)
            if sr != args.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
                sr = args.sr
            y = y - y.mean()
            if args.bandpass:
                y = bandpass(y, sr, args.bandpass[0], args.bandpass[1])
            # normalize peak to -1 dBFS
            peak = float(np.max(np.abs(y)) + 1e-9)
            y = 0.8913 * y / peak
            out = cache/'clean'/label/wav.name
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, y, sr)
            dur = len(y)/sr
            rows.append({'filepath': str(out), 'label': label, 'duration': dur})


    if skipped:
        print(f"Skipped {len(skipped)} unreadable files (see below):")
        for s in skipped[:10]:
            print(' -', s)
        if len(skipped) > 10:
            print(' ... (more skipped)')


    manifest = pd.DataFrame(rows)
    if len(manifest)==0:
        raise SystemExit('No valid WAVs found. Check your paths or remove non-audio files (e.g., ._*.wav).')
    manifest.to_csv(cache/'moths_manifest.csv', index=False)


    # segmentation
    seg_rows = []
    for _,r in manifest.iterrows():
        y, sr = sf.read(r['filepath'], dtype='float32')
        win = int(args.segment_sec*sr)
        hop = int(args.hop*sr)
        for s in range(0, max(len(y)-win+1, 0), hop):
            seg = y[s:s+win]
            if rms_dbfs(seg) < args.silence_dbfs: continue
            rel = os.path.relpath(r['filepath'], cache/'clean')
            seg_rows.append({'wav_relpath': rel, 'label': r['label'], 'start': s/sr, 'duration': args.segment_sec})
    seg_df = pd.DataFrame(seg_rows)
    if len(seg_df)==0:
        raise SystemExit('No segments above silence threshold; try raising --silence_dbfs (e.g., -55).')


    # split
    from sklearn.model_selection import train_test_split
    train_df, tmp = train_test_split(seg_df, test_size=0.3, stratify=seg_df['label'], random_state=42)
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=42)
    train_df['split']='train'; val_df['split']='val'; test_df['split']='test'
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df.to_csv(cache/'segments.csv', index=False)
    print('Segments:', all_df['split'].value_counts().to_dict())


if __name__ == '__main__':
    main()