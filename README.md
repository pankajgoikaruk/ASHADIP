# ASHADIP v0.1 — Audio (Moth Wingbeats)


### Quickstart
1. Put raw WAVs into `data/male/` and `data/female/`.
2. Create a virtual env, then:
```bash
pip install -r requirements.txt

3. Build segments & manifest:
python scripts/prep_segments.py --root data --cache data_cache \
--sr 16000 --segment_sec 1.0 --hop 0.5 --silence_dbfs -40 \
--bandpass 100 3000

4. Extract log-mel features:
python scripts/extract_features.py --cache data_cache --n_mels 64 \
--n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn

5. Train ExitNet:
python training/train.py --config configs/audio_moth.yaml

6. Calibrate and select offline thresholds (auto-run at end of training). To re-run explicitly:
python training/calibrate.py --run_dir runs/latest
python training/thresholds_offline.py --run_dir runs/latest

