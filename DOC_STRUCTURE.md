# ASHADIP_V0 – Documentation & Mini-Book Structure

This document defines the **standard structure** for the “mini-book” that
describes the ASHADIP_V0 project. You can reuse the same structure for future
projects by just swapping the project-specific details.

The goal is simple:

- A new researcher should be able to read the mini-book and understand  
  **what ASHADIP_V0 does, why it exists, and how the code fits together**,
  without needing you.
- The mini-book mirrors the **actual code structure**:
  - `scripts/`, `data/`, `adapters/`, `models/`, `training/`, `utils/`, `policies/`, `tests/`.
- It is written in **LaTeX** (Overleaf) but this file is the *master outline*.


---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain *why* ASHADIP_V0 exists and *what* it solves.

Content to include:

- **Problem context**
  - Domain: acoustic classification of moth wingbeat sounds.
  - Task: binary classification (male vs female) from short audio segments.
  - Motivation: efficient, accurate, and deployable classification on
    resource-constrained devices (first step for the general ASHADIP framework).

- **Project goals**
  - Build a **complete, reproducible baseline pipeline** from raw audio to
    calibrated, early-exit predictions.
  - Evaluate **accuracy vs. efficiency trade-offs** (FLOPs, exit depth).
  - Provide a **modular codebase** that can later be extended to gunshot/wildlife
    acoustics and more advanced early-exit policies.

- **High-level idea**
  - Raw moth audio → preprocessing & segmentation → log-mel features →
    multi-exit CNN (TinyAudioCNN + ExitNet) → temperature calibration →
    offline τ threshold selection → early-exit policy evaluation with FLOPs.

- **Contributions / novelty (already drafted in LaTeX)**
  - C1: First *calibrated multi-exit baseline* for moth wingbeat acoustics.
  - C2: Integrated **per-exit temperature scaling + offline τ selection**.
  - C3: **Compute-aware** evaluation (exit mix, expected FLOPs, calibration).
  - C4: Modular, **reusable design** as a foundation for later ASHADIP variants.

**Main supporting files**

- Conceptual only (no direct code); cite code at a high level:
  - Overall repo structure.
  - Mention that details are in the next chapters.

---

## Chapter 2 – System Overview of ASHADIP_V0

**Goal:** Give a **top-down view** of the entire pipeline before details.

Content:

- **End-to-end pipeline (step list)**

  1. Raw audio in `data/male`, `data/female`.
  2. `scripts/prep_segments.py`: clean, segment, create `segments.csv`.
  3. `scripts/extract_features.py`: compute log-mel `.npy` features.
  4. `data/datasets.py`: build PyTorch datasets / loaders.
  5. `adapters/TinyAudioCNN` + `models/ExitNet`: define multi-exit model.
  6. `training/train.py`: train multi-exit model, save `best.pt`.
  7. `training/calibrate.py`: per-exit temperature scaling, `temperature.json`.
  8. `training/thresholds_offline.py`: select τ on validation, `thresholds.json`.
  9. `training/eval.py`: per-exit test metrics, `report.json`.
  10. `scripts/policy_test.py` + `scripts/summarize_run.py`:
      early-exit policy performance, FLOPs, calibration plots, `summary.json`,
      `experiments.csv`.

- **Configuration & experiment management**
  - `configs/audio_moth.yaml`
    - `paths`: `data_root`, `cache_root`, `runs_root`.
    - `audio`, `features`, `train`, `model`, `calibration`, `thresholds`, `split`.
  - `utils/config.py`: `parse_args_with_config`, `ensure_dirs`.
  - `utils/logging.py`: `make_run_dir`, `save_json`.
  - `scripts/set_paths.ps1`: switch between v0/v1 cache/run directories.
  - `scripts/run_fresh.ps1`: end-to-end runner (steps 1→6).

**Main files to reference**

- `configs/audio_moth.yaml`
- `scripts/run_fresh.ps1`
- `scripts/set_paths.ps1`
- `utils/config.py`
- `utils/logging.py`

---

## Chapter 3 – Data and Preprocessing Pipeline

**Goal:** Describe raw data, cleaning, segmentation, and the manifest.

Content:

1. **Raw data structure**
   - Assume structure:
     - `data/male/*.wav`
     - `data/female/*.wav`
   - Describe sampling rates, typical durations, expected noise.

2. **Preprocessing steps**
   - Implemented in `scripts/prep_segments.py` using:
     - `safe_read_audio` (robust I/O, skip corrupted files).
     - Resampling to 16 kHz (via `librosa.resample`).
     - Mono conversion and mean removal.
     - Band-pass filtering (`data/transforms_audio.bandpass`).
     - Peak normalisation to ≈ −1 dBFS.
   - Segmentation:
     - Window length (e.g. 1.0s) and hop (0.5s).
     - Silence filtering using `rms_dbfs` and a `silence_dbfs` threshold.
   - Manifest:
     - `moths_manifest.csv` – all cleaned WAVs with label + duration.
     - `segments.csv` – per-segment with:
       - `wav_relpath`, `label`, `start`, `duration`, `split`.

3. **Train/val/test split**
   - Done in `scripts/prep_segments.py` (stratified via scikit-learn).

**Main files**

- `scripts/prep_segments.py`
- `data/transforms_audio.py` (for `bandpass`)

---

## Chapter 4 – Features and Representation

**Goal:** Explain log-mel representation and how data becomes PyTorch batches.

Content:

1. **Feature choice – log-mel spectrograms**
   - Why log-mel (perceptual, standard in audio ML).
   - Parameters: `n_mels`, `n_fft`, `win_ms`, `hop_ms`, `cmvn`.

2. **Feature extraction**
   - Implemented in `scripts/extract_features.py`:
     - For each row in `segments.csv`:
       - Read cleaned WAV, cut segment `[start, start+duration]`.
       - `to_logmel` → mel spectrogram in dB.
       - Optional `cmvn_feat` → per-frequency normalisation.
       - Save to `.npy` under `cache/features/...`.
       - Add `feat_relpath` column to `segments.csv`.

3. **Dataset & loaders**
   - `data/datasets.py`:
     - `LogMelDataset`:
       - Loads `segments.csv`, filters by `split`.
       - Loads `.npy` file, wraps as tensor `(1, M, T)`.
       - Encodes label with `label2id`.
     - `make_loaders`:
       - Returns `dl_tr`, `dl_va`, `dl_te`, `label2id`.

**Main files**

- `scripts/extract_features.py`
- `data/transforms_audio.py` (`to_logmel`, `cmvn_feat`)
- `data/datasets.py`

---

## Chapter 5 – Model Architecture (TinyAudioCNN + ExitNet)

**Goal:** Describe the multi-exit network itself.

Content:

1. **Backbone – TinyAudioCNN**
   - `adapters/audio_adapter.py`:
     - Conv block 1:
       - 1 → 16 channels, BN, ReLU, MaxPool(2×2).
     - Conv block 2:
       - 16 → 32 channels, BN, ReLU, MaxPool(2×2).
     - Conv block 3:
       - 32 → 64 channels, BN, ReLU, AdaptiveAvgPool(1×1).
     - Outputs:
       - `t1` = pooled + averaged features after block1 (ℝ¹⁶).
       - `t2` = pooled + averaged features after block2 (ℝ³²).
       - `t3` = flattened final feature (ℝ⁶⁴).

2. **Multi-exit wrapper – ExitNet**
   - `models/exit_net.py`:
     - Three heads:
       - `exit1`: Linear(16 → C)
       - `exit2`: Linear(32 → C)
       - `final`: Linear(64 → C)
     - `forward(x)`:
       - Gets `(final_feat, [t1, t2])` from backbone.
       - Returns `[logits1, logits2, logits3]`.

3. **Role of exits**
   - Exit 1: shallow, cheapest, least accurate.
   - Exit 2: mid-level.
   - Exit 3: full model (deepest, most accurate).

**Main files**

- `adapters/audio_adapter.py`
- `models/exit_net.py`

---

## Chapter 6 – Training, Calibration, and Threshold Selection

**Goal:** Explain how the model is trained, calibrated, and how τ is chosen.

### 6.1 Training (training/train.py)

- Uses:
  - `parse_args_with_config` to load `audio_moth.yaml`.
  - `make_loaders` to get train/val/test loaders.
  - `TinyAudioCNN` + `ExitNet` with `num_classes`.
- Training loop:
  - `train_one_epoch`:
    - Takes outputs `[logits1, logits2, logits3]`.
    - Computes `cross_entropy` per exit.
    - Uses weighted sum with `loss_weights` (e.g. `[0.3, 0.3, 1.0]`).
    - Tracks per-exit accuracy.
  - `evaluate`:
    - Computes per-exit accuracy on validation set.
  - Saves `ckpt/best.pt` when final exit (exit3) val accuracy improves.
- Logs metrics to `metrics.json`.

### 6.2 Temperature calibration (training/calibrate.py)

- Uses validation loader only.
- `collect_val_logits`:
  - Caches logits per exit + labels (no gradients).
- `TempScale`:
  - Scalar temperature T, parameterised via `log_t` for positivity.
- `fit_temperature_for_exit`:
  - Optimises T per exit via LBFGS to minimise cross-entropy.
- Outputs:
  - `temperature.json` with `"temperatures": [T1, T2, T3]`.

### 6.3 Offline τ selection (training/thresholds_offline.py)

- Loads:
  - `best.pt`
  - `temperature.json` (if available; clamps T ≥ 0.5)
  - Validation loader.
- `collect_val_logits`:
  - Caches temperature-scaled logits per exit + labels.
- `eval_policy_for_tau`:
  - For each candidate τ:
    - Greedy early-exit:
      - Check exit1 → exit2 → exit3; exit at first where max prob ≥ τ.
      - Fallback: if none, use final exit.
    - Compute macro-F1 and accuracy on validation set.
- Chooses τ with best macro-F1.
- Saves `thresholds.json` with `"tau"`, `"f1"`, `"acc"`.

### 6.4 Future policy extension stub (policies/early_exit.py)

- `should_exit(prob_k, pval_k, tau_k, alpha_k)`:
  - Combines:
    - Confidence: `maxp >= tau_k`.
    - Conformal p-value: `pval_k >= 1 - alpha_k`.
- Not yet used in V0; kept as a hook for future conformal/uncertainty-aware
  ASHADIP variants.

**Main files**

- `training/train.py`
- `training/calibrate.py`
- `training/thresholds_offline.py`
- `policies/early_exit.py`

---

## Chapter 7 – Evaluation, Metrics, and Reporting

**Goal:** Show how we evaluate exits, the early-exit policy, FLOPs and calibration.

### 7.1 Per-exit evaluation (training/eval.py)

- Loads `best.pt` and test loader.
- For each exit:
  - Computes predictions on test set.
  - Uses `sklearn.classification_report` to get precision/recall/F1/support.
- Saves `report.json` with sections:
  - `"exit1"`, `"exit2"`, `"exit3"`.

### 7.2 Policy test (scripts/policy_test.py)

- Loads:
  - `thresholds.json` (τ),
  - `temperature.json` (per-exit T),
  - `best.pt`,
  - test loader.
- Applies:
  - Temperature scaling to logits.
  - Greedy early-exit with τ.
- Computes:
  - Test accuracy under policy.
  - Average exit depth.
- Prints results (no JSON output).

### 7.3 Comprehensive summary (scripts/summarize_run.py)

- Loads:
  - `metrics.json`, `report.json`, `temperature.json`, `thresholds.json`.
- Calls `policy_eval`:
  - Repeats policy evaluation on test set.
  - Computes:
    - Exit mix: fractions of samples exiting at 1, 2, 3.
    - FLOPs per exit via `utils/profiling.estimate_flops_tiny_audiocnn`.
    - Expected FLOPs and compute saving % vs always exit3.
    - Calibration stats for:
      - Policy decisions.
      - Each exit head (ECE, histograms, reliability curves).
- Produces plots:
  - Confidence histograms.
  - Reliability diagrams.
  - Confidence vs correctness scatter plots.
- Saves:
  - `summary.json` in run directory.
  - Appends row to global `experiments.csv` with:
    - τ, temperatures, test accuracy, exit mix, FLOPs, compute savings, ECE, etc.

**Main files**

- `training/eval.py`
- `scripts/policy_test.py`
- `scripts/summarize_run.py`
- `utils/profiling.py`

---

## Chapter 8 – Implementation Details and Reproducibility

**Goal:** Document config, scripts, directory structure, and how to rerun everything.

Content:

1. **Config structure**
   - `configs/audio_moth.yaml`:
     - `paths`, `audio`, `features`, `train`, `model`,
       `calibration`, `thresholds`, `split`.

2. **Versioned caches and runs**
   - `scripts/set_paths.ps1`:
     - Switches between:
       - `data_cache_v0`, `runs_v0`
       - `data_cache_v1`, `runs_v1`.
   - Encourages clean separation of experimental versions.

3. **End-to-end runner**
   - `scripts/run_fresh.ps1`:
     - Step [1/6] prep segments.
     - Step [2/6] extract features.
     - Step [3/6] train ExitNet.
     - Step [4/6] calibrate temperatures.
     - Step [5/6] select τ.
     - Step [6/6] policy test & summarise.

4. **Directory overview**

   - `adapters/` – domain-specific input backbones (e.g. `audio_adapter.py`).
   - `configs/` – YAML experiment configs.
   - `data/` – dataset and audio transform utilities.
   - `models/` – model architectures (e.g. `exit_net.py`).
   - `policies/` – early-exit / decision policies (future extension).
   - `scripts/` – pipeline and analysis scripts.
   - `training/` – training, calibration, threshold selection, eval.
   - `utils/` – generic utilities (config, logging, profiling).
   - `tests/` – placeholder test scripts (currently extra eval scripts).

5. **Reproducibility statement**
   - Specify:
     - Python version, main library versions (PyTorch, librosa, sklearn).
     - How random seeds are set (if/when added).
     - Exact commands to reproduce:
       - `.\scripts\set_paths.ps1` (if needed).
       - `.\scripts\run_fresh.ps1`.

---

## Chapter 9 – Limitations and Outlook

**Goal:** Be honest about what V0 does *not* do, and how it will be extended.

Content:

- **Current limitations**
  - Binary classification only (male vs female).
  - Small backbone (TinyAudioCNN) – not state-of-the-art for all audio tasks.
  - Early-exit policy is a **single global τ**, not adaptive per sample or per class.
  - `policies/early_exit.py` (conformal component) not integrated yet.
  - Only one dataset/domain (moth wingbeats).

- **Planned extensions**
  - Replace / augment backbone for other domains (gunshots, wildlife poaching).
  - More advanced policies:
    - Conformal prediction (p-values + α),
    - DilemmaExitNet, dynamic thresholds, etc.
  - Integration with broader ASHADIP framework (static/dynamic/strategic phases).

- **Role within the PhD**
  - ASHADIP_V0 is the **baseline chapter**:
    - Proves feasibility of calibrated multi-exit audio classification.
    - Provides reusable tooling and code structure.
  - Later chapters build on this:
    - New datasets, domains, policies, and dynamic neural network ideas.

---

## How to Use This Document

- When you write the mini-book in LaTeX, **each chapter above becomes a section**,
  and each bullet becomes a paragraph or subsection.
- When the code changes, update **both**:
  - The actual module docstrings/comments.
  - The relevant section in this `DOC_STRUCTURE.md` file so future you (or students)
    know where everything is.

This structure is general enough that you can **reuse it for other projects**:
just copy this file, rename it, and update:
- the domain & task,
- the pipeline steps,
- the module mappings.
