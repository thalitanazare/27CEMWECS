# Ensemble system identification for recursive WEC simulation

Companion code for *Ensemble System Identification for Recursive Simulation of Wave Energy Converters*, by Thalita Nazaré and Erivelton Nepomuceno.

The method combines sixteen independently simulated models from eight families. A calibration-trained Extra Trees regressor predicts each model's forward simulation risk from past excitation and PTO torques. The resulting weights combine angular-velocity trajectories without feeding the combined output back into the models.

This repository contains the numerical method, the experiments reported in the article and their per-record metrics. Exploratory searches and abandoned variants are excluded.

## Contents

| Path | Purpose |
|---|---|
| `code/run.py` | Fit experts, calibrate combinations and evaluate a layout |
| `code/experts.py`, `code/catalogue.py` | Eight model families and the candidate grid |
| `code/ensemble.py`, `code/stacking.py` | Risk weighting and constant combinations |
| `code/dataset.py` | Preprocessing, partitions and evaluation metrics |
| `code/neural.py`, `code/refinement.py` | Additional refined neural comparator |
| `code/narmax.py`, `code/ocean.py` | NARMAX and native-layout physical references |
| `code/diagnostics.py`, `code/summarise.py` | Risk ranks and clustered paired statistics |
| `code/*config.json` | Numerical settings required for reproduction |
| `data/manifest.json` | Source URLs and SHA-256 checksums |
| `reference/` | Selected numerical results from the article |
| `code/tests/` | Numerical and causality checks |

## Installation

Use Python 3.13 and a CPU environment. PyTorch is used for the temporal convolution and refined neural comparator. Downloading data requires `curl` on the system path.

```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -r code/requirements.txt
```

`code/requirements.lock.txt` records the environment used to validate this release. Floating-point results and stochastic fits can vary with package versions and numerical libraries.

## Data

The [SWELL v2 dataset](https://doi.org/10.17632/n34wcksmts.2) supplies the D1 Wavestar records. The [OCEAN dataset](https://doi.org/10.17632/97cpczdg2r.1) supplies the L0 state-space reference. Raw measurements and fitted models are downloaded or generated locally and are not committed.

```sh
python code/download.py --layout L3
# All six layouts
python code/download.py
# L0 measurements and the physical model
python code/download.py --layout L0 --ocean
```

The downloader verifies SHA-256 checksums against `data/manifest.json`. Add `--verify-only` to verify existing files without downloading.

## Reproduction

Inspect the reported paired comparisons without downloading measurements or fitting models:

```sh
python code/summarise.py
```

Refit the matched-state procedure for one held-out layout:

```sh
python code/run.py --layout L3 --narmax --ablations
python code/summarise.py --input results/L3/matched/metrics.csv
```

Repeat for L0–L5 to reproduce all matched-state comparisons. Candidate fits are cached, so an interrupted run can resume. Full fitting is computationally expensive and can take hours. Delete a run's generated directory before changing its settings.

Excluded-state evaluation removes that state from both training and calibration and refits the complete procedure:

```sh
python code/run.py --layout L3 --exclude ISS1
```

Repeat for ISS1–ISS3 and L3–L5 for the nine excluded-state hold-out folds. Development excluded-state cases use the same command on L0–L2. Family removal and single-channel gate recalibration are enabled with `--ablations`. OCEAN replay is restricted to matched-state L0:

```sh
python code/run.py --layout L0 --ocean
```

Results are written under `results/<layout>/<matched-or-excluded-state>/`, including selected configurations, training/calibration partitions, models, risk-rank diagnostics and metrics. Use repeated `--input` arguments to combine layouts in the summary.

## Numerical details

The article gives the main settings. The following implementation details make the reproduction explicit.

| Setting | Value |
|---|---|
| Sampling and trimming | 200Hz to 20Hz, discard first 30s and last 10s, then 2s at each filtered edge |
| Partition | First-realisation UM/P and broadband UM, chronological 65% split with 80 samples removed on each side |
| Scoring burn-in | 80 samples |
| Expert replicas | Seeds 11, 22, 33 for stochastic families, seed 11 for deterministic fits |
| Context and risk | 128 past samples, lags 1/4/8/16/32, 32-sample forward labels every 8 samples |
| Risk forest | 64 trees, minimum leaf 24, random seed 11, one worker |
| Gate calibration | Leave one calibration record out, temperature 0.03/0.1/0.3 and prior shrinkage 0/0.5/1 |
| Calibration objective | Mean joint temporal/spectral error plus 0.25 times the applicable energy error |
| Constant risk prior | Softmax temperature 0.1 |
| Spectral metric | 256-sample nonoverlapping demeaned blocks with periodic Hann windows |
| Excited support | Union of training PSD bins within 25dB of each record's peak, excluding DC and frequencies at or above 0.4 times the sample rate |
| Energy safeguard | Signed mean power divided by power RMS threshold 0.05 |
| Numerical safeguards | 1e−8 for context variance/risk scaling, 1e−12 for metric norm denominators |
| NARMAX estimation | Three ELS updates, five noise lags, process term counts 5/10/15/20/25/30/35/40/50/60 |
| Cluster uncertainty | 10,000 bootstrap resamples, seed 0, clusters defined by state and policy |

The full grid is in `code/catalogue.py`. Each family has twelve configurations across the two input conventions, giving 192 candidate fits per layout and fold. One configuration per family is retained, including three replicas for each stochastic family, for sixteen experts. The refined neural comparator adds fifteen fits and stays outside this bank. Its teacher fit uses a 16-unit tanh network, 500 maximum L-BFGS iterations and regularisation 1e−4. Recursive refinement uses 256-sample scored windows, Adam learning rate 0.001, gradient clipping 1, window seed 1701 and checkpoints at 25 and 50 updates. Its complete loss settings are in `code/neural_config.json` and `code/refinement_config.json`.


## Validation and licence

```sh
python -m pytest code/tests -q
```

Code is licensed under [MIT](LICENSE). SWELL and OCEAN measurements remain subject to their original dataset licences.