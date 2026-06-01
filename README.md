# HAR-Flower: Federated Learning for Human Activity Recognition

This repository contains the implementation for a federated learning (FL) simulation for Human Activity Recognition (HAR) using accelerometer data from multiple real-world datasets. The system is built on the [Flower (flwr)](https://flower.ai) framework and supports FedAvg, FedPer, and Experience Replay strategies across CNN and MLP model architectures.

> **Thesis:** *Federated Learning for Human Activity Recognition: Evaluating FedAvg, FedPer, and Experience Replay under Sequential, Cross-Dataset Conditions*

---

## Repository Structure

```
.
├── harflwr/                        # Main package
│   ├── __init__.py
│   ├── preprocess_windows.py       # Preprocessing pipeline (windowing, splitting, normalisation)
│   ├── data_precomputed.py         # Data loading utilities and chunk indexing
│   ├── task.py                     # Model definitions (HARNetCNN, HARNetMLP) and train/eval functions
│   ├── centralized.py              # Centralized training loop with early stopping
│   ├── server_app.py               # Flower server: round management, aggregation, output saving
│   ├── client_app.py               # Flower client: chunk loading, FedPer head management, replay logic
│   ├── customized_strategies.py    # SequentialChunkFedAvg extending Flower's FedAvg
│   ├── evaluate_fl.py              # Post-simulation evaluation on held-out test clients (FL)
│   ├── evaluate_centralized.py     # Post-training evaluation on held-out test clients (centralized)
│   └── experiment_utils.py         # Run directory creation and CSV saving helpers
├── run_experiments_FL.py           # Automates full FL ablation matrix across repetitions
├── run_experiments_centralized.py  # Automates centralized LR ablation
├── run_experiments_per_dataset.py  # Per-dataset centralized and FL runner
├── run_experiments_window_sizes.py # Centralized window size ablation runner
├── pyproject.toml                  # Flower app config, hyperparameters, component registration
├── precomputed/                    # Precomputed .npz files (see Preprocessing section)
└── outputs/                        # Experiment results (auto-generated)
```

---

## Requirements

- Python 3.10+
- `flwr[simulation] >= 1.29.0`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `tqdm`

Install the package in editable mode:

```bash
pip install -e .
```

---

## Data

Raw datasets are **not included**. The pipeline expects preprocessed per-client CSV files in a `data/` directory. Each CSV file corresponds to one subject and must contain the following columns:

| Column | Description |
|---|---|
| `timestamp_ms` | Relative timestamp in milliseconds |
| `acc_x`, `acc_y`, `acc_z` | Accelerometer axes in m/s² or g |
| `activity` | Activity label string |
| `dataset` | Dataset name string (used for inter-subject splitting) |
| `segment_id` | Segment identifier (assigned during earlier preprocessing) |

The nine datasets used in this thesis are:

**Waist placement:** FLAAP, HHAR, HAPT, KU-HAR, RealWorld

**Right front thigh placement:** HARTH, WISDM AP, WISDM Actitracker, UT-SAD

---

## Preprocessing

Preprocessing converts raw per-client CSVs into per-client `.npz` files ready for FL simulation. Run once per experimental configuration. The script applies gravity removal, feature extraction, resampling, segmentation, sliding-window extraction, train/val/test splitting, and global min-max normalisation.

```bash
python -m harflwr.preprocess_windows \
  --data-dir data/partA_blockB \
  --output-dir precomputed/partA_blockB_magstd \
  --window-seconds 5.0 \
  --overlap-ratio 0.5 \
  --val-ratio 0.2 \
  --channel-config mag_std \
  --label-mapping union \
  --inter-subject-test-ratio 0.2 \
  --inter-subject-split-seed 42 \
  --datasets-in-g "HAPT"
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Directory containing per-client CSV files |
| `--output-dir` | `precomputed` | Output directory for `.npz` files |
| `--window-seconds` | `5.0` | Sliding window size in seconds |
| `--overlap-ratio` | `0.5` | Fraction of overlap between consecutive windows |
| `--channel-config` | `mag_std` | Feature configuration (see below) |
| `--label-mapping` | `intersection` | `intersection` (shared activities only) or `union` (all activities) |
| `--inter-subject-test-ratio` | `None` | Fraction of subjects assigned to test role per dataset |
| `--datasets-in-g` | `HAPT` | Comma-separated dataset names whose accelerometer is in g-units |
| `--datasets-gravity-removed` | `` | Datasets where gravity was already removed prior to this script (no default value)|

**Channel configurations** (`--channel-config`):

More than the `mag_std` channel configuration that was used in the thesis is available in the codebase. Some examples:

| Config | Channels | Description |
|---|---|---|
| `mag_std` | 2 | Acceleration magnitude + rolling std *(used in all experiments)* |
| `mag` | 1 | Magnitude only |
| `mag_deriv` | 2 | Magnitude + derivative |
| `raw` | 3 | Raw x, y, z axes |
| `mag_std_grav` | 5 | mag_std + gravity x, y, z |

The preprocessing script produces the following output structure:

```
precomputed/<name>/
├── client_000_train.npz    # Training windows for train clients
├── client_000_val.npz      # Validation windows for train clients
├── client_001_test.npz     # All windows for test clients
├── ...
├── manifest.csv            # Per-client metadata (role, window counts, norm stats)
└── metadata.json           # Global config, label map, sensor columns, norm parameters
```

**Experimental configurations used in the thesis:**

| Directory | Placement | Label space |
|---|---|---|
| `precomputed/partA_blockA_magstd` | Waist | Intersection |
| `precomputed/partA_blockB_magstd` | Waist | Union |
| `precomputed/partB_blockA_magstd` | Right front thigh | Intersection |
| `precomputed/partB_blockB_magstd` | Right front thigh | Union |

---

## Running Experiments

### 1. Full FL Ablation

Runs all conditions (FedAvg baseline, FedPer, Replay) × (CNN, MLP) × (all four placement/label-space combinations), each for 5 repetitions:

```bash
python run_experiments_FL.py
```

To resume from a specific experiment index:

```bash
python run_experiments_FL.py --start-from 6
```

To re-run FedPer conditions only (e.g. after updating the evaluation protocol):

```bash
python run_experiments_FL.py --fedper-only
```

Results are saved to `outputs/<experiment_name>/` with one subdirectory per repetition.

### 2. Centralized Learning Rate Ablation

```bash
python run_experiments_centralized.py
```

Evaluates learning rates `[0.1, 0.001, 0.0001, 0.00001]` with CNN and MLP on the union label space (Part B, Block B). Results go to `outputs/c_results/`.

### 3. Window Size Ablation

```bash
python run_experiments_window_sizes.py --part both --model both
```

Evaluates window sizes `[2, 4, 5, 8]` seconds. Requires separate precomputed directories for each window size (e.g. `precomputed/partA_blockB_magstd_2s`). Results go to `outputs/window_results/`.

### 4. Per-Dataset Centralized + FL

Runs centralized and FL experiments on each dataset individually to establish per-dataset performance ceilings:

```bash
python run_experiments_per_dataset.py --part both
```

Or for a single dataset:

```bash
python run_experiments_per_dataset.py --dataset HARTH
```

Results go to `outputs/per_dataset/<dataset_name>/`.

### 5. Single FL Run (Manual)

Configure `pyproject.toml` directly, then:

```bash
flwr run . --stream
```

Then evaluate:

```bash
python -m harflwr.evaluate_fl \
  --run-dir outputs/fl_run_<timestamp> \
  --channel-config mag_std
```

---

## Configuring a Run via `pyproject.toml`

The `[tool.flwr.app.config]` section controls all hyperparameters for a single FL run:

```toml
[tool.flwr.app.config]
precomputed-dir     = "precomputed/partB_blockA_magstd"
channel-config      = "mag_std"
model-name          = "cnn"           # "cnn" or "mlp"
fraction-train      = 1.0
fraction-evaluate   = 1.0
local-epochs        = 1
learning-rate       = 0.0001
batch-size          = 32
window-seconds      = 5.0
overlap-ratio       = 0.5
chunk-size          = 32
use-replay-buffer   = false           # true for Experience Replay
replay-buffer-capacity = 16          # windows stored per class per client
replay-sample-size  = 16             # replay windows injected per round
personalization-mode = "none"        # "none", "fedper"
fedper-adapt-epochs = 1
```

To enable the Flower simulation backend, uncomment and fill in the federation block:

```toml
[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 113
address = ":local:"
options.backend.client-resources.num-cpus = 1
options.backend.client-resources.num-gpus = 0.25
```

The `num-supernodes` value should match the number of train clients in the selected precomputed directory. Values used in the thesis:

| Precomputed dir | num-supernodes |
|---|---|
| `partA_blockA_magstd` | 112 |
| `partA_blockB_magstd` | 115 |
| `partB_blockA_magstd` | 113 |
| `partB_blockB_magstd` | 113 |

---

## Output Files

Each run directory (`outputs/<run_name>/`) contains:

| File | Source | Description |
|---|---|---|
| `config.json` | FL + centralized | Full run configuration including hyperparameters, client split, label map, and strategy flags |
| `history.csv` | FL + centralized | Round-level (FL) or epoch-level (centralized) train and validation metrics |
| `summary_metrics.csv` | FL + centralized | Single-row summary of the final round or best epoch |
| `per_client_round_metrics.csv` | FL only | Per-client per-round metrics including chunk indices, class counts, and replay statistics |
| `predictions.csv` | Centralized only | Per-window validation set predictions from the best epoch |
| `final_model.pt` | FL + centralized | Saved model weights after the final round or best epoch |
| `per_client_metrics_test.csv` | FL + centralized | Per-client test set metrics |
| `final_eval_summary_metrics_test.csv` | FL + centralized | Aggregated test set summary across all test clients |
| `predictions_test.csv` | FL + centralized | Per-window test set predictions and true labels |

---

## FL Strategy Details

### FedAvg (Baseline)
Standard Federated Averaging. All clients train on their current chunk and return full model weights for aggregation. Implemented via `SequentialChunkFedAvg`, a subclass of Flower's `FedAvg` that injects each client's cumulative selection count into the training config and filters out exhausted clients from aggregation.

### FedPer
The model is split into a shared feature extractor (`features.*`) and a private classifier head (`classifier.*`). At each round, the head is fine-tuned for one epoch with the base layers frozen, then all layers train jointly. Only the backbone is returned to the server. Each client's head is saved locally in `precomputed/_personal_heads/client_XXX_head.pt`.

For test-client evaluation, a temporal 80/20 split is applied: the first 20% of windows adapt the head (base layers frozen), and the remaining 80% are used for evaluation.

### Experience Replay
A class-aware replay buffer is stored per client as a compressed `.npz` file in `precomputed/_replay_buffers/`. The buffer holds up to 16 windows per observed class and is updated after each chunk. At training time, up to 16 replay windows are sampled exclusively from classes absent in the current chunk, with budget distributed equally across missing classes. Replay windows are concatenated with the current chunk before training.

---

## Sequential Chunk Serving

Each client's training windows are served in temporal order, divided into fixed-size chunks of 32 windows. One chunk is used per round. A client that has exhausted all its chunks sends a zero-example reply and is excluded from further aggregation. The number of FL rounds equals the maximum number of chunks across all training clients.

---

## Models

Both models share a two-part structure: a feature extractor and a classifier head, with a 128-dimensional bottleneck at their boundary. This boundary is used as the FedPer split point.

**HARNetCNN** — Three 1D convolutional blocks (64→128→128 filters, kernel sizes 15→9→5) with max pooling after the first two blocks and global average pooling at the end. Input shape: `[B, C, 100]`.

**HARNetMLP** — Two fully connected blocks (input→256→128) with BatchNorm, ReLU, and Dropout. The input is flattened before the first layer. Input shape: `[B, C×100]`. Fixed to `C=2`, `window=100` in all experiments.

Both are defined in `harflwr/task.py`.

---

## Reproducibility

- Train/test client split: seed 42, stratified by dataset
- DataLoader shuffling: seed 42
- Replay sampling: deterministic seed derived from server round and client ID
- Centralized training: fully deterministic with seed 42 (single run per configuration)
- FL repetitions: 5 per condition — Ray's client scheduling is non-deterministic within a round, so aggregated weights can differ between repetitions under otherwise identical configurations

---

## AI Assistance Disclosure

Claude (Anthropic) was used for:
- Advice on model structure and debugging (`task.py`)
- Logic review and cleanup of the preprocessing pipeline (`preprocess_windows.py`)
- FedPer head adaptation logic for test clients (`evaluate_fl.py`)
- Extending FedAvg to support sequential chunk serving (`customized_strategies.py`)
