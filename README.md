# WiSense-MCP

<p align="center">
  <img src="graphical_abstract.png" alt="WiSense-MCP graphical abstract showing the WiFi CSI sensing pipeline, temporal deep-learning model, predictions, evaluation, dashboard, and MCP interface." width="100%">
</p>

<p align="center"><em>WiFi CSI to multi-user presence and activity recognition, with an interactive dashboard and MCP interface.</em></p>

WiSense-MCP is a reproducible WiFi Channel State Information (CSI) sensing project for multi-user presence detection and activity recognition. The project uses the public WiMANS dataset, trains a temporal deep-learning model from CSI amplitude measurements, combines several independently trained models with a soft ensemble, and exposes the finished system through both a browser dashboard and a Model Context Protocol (MCP) server. The dashboard can analyze samples already in WiMANS or run inference on a compatible CSI capture uploaded from another sensor.

The repository is organized as a complete pipeline rather than a collection of isolated notebooks. A new user should be able to place the dataset in the configured directory, run the scripts in order, reproduce the preprocessing and training workflow, evaluate the final ensemble, and then start the dashboard and MCP endpoint without editing the Python source code. Once the model is available, a user can also upload a compatible CSI file through the browser and obtain a prediction without adding that sample to the project dataset.

## What the system does

Each WiMANS sample contains approximately three seconds of WiFi CSI. The model receives the CSI amplitude from 270 channels and predicts two things for each of six anonymized user slots:

- whether the user is present;
- which of nine activities the user is performing when present.

The nine activity classes are `jump`, `lie_down`, `nothing`, `pick_up`, `rotation`, `sit_down`, `stand_up`, `walk`, and `wave`.

The final model uses a temporal CNN followed by a packed bidirectional LSTM. Instead of reducing the entire sequence to a single recurrent state, it uses temporal-pyramid attention to retain information from the full sequence as well as its early, middle, and late portions. Three independently trained models are then combined by averaging their output probabilities. The same inference path is used for cached WiMANS samples and compatible CSI files uploaded through the dashboard.

```text
CSI amplitude: 270 channels x up to 3000 packets
                     |
                     v
              Temporal CNN
                     |
                     v
          Packed bidirectional LSTM
                     |
          +----------+----------+----------+
          |          |          |          |
        Global     Early      Middle      Late
       attention  attention   attention   attention
          |          |          |          |
          +----------+----------+----------+
                     |
                     v
             Feature fusion
                     |
          +----------+----------+
          |                     |
      Presence head          Activity head
       6 sigmoid             6 x 9 softmax
          |                     |
          +----------+----------+
                     |
                     v
          Equal-weight seed ensemble
```

## Dataset

This project uses **WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing**, introduced at ECCV 2024. WiMANS contains 11,286 three-second samples collected using 2.4 GHz and 5 GHz WiFi, with synchronized video reference data and annotations for multiple simultaneous users.

Official resources:

- WiMANS repository: https://github.com/huangshk/WiMANS
- Dataset download: https://www.kaggle.com/datasets/shuokanghuang/wimans
- ECCV paper page: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5826_ECCV_2024_paper.php

The dataset itself is not redistributed in this repository. Download it separately and follow the WiMANS dataset terms.

Only `annotation.csv` and the supplied CSI amplitude `.npy` files are required by the training pipeline in this repository. Raw `.mat` CSI files and synchronized videos are not required unless you want to perform additional analyses outside the provided pipeline.

Place the data in the following structure by default:

```text
data/
└── raw/
    └── WiMANS/
        ├── annotation.csv
        └── wifi_csi/
            └── amp/
                ├── act_1_1.npy
                ├── act_1_2.npy
                └── ...
```

If you prefer a different location, change `dataset_dir` in `config.ini`. No Python file needs to be edited.

## Reference results

The current reference run used a stratified 70/10/20 train/validation/test split and a three-model equal-weight ensemble. The final test split was opened only after the model configuration, checkpoints, presence threshold, and ensemble rule had been frozen.

| Metric | Final test result |
| --- | ---: |
| Presence Macro F1 | 0.94495 |
| Presence Precision | 0.93468 |
| Presence Recall | 0.95568 |
| Activity Top-1 Accuracy | 0.40442 |
| Activity Macro F1 | 0.40455 |
| Structured Macro F1 | 0.44136 |
| Structured Accuracy | 0.71420 |
| Absent-user false-positive rate | 0.04869 |

The structured metric treats each user slot as a ten-state prediction problem: one absent state plus the nine activity states.

These results should not be treated as a direct leaderboard comparison with the benchmark numbers reported in the WiMANS paper. This repository uses its own split and structured multi-user evaluation protocol.

## Repository layout

```text
.
├── config.ini
├── requirements.txt
├── artifacts/
│   └── best_values.json
├── scripts/
│   ├── build_splits.py
│   ├── compute_normalization.py
│   ├── build_cache.py
│   ├── train.py
│   ├── build_ensemble.py
│   ├── evaluate_test.py
│   ├── verify_inference.py
│   └── run_server.py
└── src/
    ├── settings.py
    ├── ml/
    │   ├── dataset.py
    │   ├── factory.py
    │   ├── labels.py
    │   ├── metrics.py
    │   ├── model.py
    │   └── training.py
    └── wisense_mcp/
        ├── diagnostics.py
        ├── inference.py
        ├── mcp_server.py
        ├── repository.py
        ├── sample_catalog.py
        ├── web_app.py
        └── static/
            ├── index.html
            ├── app.css
            └── app.js
```

`config.ini` contains the user-editable project settings. `artifacts/best_values.json` contains the selected hyperparameters from the model-selection stage used to reproduce the reference model. Training reads this file automatically; the selected values are not duplicated in the training source code.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for training, although the code can fall back to CPU execution.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/abhinz16/WiSense.git
cd WiSense
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU acceleration, make sure your PyTorch installation is compatible with your NVIDIA driver and CUDA environment. The PyTorch installation selector at https://pytorch.org/get-started/locally/ is the safest way to choose the appropriate build.

## Configuration

The normal workflow should not require editing any Python code. User settings are collected in `config.ini`.

The most commonly changed entries are:

```ini
[paths]
dataset_dir = data/raw/WiMANS

[training]
seeds = 42, 43, 44
max_epochs = 200
num_workers = 8
use_amp = true

[server]
host = 127.0.0.1
port = 8000
external_sample_max_mb = 50
external_sample_formats = .npy, .npz, .csv, .txt
```

The default split is 70% training, 10% validation, and 20% test. Global per-channel normalization statistics are computed from the training split only.

The tracked `artifacts/best_values.json` file is not intended as a user-input file for normal reproduction. It records the selected hyperparameters used by the reference model, and `scripts/train.py` reads it automatically. Editing it defines a different model configuration.

## Running the full pipeline

Run the scripts from the repository root and in the order shown below.

### 1. Build the train, validation, and test splits

```bash
python scripts/build_splits.py
```

This reads `annotation.csv`, creates the multi-user labels, stratifies by environment, WiFi band, and simultaneous user count, and writes the split manifests to the directory configured as `split_dir`.

Default outputs:

```text
data/processed/splits/
├── manifest.csv
├── train.csv
├── validation.csv
├── test.csv
└── split_summary.json
```

### 2. Compute normalization statistics

```bash
python scripts/compute_normalization.py
```

The channel means and standard deviations are calculated using the training split only. This prevents validation or test information from entering preprocessing.

Default output:

```text
data/processed/normalization/training_channel_stats.npz
```

### 3. Build the tensor cache

```bash
python scripts/build_cache.py
```

Each CSI amplitude sequence is normalized with the training statistics, padded or truncated to the configured target length, and stored in memory-mappable NumPy arrays. True packet lengths are saved separately so the recurrent model can ignore padded timesteps.

Default outputs include:

```text
data/processed/tensor_cache/
├── train_x.npy
├── train_y.npy
├── train_lengths.npy
├── validation_x.npy
├── validation_y.npy
├── validation_lengths.npy
├── test_x.npy
├── test_y.npy
├── test_lengths.npy
└── cache_config.json
```

With the full WiMANS dataset and the default `float16` cache, the input tensors require roughly 17 GiB of disk space, so check available storage before running this step.

### 4. Train the seeded models

```bash
python scripts/train.py
```

The script reads the seeds and training controls from `config.ini` and the selected model hyperparameters from `artifacts/best_values.json`. Each seed is trained independently. The best checkpoint for each seed is selected using validation activity Macro F1, with early stopping and a learning-rate scheduler.

Default checkpoints:

```text
models/
├── model_seed_42.pt
├── model_seed_43.pt
└── model_seed_44.pt
```

Training histories and summary metrics are written under `results/training/`.

### 5. Build the validation ensemble

```bash
python scripts/build_ensemble.py
```

This runs the trained checkpoints on the validation split and averages their presence and activity probabilities with equal weight. It does not retrain the models or optimize ensemble weights.

Default outputs are written to:

```text
results/ensemble/
```

The saved validation probabilities are later used to verify that the production inference path reproduces the same model behavior.

### 6. Run the locked final test evaluation

```bash
python scripts/evaluate_test.py
```

This is intentionally a separate step. Before reading the test data, the script writes a model-lock file containing the checkpoint paths, SHA-256 hashes, seeds, ensemble rule, and decision threshold. It then performs the final test evaluation and saves the reporting artifacts.

Default outputs:

```text
results/final/
├── model_lock.json
├── metrics.json
├── report.json
├── individual_metrics.csv
├── per_class_metrics.csv
├── confusion_matrix.csv
└── predictions.npz
```

By default, `prevent_test_overwrite = true` in `config.ini`. If final metrics already exist, the script stops instead of silently reevaluating the test set. A deliberate full rerun requires removing the previous final-results directory first.

### 7. Verify production inference

```bash
python scripts/verify_inference.py
```

This compares live inference with the probabilities saved during the validation and final test evaluations. The number of checked samples and numerical tolerance are controlled by `config.ini`.

A successful run ends with output similar to:

```text
Verified 6/6 samples
```

Small floating-point differences are expected when batched evaluation and single-sample GPU inference use different CUDA kernels. The default tolerance is `0.001`, while the presence decisions and activity argmax values are also checked for agreement.

## Starting the dashboard and MCP server

The browser interface and MCP server are hosted by the same Starlette/Uvicorn application.

Start it with:

```bash
python scripts/run_server.py
```

With the default configuration, the following endpoints become available:

```text
Dashboard:     http://127.0.0.1:8000/
Health check:  http://127.0.0.1:8000/api/health
MCP endpoint:  http://127.0.0.1:8000/mcp
```

The dashboard provides sample browsing, live model inference, final-test analytics, error inspection, and CSI diagnostics. In **Live Analyze**, the input source can be switched between a sample already in the WiMANS dataset and a compatible CSI file collected from another sensor. The neural-network ensemble is loaded lazily, so opening the dashboard does not immediately allocate the models on the GPU.

## Running inference on a new sensor sample

The **New Sensor Sample** option in the dashboard is intended for testing CSI collected outside the WiMANS dataset. The uploaded file is read directly for inference and is not added to the dataset or written into the project results. Ground-truth labels are not required.

### Input requirements

The trained network expects **270 CSI channels per packet**, matching the channel meaning and antenna/subcarrier ordering used during training. With the WiMANS representation, those 270 values correspond to the flattened `3 x 3 x 30` CSI dimensions for each packet.

The uploader accepts the following file types by default:

| Format | Expected contents |
| --- | --- |
| `.npy` | One NumPy array containing real amplitude values or complex CSI |
| `.npz` | A NumPy archive containing a numeric CSI array |
| `.csv` | UTF-8 numeric CSI data, comma separated |
| `.txt` | UTF-8 numeric CSI data, whitespace separated |

The allowed extensions and maximum upload size are controlled by `config.ini`:

```ini
[server]
external_sample_max_mb = 50
external_sample_formats = .npy, .npz, .csv, .txt
```

For `.npz` files, the loader first looks for arrays named `csi`, `amplitude`, `amp`, `data`, or `sample`. If none of those keys are present, it uses the first numeric array in the archive.

### Supported array layouts

The simplest and recommended layout is one row per packet:

```text
[packet_count, 270]
```

For example, a three-second capture with 3000 packets can be saved as:

```python
import numpy as np

# csi_amplitude has shape (3000, 270)
np.save("new_sensor_sample.npy", csi_amplitude)
```

The loader also accepts:

```text
[270, packet_count]
[packet_count, 3, 3, 30]
[3, 3, 30, packet_count]
```

More generally, a multidimensional array is accepted when exactly one axis can be interpreted as the packet axis and the product of all remaining dimensions is 270. A flat one-dimensional array is also accepted when its total length is an exact multiple of 270.

Complex-valued CSI is converted to amplitude with `abs(CSI)` before normalization. Real-valued input is treated as CSI amplitude. Non-finite values are replaced using the training-set channel means. The sample is then normalized with the same training-only per-channel mean and standard deviation used by the model.

The default model input length is 3000 packets. Shorter captures are zero-padded after normalization, while longer captures are truncated to the configured target length. The valid packet length is still passed to the recurrent model so padded timesteps are excluded from sequence processing.

### Uploading a sample from the dashboard

Start the application normally:

```bash
python scripts/run_server.py
```

Open:

```text
http://127.0.0.1:8000/
```

Then open **Live Analyze**, select **New Sensor Sample**, and either drag the CSI file onto the upload area or click **Choose file**. After the file is selected, click **Run Neural Inference**.

The result view shows the detected users, presence probabilities, predicted activity for each present user, top activity probabilities, inference latency, device information, the detected input layout, packet counts, and a compact CSI-energy preview. Because an uploaded sensor sample has no project annotation, the dashboard reports predictions only and does not calculate ground-truth correctness for that sample.

### Sensor compatibility

A file being accepted by the uploader does not guarantee that its predictions are calibrated. The model was trained on the WiMANS CSI representation, so a new sensor should preserve the same 270-channel interpretation and ordering as closely as possible. Changes in WiFi hardware, antenna layout, subcarrier selection, CSI extraction method, sampling behavior, environment, or signal scaling can introduce domain shift.

If a sensor produces a different number of CSI channels, the current model cannot use that sample directly. If it produces 270 channels with a different physical meaning or ordering, the code may run but the resulting prediction should not be treated as reliable without validation, domain adaptation, or retraining.

The browser upload workflow is currently separate from the MCP sample tools. MCP tools operate on samples already available through the project dataset/cache, while external sensor files are uploaded through the dashboard.

## Using the MCP server

The MCP endpoint uses Streamable HTTP and is mounted at `/mcp`. Any MCP client that supports Streamable HTTP can connect to:

```text
http://127.0.0.1:8000/mcp
```

Keep `scripts/run_server.py` running while the MCP client is connected.

### Testing with MCP Inspector

The easiest way to inspect the server locally is the official MCP Inspector. It requires Node.js and `npx`.

Start WiSense in one terminal:

```bash
python scripts/run_server.py
```

In another terminal, start the Inspector:

```bash
npx -y @modelcontextprotocol/inspector
```

Open the URL printed by the Inspector, choose the Streamable HTTP transport, and connect to:

```text
http://127.0.0.1:8000/mcp
```

The Inspector will discover the tools exposed by the server and lets you call them manually.

### MCP tools

The MCP interface is read-only. It exposes the following tools:

| Tool | Purpose |
| --- | --- |
| `get_dataset_summary` | Dataset labels, environments, WiFi bands, and split counts |
| `get_final_model_metrics` | Locked final test metrics and model-lock metadata |
| `get_activity_performance` | Per-class precision, recall, F1, and support |
| `get_top_activity_confusions` | Largest directional activity confusions |
| `compare_wifi_bands` | Performance by 2.4/5 GHz band |
| `compare_environments` | Performance by sensing environment |
| `compare_user_counts` | Performance by simultaneous user count |
| `get_user_performance` | Presence and activity performance by anonymized user slot |
| `get_error_analysis_summary` | Summary of activity and presence errors |
| `list_samples` | Browse samples with optional filters |
| `get_sample` | Resolve a sample alias or raw dataset ID |
| `get_frozen_prediction` | Retrieve stored final ensemble probabilities for a test sample |
| `predict_sample` | Run live ensemble inference on a cached sample |
| `verify_live_prediction` | Compare live output with saved ensemble probabilities |
| `get_model_status` | Device, checkpoint, loading, and artifact status |
| `diagnose_sample` | Inspect CSI integrity and prediction-vs-ground-truth behavior |
| `find_model_errors` | Search test samples by error type |
| `find_difficult_samples` | Rank samples with the retrospective uncertainty heuristic |
| `get_capabilities` | Describe the server and available analysis functions |

For example, an MCP-capable assistant can answer requests such as:

```text
Show me the final model metrics.

Which activities are most often confused?

Compare performance between 2.4 GHz and 5 GHz.

List test samples from the classroom with three users.

Run live inference on a sample and explain the prediction.

Find difficult samples where the model made an activity error.
```

The exact way an MCP endpoint is added depends on the client. Use the Streamable HTTP URL above rather than trying to launch `mcp_server.py` directly; the MCP application is mounted inside the main web server.

### Local-only default and remote deployment

The default server host is `127.0.0.1`, which intentionally limits access to the local machine. Do not expose the MCP endpoint directly to the public internet by simply changing the host to `0.0.0.0`. A remote deployment should add appropriate authentication, TLS, origin validation, and network controls.

## Generated files and Git

The repository `.gitignore` excludes the dataset, tensor cache, trained checkpoints, and generated results:

```text
data/
models/
results/
```

This is intentional. The Git repository contains the code, project configuration, and selected hyperparameter artifact, while large or generated files are created locally by the pipeline.

If you want to distribute pretrained checkpoints separately, use a release asset or external model-storage service rather than committing large binary files directly to the repository.

## Reproducibility notes

Several choices in the pipeline are deliberate:

- Train/validation/test manifests are created with fixed random seeds.
- Normalization statistics are calculated from the training split only.
- Variable packet lengths are preserved and supplied to a packed BiLSTM.
- The three training seeds are configured in `config.ini`.
- Deterministic PyTorch execution can be enabled through `config.ini` and is enabled by default.
- Ensemble weights are fixed and equal; they are not tuned on the validation or test split.
- The presence threshold is configured before final testing.
- The final evaluator hashes the selected checkpoints before opening the test data.
- The test split is reporting-only and should not be used to make further model-selection decisions.

Exact floating-point values can still vary slightly across PyTorch, CUDA, cuDNN, GPU, and driver versions. Reproducibility should therefore be judged using the saved configuration, model-selection procedure, close numerical agreement, and consistent discrete predictions rather than assuming bit-for-bit equality across every platform.

## Model limitations

The system is a research implementation, not a safety-critical people-sensing product. Presence detection is substantially stronger than fine-grained activity recognition. In the reference test evaluation, highly dynamic activities such as jump, rotation, and wave remained among the more difficult classes.

The current model input is CSI amplitude only. Complex CSI uploaded through the dashboard is reduced to magnitude before inference. Raw phase calibration, Doppler representations, additional antenna/subcarrier structure, domain adaptation, and evaluation on unseen environments or identities are reasonable directions for future work.

External sensor inference should be treated as a compatibility feature rather than a claim of hardware-independent generalization. The model expects the same 270-channel semantics and ordering used during training, and performance can degrade when the acquisition hardware or sensing domain differs from WiMANS.

WiMANS does not provide session identifiers suitable for claiming a session-independent split in this implementation. The repository therefore describes its split as stratified by environment, WiFi band, and user count rather than making a session-independence claim.

## Citation

Please also cite this repository if you build on the model, preprocessing pipeline, or MCP application in your own work.

## Questions and contributions

Issues and pull requests are welcome. If you report a problem, include your operating system, Python version, PyTorch version, whether CUDA is enabled, the command that failed, and the relevant traceback. That information is usually enough to reproduce most setup or runtime problems quickly.
