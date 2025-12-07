# PaDiM Anomaly Detection & Localization — Quick Start

This repository contains a PaDiM-style anomaly detection / localization implementation built around PyTorch feature embeddings. The main entrypoint is `run_experiment.py` which drives dataset loading, feature extraction, distribution learning and evaluation.

This code is built on top of the original implementation found at https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master.git

## Quick start (Conda on WSL)

These steps assume you're on WSL (Ubuntu) and have Conda/Miniconda installed. If you use native Linux, the steps are the same. For GPU support install the matching PyTorch + CUDA toolkit from the official instructions

1. Create and activate a conda environment:

```bash
conda create -n padim python=3.12 -y
conda activate padim
```

2. Install Python requirements used by the repo:

```bash
pip install -r requirements.txt
```

The requirements include:

- torch
- tqdm
- scikit-learn
- matplotlib
- scikit-image
- torchvision
- seaborn
- statsmodels
- pandas
- opencv-python
- scipy

4. (Optional) If you plan to run with GPU and WSL2, ensure Windows + WSL have GPU passthrough set up (NVIDIA drivers + WSL CUDA). Verify `torch.cuda.is_available()` in Python.

5. Prepare your dataset. Then run the experiment.

## Downloading MVTec
MVTec dataset can be downloaded from: https://www.mvtec.com/company/research/datasets/mvtec-ad/ — this is the recommended dataset to start with.

It is used in the original paper and has a well-defined structure. See the "Dataset layout" section below for details on how to organize the data.

Using custom dataset helpers (what these files do)
-----------------------------------------------
This repo includes a few small helper scripts in `data/` to convert and organize a custom  dataset so it matches the expected layout (one folder per class, each with `train/` and `test/`). If you are using a custom dataset (e.g. the BowTie CSVs and image folders), follow these steps:

- `data/convert_excel.py` — converts XLSX files into cleaned CSVs. If you have Excel metadata sheets (the original BowTie inputs), run this first to produce the CSVs the pipeline expects. The script sets a clean header, selects the expected columns and writes CSVs. Example:

```bash
python data/convert_excel.py -i data/BowTie/round-2/ -o data/BowTie/round-2/converted
```

- `data/map.json` — a small JSON mapping included in the repo that maps CSV filenames to their corresponding source image folder names. You can edit or replace this map to match your local layout. `process_data.py` reads this file to know which folder holds the images for each CSV.

- `data/process_data.py` — the main organizer/preprocessor. It reads CSVs (or the converted CSVs), applies optional preprocessing (grayscale, Canny, histogram equalization), and sorts images into the standard `train/good` and `test/*` folders. It also supports a `--combine` mode that merges multiple folders into one dataset (useful for creating a larger training pool). Example usage:

```bash
# Sort/prepare using the provided map.json
python data/process_data.py -i data/BowTie/round-2/ -o data/BowTie-New/original -m data/map.json

# Combine a list of folders into one dataset
python data/process_data.py --combine 116 117 118 -i data/BowTie/round-2/ -o data/BowTie-New/combined -m data/map.json
```

Output of `process_data.py` is suitable for consumption by the dataset loader in `datasets/bowtie.py` and for direct use with `run_experiment.py`. By default this repo's `run_experiment.py` points to `../anomaly_detection/data/BowTie-New/original` — if you follow the examples above the produced `data/BowTie-New/original` folder will match the default `--data_path` used in the script.

Once these scripts are run, the data is organized in a similar way to the MVTec dataset structure described, making it compatible with the main experiment script and generally more usable.

## Run the example

Basic usage (defaults are reasonable):

```bash
python run_experiment.py --dataset mvtec --results_subdir mvtec_test --data_path data/MVTec
```

This will create `./results/mvtec_test` (or a name derived from the model/resize/crop settings) and write summary outputs there.

## Command line arguments

Run `python run_experiment.py -h` to see all flags. Here are the main ones (name — description — default):

- `--model_architecture` (str) — Which pre-trained model to use. Options in code: `wide_resnet50_2`, `resnet18`, `efficientnet_b5`. Default: `wide_resnet50_2`.
- `--dataset` (str) — dataset loader to use: `bowtie` or `mvtec`. Default: `bowtie`.
- `--data_path` (str) — Root path to dataset. Default: `../anomaly_detection/data/BowTie-New/original` in this repo. Provide absolute path for clarity.
- `--base_results_dir` (str) — Root directory for results. Default: `./results`.
- `--resize` (int) — Image resize short side before crop. Default: `256`.
- `--cropsize` (int) — Center-crop size used for the model input. Default: `256`.
- `--seed` (int) — Random seed. Default: `1024`.
- `--save_distribution` — If present, signal to save learned distribution.
- `--batch_size` (int) — Batch size for dataloaders. Default: `32`.
- `--augmentations_enabled` — Enable augmentations for training loader.
- `--horizontal_flip`, `--vertical_flip` — Enable flips for augmentations.
- `--augmentation_prob` (float) — Probability of augmentations. Default: `0.5`.
- `--results_subdir` (str) — Append a subdir under `--base_results_dir` for this run.
- `--test_class_name` / `--test_class_path` — Use a different class for testing than training (or point to a specific class folder).
- `--mahalanobis_on_gpu` — Compute Mahalanobis distance on the GPU (if supported).
- `--stats_on_gpu` — Compute mean/covariance on the GPU (if supported).
- `--test_sample_ratio` (float) — For sampling normal test images when building test dataset (default `0.2`).

Notes: the script also uses the pre-trained model's intermediate layers by registering forward hooks to collect feature maps — these are concatenated and a subset of channels is randomly selected (random projection) to build a lower-dimensional embedding used by PaDiM.

## Repository / file overview

- `run_experiment.py` — CLI & experiment orchestration. Parses args, builds model, registers forward hooks to capture intermediate features, and iterates classes.
- `trainer.py` — heavy lifting: dataset handling, embedding extraction, covariance/mean learning, scoring, metrics, and plotting/saving results.
- `datasets/` — dataset loaders (`mvtec.py`, `bowtie.py`) and helpers.
- `utils/` — plotting and embedding helper functions.
- `requirements.txt` — Python packages used by this project.

## Dataset layout

This code assumes a dataset layout where each class has `train` and `test` subfolders. Two common layouts are described below.

1) MVTec-style dataset (recommended when using `--dataset mvtec`):

```
/path/to/dataset/
    ├── bottle/                  <-- Class Name
    │   ├── train/
    │   │   └── good/            <-- Normal training images
    │   └── test/
    │       ├── good/            <-- (Optional) Normal test images
    │       ├── broken_large/    <-- Defect type 1
    │       └── contamination/   <-- Defect type 2
    └── cable/
```

2) Custom dataset (BowTie / other): same pattern — the code expects one folder per class, and within each class a `train/` and `test/` folder. Example:

```
/path/to/dataset/
    ├── class_name_1/
    │   ├── train/
    │   │   └── good/            <-- All normal images
    │   └── test/
    │       ├── defect_type_A/   <-- Anomalous images
    │       └── defect_type_B/
    └── class_name_2/
        ...
```

If you have a single-class folder structure already (i.e. your `--data_path` directly points to a class folder containing `train/` and `test/`), the code detects that and processes only that class.

## Results / Output structure

When you run the experiment for a class, a directory will be created under the `--master_save_dir` (default `./results/<model>_resize-...`) with one subfolder per processed class (or `train_<train>_test_<test>` when mixing train/test pairs). Inside each class folder you'll typically find:

- `learned_distribution.pkl` — saved mean/covariance learned on training features (used later for scoring). Only saved if specified.
- `visualizations/` — per-image visualizations, overlays and example anomaly maps.
- `master_results.csv` (in the master results folder) — when processing multiple classes, a CSV summarizing per-class metrics is created by `run_experiment.py`.
- plots: ROC curve, PR curve, mean anomaly maps, patch score distributions (saved as images by the plotting utilities).

## How the code works

1. `run_experiment.py` parses CLI args, constructs the pre-trained model (`torchvision` models are used), registers forward hooks on selected intermediate layers to capture feature maps, and builds `random_feature_indices` used to select a subset of features.
2. The script enumerates classes (or accepts a single-class folder). For each class it calls `trainer.run_class_processing(...)`.
3. `trainer.run_class_processing`:
   - Loads train/test datasets using the `datasets/` loaders.
   - Runs the model (forward) over the train set capturing the intermediate feature maps.
   - Concatenates per-layer feature maps into per-patch embeddings (H x W grid per image -> total_patches).
   - Learns distribution: per-patch mean and covariance (regularized) across the training dataset and saves it (`learned_distribution.pkl`).
   - For each test image, computes Mahalanobis distance from learned distribution for each patch, producing an anomaly map.
   - Upscales maps to image size, computes image-level anomaly scores (max pooling), computes ROC AUC, PR AUC and other metrics, and saves visualizations & plots.

If `--mahalanobis_on_gpu` or `--stats_on_gpu` are provided and CUDA is available, parts of the computation can be performed on GPU for speed. There are also instances where running stats on RAM and mahalanobis on GPU allows for avoiding OOM errors 

