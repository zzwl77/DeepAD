# Deep Learning-Driven 3D Eye Tracking as a Biomarker for Alzheimer’s Disease Detection and Assessment

Reference implementation of **DeepAD** for binocular eye-movement–based Alzheimer’s disease (AD) classification and cognitive assessment. The repository includes model code, training/evaluation scripts, and simple visualization utilities.

## Quick Start

### 1) Environment

```bash
git clone https://github.com/zzwl77/DeepAD.git
cd DeepAD
````

Create a clean Python environment and install the required packages used by the training scripts (e.g., PyTorch, torchvision, pandas, numpy, scipy, scikit-learn, matplotlib).

### 2) Data

This repository is associated with the study **“Deep Learning-Driven 3D Eye Tracking as a Biomarker for Alzheimer’s Disease Detection and Assessment.”**

If you need access to the research data or have data-related questions, **please contact the paper’s corresponding author (as listed in the manuscript).**

A publicly shared diagnostic test dataset related to this study, **ADEM_TEST**, is available via Zenodo.

To use your own data, follow the expected structure in `AD_Dataloader.py` and prepare the CSV split files required by the training scripts (e.g., `ad_10f4`, `adnc_10f4`).

### 3) Training

Training scripts are provided in `DeepAD/train/`.

* **Regression (MMSE / MoCA, AD only):** `main.py`
* **Regression (MMSE / MoCA, AD + NC):** `mainADNC.py`
* **Classification (AD vs. NC):** `main_cls.py`

Example classification run:

```bash
python main_cls.py \
  --dataset /path/to/data \
  --datacsv adnc_10f4 \
  --questionnaire cls \
  --n_fold 0 \
  --train-batch 20 \
  --test-batch 20 \
  --gpu-id 0 \
  --epochs 100
```

Example regression run (MMSE / MoCA):

```bash
python main.py \
  --dataset /path/to/data \
  --datacsv ad_10f4 \
  --questionnaire mmse \
  --n_fold 0 \
  --train-batch 10 \
  --test-batch 10 \
  --gpu-id 0 \
  --epochs 100
```

Example regression run with AD + NC:

```bash
python mainADNC.py \
  --dataset /path/to/data \
  --datacsv adnc_10f4 \
  --questionnaire moca \
  --n_fold 0 \
  --train-batch 20 \
  --test-batch 20 \
  --gpu-id 0 \
  --epochs 100
```

Shell launchers are also provided for repeated experiments:

* `main_cls.sh`
* `main_mmse_ad.sh`
* `main_mmse_adnc.sh`
* `main_moca_ad.sh`
* `main_moca_adnc.sh`

Please update the dataset paths in these scripts before use.

### 4) Evaluation & Visualization

* **Classification test:** `test/test_cls.py`
* **Regression test:** `test/test_reg.py`
* **Bar plots:** `plot/plot_bar.py`
* **Correlation heatmaps:** `plot/plot_corr_heat.py`
* **EEG / signal plotting:** `plot/plot_eeg.py`
* **t-SNE visualization:** `plot/plot_tsne.py`

Run these scripts with your own checkpoints, paths, and settings as needed.

## Repository Layout

```text
DeepAD/
├─ DeepAD/
│  ├─ AD_Dataloader.py      # data loading / preprocessing
│  ├─ models/               # model definitions
│  │  ├─ cls/               # classification models and ablations
│  │  ├─ regression/        # regression models and ablations
│  │  ├─ vit/               # transformer-related modules
│  │  └─ utils.py
│  ├─ plot/                 # plotting / visualization scripts
│  ├─ test/                 # evaluation scripts
│  ├─ train/                # training entry scripts and launchers
│  └─ utils/                # logging, LR schedule, helpers, progress
└─ README.md
```

Files are organized as listed in the repository.

## Citation

If this code helps your research, please cite the associated manuscript:

```bibtex
@article{DeepAD2025,
  title   = {Deep Learning-Driven 3D Eye Tracking as a Biomarker for Alzheimer’s Disease Detection and Assessment},
  author  = {…},
  journal = {…},
  year    = {2025}
}
```

[3]: https://raw.githubusercontent.com/zzwl77/DeepAD/main/DeepAD/train/main.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/zzwl77/DeepAD/main/DeepAD/train/main_cls.sh "raw.githubusercontent.com"
