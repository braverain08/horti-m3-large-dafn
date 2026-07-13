# Horti-M3-Large & DAFN-T

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17217565.svg)](https://zenodo.org/records/17217565)

**Code for:** *"Horti-M3-Large: A Large-Scale Benchmark and Adaptive Temporal Fusion Framework for Tomato Stress Diagnosis in Cold-Region Greenhouses"*

This repository contains the complete pipeline for constructing the **Horti-M3-Large** dataset and reproducing all experiments with the **DAFN-T** framework for multimodal stress diagnosis in cold-region greenhouses.

---

## Overview

Cold-region greenhouses face unique challenges: extreme temperature fluctuations, prolonged low-light periods, and condensation that degrades visual data. This work makes three contributions:

1. **Horti-M3-Large** — The largest known benchmark for cold-region greenhouse stress diagnosis (9,384 annotated samples, 108 tomato plants, 3 growing seasons, 3 modalities)
2. **Systematic evaluation** — Demonstrates frozen ImageNet features achieve only 37.7% accuracy, while fine-tuning on Horti-M3-Large yields 97.4%
3. **DAFN-T** — A lightweight adaptive temporal fusion framework (~80K parameters, 1.5ms CPU inference) designed for robust performance under visual data degradation

---

## Repository Structure

```
├── data/
│   ├── 01_unify_csv.py         # Unify raw agronomic CSVs
│   ├── 02_parse_labels.py      # Parse stress labels from growth records
│   ├── 03_parse_sensors.py     # Process environmental sensor data
│   ├── 04_build_dataset.py     # Build final dataset with plant-wise split
│   ├── 05_extract_features.py  # Extract frozen ResNet50 features
│   ├── 06_finetune_resnet.py   # Fine-tune ResNet50 on Horti-M3-Large
│   ├── 07_extract_logits.py    # Extract logits (2 modalities: image + agronomic)
│   ├── 08_extract_logits3d.py  # Extract logits (3 modalities: + sensor)
│   └── 99_gradcam.py           # Grad-CAM visualization
├── experiments/
│   └── run.py                  # Main experiment runner
├── models/
│   ├── __init__.py
│   ├── dafn.py                 # Original DAFN (dual-branch, frozen backbone)
│   └── dafn_ft.py              # DAFN-ImageNet & DAFN-Proj (fine-tuned backbone)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24
scikit-learn>=1.3
Pillow>=10.0
tqdm>=4.65
openpyxl>=3.1
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Reproducing Experiments

### 1. Download the Dataset

The experiment uses the **Horti-M3-Large** dataset (Gong et al., 2026), which consists of:

- **RGB images** — 9,384 annotated greenhouse plant images (3 growing seasons)
- **Agronomic measurements** — 8 physiological indices per observation
- **Environmental sensor data** — temperature, humidity, light intensity, CO₂ (hourly)

The dataset is hosted on **Zenodo**:

```
🔗 https://zenodo.org/records/17217565
```

Download the ZIP file, extract to a local directory, and set `--data-dir` accordingly.

**Expected directory structure after extraction:**

```
2023-2025 Tomato dataset/
├── images/       # RGB images: {plant_id}_{date}_{index}.jpg
├── agronomic/    # Growth index CSVs per season
├── sensor/       # Environmental sensor data
└── metadata/     # Plant-wise train/val/test split
```

> 📌 If you use this dataset in your research, please cite the original paper (see [Citation](#citation)).

### 2. Build the Dataset (from raw data)

```bash
# Step 1: Unify raw agronomic CSVs
python data/01_unify_csv.py

# Step 2: Parse stress labels
python data/02_parse_labels.py

# Step 3: Process sensor data
python data/03_parse_sensors.py

# Step 4: Build final dataset with plant-wise split
python data/04_build_dataset.py
```

### 3. Extract Features (GPU recommended)

```bash
# Extract frozen ResNet50 features
python data/05_extract_features.py \
    --input data/dataset_ready.csv \
    --output data/image_features.npy \
    --data-dir /path/to/dataset

# Fine-tune ResNet50 on the dataset
python data/06_finetune_resnet.py

# Extract fine-tuned logits (2 modalities)
python data/07_extract_logits.py

# Extract fine-tuned logits (3 modalities, optional)
python data/08_extract_logits3d.py
```

### 4. Run Experiments

```bash
# Main experiments (image + agronomic)
python experiments/run.py

# With sensor data
python experiments/run.py --use_sensor

# Grad-CAM visualization
python data/99_gradcam.py
```

---

## Citation

If you use this code or the Horti-M3-Large dataset in your research, please cite:

```bibtex
@article{gong2026hortim3large,
  title={Horti-M3-Large: A Large-Scale Benchmark and Adaptive Temporal Fusion Framework for Tomato Stress Diagnosis in Cold-Region Greenhouses},
  author={Gong, Yu and He, Yifei and Zhang, Xuefeng},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  note={Dataset: \url{https://doi.org/10.5281/zenodo.17217565}}
}
```

---

## Dataset DOI

The **Horti-M3-Large** dataset is archived on Zenodo:

| Identifier | Link |
|------------|------|
| **Dataset DOI** | [10.5281/zenodo.17217565](https://zenodo.org/records/17217565) |

To cite the dataset directly, use:

> Gong, Y., He, Y., & Zhang, X. (2026). *Horti-M3-Large: Tomato Stress Dataset for Cold-Region Greenhouses (2023–2025)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17217565

> **Tip:** After making this repository public (post-publication), enable the [Zenodo–GitHub integration](https://zenodo.org/account/settings/github/) to auto-archive releases and obtain a separate DOI for the code.

---

## License

This project is licensed under the MIT License.

## Contact

Yu Gong — gongyu6982@163.com  
Heilongjiang Academy of Agricultural Sciences, Harbin, China
