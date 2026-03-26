# 🚗 Hierarchical Vehicle Metadata Identification via Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11n--cls-green.svg)](https://github.com/ultralytics/ultralytics)
[![Paper](https://img.shields.io/badge/Paper-FLAIRS--39-red.svg)](#citation)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

> **A hierarchical deep learning framework for automated vehicle make, model, and body type classification — designed for law enforcement and intelligent transportation applications.**

Published research — *[Your Paper Title]*, **FLAIRS-39, 2025** | Affiliation: University of Technology Sydney & Daffodil International University

---

## 📌 Overview

This repository contains the full implementation of a hierarchical vehicle metadata identification system using **YOLOv11n-cls** (YOLO nano classification). The system is designed to classify vehicles from surveillance and dashcam imagery across three levels:

- **Make classification** — 15 car brands (Audi, BMW, Toyota, Tesla, etc.)
- **Body type classification** — 11 vehicle body types (83.69% Top-1 accuracy)
- **Viewpoint-aware inference** — rear-only, front+rear, and all-view variants

The primary application domain is **law enforcement** — specifically automated vehicle identification from CCTV and road surveillance footage — though the framework is general-purpose for any intelligent transport system.

---

## 🏆 Key Results

| Task | Model | Dataset Split | Top-1 Accuracy |
|---|---|---|---|
| Car Body Type Classification | YOLOv11n-cls | Test | **83.69%** |
| Car Make Classification — Rear View (14 brands) | YOLOv11n-cls | Test | **[XX]%** |
| Car Make Classification — Front+Rear (14 brands) | YOLOv11n-cls | Test | **[XX]%** |
| Car Make Classification — All Views (15 brands) | YOLOv11n-cls | Test | **[XX]%** |

> **Note:** All experiments use YOLOv11n-cls (nano variant). Full training logs and confusion matrices are in `runs/`.

---

## 🎯 Supported Vehicle Classes

### Car Makes (15 Brands)
| | | | | |
|---|---|---|---|---|
| Audi | BMW | Mercedes-Benz | Toyota | Honda |
| Hyundai | Kia | Lexus | Mazda | Nissan |
| Ford | Suzuki | BYD | Mini | Tesla |

### Body Types (11 Categories)
Sedan · SUV · Hatchback · Coupe · Wagon · Convertible · Van · Pickup · Crossover · MPV · Sports

---

## 🏗️ System Architecture

The pipeline follows a **hierarchical classification** strategy:

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Viewpoint Detection │  ← Front / Rear / Side
└─────────┬───────────┘
          │
    ┌─────▼──────┐
    │ Body Type  │  ← 11 classes (83.69% acc.)
    │Classifier  │
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │   Make     │  ← 14–15 brands, viewpoint-specific
    │Classifier  │
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │Brand-Spec. │  ← Optional: brand-specific model
    │   Model    │
    └────────────┘
```

Each stage uses a dedicated **YOLOv11n-cls** model, enabling modular replacement or fine-tuning of individual stages without retraining the full pipeline.

---

## 📁 Project Structure

```
car_project_dev/
│
├── notebooks/                          ← Training & experimentation notebooks
│   ├── YoLo_classification.ipynb       ← Base make classification
│   ├── YoLo_classification_font_rear.ipynb  ← Front/rear variant
│   └── YoLo_classification car model classification.ipynb
│
├── yolo_cls_car_makes/                 ← Dataset: rear-view (14 brands)
│   ├── train/ val/ test/
│   └── label_vocabulary.csv
│
├── yolo_cls_car_makes_front_rear/      ← Dataset: front+rear (14 brands)
│   ├── train/ val/ test/
│   └── label_vocabulary.csv
│
├── yolo_cls_car_makes_allview/         ← Dataset: all viewpoints (15 brands)
│   ├── train/ val/ test/
│   └── label_vocabulary.csv
│
├── yolo_cls_car_type/                  ← Dataset: body type (11 classes)
├── brand_specific_models/              ← Per-brand fine-tuned models
├── models/                             ← Saved model weights
├── runs/                               ← Training outputs (metrics, plots)
├── reports/figures/                    ← Confusion matrices, training curves
├── MVP/                                ← Minimum viable product demo
├── docs/                               ← Additional documentation
│
├── model_config.json                   ← Centralized model configuration
├── requirements.txt                    ← Python dependencies
├── requirements_streamlit.txt          ← Demo app dependencies
├── run_app.sh                          ← One-command demo launcher
├── Makefile                            ← Convenience commands
└── pyproject.toml                      ← Project metadata
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~4GB disk space for datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/afraz-rupak/car_project_dev.git
cd car_project_dev

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo App

```bash
bash run_app.sh
```

Or manually:

```bash
pip install -r requirements_streamlit.txt
streamlit run MVP/app.py
```

### Training a Model

Open any notebook in `notebooks/` and follow the step-by-step cells. Each notebook is self-contained:

```bash
jupyter notebook notebooks/YoLo_classification.ipynb
```

For a specific viewpoint variant:
- **Rear-only:** `YoLo_classification.ipynb`
- **Front+Rear:** `YoLo_classification_font_rear.ipynb`
- **All views:** `YoLo_classification car model classification.ipynb`

### Quick Inference (Python)

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("models/yolo11n_car_makes_allview.pt")

# Run inference
results = model.predict("path/to/car_image.jpg")
print(results[0].probs.top1)   # Top predicted class index
print(results[0].probs.top5)   # Top-5 predictions
```

---

## 📊 Datasets

Three viewpoint-specific datasets are included, all formatted in **YOLO classification format** (`train/class_name/image.jpg`):

| Dataset | Viewpoints | Brands | Split |
|---|---|---|---|
| `yolo_cls_car_makes` | Rear only | 14 | 70/15/15 |
| `yolo_cls_car_makes_front_rear` | Front + Rear | 14 | 70/15/15 |
| `yolo_cls_car_makes_allview` | All views | 15 (+ Tesla) | 70/15/15 |
| `yolo_cls_car_type` | All views | — | 11 body types |

Data was collected from publicly available vehicle image databases and curated for balanced class representation.

---

## 🧪 Reproducibility

All experiments are fully reproducible. To replicate the paper results:

```bash
# Using Makefile
make train_rear       # Train rear-view model
make train_allview    # Train all-view model
make evaluate         # Run evaluation on test split
```

Model configuration (epochs, batch size, image size) is centralized in `model_config.json`.

---

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{afraz2025vehicle,
  title     = {[Your Paper Title]},
  author    = {Afraz Ul Haque and [Co-authors]},
  booktitle = {Proceedings of the 38th International Florida Artificial Intelligence Research Society Conference (FLAIRS-39)},
  year      = {2025},
  note      = {[DOI or URL when available]}
}
```

---

## 👤 Author

**Afraz Ul Haque**
Master of Data Science & Innovation — University of Technology Sydney
Previously: ML Engineer, InflexionPoint Technologies | Researcher, DIU NLP & ML Research Lab

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.com/citations?user=tQ4Ur6UAAAAJ&hl=en)
[![GitHub](https://img.shields.io/badge/GitHub-afraz--rupak-black?logo=github)](https://github.com/afraz-rupak)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/afraz-ul-haque-rupak-89b8a1194/)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. This is active research code — if you use it and find bugs or want to add support for new vehicle brands, please open an issue or PR.

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

> **Research Impact:** This system is designed to assist law enforcement agencies in automating vehicle identification from surveillance footage, reducing manual review time and improving incident response accuracy.



