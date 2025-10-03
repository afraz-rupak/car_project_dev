# Car Make Classification Using YOLO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project focuses on developing and training YOLO (You Only Look Once) classification models for identifying car makes from images. The project supports classification of multiple car brands including Audi, BMW, Mercedes-Benz, Toyota, Honda, and others, using different viewpoints (rear, front, and all-view datasets).

## Supported Car Makes

The model can classify the following car brands:
- Audi
- Mercedes-Benz (Benz)
- BMW
- BYD
- Ford
- Honda
- Hyundai
- Kia
- Lexus
- Mazda
- Mini
- Nissan
- Suzuki
- Tesla
- Toyota

## Project Organization

```
├── LICENSE                    <- Open-source license
├── Makefile                   <- Makefile with convenience commands
├── README.md                  <- Project documentation
├── requirements.txt           <- Python dependencies
├── pyproject.toml            <- Project configuration and metadata
├── car_project_dev/          <- Source code module
│   └── __init__.py
│
├── data/                     <- Project datasets
│   ├── car_csv/             <- CSV files with car data by brand
│   │   ├── audi.csv
│   │   ├── bmw.csv
│   │   ├── benz.csv
│   │   ├── toyota.csv
│   │   ├── tesla.csv
│   │   └── ... (other brand CSV files)
│   ├── generated_csv/       <- Generated datasets
│   ├── interim/             <- Intermediate processed data
│   ├── processed/           <- Final datasets for modeling
│   └── test_data/          <- Sample test images
│
├── notebooks/               <- Jupyter notebooks for experimentation
│   ├── YoLo_classification car model classification.ipynb
│   ├── YoLo_classification_font_rear.ipynb
│   ├── YoLo_classification.ipynb
│   └── yolo11n-cls.pt      <- Pre-trained YOLO model
│
├── models/                  <- Trained models and predictions
├── reports/                 <- Analysis reports
│   └── figures/            <- Generated plots and visualizations
│
├── runs/                    <- Training run outputs
│   ├── classify/           <- YOLO classification training runs
│   ├── classify2/
│   └── classify3/
│
├── yolo_cls_car_makes/      <- Rear-view car classification dataset
│   ├── label_vocabulary.csv
│   ├── train/              <- Training images by brand
│   ├── val/                <- Validation images by brand
│   └── test/               <- Test images by brand
│
├── yolo_cls_car_makes_front_rear/  <- Front and rear view dataset
│   ├── label_vocabulary.csv
│   ├── train/
│   ├── val/
│   └── test/
│
└── yolo_cls_car_makes_allview/     <- All viewpoints dataset
    ├── label_vocabulary.csv
    ├── train/
    ├── val/
    └── test/
```

## Datasets

The project includes three main datasets:

1. **yolo_cls_car_makes**: Rear-view car images for 14 car brands
2. **yolo_cls_car_makes_front_rear**: Combined front and rear view images
3. **yolo_cls_car_makes_allview**: All viewpoints including Tesla (15 brands total)

Each dataset is organized in YOLO classification format with separate train, validation, and test splits for each car brand.

## Model Training

The project uses YOLO11n (YOLOv11 nano) for car make classification. Training configurations and results are stored in the `runs/` directory with different experiment iterations.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore the notebooks in the `notebooks/` directory to understand the data and training process

3. Train models using the prepared datasets in `yolo_cls_car_makes*` directories

## Results

Training results and model performance metrics are stored in the `runs/` directory, organized by experiment type and iteration.

--------

