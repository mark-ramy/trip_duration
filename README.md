# NYC Taxi Trip Duration Prediction

Predicts NYC taxi trip duration using engineered features (weather + OSRM routing
data) with **Ridge regression** (the official model for this assignment) and
**XGBoost** (trained alongside for comparison).

## Project structure

```
.
├── build_processed_train.py   # merges raw train data with weather + OSRM features
├── train_model.py             # fits Ridge(alpha=1) and/or XGBoost, saves artifacts
├── test_model.py              # runs saved model(s) on val/test data
├── generate_report.py         # builds a PDF report from metrics + EDA summary
├── features/
│   ├── main.py                # run_changes(): the shared feature pipeline
│   ├── features_extraction.py
│   ├── encoding.py            # one-hot encoding helpers
│   ├── external_data.py       # weather / OSRM merge logic
│   └── utilities.py
├── utils/
│   └── data_helper.py         # save/load processed data
├── notebooks/
│   ├── NYC_trip_duration_EDA.ipynb
│   └── features_engineering.ipynb
├── artifacts/                 # trained models, encoder, metrics, metadata
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Build the processed training set** (merges weather + OSRM into the raw data):

```bash
python build_processed_train.py \
    --raw-train  "path/to/split/raw/train.csv" \
    --weather    "path/to/split/weather/weather_dataset.csv" \
    --osrm       "path/to/split/OSRM/fastest_routes_train_part_1.csv" \
                 "path/to/split/OSRM/fastest_routes_train_part_2.csv" \
    --out        "path/to/split/processed/train.csv"
```

**2. Train the model(s):**

```bash
python train_model.py --data "path/to/split/processed/train.csv" \
    --weather "path/to/split/weather/weather_dataset.csv" \
    --osrm "path/to/split/OSRM/fastest_routes_train_part_1.csv" \
           "path/to/split/OSRM/fastest_routes_train_part_2.csv" \
    --out-dir artifacts
```

Use `--models ridge` or `--models xgboost` to train just one.

**3. Evaluate on val/test data:**

```bash
python test_model.py --input "path/to/raw/val.csv" \
    --artifacts-dir artifacts --out val_predictions.csv
```

**4. Generate the PDF report:**

```bash
python generate_report.py --artifacts-dir artifacts --out report.pdf
```

## Model notes

- **Ridge(alpha=1)** is the official model for this assignment (fixed per spec).
- **XGBoost** (600 trees, depth 7, lr 0.05) is trained purely as a comparison.
- Target (`trip_duration`) is log1p-transformed; outliers are filtered during
  processing (see `features/main.py::run_changes`).
- Current training-split metrics are in `artifacts/metrics.json`.

## Results

All metrics are computed on the log1p-transformed target.

| Split | Model   | RMSE   | MAE    | R²     | n_rows  |
|-------|---------|--------|--------|--------|---------|
| Train | Ridge   | 0.4350 | 0.3292 | 0.6667 | 993,553 |
| Train | XGBoost | 0.2962 | 0.2134 | 0.8454 | 993,553 |
| Val   | Ridge   | 0.4951 | 0.3433 | 0.6158 | 229,182 |
| Val   | XGBoost | 0.4037 | 0.2363 | 0.7446 | 229,182 |
| Test  | Ridge   | 0.4904 | 0.3424 | 0.6192 | 229,144 |
| Test  | XGBoost | 0.4008 | 0.2362 | 0.7456 | 229,144 |

XGBoost outperforms Ridge on every split and metric, but Ridge remains the
official model for this assignment per the spec. Val/test scores for both
models track closely with their train-split numbers, suggesting neither model
is overfitting substantially.

## Data

Raw NYC taxi trip data, weather, and OSRM routing files are not included in
this repo — point the scripts above at your local copies via `--raw-train`,
`--weather`, and `--osrm`.
