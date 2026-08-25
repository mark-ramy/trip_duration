"""
Train and save the project's models on the processed training data.

Ridge(alpha=1) remains the project's OFFICIAL model, as specified in the
assignment ("We will fix the model to Ridge, and fix its alpha to 1").
XGBoost is trained alongside it purely as a comparison model (use
--models ridge to skip it if you only want the official model).

Pipeline
--------
1. Load the processed train.csv produced by build_processed_train.py
   (already has weather + OSRM features merged in, outliers filtered, and
   the target log1p-transformed — see features/main.py::run_changes).
2. One-hot encode the remaining categorical column(s) (vendor_id).
3. Fit Ridge(alpha=1) and/or XGBoost on the full processed training data.
4. Save everything needed to reproduce predictions later:
   - the fitted model(s): ridge_model.joblib, xgb_model.joblib
   - the fitted one-hot encoder + which columns it was fit on
   - the final feature column order the model(s) expect
   - the weather/osrm paths used (so test_model.py can be sure it's
     reproducing the exact same feature set)
   - the held-out metrics for every model trained, as metrics.json
     (consumed by generate_report.py)

Run (both models, default):
    python train_model.py --data "path/to/split/processed/train.csv" \
        --weather "path/to/split/weather/weather_dataset.csv" \
        --osrm "path/to/split/OSRM/fastest_routes_train_part_1.csv" \
               "path/to/split/OSRM/fastest_routes_train_part_2.csv" \
        --out-dir artifacts

Run (Ridge only, the official model):
    python train_model.py --data "..." --models ridge --out-dir artifacts

Run (XGBoost only):
    python train_model.py --data "..." --models xgboost --out-dir artifacts
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from features.encoding import apply_encoding

TARGET = "trip_duration"
RIDGE_ALPHA = 1.0
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def evaluate(model, X, y, name):
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))

    print(f"\n{name}")
    print(f"  RMSE (log1p scale): {rmse:.4f}")
    print(f"  MAE  (log1p scale): {mae:.4f}")
    print(f"  R2                : {r2:.4f}")

    return {"split": name, "rmse": rmse, "mae": mae, "r2": r2, "n_rows": int(len(y))}


def train(processed_train_path, weather_path, osrm_paths, out_dir,
          models=("ridge", "xgboost")):
    os.makedirs(out_dir, exist_ok=True)
    models = set(models)

    print(f"Loading processed training data from {processed_train_path} ...")
    df = pd.read_csv(processed_train_path)
    print(f"  shape: {df.shape}")

    df_encoded, encoder, categorical_cols = apply_encoding(df)

    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]
    feature_columns = X.columns.tolist()

    # Training-set medians, saved so test_model.py can fill any NaNs that
    # show up in val/test (e.g. a pickup date outside the weather file's
    # range, or an id with no OSRM match) the same way every time, instead
    # of the model(s) silently erroring out on NaN.
    feature_medians = X.median(numeric_only=True).to_dict()

    n_missing = int(X.isna().sum().sum())
    if n_missing:
        print(f"  Filling {n_missing} missing value(s) in the feature matrix with "
              f"training-set medians (e.g. unmatched OSRM ids, out-of-range weather dates).")
        X = X.fillna(feature_medians)

    trained = {}
    metrics = {}

    # --- Ridge (the official model) ---
    if "ridge" in models:
        print(f"\nTraining Ridge(alpha={RIDGE_ALPHA}) on {X.shape[0]} rows, "
              f"{X.shape[1]} features ...")
        ridge_model = Ridge(alpha=RIDGE_ALPHA)
        ridge_model.fit(X, y)

        ridge_train_metrics = evaluate(ridge_model, X, y, "Ridge [Train split]")

        joblib.dump(ridge_model, os.path.join(out_dir, "ridge_model.joblib"))
        trained["ridge"] = ridge_model
        metrics["ridge"] = {"train": ridge_train_metrics}

    # --- XGBoost (comparison model) ---
    if "xgboost" in models:
        print(f"\nTraining XGBoost on {X.shape[0]} rows, "
              f"{X.shape[1]} features ...")
        xgb_model = XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(X, y)

        xgb_train_metrics = evaluate(xgb_model, X, y, "XGBoost [Train split]")

        joblib.dump(xgb_model, os.path.join(out_dir, "xgb_model.joblib"))
        trained["xgboost"] = xgb_model
        metrics["xgboost"] = {"train": xgb_train_metrics}

    # --- Save everything needed to reproduce predictions later ---
    joblib.dump(encoder, os.path.join(out_dir, "onehot_encoder.joblib"))

    metadata = {
        "models_trained": sorted(trained.keys()),
        "alpha": RIDGE_ALPHA,
        "xgb_params": XGB_PARAMS,
        "target": TARGET,
        "target_transform": "log1p",
        "categorical_cols": list(categorical_cols),
        "feature_columns": feature_columns,
        "feature_medians": feature_medians,
        "weather_path": weather_path,
        "osrm_paths": osrm_paths,
        "processed_train_path": processed_train_path,
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model(s) {sorted(trained.keys())}, encoder, metadata, "
          f"and metrics to '{out_dir}/'.")
    return trained, encoder, metadata, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save Ridge(alpha=1) and/or XGBoost on the processed training data."
    )
    parser.add_argument("--data", required=True, help="Path to the processed train.csv.")
    parser.add_argument(
        "--weather", default=None,
        help="Path to weather_dataset.csv used when building --data (saved into "
             "model_metadata.json so test_model.py can reuse the exact same path).",
    )
    parser.add_argument(
        "--osrm", default=None, nargs="*",
        help="Path(s) to the OSRM csv file(s) used when building --data.",
    )
    parser.add_argument("--out-dir", default="artifacts", help="Where to save the trained model(s).")
    parser.add_argument(
        "--models", default=["ridge", "xgboost"], nargs="+",
        choices=["ridge", "xgboost"],
        help="Which model(s) to train and save. Default: both. "
             "Use '--models ridge' for just the official model, "
             "'--models xgboost' for just the comparison model.",
    )
    args = parser.parse_args()

    train(args.data, args.weather, args.osrm, args.out_dir, models=args.models)