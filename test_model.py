"""
Load the saved model(s) — Ridge(alpha=1) and/or XGBoost — and run them on
a sample/val/test csv.

Per the assignment: "You are allowed to only test once and report your
test-score." This script does not enforce that (it's a one-time human
process rule, not something to hard-block in code), but it prints a
reminder every run and writes a timestamped log line to
artifacts/test_runs.log so you have a record of exactly when you spent
your one official run on the real test set.

The input csv is expected in the RAW schema (same columns as train.csv,
minus/optionally including trip_duration) — this script runs it through
the exact same features/main.py::run_changes(is_train=False) pipeline,
then the SAME fitted encoder saved by train_model.py, so the feature
columns line up with what the model(s) were trained on.

Run (both models, default):
    python test_model.py --input "path/to/raw/val.csv" \
        --artifacts-dir artifacts \
        --out val_predictions.csv

    python test_model.py --input "path/to/raw/test.csv" \
        --artifacts-dir artifacts \
        --out test_predictions.csv

Run (Ridge only, the official model):
    python test_model.py --input "path/to/raw/test.csv" \
        --artifacts-dir artifacts --models ridge --out test_predictions.csv

Run (XGBoost only):
    python test_model.py --input "path/to/raw/test.csv" \
        --artifacts-dir artifacts --models xgboost --out test_predictions.csv

If the input csv has a trip_duration column, RMSE/MAE/R2 are also
computed and printed for every model requested (and saved to
artifacts/test_runs.log).
"""

import argparse
import datetime as dt
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.main import run_changes
from features.encoding import apply_encoding

TARGET = "trip_duration"
MODEL_FILENAMES = {
    "ridge": "ridge_model.joblib",
    "xgboost": "xgb_model.joblib",
}


def load_artifacts(artifacts_dir, models=("ridge", "xgboost")):
    loaded = {}
    for name in models:
        path = os.path.join(artifacts_dir, MODEL_FILENAMES[name])
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            print(f"NOTE: {path} not found -> skipping '{name}' "
                  f"(train it first with train_model.py --models {name}).")

    if not loaded:
        raise FileNotFoundError(
            f"None of the requested models {list(models)} were found in "
            f"'{artifacts_dir}/'. Run train_model.py first."
        )

    encoder = joblib.load(os.path.join(artifacts_dir, "onehot_encoder.joblib"))
    with open(os.path.join(artifacts_dir, "model_metadata.json")) as f:
        metadata = json.load(f)
    return loaded, encoder, metadata


def preprocess(raw_path, encoder, metadata):
    raw_df = pd.read_csv(raw_path)

    df = run_changes(
        raw_df,
        weather_path=metadata.get("weather_path"),
        osrm_paths=metadata.get("osrm_paths"),
        is_train=False,
    )

    y = None
    if TARGET in df.columns:
        y = np.log1p(df[TARGET])
        df = df.drop(columns=[TARGET])

    df_encoded, _, _ = apply_encoding(
        df, encoder=encoder, categorical_cols=metadata["categorical_cols"]
    )

    feature_columns = metadata["feature_columns"]
    missing = [c for c in feature_columns if c not in df_encoded.columns]
    if missing:
        print(f"WARNING: {len(missing)} expected feature(s) missing from this "
              f"input, filling with training-set medians: {missing}")

    X = df_encoded.reindex(columns=feature_columns)

    # Fill any NaN (missing columns entirely, plus row-level NaNs e.g. an
    # OSRM id with no match or a pickup date outside the weather file's
    # range) with the training-set medians saved by train_model.py, so
    # Ridge never errors out on NaN and every input is scored the same way.
    feature_medians = metadata.get("feature_medians", {})
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        print(f"Filling {n_missing} missing value(s) with training-set medians.")
        X = X.fillna(feature_medians).fillna(0)

    return X, y, df.index


def run_test(input_path, artifacts_dir, out_path, models=("ridge", "xgboost")):
    loaded_models, encoder, metadata = load_artifacts(artifacts_dir, models=models)

    print(f"Loaded model(s) {list(loaded_models.keys())}, trained on "
          f"{metadata['processed_train_path']}")
    print(f"Running on: {input_path}\n")

    X, y, kept_index = preprocess(input_path, encoder, metadata)

    result_df = pd.DataFrame(index=kept_index)
    all_metrics = {}

    for name, model in loaded_models.items():
        log_preds = model.predict(X)
        result_df[f"trip_duration_pred_seconds_{name}"] = np.expm1(log_preds)

        if y is not None:
            rmse = float(np.sqrt(mean_squared_error(y, log_preds)))
            mae = float(mean_absolute_error(y, log_preds))
            r2 = float(r2_score(y, log_preds))
            all_metrics[name] = {
                "rmse_log1p": rmse, "mae_log1p": mae, "r2": r2, "n_rows": int(len(y)),
            }

            print(f"{name} performance (log1p scale, matching training target):")
            print(f"  RMSE : {rmse:.4f}")
            print(f"  MAE  : {mae:.4f}")
            print(f"  R2   : {r2:.4f}\n")

    if y is not None:
        result_df["trip_duration_actual_seconds"] = np.expm1(y.values)
        if len(all_metrics) > 1:
            comparison = pd.DataFrame(all_metrics).T.sort_values("rmse_log1p")
            print("=== Comparison (sorted by RMSE, lower is better) ===")
            print(comparison.to_string())
    else:
        print(f"'{TARGET}' not found in {input_path} -> predictions only, no score computed.")

    metrics = all_metrics if all_metrics else None

    result_df.to_csv(out_path)
    print(f"\nPredictions saved to {out_path}")

    log_path = os.path.join(artifacts_dir, "test_runs.log")
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "input_path": input_path,
            "models": list(loaded_models.keys()),
            "n_rows": int(len(X)),
            "metrics": metrics,
        }) + "\n")
    print(f"Run logged to {log_path}")

    return result_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the saved model(s) and predict/score on a raw val/test csv."
    )
    parser.add_argument("--input", required=True, help="Path to the RAW sample/val/test csv.")
    parser.add_argument("--artifacts-dir", default="artifacts",
                         help="Directory produced by train_model.py.")
    parser.add_argument("--out", default="predictions.csv",
                         help="Where to save predictions.")
    parser.add_argument(
        "--models", default=["ridge", "xgboost"], nargs="+",
        choices=["ridge", "xgboost"],
        help="Which saved model(s) to run. Default: both (any that were "
             "trained/saved by train_model.py). Missing ones are skipped "
             "with a warning rather than erroring.",
    )
    args = parser.parse_args()

    run_test(args.input, args.artifacts_dir, args.out, models=args.models)
