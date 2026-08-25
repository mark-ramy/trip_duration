"""
Build the processed training dataset — WITH weather + OSRM features baked in
— and save it to disk, ready to be handed to `modeling.py --data ...`.

This is the "save to processed files" step referenced by
features/main.py::run_changes and utils/data_helper.py::save_processed_data.
The EDA notebook explores weather (section 9) and OSRM (section 10)
separately; this script is what actually merges both into the dataset used
for training, instead of running the pipeline with weather_path=None and
osrm_paths=None.

Usage
-----
    python build_processed_train.py \
        --raw-train  "path/to/split/raw/train.csv" \
        --weather    "path/to/split/weather/weather_dataset.csv" \
        --osrm       "path/to/split/OSRM/fastest_routes_train_part_1.csv" \
                     "path/to/split/OSRM/fastest_routes_train_part_2.csv" \
        --out        "path/to/split/processed/train.csv"

IMPORTANT: whatever --weather/--osrm paths you use here must be passed
again as --weather/--osrm to modeling.py, so val/test go through the exact
same feature set and the columns line up.
"""

import argparse

import pandas as pd

from features.main import run_changes
from utils.data_helper import save_processed_data


def build_processed_train(raw_train_path, weather_path, osrm_paths, out_path):
    print(f"Loading raw training data from {raw_train_path} ...")
    df = pd.read_csv(raw_train_path)
    print(f"  raw shape: {df.shape}")

    df = run_changes(
        df,
        weather_path=weather_path,
        osrm_paths=osrm_paths,
        is_train=True,
    )
    print(f"  processed shape (after outlier filtering + feature engineering): {df.shape}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("  NOTE: columns with missing values after the merges (e.g. OSRM ids "
              "with no match, or dates outside the weather file's range):")
        print(missing.to_string())

    save_processed_data(df, out_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge weather + OSRM features into the raw training data and save it."
    )
    parser.add_argument("--raw-train", required=True, help="Path to the RAW train.csv.")
    parser.add_argument("--weather", required=True, help="Path to weather_dataset.csv.")
    parser.add_argument(
        "--osrm", required=True, nargs="+",
        help="Path(s) to the OSRM fastest-route csv file(s) "
             "(e.g. fastest_routes_train_part_1.csv fastest_routes_train_part_2.csv).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Where to save the processed csv, e.g. '.../split/processed/train.csv'.",
    )
    args = parser.parse_args()

    build_processed_train(args.raw_train, args.weather, args.osrm, args.out)
