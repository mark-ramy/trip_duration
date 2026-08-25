import numpy as np

from features.utilities import extract_datetime_features
from features.features_extraction import (
    calculate_bearing,
    calculate_manhattan_distance,
    calculate_haversine_distance,
    calculate_speed,
    calculate_airport_distance,
)
from features.external_data import add_weather_features, add_osrm_features


AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
}

AIRPORT_TRIP_THRESHOLD_M = 2000
MAX_DURATION_SECONDS = 22 * 3600
MAX_SPEED_KMH = 100


def run_changes(df, weather_path=None, osrm_paths=None, is_train=True):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip data (train or test).
    weather_path : str, optional
        Path to weather_dataset.csv. If given, daily weather features
        (rain, snowfall, temperature...) are merged in on pickup date.
    osrm_paths : str or list[str], optional
        Path(s) to the OSRM fastest-route csv file(s). If given, OSRM
        features (fastest-route distance/time/step count) are merged in
        on trip "id".
    is_train : bool, default True
        Whether `trip_duration` is a real, trustworthy target column.
        Duration/speed-based outlier filtering and the log1p transform of
        the target are skipped when False, since a test set either has no
        trip_duration or it isn't meant to drive filtering decisions.

    Steps
    -----
    1. Coordinate bounds filter -> keep only valid NYC-area rows.
    2. External data merges (weather by date, OSRM by id) -> done BEFORE
       "id"/"pickup_datetime" are dropped, since both merges need them.
    3. Feature engineering: datetime parts, distances, bearing, airport flags.
    4. Outlier filtering (train only).
    5. Drop leaky/unused columns.
    6. log1p transform of the skewed distance columns, and of the target
       (train only).
    """
    df = df.copy()

    # 1. Coordinate bounds filter
    lat_min, lat_max = 40.50, 41.00
    lon_min, lon_max = -74.25, -73.70
    df = df[
        (df["pickup_latitude"].between(lat_min, lat_max)) &
        (df["dropoff_latitude"].between(lat_min, lat_max)) &
        (df["pickup_longitude"].between(lon_min, lon_max)) &
        (df["dropoff_longitude"].between(lon_min, lon_max))
    ]

    # 2. External data merges (need "id" / pickup_datetime, so before they're dropped)
    if osrm_paths is not None and "id" in df.columns:
        df = add_osrm_features(df, osrm_paths)

    if weather_path is not None:
        df = add_weather_features(df, weather_path)

    df = df.drop(columns=["id"], errors="ignore")
    df = extract_datetime_features(df)
    df = df.drop(columns=["pickup_datetime"], errors="ignore")

    # 3. Feature engineering
    df = calculate_haversine_distance(df)
    df = calculate_manhattan_distance(df)
    df = calculate_bearing(df)

    locations = {
        "pickup": ("pickup_latitude", "pickup_longitude"),
        "dropoff": ("dropoff_latitude", "dropoff_longitude"),
    }
    for airport, (airport_lat, airport_lon) in AIRPORTS.items():
        for loc, (lat_col, lon_col) in locations.items():
            df = calculate_airport_distance(
                df, lat_col, lon_col, airport_lat, airport_lon, f"{airport}_{loc}_dist"
            )

    df["jfk_trip"] = (
        (df["jfk_pickup_dist"] <= AIRPORT_TRIP_THRESHOLD_M) |
        (df["jfk_dropoff_dist"] <= AIRPORT_TRIP_THRESHOLD_M)
    )
    df["lga_trip"] = (
        (df["lga_pickup_dist"] <= AIRPORT_TRIP_THRESHOLD_M) |
        (df["lga_dropoff_dist"] <= AIRPORT_TRIP_THRESHOLD_M)
    )

    # 4. Outlier filtering - train only (needs trustworthy trip_duration)
    if is_train and "trip_duration" in df.columns:
        df = calculate_speed(df)

        df = df[
            (df["trip_duration"] < MAX_DURATION_SECONDS) &
            ((df["manhattan_distance_km"] > 0) |
             ((df["manhattan_distance_km"] <= 0.01) & (df["trip_duration"] < 60))) &
            (df["trip_duration"] > 10) &
            (df["speed_kmh"] < MAX_SPEED_KMH)
        ]

        if "passenger_count" in df.columns:
            df = df[df["passenger_count"] > 0]

    # 5. Drop leaky / unused columns
    # store_and_fwd_flag has no relationship with trip duration.
    # speed_kmh is derived from trip_duration itself -> would leak the target.
    df = df.drop(columns=["store_and_fwd_flag", "speed_kmh"], errors="ignore")

    # 6. Log-transform right-skewed distance columns (and target, train only)
    df["haversine_distance_km"] = np.log1p(df["haversine_distance_km"])
    df["manhattan_distance_km"] = np.log1p(df["manhattan_distance_km"])

    if is_train and "trip_duration" in df.columns:
        df["trip_duration"] = np.log1p(df["trip_duration"])

    return df
