import pandas as pd


def add_weather_features(df, weather_path, datetime_col="pickup_datetime"):
    """
    Merge daily NYC weather data (precipitation, snowfall, temperature) onto df.

    Expects a raw NOAA-style csv with columns:
        date, precipitation, snow fall, snow depth,
        maximum temperature, minimum temperature
    (same file used in the EDA notebook, section 9 - Weather EDA).

    The merge key is the calendar date of `datetime_col`, so this must be
    called while `datetime_col` (e.g. pickup_datetime) is still present.
    """
    df = df.copy()

    weather = pd.read_csv(weather_path)
    weather["date"] = pd.to_datetime(weather["date"], format="%d-%m-%Y").dt.date

    # "T" = trace amount in NOAA data, treated as a tiny non-zero value
    for col in ["precipitation", "snow fall", "snow depth"]:
        weather[col] = weather[col].replace("T", 0.01).astype(float)

    weather = weather.rename(columns={
        "precipitation": "rain",
        "snow fall": "s_fall",
        "snow depth": "s_depth",
        "maximum temperature": "max_temp",
        "minimum temperature": "min_temp",
    })

    weather["all_precip"] = weather["rain"] + weather["s_fall"]
    weather["has_snow"] = (weather["s_fall"] > 0).astype(int)
    weather["has_rain"] = (weather["rain"] > 0).astype(int)

    weather_features = weather[[
        "date", "rain", "s_fall", "all_precip",
        "has_snow", "has_rain", "s_depth", "max_temp", "min_temp",
    ]]

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["date"] = df[datetime_col].dt.date

    df = df.merge(weather_features, on="date", how="left")
    df = df.drop(columns=["date"])

    return df


def add_osrm_features(df, osrm_paths, id_col="id"):
    """
    Merge OSRM fastest-route features onto df by trip id: the distance,
    travel time, and number of steps of the theoretically fastest route
    between pickup and dropoff (see EDA notebook, section 10).

    osrm_paths: a single path, or a list of paths to concatenate
    (the Kaggle NYC taxi extra dataset ships this split into two parts).

    Must be called while `id_col` is still present in df.
    """
    df = df.copy()

    if isinstance(osrm_paths, str):
        osrm_paths = [osrm_paths]

    osrm = pd.concat([pd.read_csv(p) for p in osrm_paths], ignore_index=True)

    keep_cols = [c for c in ["id", "total_distance", "total_travel_time", "number_of_steps"]
                 if c in osrm.columns]
    osrm = osrm[keep_cols]

    df = df.merge(osrm, on=id_col, how="left")

    df = df.rename(columns={
        "total_distance": "osrm_distance",
        "total_travel_time": "osrm_travel_time",
        "number_of_steps": "osrm_num_steps",
    })

    return df
