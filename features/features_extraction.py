import numpy as np
import pandas as pd


def calculate_airport_distance(
        df,
        lat1_col,
        lon1_col,
        fixed_lat,
        fixed_lon,
        output_col
):
    """
    Compute Haversine distance in meters to a fixed coordinate pair.
    """
    df = df.copy()

    lat1 = np.radians(df[lat1_col])
    lon1 = np.radians(df[lon1_col])

    lat2 = np.radians(fixed_lat)
    lon2 = np.radians(fixed_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0, 1)

    # Multiplying by 6371000 to get meters (to match the 2000m threshold in the plot)
    df[output_col] = 2 * 6371000 * np.arcsin(np.sqrt(a))

    return df


def calculate_haversine_distance(
    df,
    lat1_col='pickup_latitude',
    lon1_col='pickup_longitude',
    lat2_col='dropoff_latitude',
    lon2_col='dropoff_longitude',
    output_column = 'haversine_distance_km'
):
    """
    Compute great-circle (Haversine) distance in km between two coordinate pairs.
    """
    df = df.copy()

    lat1 = np.radians(df[lat1_col])
    lon1 = np.radians(df[lon1_col])
    lat2 = np.radians(df[lat2_col])
    lon2 = np.radians(df[lon2_col])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0, 1)

    df[output_column] = 2 * 6371 * np.arcsin(np.sqrt(a))
    return df


def calculate_manhattan_distance(
    df,
    lat1_col='pickup_latitude',
    lon1_col='pickup_longitude',
    lat2_col='dropoff_latitude',
    lon2_col='dropoff_longitude'
):
    """
    Approximate Manhattan/grid distance in km using latitude and longitude differences.
    """
    df = df.copy()

    # 1 degree latitude ≈ 111 km
    lat_diff = np.abs(df[lat1_col] - df[lat2_col]) * 111.0

    # 1 degree longitude ≈ 111 * cos(latitude) km
    avg_lat = np.radians((df[lat1_col] + df[lat2_col]) / 2.0)
    lon_diff = np.abs(df[lon1_col] - df[lon2_col]) * 111.0 * np.cos(avg_lat)

    df['manhattan_distance_km'] = lat_diff + lon_diff
    return df


def calculate_bearing(
    df,
    lat1_col='pickup_latitude',
    lon1_col='pickup_longitude',
    lat2_col='dropoff_latitude',
    lon2_col='dropoff_longitude',
):
    """
    Compute compass bearing (0–360°) from pickup to dropoff.
    """
    df = df.copy()

    lat1 = np.radians(df[lat1_col])
    lon1 = np.radians(df[lon1_col])
    lat2 = np.radians(df[lat2_col])
    lon2 = np.radians(df[lon2_col])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))
    df['bearing'] = (bearing + 360) % 360
    return df


def calculate_speed(df):
    """
    Compute speed in km/h using haversine distance and trip duration in seconds.
    """
    df = df.copy()
    df["speed_kmh"] = np.nan

    mask = (df["trip_duration"] > 0) & df["haversine_distance_km"].notnull()
    df.loc[mask, "speed_kmh"] = (
        df.loc[mask, "haversine_distance_km"] /
        (df.loc[mask, "trip_duration"] / 3600.0)
    )

    return df


