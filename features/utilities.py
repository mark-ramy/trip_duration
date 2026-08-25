import numpy as np
import pandas as pd

def apply_log(feature):
    """
    :param feature: A feature to apply log on it
    :return: the same feature but logged
    """
    return np.log1p(feature)

def extract_datetime_features(df, time_col='pickup_datetime'):
    """
    Extracts hour, day of week, and month from a datetime column.
    """
    df = df.copy()

    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour

    df['day_of_week'] = df[time_col].dt.dayofweek

    df['month'] = df[time_col].dt.month

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Sat/Sun = 1

    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    return df


