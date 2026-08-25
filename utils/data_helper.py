def save_processed_data(df, path):
    """
    Save the processed DataFrame to a given path.

    Args:
        df (pd.DataFrame): Data to save
        path (str): Full path including filename (e.g. 'data/processed/train_processed.csv')
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)
    print(f" Processed data saved to {path}")