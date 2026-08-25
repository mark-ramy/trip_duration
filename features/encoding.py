from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def apply_encoding(df, encoder=None, categorical_cols=None):
    """
    One-hot encode the object-dtype columns of df.

    Pass `encoder` (and the `categorical_cols` it was fit on) when encoding
    a val/test set, so the SAME fitted encoder is reused instead of a new
    one being fit from scratch (which could silently produce a different
    set/order of dummy columns than the training data used).

    Parameters
    ----------
    df : pd.DataFrame
    encoder : fitted sklearn OneHotEncoder, optional
        If None, a new encoder is fit on `df` (this is the training-set path).
    categorical_cols : Index/list[str], optional
        Which columns to encode. If None, inferred from df's object dtypes.
        Must be passed explicitly (from the training call) when reusing an
        encoder, so val/test are encoded on the same columns even if their
        dtypes happen to differ.

    Returns
    -------
    df_encoded : pd.DataFrame
    encoder : the fitted OneHotEncoder (fit here, or the one passed in)
    categorical_cols : the columns that were encoded (fit here, or passed in)
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) == 0:
        return df.copy(), encoder, categorical_cols

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df[categorical_cols])
    else:
        encoded_array = encoder.transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df_encoded = pd.concat(
        [df.drop(columns=categorical_cols), encoded_df],
        axis=1
    )

    return df_encoded, encoder, categorical_cols