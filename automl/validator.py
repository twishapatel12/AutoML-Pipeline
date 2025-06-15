import pandas as pd

def validate_data(df: pd.DataFrame):
    if df.isnull().any().any():
        raise ValueError("Dataset contains null values.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature and one label.")
 