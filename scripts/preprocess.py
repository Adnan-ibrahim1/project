import pandas as pd

def handle_missing_values(df):
    return df.fillna(df.median())

def encode_target(df, target_column):
    df[target_column] = df[target_column].astype("category").cat.codes
    return df
