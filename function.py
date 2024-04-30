import pandas as pd

def create_df(df, feature):
    df = df.dropna(subset=[feature]).query("Path.str.contains('frontal')")
    df = df[['Path', feature]].reset_index(drop=True)
    df.rename(columns={feature: 'Feature'}, inplace=True)
    return df