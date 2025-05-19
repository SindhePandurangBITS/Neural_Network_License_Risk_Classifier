# src/preprocessing.py
import pandas as pd
from sklearn.cluster import KMeans
from config import PLOTS_DIR

def impute_categoricals(df, cols):
    for c in cols:
        df[f'{c}_MISSING'] = df[c].isna().astype(int)
        mode = df[c].mode()[0]
        df[c] = df[c].fillna(mode).astype('category')
    return df

def cluster_geo_impute(df, n_clusters=50, seed=42):
    coords = df[['LATITUDE','LONGITUDE']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(coords)
    mask = df['LATITUDE'].isna() | df['LONGITUDE'].isna()
    df.loc[mask,['LATITUDE','LONGITUDE']] = kmeans.cluster_centers_[kmeans.predict(df.loc[mask,['LATITUDE','LONGITUDE']].fillna(0))]
    assert df['LATITUDE'].notna().all()
    return df

def encode_flags_and_zip(df, flag_cols):
    for f in flag_cols:
        df[f] = df[f].map({'Y':1,'N':0}).fillna(0).astype(int)
    df['ZIP_CODE'] = df['ZIP_CODE'].str.zfill(5)
    return df
