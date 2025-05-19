# src/feature_engineering.py
import pandas as pd

def compute_delta_features(df):
    df['DAYS_TO_PAYMENT'] = (df['PAYMENT_DATE'] - df['APPLICATION_CREATED_DATE']).dt.days.fillna(-1)
    df['TERM_LENGTH_DAYS'] = (df['EXPIRATION_DATE'] - df['DATE_ISSUED']).dt.days.fillna(-1)
    df['DAYS_TO_ISSUE'] = (df['DATE_ISSUED'] - df['APPLICATION_CREATED_DATE']).dt.days.fillna(-1)
    return df

def extract_date_parts(df):
    dt = df['APPLICATION_CREATED_DATE']
    df['APP_YEAR'] = dt.dt.year.fillna(0).astype(int).replace(0,-1)
    df['APP_MONTH'] = dt.dt.month.fillna(0).astype(int).replace(0,-1)
    df['APP_DOW'] = dt.dt.dayofweek.fillna(0).astype(int).replace(0,-1)
    return df
