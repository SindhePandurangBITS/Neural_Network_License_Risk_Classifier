# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, RANDOM_SEED

def load_raw(path=RAW_DATA_PATH):
    df = pd.read_csv(
        path, low_memory=False,
        parse_dates=['APPLICATION_CREATED_DATE','PAYMENT_DATE','DATE_ISSUED','EXPIRATION_DATE'],
        dtype={'APPLICATION_ID': str,'ACCOUNT_ID': str,'ZIP_CODE': str,'LICENSE_STATUS': 'category'}
    )
    df.info()
    return df

def split_data(df, target_col='LICENSE_STATUS'):
    train_val, test = train_test_split(df, test_size=0.15, stratify=df[target_col], random_state=RANDOM_SEED)
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val[target_col], random_state=RANDOM_SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

