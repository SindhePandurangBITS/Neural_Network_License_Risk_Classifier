# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOTS_DIR

def plot_missingness(df):
    missing = df.isna().sum().sort_values(ascending=False)
    print("Missingness ranking:\n", missing)
    plt.figure(figsize=(10,6))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'missingness.png'); plt.close()

def plot_cardinality(df):
    cat_cols = df.select_dtypes(include=['category','object']).columns
    cardinality = df[cat_cols].nunique().sort_values(ascending=False)
    print("Cardinality ranking:\n", cardinality)
    plt.figure(figsize=(10,6))
    sns.barplot(x=cardinality.values, y=cardinality.index)
    plt.title('Feature Cardinality')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cardinality.png'); plt.close()

def plot_geo_scatter(df):
    sub = df.dropna(subset=['LATITUDE','LONGITUDE'])
    plt.figure(figsize=(8,8))
    sns.scatterplot(x='LONGITUDE',y='LATITUDE',hue='LICENSE_STATUS',data=sub,alpha=0.5,s=10)
    plt.title('Geo Scatter'); plt.tight_layout(); plt.savefig(PLOTS_DIR/'geo_scatter.png'); plt.close()

def plot_temporal_trends(df):
    for col in ['APPLICATION_CREATED_DATE','PAYMENT_DATE','DATE_ISSUED']:
        daily = df[col].dt.date.value_counts().sort_index()
        plt.figure(figsize=(12,4)); daily.plot()
        plt.title(f'Daily count of {col}'); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'daily_{col}.png'); plt.close()

def contingency_table(df, col):
    table = pd.crosstab(df[col], df['LICENSE_STATUS'], normalize='index')
    print(table)
    return table
