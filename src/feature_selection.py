# src/feature_selection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from config import PLOTS_DIR

def variance_threshold_selection(df, thresh=0.01):
    num = df.select_dtypes(include=[np.number])
    sel = VarianceThreshold(threshold=thresh).fit(num)
    cols = num.columns[sel.get_support()]
    print(f"Retained {len(cols)} of {num.shape[1]}")
    return df[cols]

def corr_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,cmap='coolwarm',center=0,square=True)
    plt.title('Feature Correlation')
    plt.tight_layout(); plt.savefig(PLOTS_DIR/'correlation.png'); plt.close()
