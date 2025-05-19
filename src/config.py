# src/config.py
import os
from pathlib import Path

# Paths
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
RAW_DATA_PATH = DATA_DIR / 'raw' / 'applications.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'applications_processed.csv'
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'models'))
PLOTS_DIR = Path(os.getenv('PLOTS_DIR', 'plots'))

# Reproducibility
RANDOM_SEED = 42

# Class weight
def get_class_weights(y):
    from collections import Counter
    counts = Counter(y)
    total = sum(counts.values())
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}

# Scheduler parameters
SCHEDULER_PARAMS = {'factor': 0.5, 'patience': 3, 'min_lr': 1e-5}
