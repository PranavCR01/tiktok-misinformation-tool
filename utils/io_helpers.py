#utils/ io_helpers.py

import os
import pandas as pd

def save_dataframe(df, path: str):
    df.to_csv(path, index=False)

def load_dataframe(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
