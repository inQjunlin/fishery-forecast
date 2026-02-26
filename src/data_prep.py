import pandas as pd
from typing import Tuple

def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from CSV/TSV path."""
    return pd.read_csv(path)

def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a dataframe to CSV."""
    df.to_csv(path, index=False)
