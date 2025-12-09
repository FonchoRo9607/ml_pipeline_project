import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Carga dataset desde CSV."""
    return pd.read_csv(path)