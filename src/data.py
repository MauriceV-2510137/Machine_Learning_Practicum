"""
Data loading, cleaning, and structure derived from the actual loaded data (feature groups, class labels) rather than hardcoded.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    RANDOM_STATE,
    SEM1_MARKER,
    SEM2_MARKER,
    TARGET_COL,
    TEST_SIZE,
)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and do minimal, deterministic cleaning."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def infer_feature_groups(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Split columns into (enrollment, sem1, sem2)
    """
    cols = [c for c in df.columns if c != target_col]
    sem1 = [c for c in cols if SEM1_MARKER in c]
    sem2 = [c for c in cols if SEM2_MARKER in c]
    enrollment = [c for c in cols if c not in sem1 and c not in sem2]
    return enrollment, sem1, sem2


def get_classes(df: pd.DataFrame, target_col: str = TARGET_COL) -> list:
    """Class labels, derived from the data."""
    return sorted(df[target_col].unique().tolist())


def get_feature_target_split(
    df: pd.DataFrame, feature_cols: list, target_col: str = TARGET_COL
):
    """Select a feature subset and the target."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def split_train_test(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """
    Stratified train/test split
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)