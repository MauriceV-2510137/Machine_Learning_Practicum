"""Preprocessing: ColumnTransformer for a feature list; which columns are categorical is fixed domain knowledge."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES


def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    """Categorical columns get one-hot encoded, everything else gets standard-scaled."""
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURES]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
