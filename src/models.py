"""
Pipeline builders: combine the shared preprocessor with a model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_logistic_regression_pipeline(feature_cols: list) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])