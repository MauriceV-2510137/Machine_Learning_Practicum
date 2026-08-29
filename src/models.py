"""
Pipeline builders: combine the shared preprocessor with a model.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def build_logistic_regression_pipeline(
    feature_cols: list, class_weight=None
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_cols)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight
                ),
            ),
        ]
    )


def build_random_forest_pipeline(feature_cols: list, class_weight=None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline(feature_cols: list, class_weight=None) -> Pipeline:
    # GradientBoostingClassifier has no class_weight parameter (unlike LogisticRegression/RandomForestClassifier)
    if class_weight is not None:
        raise NotImplementedError(
            "GradientBoostingClassifier has no class_weight parameter; "
            "use sample_weight at fit time instead."
        )
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_cols)),
            (
                "model",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )
