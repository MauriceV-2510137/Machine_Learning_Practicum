"""Pipeline builders: combine the shared preprocessor with a model."""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def _build_pipeline(feature_cols: list, estimator) -> Pipeline:
    """Wrap the shared preprocessor and a given estimator into one Pipeline."""
    return Pipeline(
        [("preprocessor", build_preprocessor(feature_cols)), ("model", estimator)]
    )


def build_logistic_regression_pipeline(
    feature_cols: list, class_weight=None
) -> Pipeline:
    return _build_pipeline(
        feature_cols,
        LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight
        ),
    )


def build_decision_tree_pipeline(feature_cols: list, class_weight=None) -> Pipeline:
    return _build_pipeline(
        feature_cols,
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=class_weight),
    )


def build_random_forest_pipeline(
    feature_cols: list,
    class_weight=None,
    n_jobs=-1,
    max_features="sqrt",
    n_estimators=300,
) -> Pipeline:
    return _build_pipeline(
        feature_cols,
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            n_jobs=n_jobs,
            max_features=max_features,
        ),
    )


def build_bagging_pipeline(feature_cols: list, class_weight=None) -> Pipeline:
    """Bagging is Random Forest with max_features=None (m=p, every split considers all features)."""
    return _build_pipeline(
        feature_cols,
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            max_features=None,
            n_jobs=-1,
        ),
    )


def build_gradient_boosting_pipeline(feature_cols: list, class_weight=None) -> Pipeline:
    """GradientBoostingClassifier has no class_weight param -- balancing would need sample_weight at fit time instead."""
    if class_weight is not None:
        raise NotImplementedError(
            "GradientBoostingClassifier has no class_weight parameter."
        )
    return _build_pipeline(
        feature_cols, GradientBoostingClassifier(random_state=RANDOM_STATE)
    )
