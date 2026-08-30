"""Shared train/evaluate helpers, used by both per-model tuning (experiments.py) and consolidation (analysis.py)."""

from sklearn.utils.class_weight import compute_sample_weight

from src.config import OUTPUT_DIR
from src.data import split_train_test
from src.evaluate import compute_confusion_matrix, print_classification_report
from src.plots import plot_confusion_matrix


def train_model(
    X, y, pipeline_builder, feature_cols, class_weight=None, extra_params=None,
    balanced_sample_weight=False, label="Model",
):
    """Train/test split + fit; extra_params applied via set_params, balanced_sample_weight for models without class_weight."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = pipeline_builder(feature_cols, class_weight=class_weight)
    if extra_params:
        model.set_params(**extra_params)

    fit_kwargs = {}
    if balanced_sample_weight:
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, **fit_kwargs)

    accuracy = model.score(X_test, y_test)
    print(f"\n{label} — test accuracy: {accuracy:.3f}")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, classes, model_name):
    """Classification report + confusion matrix for a trained model."""
    print_classification_report(model, X_test, y_test, classes)
    cm = compute_confusion_matrix(model, X_test, y_test, classes)
    plot_confusion_matrix(cm, classes, OUTPUT_DIR, filename=f"confusion_matrix_{model_name}.png")