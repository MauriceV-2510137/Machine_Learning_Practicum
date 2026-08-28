"""
Pipeline steps for the student dropout project. Each function does one
job; run_pipeline() calls them in order.
"""

import pandas as pd

from src.config import OUTPUT_DIR
from src.data import (
    get_classes,
    get_feature_target_split,
    infer_feature_groups,
    load_data,
    split_train_test,
)
from src.evaluate import compute_confusion_matrix, print_classification_report
from src.models import build_logistic_regression_pipeline
from src.plots import (
    plot_confusion_matrix,
    plot_feature_distributions_by_target,
    plot_target_distribution,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def load_and_summarize():
    """Load the data, print a quick overview, return df + derived structure."""
    df = load_data()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    enrollment_cols, sem1_cols, sem2_cols = infer_feature_groups(df)
    classes = get_classes(df)
    print(
        f"Feature groups -> enrollment: {len(enrollment_cols)}, sem1: {len(sem1_cols)}, sem2: {len(sem2_cols)}"
    )
    print(f"Classes: {classes}")

    print("\n--- Missing values (columns with >0 missing) ---")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing if len(missing) else "None")

    print("\n--- Target class distribution ---")
    counts = df["Target"].value_counts().reindex(classes)
    pct = (df["Target"].value_counts(normalize=True) * 100).round(1).reindex(classes)
    print(pd.concat([counts, pct], axis=1, keys=["count", "pct"]))

    return df, enrollment_cols, sem1_cols, sem2_cols, classes


def generate_plots(df, classes):
    """Save the EDA plots for this dataset."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_target_distribution(df, classes, OUTPUT_DIR)

    numeric_preview = [
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
    ]
    plot_feature_distributions_by_target(df, classes, OUTPUT_DIR, numeric_preview)


def prepare_early_warning_features(df, enrollment_cols, sem1_cols):
    """Build X, y for the primary (early-warning) feature set."""
    early_warning_features = enrollment_cols + sem1_cols
    X, y = get_feature_target_split(df, early_warning_features)
    print(f"\nEarly-warning feature matrix: {X.shape}")
    return X, y, early_warning_features


def train_baseline_model(X, y, feature_cols):
    """Train/test split + Logistic Regression baseline, accuracy as a sanity check."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_logistic_regression_pipeline(feature_cols)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nBaseline Logistic Regression — test accuracy: {accuracy:.3f}")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, classes, model_name):
    """Classification report + confusion matrix for a trained model."""
    print_classification_report(model, X_test, y_test, classes)
    cm = compute_confusion_matrix(model, X_test, y_test, classes)
    plot_confusion_matrix(
        cm, classes, OUTPUT_DIR, filename=f"confusion_matrix_{model_name}.png"
    )


def run_pipeline():
    df, enrollment_cols, sem1_cols, _, classes = load_and_summarize()
    generate_plots(df, classes)
    X, y, early_warning_features = prepare_early_warning_features(
        df, enrollment_cols, sem1_cols
    )
    model, X_test, y_test = train_baseline_model(X, y, early_warning_features)
    evaluate_model(
        model, X_test, y_test, classes, model_name="logistic_regression_baseline"
    )
    return model, X_test, y_test, classes
