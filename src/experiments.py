"""
Per-model experiments. Each function trains (and tunes, where applicable)
one model family, evaluates it, and returns its fitted model(s) as a dict.

Kept separate from pipeline.py so individual models can be worked on and
selectively re-run without touching the orchestration logic.
"""

from src.config import OUTPUT_DIR
from src.data import split_train_test
from src.evaluate import compute_confusion_matrix, print_classification_report
from src.models import (
    build_decision_tree_pipeline,
    build_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)
from src.plots import plot_confusion_matrix, plot_validation_curve
from src.tuning import get_ccp_alpha_candidates, tune_hyperparameter


def train_model(
    X,
    y,
    pipeline_builder,
    feature_cols,
    class_weight=None,
    extra_params=None,
    label="Model",
):
    """Train/test split + fit a pipeline built by pipeline_builder, accuracy as a sanity check.
    extra_params (e.g. {"model__ccp_alpha": 0.001}) are applied via set_params
    before fitting -- used to plug in a cross-validated hyperparameter choice."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = pipeline_builder(feature_cols, class_weight=class_weight)
    if extra_params:
        model.set_params(**extra_params)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\n{label} — test accuracy: {accuracy:.3f}")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, classes, model_name):
    """Classification report + confusion matrix for a trained model."""
    print_classification_report(model, X_test, y_test, classes)
    cm = compute_confusion_matrix(model, X_test, y_test, classes)
    plot_confusion_matrix(
        cm, classes, OUTPUT_DIR, filename=f"confusion_matrix_{model_name}.png"
    )


def run_logistic_regression(X, y, feature_cols, classes):
    """Baseline + class-balanced Logistic Regression."""
    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_logistic_regression_pipeline,
        feature_cols,
        class_weight=None,
        label="Logistic Regression (baseline)",
    )
    evaluate_model(
        baseline_model,
        X_test,
        y_test,
        classes,
        model_name="logistic_regression_baseline",
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        build_logistic_regression_pipeline,
        feature_cols,
        class_weight="balanced",
        label="Logistic Regression (class_weight=balanced)",
    )
    evaluate_model(
        balanced_model,
        X_test,
        y_test,
        classes,
        model_name="logistic_regression_balanced",
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


def run_decision_tree(X, y, feature_cols, classes):
    """Cost-complexity pruning: derive candidate ccp_alpha values from the
    training data's own pruning path, pick the best via cross-validation,
    then fit and evaluate the final tree."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    alphas = get_ccp_alpha_candidates(X_train, y_train, feature_cols)
    print(
        f"\nDecision Tree — tuning ccp_alpha via 5-fold CV ({len(alphas)} candidates)..."
    )
    best_alpha, train_scores, val_scores = tune_hyperparameter(
        build_decision_tree_pipeline,
        feature_cols,
        X_train,
        y_train,
        param_name="ccp_alpha",
        param_range=alphas,
    )
    print(f"Decision Tree — best ccp_alpha (CV balanced accuracy): {best_alpha:.5f}")

    plot_validation_curve(
        alphas,
        train_scores,
        val_scores,
        param_label="ccp_alpha",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_decision_tree_ccp_alpha.png",
    )

    model, X_test, y_test = train_model(
        X,
        y,
        build_decision_tree_pipeline,
        feature_cols,
        extra_params={"model__ccp_alpha": best_alpha},
        label=f"Decision Tree (pruned, ccp_alpha={best_alpha:.5f})",
    )
    evaluate_model(model, X_test, y_test, classes, model_name="decision_tree_pruned")

    return {"pruned": model}


def run_random_forest(X, y, feature_cols, classes):
    """Baseline + class-balanced Random Forest."""
    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_random_forest_pipeline,
        feature_cols,
        class_weight=None,
        label="Random Forest (baseline)",
    )
    evaluate_model(
        baseline_model, X_test, y_test, classes, model_name="random_forest_baseline"
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        build_random_forest_pipeline,
        feature_cols,
        class_weight="balanced",
        label="Random Forest (class_weight=balanced)",
    )
    evaluate_model(
        balanced_model, X_test, y_test, classes, model_name="random_forest_balanced"
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


def run_gradient_boosting(X, y, feature_cols, classes):
    """Baseline Gradient Boosting (no class_weight support -- see models.py)."""
    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_gradient_boosting_pipeline,
        feature_cols,
        class_weight=None,
        label="Gradient Boosting (baseline)",
    )
    evaluate_model(
        baseline_model, X_test, y_test, classes, model_name="gradient_boosting_baseline"
    )

    return {"baseline": baseline_model}
