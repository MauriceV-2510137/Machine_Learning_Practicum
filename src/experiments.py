"""Per-model experiments: train (and tune, where applicable), evaluate, return fitted model(s)."""

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
    """Train/test split + fit; extra_params (e.g. {"model__ccp_alpha": 0.001}) applied via set_params before fit."""
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


def run_baseline_and_balanced(
    pipeline_builder, X, y, feature_cols, classes, model_name, label
):
    """Train + evaluate a baseline and a class_weight="balanced" variant of the same model."""
    baseline_model, X_test, y_test = train_model(
        X,
        y,
        pipeline_builder,
        feature_cols,
        class_weight=None,
        label=f"{label} (baseline)",
    )
    evaluate_model(
        baseline_model, X_test, y_test, classes, model_name=f"{model_name}_baseline"
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        pipeline_builder,
        feature_cols,
        class_weight="balanced",
        label=f"{label} (class_weight=balanced)",
    )
    evaluate_model(
        balanced_model, X_test, y_test, classes, model_name=f"{model_name}_balanced"
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


def run_logistic_regression(X, y, feature_cols, classes):
    return run_baseline_and_balanced(
        build_logistic_regression_pipeline,
        X,
        y,
        feature_cols,
        classes,
        "logistic_regression",
        "Logistic Regression",
    )


def run_random_forest(X, y, feature_cols, classes):
    return run_baseline_and_balanced(
        build_random_forest_pipeline,
        X,
        y,
        feature_cols,
        classes,
        "random_forest",
        "Random Forest",
    )


def run_decision_tree(X, y, feature_cols, classes):
    """Cross-validated cost-complexity pruning, then fit and evaluate the final tree."""
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


def run_gradient_boosting(X, y, feature_cols, classes):
    """Baseline only -- GradientBoostingClassifier has no class_weight support (see models.py)."""
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
