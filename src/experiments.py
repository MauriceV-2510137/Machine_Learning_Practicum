"""Per-model experiments: train (and tune, where applicable), evaluate, return fitted model(s)."""

import warnings
from functools import partial

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.config import OUTPUT_DIR
from src.data import split_train_test
from src.evaluate import compute_confusion_matrix, print_classification_report
from src.models import (
    build_bagging_pipeline,
    build_decision_tree_pipeline,
    build_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)
from src.plots import (
    plot_confusion_matrix,
    plot_cv_score_comparison,
    plot_top_coefficients,
    plot_validation_curve,
)
from src.tuning import (
    get_ccp_alpha_candidates,
    get_max_features_candidates,
    tune_hyperparameter,
)

# Trees used only during max_features tuning -- fewer than the final 300, since the ranking of
# candidate m values barely changes with tree count, and it keeps CV tuning fast.
TUNING_N_ESTIMATORS = 50


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
    """Cross-validated tuning of C for L1 vs L2 (via l1_ratio, solver=saga), then baseline + balanced fits + coefficients."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    C_values = np.logspace(-2, 2, num=5)
    cv_results = {}
    best_overall = None  # (mean_val_score, l1_ratio, C, label)
    for label, l1_ratio in (("L2", 0.0), ("L1", 1.0)):
        tuning_builder = partial(
            build_logistic_regression_pipeline, l1_ratio=l1_ratio, solver="saga"
        )
        print(
            f"\nLogistic Regression — tuning C for {label} via 5-fold CV ({len(C_values)} candidates)..."
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            best_C, _, val_scores = tune_hyperparameter(
                tuning_builder,
                feature_cols,
                X_train,
                y_train,
                param_name="C",
                param_range=C_values,
            )
        mean_val_score = val_scores.mean(axis=1).max()
        print(
            f"Logistic Regression — best C for {label} (CV balanced accuracy): {best_C:.4g} -> {mean_val_score:.3f}"
        )
        cv_results[label] = val_scores
        if best_overall is None or mean_val_score > best_overall[0]:
            best_overall = (mean_val_score, l1_ratio, best_C, label)

    plot_cv_score_comparison(
        C_values,
        cv_results,
        param_label="C",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_logistic_regression_C.png",
        log_x=True,
    )

    _, best_l1_ratio, best_C, best_label = best_overall
    print(f"Logistic Regression — overall best: {best_label}, C={best_C:.4g}")

    tuned_builder = partial(
        build_logistic_regression_pipeline, l1_ratio=best_l1_ratio, solver="saga"
    )
    tuned_params = {"model__C": best_C}
    label_suffix = f"{best_label}, C={best_C:.4g}"

    baseline_model, X_test, y_test = train_model(
        X,
        y,
        tuned_builder,
        feature_cols,
        class_weight=None,
        extra_params=tuned_params,
        label=f"Logistic Regression (tuned, {label_suffix}, baseline)",
    )
    evaluate_model(
        baseline_model,
        X_test,
        y_test,
        classes,
        model_name="logistic_regression_tuned_baseline",
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        tuned_builder,
        feature_cols,
        class_weight="balanced",
        extra_params=tuned_params,
        label=f"Logistic Regression (tuned, {label_suffix}, balanced)",
    )
    evaluate_model(
        balanced_model,
        X_test,
        y_test,
        classes,
        model_name="logistic_regression_tuned_balanced",
    )

    raw_names = baseline_model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [name.split("__", 1)[-1] for name in raw_names]
    coefficients = baseline_model.named_steps["model"].coef_
    plot_top_coefficients(
        feature_names,
        coefficients,
        classes,
        output_dir=OUTPUT_DIR,
        filename="coefficients_logistic_regression_tuned.png",
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


def run_random_forest(X, y, feature_cols, classes):
    """Cross-validated tuning of max_features (m), then n_estimators, then baseline + balanced fits."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    m_values = get_max_features_candidates(X_train, feature_cols)
    tuning_builder = partial(
        build_random_forest_pipeline, n_jobs=1, n_estimators=TUNING_N_ESTIMATORS
    )
    print(
        f"\nRandom Forest — tuning max_features (m) via 5-fold CV ({len(m_values)} candidates)..."
    )
    best_m, m_train_scores, m_val_scores = tune_hyperparameter(
        tuning_builder,
        feature_cols,
        X_train,
        y_train,
        param_name="max_features",
        param_range=m_values,
    )
    print(
        f"Random Forest — best max_features (CV balanced accuracy): {best_m} of {m_values[-1]}"
    )
    plot_validation_curve(
        m_values,
        m_train_scores,
        m_val_scores,
        param_label="max_features (m)",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_random_forest_max_features.png",
        log_x=True,
    )

    n_estimators_values = [50, 100, 200, 300]
    tuning_builder_fixed_m = partial(
        build_random_forest_pipeline, n_jobs=1, max_features=int(best_m)
    )
    print(
        f"Random Forest — checking n_estimators plateau via 5-fold CV ({len(n_estimators_values)} candidates, m={best_m})..."
    )
    _, n_train_scores, n_val_scores = tune_hyperparameter(
        tuning_builder_fixed_m,
        feature_cols,
        X_train,
        y_train,
        param_name="n_estimators",
        param_range=n_estimators_values,
    )
    plot_validation_curve(
        n_estimators_values,
        n_train_scores,
        n_val_scores,
        param_label="n_estimators",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_random_forest_n_estimators.png",
    )
    # Unlike max_features/ccp_alpha, more trees can't overfit (averaging only reduces variance),
    # so there's no real peak to chase here -- the curve just confirms the score has plateaued.
    # Keep n_estimators at the project's existing default rather than following CV noise.
    final_n_estimators = 300
    print(
        f"Random Forest — n_estimators kept at {final_n_estimators} (curve confirms plateau, more trees can't overfit)"
    )

    tuned_params = {
        "model__max_features": int(best_m),
        "model__n_estimators": final_n_estimators,
    }
    label_suffix = f"m={best_m}, n_estimators={final_n_estimators}"

    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_random_forest_pipeline,
        feature_cols,
        class_weight=None,
        extra_params=tuned_params,
        label=f"Random Forest (tuned, {label_suffix}, baseline)",
    )
    evaluate_model(
        baseline_model,
        X_test,
        y_test,
        classes,
        model_name="random_forest_tuned_baseline",
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        build_random_forest_pipeline,
        feature_cols,
        class_weight="balanced",
        extra_params=tuned_params,
        label=f"Random Forest (tuned, {label_suffix}, balanced)",
    )
    evaluate_model(
        balanced_model,
        X_test,
        y_test,
        classes,
        model_name="random_forest_tuned_balanced",
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


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


def run_bagging(X, y, feature_cols, classes):
    return run_baseline_and_balanced(
        build_bagging_pipeline,
        X,
        y,
        feature_cols,
        classes,
        "bagging",
        "Bagging",
    )


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
