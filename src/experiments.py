"""Per-model tuning experiments: train, tune (where applicable), evaluate, persist best hyperparameters."""

import warnings
from functools import partial

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.config import OUTPUT_DIR
from src.data import split_train_test
from src.hyperparameter_store import save_hyperparameters
from src.models import (
    build_bagging_pipeline,
    build_decision_tree_pipeline,
    build_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)
from src.plots import (
    plot_cv_score_comparison,
    plot_feature_importances,
    plot_top_coefficients,
    plot_validation_curve,
)
from src.preprocessing import get_feature_names
from src.training import evaluate_model, train_model
from src.tuning import (
    get_ccp_alpha_candidates,
    get_max_features_candidates,
    tune_hyperparameter,
)

# Trees used only during max_features tuning -- fewer than the final 300, since the ranking of
# candidate m values barely changes with tree count, and it keeps CV tuning fast.
TUNING_N_ESTIMATORS = 50


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
    X_train, _, y_train, _ = split_train_test(X, y)

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
    save_hyperparameters(
        "Logistic Regression",
        {"l1_ratio": best_l1_ratio, "solver": "saga", "C": float(best_C)},
    )

    tuned_params = {
        "model__l1_ratio": best_l1_ratio,
        "model__solver": "saga",
        "model__C": float(best_C),
    }
    label_suffix = f"{best_label}, C={best_C:.4g}"

    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_logistic_regression_pipeline,
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
        build_logistic_regression_pipeline,
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

    feature_names = get_feature_names(baseline_model.named_steps["preprocessor"])
    coefficients = baseline_model.named_steps["model"].coef_
    plot_top_coefficients(
        feature_names,
        coefficients,
        classes,
        output_dir=OUTPUT_DIR,
        filename="coefficients_logistic_regression_tuned.png",
    )

    return {"baseline": baseline_model, "balanced": balanced_model}


def run_decision_tree(X, y, feature_cols, classes):
    """Cross-validated cost-complexity pruning, then fit and evaluate the final tree."""
    X_train, _, y_train, _ = split_train_test(X, y)

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
    save_hyperparameters("Decision Tree", {"ccp_alpha": float(best_alpha)})

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
        extra_params={"model__ccp_alpha": float(best_alpha)},
        label=f"Decision Tree (pruned, ccp_alpha={best_alpha:.5f})",
    )
    evaluate_model(model, X_test, y_test, classes, model_name="decision_tree_pruned")

    return {"pruned": model}


def run_bagging(X, y, feature_cols, classes):
    """Baseline + balanced Bagging (Random Forest with max_features=None) -- not tuned, no hyperparameters to persist."""
    return run_baseline_and_balanced(
        build_bagging_pipeline,
        X,
        y,
        feature_cols,
        classes,
        "bagging",
        "Bagging",
    )


def run_random_forest(X, y, feature_cols, classes):
    """Cross-validated tuning of max_features (m), then n_estimators, then baseline + balanced fits."""
    X_train, _, y_train, _ = split_train_test(X, y)

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
        f"\nRandom Forest — checking n_estimators plateau via 5-fold CV ({len(n_estimators_values)} candidates, m={best_m})..."
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
    save_hyperparameters(
        "Random Forest",
        {"max_features": int(best_m), "n_estimators": final_n_estimators},
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

    feature_names = get_feature_names(baseline_model.named_steps["preprocessor"])
    importances = baseline_model.named_steps["model"].feature_importances_
    plot_feature_importances(
        feature_names,
        importances,
        output_dir=OUTPUT_DIR,
        filename="feature_importances_random_forest.png",
        title="Random Forest — feature importances",
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


def run_gradient_boosting(X, y, feature_cols, classes):
    """Cross-validated tuning of max_depth, learning_rate, n_estimators (in that order), then baseline + balanced fits."""
    X_train, _, y_train, _ = split_train_test(X, y)

    max_depth_values = [1, 2, 3, 4, 5, 6]
    print(
        f"\nGradient Boosting — tuning max_depth via 5-fold CV ({len(max_depth_values)} candidates)..."
    )
    best_max_depth, md_train, md_val = tune_hyperparameter(
        build_gradient_boosting_pipeline,
        feature_cols,
        X_train,
        y_train,
        param_name="max_depth",
        param_range=max_depth_values,
    )
    print(
        f"Gradient Boosting — best max_depth (CV balanced accuracy): {best_max_depth}"
    )
    plot_validation_curve(
        max_depth_values,
        md_train,
        md_val,
        param_label="max_depth",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_gradient_boosting_max_depth.png",
    )

    learning_rate_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    tuning_builder_depth = partial(
        build_gradient_boosting_pipeline, max_depth=int(best_max_depth)
    )
    print(
        f"\nGradient Boosting — tuning learning_rate via 5-fold CV ({len(learning_rate_values)} candidates, max_depth={best_max_depth})..."
    )
    best_lr, lr_train, lr_val = tune_hyperparameter(
        tuning_builder_depth,
        feature_cols,
        X_train,
        y_train,
        param_name="learning_rate",
        param_range=learning_rate_values,
    )
    print(f"Gradient Boosting — best learning_rate (CV balanced accuracy): {best_lr}")
    plot_validation_curve(
        learning_rate_values,
        lr_train,
        lr_val,
        param_label="learning_rate",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_gradient_boosting_learning_rate.png",
        log_x=True,
    )

    n_estimators_values = [50, 100, 200, 300, 500]
    tuning_builder_depth_lr = partial(
        build_gradient_boosting_pipeline,
        max_depth=int(best_max_depth),
        learning_rate=float(best_lr),
    )
    print(
        f"\nGradient Boosting — tuning n_estimators via 5-fold CV ({len(n_estimators_values)} candidates, max_depth={best_max_depth}, learning_rate={best_lr})..."
    )
    best_n_estimators, n_train, n_val = tune_hyperparameter(
        tuning_builder_depth_lr,
        feature_cols,
        X_train,
        y_train,
        param_name="n_estimators",
        param_range=n_estimators_values,
    )
    print(
        f"Gradient Boosting — best n_estimators (CV balanced accuracy): {best_n_estimators}"
    )
    plot_validation_curve(
        n_estimators_values,
        n_train,
        n_val,
        param_label="n_estimators",
        output_dir=OUTPUT_DIR,
        filename="validation_curve_gradient_boosting_n_estimators.png",
    )
    # Unlike Random Forest, boosting CAN overfit with too many estimators (each new tree
    # fits the remaining residuals), so -- unlike the RF plateau check -- we use the
    # actual CV-selected value here rather than overriding it.
    save_hyperparameters(
        "Gradient Boosting",
        {
            "max_depth": int(best_max_depth),
            "learning_rate": float(best_lr),
            "n_estimators": int(best_n_estimators),
        },
    )

    tuned_params = {
        "model__max_depth": int(best_max_depth),
        "model__learning_rate": float(best_lr),
        "model__n_estimators": int(best_n_estimators),
    }
    label_suffix = f"max_depth={best_max_depth}, learning_rate={best_lr}, n_estimators={best_n_estimators}"

    baseline_model, X_test, y_test = train_model(
        X,
        y,
        build_gradient_boosting_pipeline,
        feature_cols,
        extra_params=tuned_params,
        label=f"Gradient Boosting (tuned, {label_suffix}, baseline)",
    )
    evaluate_model(
        baseline_model,
        X_test,
        y_test,
        classes,
        model_name="gradient_boosting_tuned_baseline",
    )

    feature_names = get_feature_names(baseline_model.named_steps["preprocessor"])
    importances = baseline_model.named_steps["model"].feature_importances_
    plot_feature_importances(
        feature_names,
        importances,
        output_dir=OUTPUT_DIR,
        filename="feature_importances_gradient_boosting.png",
        title="Gradient Boosting — feature importances",
    )

    balanced_model, X_test, y_test = train_model(
        X,
        y,
        build_gradient_boosting_pipeline,
        feature_cols,
        extra_params=tuned_params,
        balanced_sample_weight=True,
        label=f"Gradient Boosting (tuned, {label_suffix}, balanced via sample_weight)",
    )
    evaluate_model(
        balanced_model,
        X_test,
        y_test,
        classes,
        model_name="gradient_boosting_tuned_balanced",
    )

    return {"baseline": baseline_model, "balanced": balanced_model}
