"""Consolidation analyses reusing each model's tuned hyperparameters: full-year comparison, final summary."""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from src.config import OUTPUT_DIR
from src.data import get_feature_target_split, split_train_test
from src.evaluate import get_classification_report_dict
from src.experiments import (
    run_decision_tree,
    run_gradient_boosting,
    run_logistic_regression,
    run_random_forest,
)
from src.hyperparameter_store import load_hyperparameters
from src.models import (
    build_bagging_pipeline,
    build_decision_tree_pipeline,
    build_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
)
from src.plots import plot_grouped_bar_comparison
from src.training import train_model

# (display name, base pipeline builder, fallback tuning step (None if not tuned), needs sample_weight)
MODEL_REGISTRY = [
    (
        "Logistic Regression",
        build_logistic_regression_pipeline,
        run_logistic_regression,
        False,
    ),
    ("Decision Tree", build_decision_tree_pipeline, run_decision_tree, False),
    ("Bagging", build_bagging_pipeline, None, False),
    ("Random Forest", build_random_forest_pipeline, run_random_forest, False),
    (
        "Gradient Boosting",
        build_gradient_boosting_pipeline,
        run_gradient_boosting,
        True,
    ),
]


def ensure_hyperparameters(model_name, tuning_step, X, y, feature_cols, classes):
    """Stored hyperparameters for model_name, running its tuning step first (which then persists them) if missing."""
    stored = load_hyperparameters()
    if model_name not in stored:
        print(
            f"\nHyperparameters for {model_name} not yet computed — running its tuning step first..."
        )
        tuning_step(X, y, feature_cols, classes)
        stored = load_hyperparameters()
    return stored[model_name]


def _resolve_extra_params(model_name, tuning_step, X, y, feature_cols, classes):
    """extra_params dict ready for train_model, empty for untuned models (e.g. Bagging)."""
    if tuning_step is None:
        return {}
    params = ensure_hyperparameters(
        model_name, tuning_step, X, y, feature_cols, classes
    )
    return {f"model__{k}": v for k, v in params.items()}


def run_full_year_comparison(df, enrollment_cols, sem1_cols, sem2_cols, classes):
    """Refit every model's already-tuned config on early-warning vs full-year features to isolate the feature-set effect."""
    early_warning_features = enrollment_cols + sem1_cols
    full_year_features = enrollment_cols + sem1_cols + sem2_cols
    X_early, y_early = get_feature_target_split(df, early_warning_features)
    feature_sets = {
        "early_warning": early_warning_features,
        "full_year": full_year_features,
    }

    accuracies = {}  # {(model_name, fs_name, variant): accuracy}
    for model_name, builder, tuning_step, use_sample_weight in MODEL_REGISTRY:
        extra_params = _resolve_extra_params(
            model_name, tuning_step, X_early, y_early, early_warning_features, classes
        )

        for fs_name, feature_cols in feature_sets.items():
            X, y = get_feature_target_split(df, feature_cols)

            baseline_model, X_test, y_test = train_model(
                X,
                y,
                builder,
                feature_cols,
                class_weight=None,
                extra_params=extra_params,
                label=f"{model_name} ({fs_name}, baseline)",
            )
            accuracies[(model_name, fs_name, "baseline")] = baseline_model.score(
                X_test, y_test
            )

            if use_sample_weight:
                balanced_model, X_test, y_test = train_model(
                    X,
                    y,
                    builder,
                    feature_cols,
                    extra_params=extra_params,
                    balanced_sample_weight=True,
                    label=f"{model_name} ({fs_name}, balanced)",
                )
            else:
                balanced_model, X_test, y_test = train_model(
                    X,
                    y,
                    builder,
                    feature_cols,
                    class_weight="balanced",
                    extra_params=extra_params,
                    label=f"{model_name} ({fs_name}, balanced)",
                )
            accuracies[(model_name, fs_name, "balanced")] = balanced_model.score(
                X_test, y_test
            )

    print("\n--- Early-warning vs full-year (baseline variant, test accuracy) ---")
    model_names = [cfg[0] for cfg in MODEL_REGISTRY]
    early_scores = [
        accuracies[(name, "early_warning", "baseline")] for name in model_names
    ]
    full_scores = [accuracies[(name, "full_year", "baseline")] for name in model_names]
    for name, ew, fy in zip(model_names, early_scores, full_scores):
        print(
            f"{name:<20} early-warning: {ew:.3f}   full-year: {fy:.3f}   delta: {fy - ew:+.3f}"
        )

    plot_grouped_bar_comparison(
        model_names,
        early_scores,
        full_scores,
        label_a="Early-warning (30 features)",
        label_b="Full-year (36 features)",
        y_label="Test accuracy",
        title="Early-warning vs full-year features",
        output_dir=OUTPUT_DIR,
        filename="feature_set_comparison_accuracy.png",
    )

    return accuracies


def run_final_summary(df, enrollment_cols, sem1_cols, classes):
    """Consolidate every final (already-tuned) early-warning model into one CV + held-out test table and plot."""
    feature_cols = enrollment_cols + sem1_cols
    X, y = get_feature_target_split(df, feature_cols)
    X_train, _, y_train, _ = split_train_test(X, y)

    rows = []
    for model_name, builder, tuning_step, use_sample_weight in MODEL_REGISTRY:
        extra_params = _resolve_extra_params(
            model_name, tuning_step, X, y, feature_cols, classes
        )

        for variant, class_weight, sample_weight_flag in (
            ("baseline", None, False),
            ("balanced", None if use_sample_weight else "balanced", use_sample_weight),
        ):
            pipeline = builder(feature_cols, class_weight=class_weight)
            if extra_params:
                pipeline.set_params(**extra_params)

            if sample_weight_flag:
                weights = compute_sample_weight("balanced", y_train)
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                    params={"model__sample_weight": weights},
                )
            else:
                # scoring="accuracy" here (not "balanced_accuracy" like the tuning steps used) so
                # this is a fair like-for-like comparison against the held-out test accuracy below.
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
                )
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

            model, X_test_, y_test_ = train_model(
                X,
                y,
                builder,
                feature_cols,
                class_weight=class_weight,
                extra_params=extra_params,
                balanced_sample_weight=sample_weight_flag,
                label=f"{model_name} ({variant})",
            )
            report = get_classification_report_dict(model, X_test_, y_test_, classes)

            row = {
                "model": model_name,
                "variant": variant,
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
                "test_accuracy": model.score(X_test_, y_test_),
            }
            for cls in classes:
                row[f"{cls}_precision"] = report[cls]["precision"]
                row[f"{cls}_recall"] = report[cls]["recall"]
                row[f"{cls}_f1"] = report[cls]["f1-score"]
            rows.append(row)

    results_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "final_results_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print(
        "\n--- Final model summary (5-fold CV accuracy vs held-out test accuracy) ---"
    )
    for _, r in results_df.iterrows():
        print(
            f"{r['model']:<20} {r['variant']:<10} CV: {r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f}   Test acc: {r['test_accuracy']:.3f}"
        )

    labels = [f"{r['model']} ({r['variant']})" for _, r in results_df.iterrows()]
    cv_values = results_df["cv_accuracy_mean"].tolist()
    test_values = results_df["test_accuracy"].tolist()
    plot_grouped_bar_comparison(
        labels,
        cv_values,
        test_values,
        label_a="5-fold CV accuracy (train)",
        label_b="Held-out test accuracy",
        y_label="Accuracy",
        title="Final models: cross-validated vs held-out performance",
        output_dir=OUTPUT_DIR,
        filename="final_summary_comparison.png",
    )

    return results_df
