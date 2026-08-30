"""Plots, saved to OUTPUT_DIR as PNG files."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import PROJECT_ROOT


def _save_figure(fig, output_dir, filename):
    """Save a figure, close it, and print a project-relative path (not the full absolute path)."""
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = path
    print(f"Saved: {rel_path}")


def plot_target_distribution(df, classes, output_dir):
    counts = df["Target"].value_counts().reindex(classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Target class distribution")
    ax.set_ylabel("Number of students")
    _save_figure(fig, output_dir, "target_distribution.png")


def plot_feature_distributions_by_target(df, classes, output_dir, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for ax, col in zip(axes, columns, strict=True):
        for cls in classes:
            subset = df.loc[df["Target"] == cls, col]
            ax.hist(subset, bins=20, alpha=0.5, label=cls)
        ax.set_title(col)
        ax.legend(fontsize=8)
    _save_figure(fig, output_dir, "feature_distributions_by_target.png")


def plot_confusion_matrix(cm, classes, output_dir, filename="confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    _save_figure(fig, output_dir, filename)


def plot_validation_curve(
    param_range,
    train_scores,
    val_scores,
    param_label,
    output_dir,
    filename,
    log_x=False,
):
    """Train vs cross-validation score across a hyperparameter range (mean +/- std across folds)."""
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(param_range, train_mean, label="Train", color="#1f77b4")
    ax.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="#1f77b4",
    )
    ax.plot(param_range, val_mean, label="Cross-validation", color="#ff7f0e")
    ax.fill_between(
        param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color="#ff7f0e"
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(param_label)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Validation curve: {param_label}")
    ax.legend()
    _save_figure(fig, output_dir, filename)


def plot_cv_score_comparison(
    param_range, results, param_label, output_dir, filename, log_x=False
):
    """Cross-validation score vs a hyperparameter for multiple named variants, overlaid (results: {label: val_scores})."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, val_scores in results.items():
        mean = val_scores.mean(axis=1)
        std = val_scores.std(axis=1)
        ax.plot(param_range, mean, label=label)
        ax.fill_between(param_range, mean - std, mean + std, alpha=0.2)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(param_label)
    ax.set_ylabel("Balanced accuracy (cross-validation)")
    ax.set_title(f"{param_label} — variant comparison")
    ax.legend()
    _save_figure(fig, output_dir, filename)


def plot_top_coefficients(
    feature_names, coefficients, classes, output_dir, filename, top_n=12
):
    """Horizontal bar chart of the top-|coefficient| features per class, colored by sign."""
    fig, axes = plt.subplots(
        1, len(classes), figsize=(6 * len(classes), 5), gridspec_kw={"wspace": 0.7}
    )
    if len(classes) == 1:
        axes = [axes]
    for ax, cls, coefs in zip(axes, classes, coefficients, strict=True):
        order = np.argsort(np.abs(coefs))[-top_n:]
        names = [feature_names[i] for i in order]
        values = coefs[order]
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cls)
        ax.set_xlabel("Coefficient")
        ax.tick_params(axis="y", labelsize=8)
    _save_figure(fig, output_dir, filename)


def plot_feature_importances(
    feature_names,
    importances,
    output_dir,
    filename,
    top_n=15,
    title="Feature importances",
):
    """Horizontal bar chart of the top-N features by importance (e.g. tree-based .feature_importances_)."""
    order = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in order]
    values = importances[order]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, values, color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    _save_figure(fig, output_dir, filename)


def plot_multi_bar_comparison(categories, series, y_label, title, output_dir, filename):
    """Grouped bar chart comparing any number of named series across categories (series: {label: values})."""
    x = np.arange(len(categories))
    n_series = len(series)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(2.2 * len(categories) + 2, 5))
    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    _save_figure(fig, output_dir, filename)


def plot_step_runtimes(step_times, output_dir, filename):
    """Bar chart of how long each pipeline step took in this run (wall-clock seconds)."""
    names = [name.replace("_", " ").title() for name in step_times]
    times = list(step_times.values())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2, 5))
    ax.bar(x, times, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime per pipeline step (this run)")
    _save_figure(fig, output_dir, filename)
