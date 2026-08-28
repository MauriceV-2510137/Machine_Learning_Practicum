"""
Plots, saved to OUTPUT_DIR as PNG files.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_target_distribution(df, classes, output_dir):
    counts = df["Target"].value_counts().reindex(classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Target class distribution")
    ax.set_ylabel("Number of students")
    path = output_dir / "target_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_distributions_by_target(df, classes, output_dir, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for ax, col in zip(axes, columns):
        for cls in classes:
            subset = df.loc[df["Target"] == cls, col]
            ax.hist(subset, bins=20, alpha=0.5, label=cls)
        ax.set_title(col)
        ax.legend(fontsize=8)
    path = output_dir / "feature_distributions_by_target.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion_matrix(cm, classes, output_dir, filename="confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    path = output_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
