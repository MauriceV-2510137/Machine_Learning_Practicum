"""
Student Dropout Prediction — Machine Learning Project
========================================================
Dataset: UCI "Predict Students' Dropout and Academic Success"
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Author: Maurice Vandenheede
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR
from src.data import (
    get_classes,
    get_feature_target_split,
    infer_feature_groups,
    load_data,
)
from src.timer import timer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ============================================================
    # SECTION 1 — Load & first look
    # ============================================================
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

    # ============================================================
    # SECTION 2 — Target class balance
    # ============================================================
    print("\n--- Target class distribution ---")
    counts = df["Target"].value_counts().reindex(classes)
    pct = (df["Target"].value_counts(normalize=True) * 100).round(1).reindex(classes)
    print(pd.concat([counts, pct], axis=1, keys=["count", "pct"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Target class distribution")
    ax.set_ylabel("Number of students")
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'target_distribution.png'}")

    # ============================================================
    # SECTION 3 — A few enrollment-time features, split by target (do dropout students already look different before sem 1?)
    # ============================================================
    numeric_preview = [
        "Age at enrollment",
        "Admission grade",
        "Previous qualification (grade)",
    ]

    fig, axes = plt.subplots(
        1, len(numeric_preview), figsize=(4 * len(numeric_preview), 4)
    )
    for ax, col in zip(axes, numeric_preview):
        for cls in classes:
            subset = df.loc[df["Target"] == cls, col]
            ax.hist(subset, bins=20, alpha=0.5, label=cls)
        ax.set_title(col)
        ax.legend(fontsize=8)
    plt.savefig(
        OUTPUT_DIR / "feature_distributions_by_target.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'feature_distributions_by_target.png'}")

    # ============================================================
    # SECTION 4 — Prepare the primary (early-warning) feature set
    # ============================================================
    early_warning_features = enrollment_cols + sem1_cols
    X, _ = get_feature_target_split(df, early_warning_features)
    print(f"\nEarly-warning feature matrix: {X.shape}")


if __name__ == "__main__":
    with timer("dropout_project.py"):
        main()
