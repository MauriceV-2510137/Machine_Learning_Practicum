"""
Student Dropout Prediction — Machine Learning Project
========================================================
Dataset: UCI "Predict Students' Dropout and Academic Success"
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Author: Maurice Vandenheede
"""

import pandas as pd

#import numpy as np
 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
 
from pathlib import Path
 
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
 
CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "student_dropout_data.csv",     # repo layout: data/
    SCRIPT_DIR / "student_dropout_data.csv",                # same folder as script
]
 
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find student_dropout_data.csv. Place it next to this "
        "script or update CANDIDATE_PATHS."
    )
 
df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
 
# Clean up column names: strip stray whitespace/tabs picked up from the
# original file (e.g. "Daytime/evening attendance\t").
df.columns = df.columns.str.strip()
 
print(f"Loaded from: {DATA_PATH}")
 
# ============================================================
# SECTION 1 — First look at the data
# ============================================================
 
print("\n--- Shape ---")
print(df.shape)
 
print("\n--- Column dtypes ---")
print(df.dtypes)
 
print("\n--- Missing values (columns with >0 missing) ---")
missing = df.isna().sum()
missing = missing[missing > 0]
print(missing if len(missing) else "None")
 
print("\n--- Target class distribution ---")
print(df["Target"].value_counts())
print("\n--- Target class distribution (%) ---")
print((df["Target"].value_counts(normalize=True) * 100).round(1))