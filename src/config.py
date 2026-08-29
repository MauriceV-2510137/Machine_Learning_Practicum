"""Project-wide configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_dropout_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Target"

# Marks which columns belong to which semester; data.py uses this to infer feature groups.
SEM1_MARKER = "1st sem"
SEM2_MARKER = "2nd sem"

# Categorical codes stored as int in the CSV (indistinguishable from numeric by dtype) -- domain knowledge from UCI docs.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
