# Student Dropout Prediction

Machine learning pipeline for the UCI dataset "Predict Students' Dropout and Academic
Success". See [Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
for the source.

3-class classification (Dropout / Enrolled / Graduate), trained mainly on enrollment-time
and first-semester data to simulate an early-warning setup.

This file covers how to set up and run the code. For the reasoning behind the modeling
choices, the results, and the discussion, see the report in `report/`.

## Requirements

- Python 3.14
- Packages pinned in `requirements.txt` (installed into a venv, see below)

Versions are pinned on purpose. Some results (mainly exact hyperparameter values found
during tuning) can shift slightly between library versions, so pinning keeps a run
reproducible.

## Setup

```
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt --break-system-packages
```

Place `student_dropout_data.csv` in `data/` if it is not already there.

## Quick start

```
python -m src.main
```

This runs everything: all 5 models, the early-warning vs full-year comparison, and the
final summary table. Takes a few minutes, see "Runtime" below.

## Project layout

```
data/                       dataset csv
src/
  config.py                 paths, random_state, categorical feature list
  data.py                   loading, feature groups, train/test split
  preprocessing.py          ColumnTransformer (scaling + one-hot)
  models.py                 pipeline builders, one per model type
  tuning.py                 cross-validated hyperparameter search
  hyperparameter_store.py   saves/loads best_hyperparameters.json
  training.py               shared train/evaluate helpers
  experiments.py            per-model tuning + evaluation (5 steps)
  analysis.py               full-year comparison, final summary
  plots.py                  every plot function
  evaluate.py               classification report / confusion matrix
  timer.py                  timing helper
  pipeline.py               wires everything together
  main.py                   entry point, command line args
outputs/                    generated on each run: plots, csv, json (not in git)
report/                     the actual report
.vscode/launch.json         run configs for VS Code
```

## Running the pipeline

The full pipeline has 3 parts: 5 model steps, a full-year comparison, and a final
summary. You can run any combination.

Everything:
```
python -m src.main
```

One model only:
```
python -m src.main --steps random_forest
```

A few models:
```
python -m src.main --steps decision_tree random_forest
```

Only the full-year comparison, no model steps:
```
python -m src.main --steps --full-year
```

Only the final summary:
```
python -m src.main --steps --summary
```

Model step names: `logistic_regression`, `decision_tree`, `bagging`, `random_forest`,
`gradient_boosting`.

### From VS Code

Open the Run and Debug panel, pick a configuration from the dropdown at the top, press
F5. There is one config per model plus "Run All", "Run: Full-Year Comparison" and
"Run: Final Summary". No terminal needed.

## Runtime

Actual numbers from one full run on a 16-core machine:

```
logistic_regression      12s
decision_tree              5s
bagging                    8s
random_forest             19s
gradient_boosting         82s
full_year_comparison      66s
final_summary             82s
total                  274s  (about 4.5 minutes)
```

Gradient Boosting and Final Summary are consistently the slowest. Both run a lot of
cross-validation folds and Gradient Boosting is the most expensive model type to fit.
Every run also writes `outputs/step_runtimes.png`, a plot with the exact breakdown for
that specific run, so you are not stuck with the numbers above.

Most of that time is cross-validated hyperparameter search, parallelized across CPU
cores (`n_jobs=-1`). On a machine with fewer cores this will take noticeably longer,
especially the Gradient Boosting step, since there is less parallelism to spread the
work over.

If a full run feels too slow, run steps one at a time from the VS Code dropdown instead
of everything at once. Each step is independent and reports its own timing.

## Outputs

Everything in `outputs/` is generated fresh by each run and is not tracked in git:

- one PNG per plot (EDA, validation curves, confusion matrices, feature importances,
  comparisons)
- `final_results_summary.csv`, the full results table
- `best_hyperparameters.json`, see below

## How hyperparameters are shared between steps

Each of the 4 tunable models (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting) saves its best hyperparameters to `outputs/best_hyperparameters.json`
every time it runs. Bagging is not tuned, so it has nothing to save.

The full-year comparison and final summary read this file instead of tuning again.
If a model is missing from the file (for example, on a clean checkout with no prior
run), that model's tuning step runs automatically first, and the result gets saved for
next time. Running an individual model step always recomputes and overwrites its own
entry, regardless of what was there before.

## A few things to know

- `random_state=42` everywhere a split or model needs one, for reproducibility.
- During Logistic Regression tuning you may see `ConvergenceWarning`. This is expected
  for the L1 search at some C values and does not affect the result.
- `.vscode/launch.json` has `"subProcess": false` on every config. Without it, VS
  Code's debugger tries to attach to every worker process spawned during parallel
  cross-validation, which causes crashes and instability. Ctrl+F5 (run without
  debugging) works fine either way and skips this issue entirely.
- On Windows, antivirus real-time scanning can occasionally slow down a run for no
  code-related reason. If a run is unusually slow, that is worth ruling out first.