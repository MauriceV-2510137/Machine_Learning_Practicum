"""
Cross-validated hyperparameter tuning. Everything here runs on the training
split only -- the test set is never touched during tuning, so hyperparameter
choices can't leak test-set information into the final evaluation.
"""

from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def tune_hyperparameter(
    pipeline_builder,
    feature_cols,
    X_train,
    y_train,
    param_name,
    param_range,
    class_weight=None,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1,
):
    """
    Cross-validated validation curve for one model hyperparameter.
    Returns (best_value, train_scores, val_scores); scores have shape
    (len(param_range), cv). n_jobs=-1 parallelizes fold x candidate fits
    across all CPU cores; verbose=1 prints progress via joblib.
    """
    pipeline = pipeline_builder(feature_cols, class_weight=class_weight)
    train_scores, val_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name=f"model__{param_name}",
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    best_idx = val_scores.mean(axis=1).argmax()
    best_value = param_range[best_idx]
    return best_value, train_scores, val_scores


def get_ccp_alpha_candidates(
    X_train, y_train, feature_cols, class_weight=None, max_candidates=40
):
    """
    Candidate ccp_alpha values, derived from the training data's own
    cost-complexity pruning path (not a guessed range). The path can contain
    hundreds of breakpoints, each expensive to cross-validate (a full tree
    refit per fold), so we take an even stride through the real sequence to
    cap how many get tested -- a computational trade-off, not a hardcoded
    alpha range.
    """
    preprocessor = build_preprocessor(feature_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=class_weight)
    path = tree.cost_complexity_pruning_path(X_train_transformed, y_train)
    alphas = path.ccp_alphas

    stride = max(1, len(alphas) // max_candidates)
    return alphas[::stride]
