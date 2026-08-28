"""
Evaluation: compute metrics for a trained model.
"""

from sklearn.metrics import classification_report, confusion_matrix


def print_classification_report(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    print("\n--- Classification report ---")
    print(classification_report(y_test, y_pred, labels=classes))


def compute_confusion_matrix(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=classes)
