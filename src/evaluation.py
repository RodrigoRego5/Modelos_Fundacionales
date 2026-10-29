import mlflow
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    precision_score
)


def evaluate_model(model, X_test, y_test, threshold=None):
    y_pred = model.predict(X_test) if threshold is None else None
    if not hasattr(model, "predict_proba"):
        raise TypeError("The model must implement predict_proba for these metrics.")
    y_prob = model.predict_proba(X_test)

    classes = list(model.classes_)
    if len(classes) != 2:
        raise ValueError(f"Binary classification expected; found classes {classes!r}.")
    positive_class = 1 if 1 in classes else classes[-1]
    y_prob = y_prob[:, classes.index(positive_class)]
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        y_pred = (y_prob >= threshold).astype(int)
    y_test_binary = (y_test == positive_class).astype(int)
    y_pred_binary = (y_pred == positive_class).astype(int)

    return binary_metrics(y_test_binary, y_pred_binary, y_prob)


def binary_metrics(y_test_binary, y_pred_binary, y_prob):
    """The same six metrics for ordinary runs, CV candidates and final test."""
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test_binary, y_prob
    )

    return {
        "recall": recall_score(y_test_binary, y_pred_binary, zero_division=0),
        "f1": f1_score(y_test_binary, y_pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(y_test_binary, y_prob),
        "pr_auc": auc(recall_curve, precision_curve),
        "balanced_accuracy": balanced_accuracy_score(y_test_binary, y_pred_binary),
        "precision": precision_score(y_test_binary, y_pred_binary, zero_division=0)
    }



def log_results(model_name, dataset, params, metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params or {})
        mlflow.set_tags({
            "model": model_name,
            "dataset": dataset
        })
        mlflow.log_metrics(metrics)
