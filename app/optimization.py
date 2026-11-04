"""Optional, leakage-free hyperparameter and decision-threshold search."""

from __future__ import annotations

from time import perf_counter

import mlflow
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_val_predict,
)

from .evaluation import binary_metrics, evaluate_model
from .preprocessing import build_dataset_pipeline


def optimize_threshold(y_true, probabilities, min_precision=0.30):
    """Maximize validation recall subject to a minimum validation precision."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    if not len(eligible):
        raise ValueError("No threshold reaches the requested validation precision.")
    # Highest precision breaks recall ties; then prefer the higher threshold.
    best = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return {
        "threshold": float(thresholds[best]),
        "recall_cv": float(recall[best]),
        "precision_cv": float(precision[best]),
    }


def optimize_experiment(
    *,
    dataset,
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    param_distributions,
    participant=None,
    n_iter=10,
    cv_folds=5,
    random_state=42,
    one_hot=True,
    min_precision=None,
):
    """Search on raw training data; evaluate the selected pipeline once on test.

    Obtain data with load_dataset(dataset=..., preprocess=False). MLflow stores
    one parent run and a nested child per sampled candidate. Test labels are
    consulted only after hyperparameters and threshold have been selected.
    """
    if n_iter < 1 or cv_folds < 2:
        raise ValueError("n_iter must be positive and cv_folds >= 2.")
    if min_precision is not None and not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Binary targets must use classes 0 and 1.")
    if np.bincount(np.asarray(y_train, dtype=int)).min() < cv_folds:
        raise ValueError("Each training class needs at least cv_folds samples.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipeline = build_dataset_pipeline(dataset, model, one_hot=one_hot)
    candidates = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
    if not candidates:
        raise ValueError("Parameter search space is empty.")
    history = []
    with mlflow.start_run(run_name=f"{model_name} · {dataset} · optimization") as parent:
        mlflow.set_tags({"model": model_name, "dataset": dataset, "stage": "optimization"})
        if participant:
            mlflow.set_tag("participant", participant)
        mlflow.log_params({
            "n_candidates": len(candidates),
            "cv_folds": cv_folds,
            "random_state": random_state,
            "one_hot": one_hot,
            "scoring": "average_precision",
            "n_train": len(y_train),
            "n_test": len(y_test),
        })
        if min_precision is not None:
            mlflow.log_param("min_precision", min_precision)

        best = None
        for index, params in enumerate(candidates, start=1):
            estimator = clone(pipeline).set_params(
                **{f"model__{name}": value for name, value in params.items()}
            )
            started = perf_counter()
            with mlflow.start_run(run_name=f"candidate_{index:03d}", nested=True) as child:
                mlflow.set_tags({"stage": "candidate", "dataset": dataset,
                                 "metrics_source": "cross_validation_mean"})
                mlflow.log_params(params)
                fold_metrics = []
                scores = []
                for train_indices, validation_indices in cv.split(X_train, y_train):
                    fitted = clone(estimator).fit(
                        X_train.iloc[train_indices], y_train.iloc[train_indices]
                    )
                    X_validation = X_train.iloc[validation_indices]
                    y_validation = y_train.iloc[validation_indices]
                    probabilities = fitted.predict_proba(X_validation)[:, 1]
                    predictions = fitted.predict(X_validation)
                    fold_metrics.append(binary_metrics(y_validation, predictions, probabilities))
                    scores.append(average_precision_score(y_validation, probabilities))
                elapsed = perf_counter() - started
                mean_score = float(np.mean(scores))
                # Standard names are means over validation folds, not test scores.
                mlflow.log_metrics({key: float(np.mean([fold[key] for fold in fold_metrics]))
                                    for key in fold_metrics[0]})
                mlflow.log_metrics({
                    "cv_average_precision_mean": mean_score,
                    "cv_average_precision_std": float(np.std(scores)),
                    "duration_seconds": elapsed,
                })
                result = {"index": index, "params": params, "mean": mean_score,
                          "std": float(np.std(scores)), "run_id": child.info.run_id}
                history.append(result)
                if best is None or mean_score > best["mean"]:
                    best = result

        selected = clone(pipeline).set_params(
            **{f"model__{name}": value for name, value in best["params"].items()}
        )
        threshold_result = None
        if min_precision is not None:
            # Each probability comes from a clone fitted without that row.
            oof = cross_val_predict(selected, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=1)[:, 1]
            threshold_result = optimize_threshold(y_train, oof, min_precision)
            mlflow.log_param("decision_threshold", threshold_result["threshold"])

        selected.fit(X_train, y_train)
        default_metrics = evaluate_model(selected, X_test, y_test)
        tuned_metrics = None
        if threshold_result is not None:
            tuned_metrics = evaluate_model(selected, X_test, y_test,
                                           threshold=threshold_result["threshold"])
        # Only the six final test metrics are shown on the parent run.
        # When tuned, these reflect the selected threshold.
        mlflow.log_metrics(tuned_metrics if tuned_metrics is not None else default_metrics)
        mlflow.set_tag("metrics_source", "test_tuned" if tuned_metrics is not None
                       else "test_default")
        mlflow.set_tag("best_child_run_id", best["run_id"])
        mlflow.log_dict({"candidates": history}, "search_results.json")
        mlflow.log_dict({"best_params": best["params"], "best_cv_average_precision": best["mean"],
                         "threshold": threshold_result, "test_default": default_metrics,
                         "test_tuned": tuned_metrics}, "selection.json")

    return {
        "model": selected, "best_params": best["params"],
        "best_cv_score": best["mean"], "threshold": threshold_result,
        "test_default": default_metrics, "test_tuned": tuned_metrics,
        "run_id": parent.info.run_id,
    }
