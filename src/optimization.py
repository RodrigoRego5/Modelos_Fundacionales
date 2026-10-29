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
    cross_val_score,
)

from .evaluation import evaluate_model
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
                mlflow.set_tags({"stage": "candidate", "dataset": dataset})
                mlflow.log_params(params)
                scores = cross_val_score(
                    estimator, X_train, y_train, cv=cv,
                    scoring="average_precision", n_jobs=1, error_score="raise",
                )
                elapsed = perf_counter() - started
                mean_score = float(np.mean(scores))
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
            mlflow.log_metric("cv_average_precision_oof", float(average_precision_score(y_train, oof)))
            mlflow.log_metrics(threshold_result)

        selected.fit(X_train, y_train)
        default_metrics = evaluate_model(selected, X_test, y_test)
        mlflow.log_metrics({f"test_default_{key}": value for key, value in default_metrics.items()})
        tuned_metrics = None
        if threshold_result is not None:
            tuned_metrics = evaluate_model(selected, X_test, y_test,
                                           threshold=threshold_result["threshold"])
            mlflow.log_metrics({f"test_tuned_{key}": value for key, value in tuned_metrics.items()})
        mlflow.log_metric("best_cv_average_precision", best["mean"])
        mlflow.set_tag("best_child_run_id", best["run_id"])
        mlflow.log_dict({"candidates": history}, "search_results.json")
        mlflow.log_dict({"best_params": best["params"], "threshold": threshold_result},
                        "selection.json")

    return {
        "model": selected, "best_params": best["params"],
        "best_cv_score": best["mean"], "threshold": threshold_result,
        "test_default": default_metrics, "test_tuned": tuned_metrics,
        "run_id": parent.info.run_id,
    }
