"""Optional, leakage-free hyperparameter and decision-threshold search."""

from __future__ import annotations

from time import perf_counter
from copy import deepcopy
from math import prod

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
from .preprocessing import build_preprocessing
from .conditions import apply_training_conditions
from .evaluation import evaluate_predictions
from .models import build_model


def search_folds(X, y, config, model_name):
    """Valida el diseño de búsqueda sin instanciar modelos ni consultar test."""
    settings = config["optimization"]
    if model_name not in settings["param_distributions"]:
        raise ValueError(f"Falta el espacio de búsqueda para {model_name}.")
    if y.value_counts().min() < settings["cv_folds"]:
        raise ValueError("Cada clase de entrenamiento necesita al menos cv_folds ejemplos.")
    cv = StratifiedKFold(n_splits=settings["cv_folds"], shuffle=True,
                        random_state=settings["random_state"])
    return list(cv.split(X, y))


def search_hyperparameters(X, y, config, model_name, *, on_candidate=None):
    """Búsqueda acotada sobre train crudo; ningún argumento permite acceder a test.

    Cada fold ajusta su propia perturbación, imputación y codificación. El modelo
    final se ajusta después en run_pipeline, utilizando los parámetros elegidos.
    """
    settings = config["optimization"]
    folds = search_folds(X, y, config, model_name)
    space = settings["param_distributions"][model_name]
    n_candidates = min(settings["n_iter"], prod(len(values) for values in space.values()))
    candidates = ParameterSampler(space, n_iter=n_candidates, random_state=settings["random_state"])
    history = []
    for index, params in enumerate(candidates, start=1):
        started = perf_counter()
        candidate_config = deepcopy(config)
        candidate_config.setdefault("models", {}).setdefault(model_name, {}).update(params)
        scores = []
        for train_indices, validation_indices in folds:
            fold_X, fold_y, _ = apply_training_conditions(
                X.iloc[train_indices], y.iloc[train_indices], config.get("perturbation", {})
            )
            transformer = build_preprocessing(config)
            train = transformer.fit_transform(fold_X)
            validation = transformer.transform(X.iloc[validation_indices])
            estimator = build_model(model_name, candidate_config)
            estimator.fit(train, fold_y)
            predictions = estimator.predict(validation)
            probabilities = estimator.predict_proba(validation)[:, list(estimator.classes_).index(1)]
            scores.append(evaluate_predictions(y.iloc[validation_indices], predictions, probabilities,
                                              [settings["scoring"]])[settings["scoring"]])
        result = {"index": index, "params": params, "fold_scores": scores,
                  "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                  "duration_seconds": perf_counter() - started}
        history.append(result)
        if on_candidate is not None:
            on_candidate(result)
    # Empates: primer candidato del orden reproducible, sin consultar test.
    best = max(history, key=lambda candidate: candidate["mean"])
    return {"best_params": best["params"], "best_cv_score": best["mean"],
            "scoring": settings["scoring"], "n_candidates": len(history),
            "cv_folds": len(folds), "candidates": history,
            "cv_splits": [{"train": X.index[train].tolist(), "validation": X.index[valid].tolist()}
                          for train, valid in folds]}


def threshold_folds(X, y, config):
    """Particiones internas del train para elegir umbral; nunca recibe test."""
    settings = config["threshold_optimization"]
    if y.value_counts().min() < settings["cv_folds"]:
        raise ValueError("Cada clase necesita al menos cv_folds ejemplos para ajustar el umbral.")
    cv = StratifiedKFold(n_splits=settings["cv_folds"], shuffle=True,
                        random_state=settings["random_state"])
    return list(cv.split(X, y))


def search_decision_threshold(X, y, config, model_name):
    """Elige umbral con probabilidades out-of-fold y los hiperparámetros fijados.

    Reajusta preprocesamiento y condiciones dentro de cada fold. Las puntuaciones
    OOF se usan para seleccionar, no son una estimación independiente del rendimiento
    si los hiperparámetros se han seleccionado usando este mismo entrenamiento.
    """
    settings = config["threshold_optimization"]
    folds = threshold_folds(X, y, config)
    probabilities = np.empty(len(y), dtype=float)
    for train_indices, validation_indices in folds:
        fold_X, fold_y, _ = apply_training_conditions(
            X.iloc[train_indices], y.iloc[train_indices], config.get("perturbation", {})
        )
        transformer = build_preprocessing(config)
        train = transformer.fit_transform(fold_X)
        validation = transformer.transform(X.iloc[validation_indices])
        estimator = build_model(model_name, config)
        estimator.fit(train, fold_y)
        probabilities[validation_indices] = estimator.predict_proba(validation)[
            :, list(estimator.classes_).index(1)
        ]
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    scores = f1 if settings["scoring"] == "f1" else recall
    minimum = settings.get("min_precision")
    eligible = np.arange(len(thresholds)) if minimum is None else np.flatnonzero(precision >= minimum)
    if not len(eligible):
        raise ValueError("Ningún umbral OOF alcanza min_precision; no se ha consultado el test.")
    # Empates: mayor precisión y, finalmente, mayor umbral.
    best = max(eligible, key=lambda i: (scores[i], precision[i], thresholds[i]))
    return {
        "threshold": float(thresholds[best]), "scoring": settings["scoring"],
        "score_oof": float(scores[best]), "f1_oof": float(f1[best]),
        "precision_oof": float(precision[best]), "recall_oof": float(recall[best]),
        "cv_folds": len(folds), "random_state": settings["random_state"],
        "min_precision": minimum,
        "model_params": estimator.get_params(deep=False),
        "curve": {"thresholds": thresholds.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f1": f1.tolist()},
        "oof_predictions": {"row_index": X.index.tolist(), "y_true": y.tolist(),
                            "probability_positive": probabilities.tolist()},
        "cv_splits": [{"train": X.index[train].tolist(), "validation": X.index[valid].tolist()}
                      for train, valid in folds],
    }


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
