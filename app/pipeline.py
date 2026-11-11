"""Cargar, preparar, dividir, preprocesar, entrenar, evaluar y registrar."""

import argparse
import hashlib
import json
import platform
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
import yaml

from .config import load_config, validate_config
from .conditions import apply_training_conditions, select_features
from .datasets import load_data, prepare_data, resolve_data_path, split_data
from .evaluation import evaluate_predictions
from .models import MODEL_NAMES, build_model
from .optimization import search_hyperparameters, search_decision_threshold
from .partitions import apply_partition, fingerprint
from .preprocessing import build_preprocessing
from .tracking import log_run, start_run


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl",
                    "lightgbm", "tabpfn-client", "mlflow", "PyYAML", "xlrd", "openpyxl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def run_pipeline(config, model_name, *, partition=None, run_tags=None, tracking_uri=None, nested=False):
    """Acepta ruta YAML o dict; devuelve modelo, preprocesador y resultado común.

    El modelo recibe matrices transformadas. El preprocesador recibe predictores
    preparados con las reglas de app.datasets.prepare_data.
    """
    # VALIDACIÓN DEL YAML
    config = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Modelo desconocido: {model_name!r}")
    # NUEVA EJECUCIÓN EN MLFLOW
    with start_run(config, model_name, tags=run_tags, tracking_uri=tracking_uri, nested=nested) as run:
        started = perf_counter()
        timings = {}
        variant = (run_tags or {}).get("variant", "reference")
        mlflow.log_param("variant", variant)
        # Incluso si el entrenamiento falla queda registrada la configuración.
        mlflow.log_dict(config, "config.json")
        # CARGA DE DATOS
        stage = perf_counter()
        dataset_sha256 = hashlib.sha256(resolve_data_path(config).read_bytes()).hexdigest()
        frame = load_data(config)
        X, y = prepare_data(frame, config)
        timings["data_loading_seconds"] = perf_counter() - stage
        stage = perf_counter()
        if partition is None:
            X_train, X_test, y_train, y_test = split_data(X, y, config)
        else:
            X_train, X_test, y_train, y_test = apply_partition(
                X, y, config, partition, dataset_sha256
            )
        timings["partition_seconds"] = perf_counter() - stage
        split_rows = {"train": X_train.index.tolist(), "test": X_test.index.tolist()}
        # Se conserva el diseño incluso si falla el entrenamiento o el cliente remoto.
        mlflow.log_dict(partition if partition is not None else split_rows, "split.json")
        stage = perf_counter()
        X_train = select_features(X_train, config.get("features", {}))
        X_test = select_features(X_test, config.get("features", {}))
        timings["feature_selection_seconds"] = perf_counter() - stage
        n_train_selected = len(y_train)
        # Selección de hiperparámetros exclusivamente dentro del train externo.
        search = None
        fitted_config = deepcopy(config)
        timings["search_seconds"] = 0.0
        if config.get("optimization") is not None:
            stage = perf_counter()
            search = search_hyperparameters(
                X_train, y_train, config, model_name,
                on_candidate=lambda candidate: mlflow.log_dict(
                    candidate, f"search/candidate_{candidate['index']:03d}.json"
                ),
            )
            timings["search_seconds"] = perf_counter() - stage
            fitted_config.setdefault("models", {}).setdefault(model_name, {}).update(search["best_params"])
            mlflow.log_dict(search, "search_results.json")
            mlflow.log_metric("cv_selection_score", search["best_cv_score"])
        threshold_search = None
        timings["threshold_search_seconds"] = 0.0
        if config.get("threshold_optimization") is not None:
            stage = perf_counter()
            threshold_search = search_decision_threshold(X_train, y_train, fitted_config, model_name)
            timings["threshold_search_seconds"] = perf_counter() - stage
            mlflow.log_dict(threshold_search, "threshold_search.json")
            mlflow.log_param("decision_threshold", threshold_search["threshold"])
            mlflow.log_metric("threshold_oof_score", threshold_search["score_oof"])
            mlflow.set_tag("metrics_source", "test_tuned_threshold")
        stage = perf_counter()
        X_train, y_train, condition_context = apply_training_conditions(
            X_train, y_train, config.get("perturbation", {})
        )
        timings["training_conditions_seconds"] = perf_counter() - stage
        mlflow.log_dict(condition_context, "conditions.json")
        split_rows["train"] = X_train.index.tolist()
        # PREPROCESAMIENTO
        stage = perf_counter()
        preprocessor = build_preprocessing(config)
        train = preprocessor.fit_transform(X_train)
        timings["preprocessing_fit_transform_seconds"] = perf_counter() - stage
        stage = perf_counter()
        test = preprocessor.transform(X_test)
        timings["preprocessing_test_transform_seconds"] = perf_counter() - stage
        timings["preprocessing_seconds"] = (
            timings["preprocessing_fit_transform_seconds"]
            + timings["preprocessing_test_transform_seconds"]
        )
        # CREACIÓN MODELO, ENTRENAMIENTO Y EVALUACIÓN
        stage = perf_counter()
        estimator = build_model(model_name, fitted_config)
        timings["model_initialization_seconds"] = perf_counter() - stage
        stage = perf_counter()
        estimator.fit(train, y_train)
        timings["training_seconds"] = perf_counter() - stage
        stage = perf_counter()
        predictions = np.asarray(estimator.predict(test)).reshape(-1)
        timings["inference_predict_seconds"] = perf_counter() - stage
        stage = perf_counter()
        probabilities = estimator.predict_proba(test)[:, list(estimator.classes_).index(1)]
        timings["inference_predict_proba_seconds"] = perf_counter() - stage
        timings["inference_seconds"] = (
            timings["inference_predict_seconds"] + timings["inference_predict_proba_seconds"]
        )
        timings["inference_seconds_per_row"] = timings["inference_seconds"] / len(y_test)
        stage = perf_counter()
        default_threshold_metrics = None
        default_predictions = predictions.copy()
        if threshold_search is not None:
            default_threshold_metrics = evaluate_predictions(
                y_test, default_predictions, probabilities, config["evaluation"]["metrics"]
            )
            predictions = (probabilities >= threshold_search["threshold"]).astype(int)
        metrics = evaluate_predictions(y_test, predictions, probabilities, config["evaluation"]["metrics"])
        timings["evaluation_seconds"] = perf_counter() - stage
        # CONTEXTO DEL EXPERIMENTO
        split_hash = fingerprint(split_rows)
        n_train_pool = len(partition["train_pool"]) if partition is not None else n_train_selected
        context = {
            "dataset_path": str(resolve_data_path(config)),
            "dataset_sha256": dataset_sha256,
            "split_sha256": split_hash,
            "test_sha256": fingerprint(split_rows["test"]),
            "partition_id": partition["partition_id"] if partition is not None else split_hash,
            "split_seed": config["split"]["random_state"],
            "model_seed": estimator.get_params(deep=False).get("random_state"),
            "sampling_seed": partition["sampling_seed"] if partition is not None else None,
            "train_size_requested": partition["train_size_requested"] if partition is not None else 1.0,
            "n_train_pool": n_train_pool,
            "n_train_selected": n_train_selected,
            "n_input_features": X_train.shape[1],
            "training_data_sha256": condition_context["training_data_sha256"],
            "variant": variant,
            "features": config.get("features", {}),
            "perturbation": config.get("perturbation", {}),
            "train_fraction_actual": len(y_train) / n_train_pool,
            "n_rows_loaded": len(frame), "n_rows_dropped_target": len(frame) - len(y),
            "n_train": len(y_train), "n_test": len(y_test), "n_features": train.shape[1],
            "positive_class": 1, "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "feature_names": preprocessor.get_feature_names_out().tolist(),
            "versions": _versions(),
            "platform": platform.platform(),
            "inference_protocol": "predict_then_predict_proba; no_warmup; one_test_batch",
            "execution_context": dict(run_tags or {}),
        }
        # REGISTRO DEL RESULTADO EN MLFLOW
        timings["duration_seconds"] = perf_counter() - started
        result = {"dataset": config["name"], "model_name": model_name,
                  "run_id": run.info.run_id,
                  "experiment_id": run.info.experiment_id,
                  "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
                  "run_name": run.info.run_name, "metrics": metrics,
                  "timings": timings, "context": context}
        if search is not None:
            result["optimization"] = {key: value for key, value in search.items()
                                      if key not in ("candidates", "cv_splits")}
        if threshold_search is not None:
            result["threshold_optimization"] = {key: value for key, value in threshold_search.items()
                                                if key not in ("curve", "oof_predictions", "cv_splits")}
            result["default_threshold_metrics"] = default_threshold_metrics
        prediction_frame = pd.DataFrame({
            "row_index": X_test.index, "y_true": np.asarray(y_test),
            "y_pred": predictions, "probability_positive": probabilities,
        })
        if threshold_search is not None:
            prediction_frame["y_pred_default"] = default_predictions
        mlflow.log_text(prediction_frame.to_csv(index=False, lineterminator="\n"), "predictions.csv")
        log_run(config, result, estimator, model_name)
    return {**result, "model": estimator, "preprocessor": preprocessor}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="Ruta a la configuración YAML")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--output", type=Path, help="Exportar opcionalmente el resumen a un archivo JSON")
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(args.config, args.model)
    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.output:
        summary = {key: value for key, value in result.items() if key not in ("model", "preprocessor")}
        output = json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(f"Experimento: {result['experiment_name']}")
    print(f"Ejecucion: {result['run_name']}")
    print(f"Run ID: {result['run_id']}")
    print(f"Dataset: {result['dataset']} | Modelo: {result['model_name']}")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")
    print(f"Duracion: {result['timings']['duration_seconds']:.2f} s")
    for key in ("preprocessing_seconds", "training_seconds", "inference_seconds"):
        print(f"  {key}: {result['timings'][key]:.6f} s")
    if args.output:
        print(f"Resumen guardado en: {args.output}")


if __name__ == "__main__":
    main()
