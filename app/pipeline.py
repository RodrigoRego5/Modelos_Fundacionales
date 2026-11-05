"""Cargar, preparar, dividir, preprocesar, entrenar, evaluar y registrar."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import mlflow
import yaml

from .config import load_config, validate_config
from .datasets import load_data, prepare_data, resolve_data_path, split_data
from .evaluation import evaluate
from .models import MODEL_NAMES, build_model
from .preprocessing import build_preprocessing
from .tracking import log_run, start_run


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "tabpfn-client", "mlflow", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def run_pipeline(config, model_name):
    """Acepta ruta YAML o dict; devuelve modelo, preprocesador y resultado común.

    El modelo recibe matrices transformadas. El preprocesador recibe predictores
    preparados con las reglas de app.datasets.prepare_data.
    """
    config = load_config(config) if isinstance(config, (str, Path)) else validate_config(config)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Modelo desconocido: {model_name!r}")
    with start_run(config, model_name) as run:
        started = perf_counter()
        # Incluso si el entrenamiento falla queda registrada la configuración.
        mlflow.log_dict(config, "config.json")
        frame = load_data(config)
        X, y = prepare_data(frame, config)
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        preprocessor = build_preprocessing(config)
        train = preprocessor.fit_transform(X_train)
        test = preprocessor.transform(X_test)
        estimator = build_model(model_name, config)
        estimator.fit(train, y_train)
        metrics = evaluate(estimator, test, y_test, config["evaluation"]["metrics"])
        split_rows = {"train": X_train.index.tolist(), "test": X_test.index.tolist()}
        split_hash = hashlib.sha256(json.dumps(split_rows, sort_keys=True).encode()).hexdigest()
        context = {
            "dataset_path": str(resolve_data_path(config)),
            "dataset_sha256": hashlib.sha256(resolve_data_path(config).read_bytes()).hexdigest(),
            "split_sha256": split_hash,
            "n_rows_loaded": len(frame), "n_rows_dropped_target": len(frame) - len(y),
            "n_train": len(y_train), "n_test": len(y_test), "n_features": train.shape[1],
            "positive_class": 1, "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "feature_names": preprocessor.get_feature_names_out().tolist(),
            "versions": _versions(),
        }
        # Incluye carga, preparación, entrenamiento, evaluación y contexto;
        # excluye el envío de artefactos a MLflow y la escritura de salida CLI.
        timings = {"duration_seconds": perf_counter() - started}
        result = {"dataset": config["name"], "model_name": model_name,
                  "run_id": run.info.run_id,
                  "experiment_id": run.info.experiment_id,
                  "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
                  "run_name": run.info.run_name, "metrics": metrics,
                  "timings": timings, "context": context}
        mlflow.log_dict(split_rows, "split.json")
        log_run(config, result, estimator)
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
    if args.output:
        print(f"Resumen guardado en: {args.output}")


if __name__ == "__main__":
    main()
