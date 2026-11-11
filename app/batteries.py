"""Ejecutar baterías reproducibles y registrar sus resultados en MLflow."""

import argparse
import csv
import hashlib
import io
import json
import platform
from copy import deepcopy
from pathlib import Path

import mlflow
import yaml

from .config import _mapping, load_config, validate_config
from .conditions import apply_training_conditions, select_features
from .datasets import PROJECT_ROOT, load_data, prepare_data, resolve_data_path
from .models import MODEL_NAMES
from .optimization import search_folds
from .partitions import fingerprint, make_partitions, validate_train_size
from .pipeline import _versions, run_pipeline
from .tracking import start_run


def _seed(value, label):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{label} debe ser un entero en [0, 2**32).")


def load_battery_config(path):
    """Las rutas de datasets son relativas a PROJECT_ROOT, como las de datos."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"name", "datasets", "models", "split_seeds", "train_sizes", "tracking"}
    _mapping(config, "battery", required | {"sampling_seed", "model_seed", "model_params", "variants"}, required)
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("battery.name debe ser un texto no vacío.")
    for key in ("datasets", "models", "split_seeds", "train_sizes"):
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"battery.{key} debe ser una lista no vacía.")
        try:
            unique = {fingerprint(value) for value in values}
        except (TypeError, ValueError) as error:
            raise ValueError(f"battery.{key} contiene valores inválidos.") from error
        if len(unique) != len(values):
            raise ValueError(f"battery.{key} contiene duplicados.")
    if not all(isinstance(path, str) and path.strip() for path in config["datasets"]):
        raise ValueError("datasets debe contener rutas YAML no vacías.")
    if not all(isinstance(name, str) and name in MODEL_NAMES for name in config["models"]):
        raise ValueError(f"models debe contener nombres de {MODEL_NAMES}.")
    for seed in config["split_seeds"]:
        _seed(seed, "split_seeds")
    for key in ("sampling_seed", "model_seed"):
        config.setdefault(key, 42)
        _seed(config[key], key)
    for size in config["train_sizes"]:
        validate_train_size(size)
    _mapping(config["tracking"], "battery.tracking", {"experiment_name"}, {"experiment_name"})
    experiment = config["tracking"]["experiment_name"]
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("tracking.experiment_name debe ser un texto no vacío.")
    params = config.setdefault("model_params", {})
    _mapping(params, "battery.model_params", set(config["models"]))
    if not all(isinstance(value, dict) for value in params.values()):
        raise ValueError("Cada entrada de model_params debe ser un mapa.")
    variants = config.setdefault("variants", [{"name": "reference"}])
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants debe ser una lista no vacía.")
    names = set()
    for variant in variants:
        _mapping(variant, "variant", {"name", "preprocessing", "features", "perturbation", "optimization"}, {"name"})
        name = variant["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Cada variante necesita un nombre no vacío y único.")
        names.add(name)
        for key in ("preprocessing", "features", "perturbation"):
            if key in variant and not isinstance(variant[key], dict):
                raise ValueError(f"variant.{key} debe ser un mapa.")
    return config


def _source_fingerprint():
    paths = sorted((PROJECT_ROOT / "app").glob("*.py")) + [PROJECT_ROOT / "requirements.txt"]
    return fingerprint({str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths})


def build_plan(config_path):
    """Valida toda la matriz y materializa las filas antes de entrenar nada."""
    battery = load_battery_config(config_path)
    partitions, jobs, dataset_names, dataset_paths = {}, [], set(), set()
    for config_path in battery["datasets"]:
        path = (PROJECT_ROOT / config_path).resolve()
        config = load_config(path)
        data_path = resolve_data_path(config)
        if config["name"] in dataset_names or data_path in dataset_paths:
            raise ValueError("Los datasets deben tener nombres y archivos distintos en la batería.")
        dataset_names.add(config["name"])
        dataset_paths.add(data_path)
        dataset_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
        X, y = prepare_data(load_data(config), config)
        for split_seed in battery["split_seeds"]:
            split_config = deepcopy(config)
            split_config["split"]["random_state"] = split_seed
            shared = make_partitions(X, y, split_config, battery["train_sizes"],
                                     battery["sampling_seed"], dataset_sha256)
            for partition in shared:
                partition_id = partition["partition_id"]
                partitions[partition_id] = partition
                for variant in battery["variants"]:
                    for model_name in battery["models"]:
                        run_config = deepcopy(split_config)
                        for key in ("preprocessing", "features", "perturbation"):
                            if key in variant:
                                run_config.setdefault(key, {}).update(variant[key])
                        if "optimization" in variant:
                            run_config["optimization"] = deepcopy(variant["optimization"])
                        params = run_config.setdefault("models", {}).setdefault(model_name, {})
                        params.update(battery["model_params"].get(model_name, {}))
                        params["random_state"] = battery["model_seed"]
                        run_config["tracking"] = {
                            "experiment_name": battery["tracking"]["experiment_name"],
                            "run_name": (f"{battery['name']}__{config['name']}__{model_name}"
                                         f"__{variant['name']}__s{split_seed}__n{len(partition['train'])}"),
                        }
                        run_config = validate_config(run_config)
                        selected = select_features(X.loc[partition["train"]], run_config.get("features", {}))
                        labels = y.loc[partition["train"]]
                        apply_training_conditions(selected, labels, run_config.get("perturbation", {}))
                        if run_config.get("optimization") is not None:
                            search_folds(selected, labels, run_config, model_name)
                        job = {"config": run_config, "model_name": model_name,
                               "variant": variant["name"], "partition_id": partition_id}
                        job["job_id"] = fingerprint(job)
                        jobs.append(job)
    plan = {"schema_version": 1, "battery": battery, "partitions": partitions, "jobs": jobs,
            "source_sha256": _source_fingerprint(), "versions": _versions(),
            "platform": platform.platform()}
    plan["plan_id"] = fingerprint(plan)
    return plan


def _log_results(plan, records, run):
    """Guarda en el run padre una tabla y un resumen del progreso completo."""
    rows = []
    for job in plan["jobs"]:
        record = records.get(job["job_id"], {"status": "pending"})
        partition = plan["partitions"][job["partition_id"]]
        row = {"job_id": job["job_id"], "dataset": job["config"]["name"],
               "variant": job["variant"],
               "model": job["model_name"], "status": record["status"],
               "split_seed": partition["split_seed"],
               "sampling_seed": partition["sampling_seed"], "model_seed": plan["battery"]["model_seed"],
               "train_size_requested": partition["train_size_requested"],
               "n_train": len(partition["train"]), "n_train_pool": len(partition["train_pool"]),
               "n_train_selected": len(partition["train"]),
               "n_test": len(partition["test"]), "partition_id": job["partition_id"],
               "test_sha256": fingerprint(partition["test"])}
        if record["status"] == "completed":
            result = record["result"]
            row.update({"run_id": result["run_id"],
                        **{f"metric.{key}": value for key, value in result["metrics"].items()},
                        **result["timings"]})
            context = result.get("context", {})
            row.update({key: context[key] for key in ("n_train", "n_input_features", "n_features",
                        "train_positive_rate", "training_data_sha256") if key in context})
            if "optimization" in result:
                row["best_cv_score"] = result["optimization"]["best_cv_score"]
                row["best_params"] = json.dumps(result["optimization"]["best_params"], sort_keys=True)
        elif record["status"] == "failed":
            row["error"] = record["error"]
        rows.append(row)
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    mlflow.log_text(buffer.getvalue(), "results.csv")
    summary = {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
               "plan_id": plan["plan_id"], "total": len(rows),
               **{status: sum(row["status"] == status for row in rows)
                  for status in ("pending", "running", "completed", "failed", "interrupted")}}
    mlflow.log_dict(summary, "summary.json")
    return summary


def run_battery(config_path, *, tracking_uri=None):
    """Una invocación crea un run padre y ejecuciones hijas secuenciales.

    Validamos la matriz completa antes de entrenar. Cada job ajusta de nuevo
    modelo y preprocesador; un fallo se registra y permite continuar al resto.
    """
    plan = build_plan(config_path)
    battery = plan["battery"]
    parent_config = {"name": battery["name"], "tracking": {
        **battery["tracking"], "run_name": battery["name"]}}
    records = {}
    with start_run(parent_config, "battery", tracking_uri=tracking_uri,
                   tags={"run_kind": "battery", "battery": battery["name"],
                         "battery_id": plan["plan_id"]}) as parent:
        mlflow.log_dict(plan, "plan.json")
        summary = _log_results(plan, records, parent)
        print(f"Batería {battery['name']}: {len(plan['jobs'])} ejecuciones; "
              f"run padre de MLflow: {parent.info.run_id}", flush=True)
        for index, job in enumerate(plan["jobs"], start=1):
            record = records[job["job_id"]] = {"status": "running"}
            partition = plan["partitions"][job["partition_id"]]
            print(f"[{index}/{len(plan['jobs'])}] {job['config']['tracking']['run_name']}", flush=True)
            tags = {"run_kind": "training", "battery": battery["name"],
                    "battery_id": plan["plan_id"], "job_id": job["job_id"],
                    "partition_id": job["partition_id"], "variant": job["variant"],
                    "source_sha256": plan["source_sha256"],
                    "split_seed": partition["split_seed"], "sampling_seed": partition["sampling_seed"],
                    "model_seed": battery["model_seed"]}
            try:
                result = run_pipeline(job["config"], job["model_name"], partition=partition,
                                      run_tags=tags, tracking_uri=tracking_uri, nested=True)
                record["result"] = {key: value for key, value in result.items()
                                    if key not in ("model", "preprocessor")}
                record["status"] = "completed"
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"  Falló: {record['error']}", flush=True)
            except KeyboardInterrupt:
                record["status"] = "interrupted"
                mlflow.set_tag("battery_status", "interrupted")
                raise
            finally:
                summary = _log_results(plan, records, parent)
        mlflow.set_tag("battery_status", "completed_with_failures" if summary["failed"] else "completed")
    if summary["failed"]:
        mlflow.MlflowClient().set_terminated(parent.info.run_id, status="FAILED")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path, help="YAML de batería")
    args = parser.parse_args(argv)
    try:
        summary = run_battery(args.config)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
