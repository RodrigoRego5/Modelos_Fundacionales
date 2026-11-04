"""Registro común de configuración, contexto, métricas y tiempos."""

import logging

import mlflow

from .datasets import PROJECT_ROOT

logger = logging.getLogger(__name__)


def start_run(config, model_name):
    tracking = config["tracking"]
    uri = "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    experiment_name = tracking["experiment_name"]
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        logger.warning("Restaurando el experimento eliminado %r (id=%s).",
                       experiment_name, experiment.experiment_id)
        client.restore_experiment(experiment.experiment_id)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
    tags = {"dataset": config["name"], "model": model_name,
            "pipeline": "common", "metrics_source": "test_default"}
    return mlflow.start_run(run_name=tracking["run_name"], tags=tags)

def log_run(config, result, estimator):
    mlflow.log_params({
        "dataset_path": result["context"]["dataset_path"],
        "target": config["dataset"]["target"],
        **{f"split.{key}": value for key, value in config["split"].items()},
        **{f"preprocessing.{key}": value for key, value in config["preprocessing"].items()},
        **{f"model.{key}": value for key, value in estimator.get_params(deep=False).items()},
        **{key: result["context"][key] for key in ("n_train", "n_test", "n_features")},
        "evaluation.metrics": ",".join(config["evaluation"]["metrics"]),
    })
    mlflow.log_metrics(result["metrics"])
    mlflow.log_metrics(result["timings"])
    mlflow.log_dict(result, "result.json")
