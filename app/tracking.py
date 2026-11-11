"""Registro común de configuración, contexto, métricas y tiempos."""

import logging

import mlflow

from .datasets import PROJECT_ROOT

logger = logging.getLogger(__name__)


def start_run(config, model_name, *, tags=None, tracking_uri=None, nested=False):
    tracking = config["tracking"]
    uri = tracking_uri or "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
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
    run_tags = {"dataset": config["name"], "model": model_name,
                "pipeline": "common", "metrics_source": "test_default", **(tags or {})}
    return mlflow.start_run(run_name=tracking["run_name"], tags=run_tags, nested=nested)

def log_run(config, result, estimator, model_name):
    mlflow.log_params({
        # MLflow's Box Plot can group by parameters, but not by run tags.
        "model": model_name,
        "dataset_path": result["context"]["dataset_path"],
        "target": config["dataset"]["target"],
        **{f"split.{key}": value for key, value in config["split"].items()},
        **{f"preprocessing.{key}": value for key, value in config["preprocessing"].items()},
        **{f"features.{key}": value for key, value in config.get("features", {}).items()},
        **{f"perturbation.{key}": value for key, value in config.get("perturbation", {}).items()},
        **{f"optimization.{key}": value for key, value in (config.get("optimization") or {}).items()
           if key != "param_distributions"},
        **{f"threshold_optimization.{key}": value
           for key, value in (config.get("threshold_optimization") or {}).items()},
        **{f"model.{key}": value for key, value in estimator.get_params(deep=False).items()},
        **{key: result["context"][key] for key in ("n_train", "n_train_selected", "n_test", "n_input_features", "n_features",
                                                 "train_positive_rate", "train_size_requested")},
        "evaluation.metrics": ",".join(config["evaluation"]["metrics"]),
        **{key: result["context"][key] for key in
           ("split_seed", "model_seed", "sampling_seed", "n_train_pool", "train_fraction_actual")},
    })
    mlflow.log_metrics(result["metrics"])
    if "default_threshold_metrics" in result:
        mlflow.log_metrics({f"default_threshold_{key}": value
                            for key, value in result["default_threshold_metrics"].items()})
    mlflow.log_metrics(result["timings"])
    mlflow.log_dict(result, "result.json")
