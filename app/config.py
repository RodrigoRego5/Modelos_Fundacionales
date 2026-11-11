"""Carga segura de YAML y validación previa a datos, modelos o tracking."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

from .evaluation import METRIC_NAMES
from .models import MODEL_NAMES


def _mapping(value, label, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} debe ser un mapa.")
    if set(value).difference(allowed) or set(required).difference(value):
        raise ValueError(f"Claves inválidas en {label}; permitidas: {sorted(allowed)}; requeridas: {sorted(required)}")


def validate_config(config):
    config = deepcopy(config)
    required = {"name", "dataset", "preprocessing", "split", "evaluation", "tracking"}
    _mapping(config, "config", required | {"models", "features", "perturbation", "optimization",
                                           "threshold_optimization"}, required)
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("name debe ser un texto no vacío.")
    dataset = config["dataset"]
    _mapping(dataset, "dataset", {"path", "target", "drop_columns", "numeric_columns"}, {"path", "target"})
    for key in ("path", "target"):
        if not isinstance(dataset[key], str) or not dataset[key].strip():
            raise ValueError(f"dataset.{key} debe ser un texto no vacío.")
    for key in ("drop_columns", "numeric_columns"):
        columns = dataset.get(key, [])
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns) or len(set(columns)) != len(columns):
            raise ValueError(f"dataset.{key} debe ser una lista de nombres únicos.")
        if dataset["target"] in columns:
            raise ValueError(f"El objetivo no puede estar en dataset.{key}.")
    if set(dataset.get("numeric_columns", [])) & set(dataset.get("drop_columns", [])):
        raise ValueError("Una columna no puede convertirse y eliminarse a la vez.")
    prep = config["preprocessing"]
    _mapping(prep, "preprocessing", {"numerical_missing", "categorical_missing", "categorical_encoding"},
             {"numerical_missing", "categorical_encoding"})
    if prep["numerical_missing"] not in ("median", "mean", "most_frequent"):
        raise ValueError("numerical_missing debe ser median, mean o most_frequent.")
    if prep.get("categorical_missing", "most_frequent") != "most_frequent":
        raise ValueError("categorical_missing debe ser most_frequent.")
    prep.setdefault("categorical_missing", "most_frequent")
    if prep["categorical_encoding"] not in ("one_hot", "ordinal"):
        raise ValueError("categorical_encoding debe ser one_hot u ordinal.")
    split = config["split"]
    _mapping(split, "split", {"test_size", "random_state"}, {"test_size", "random_state"})
    if type(split["test_size"]) not in (int, float) or not 0 < split["test_size"] < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")
    if type(split["random_state"]) is not int or not 0 <= split["random_state"] < 2**32:
        raise ValueError("random_state debe ser un entero entre 0 y 2**32 - 1.")
    _mapping(config["evaluation"], "evaluation", {"metrics"}, {"metrics"})
    metrics = config["evaluation"]["metrics"]
    if (not isinstance(metrics, list) or not metrics
            or not all(isinstance(m, str) and m in METRIC_NAMES for m in metrics)
            or len(set(metrics)) != len(metrics)):
        raise ValueError(f"metrics debe ser una lista no vacía sin duplicados de: {METRIC_NAMES}")
    models = config.get("models", {})
    _mapping(models, "models", set(MODEL_NAMES))
    for name, params in models.items():
        if not isinstance(params, dict) or not all(isinstance(key, str) for key in params):
            raise ValueError(f"models.{name} debe ser un mapa de parámetros.")
        if any(word in key.lower() for key in params for word in ("token", "password", "secret", "apikey", "api_key")):
            raise ValueError("Las credenciales deben ir en variables de entorno, no en YAML.")
    tracking = config["tracking"]
    _mapping(tracking, "tracking", {"experiment_name", "run_name"}, {"experiment_name", "run_name"})
    if any(not isinstance(v, str) or not v.strip() for v in tracking.values()):
        raise ValueError("Los valores de tracking deben ser textos no vacíos.")
    _validate_conditions(config)
    return config


def _validate_conditions(config):
    features = config.get("features", {})
    _mapping(features, "features", {"mode", "categorical_columns"})
    if features.get("mode", "all") not in ("all", "numeric", "categorical"):
        raise ValueError("features.mode debe ser all, numeric o categorical.")
    _columns(features.get("categorical_columns", []), "features.categorical_columns")
    perturbation = config.get("perturbation", {})
    _mapping(perturbation, "perturbation", {"seed", "missing_fraction", "numeric_noise_std",
                                           "noise_columns", "positive_keep_fraction"})
    _seed(perturbation.get("seed", 42), "perturbation.seed")
    for key, default in (("missing_fraction", 0.0), ("numeric_noise_std", 0.0),
                         ("positive_keep_fraction", 1.0)):
        value = perturbation.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"perturbation.{key} debe ser un número finito no negativo.")
    if not 0 <= perturbation.get("missing_fraction", 0.0) < 1:
        raise ValueError("missing_fraction debe estar en [0, 1).")
    if not 0 < perturbation.get("positive_keep_fraction", 1.0) <= 1:
        raise ValueError("positive_keep_fraction debe estar en (0, 1].")
    _columns(perturbation.get("noise_columns", []), "perturbation.noise_columns")
    if perturbation.get("numeric_noise_std", 0) and not perturbation.get("noise_columns"):
        raise ValueError("El ruido requiere noise_columns explícitas (variables continuas pertinentes).")
    threshold = config.get("threshold_optimization")
    if threshold is not None:
        _mapping(threshold, "threshold_optimization",
                 {"scoring", "cv_folds", "random_state", "min_precision"},
                 {"scoring", "cv_folds", "random_state"})
        if threshold["scoring"] not in ("f1", "recall"):
            raise ValueError("threshold_optimization.scoring debe ser f1 o recall.")
        if type(threshold["cv_folds"]) is not int or threshold["cv_folds"] < 2:
            raise ValueError("threshold_optimization.cv_folds debe ser un entero >= 2.")
        _seed(threshold["random_state"], "threshold_optimization.random_state")
        minimum = threshold.get("min_precision")
        if minimum is not None and (type(minimum) not in (int, float)
                                    or not math.isfinite(minimum) or not 0 < minimum <= 1):
            raise ValueError("threshold_optimization.min_precision debe estar en (0, 1].")
        if threshold["scoring"] == "recall" and minimum is None:
            raise ValueError("Optimizar recall requiere min_precision para evitar predecir todo positivo.")
    optimization = config.get("optimization")
    if optimization is None:
        return
    _mapping(optimization, "optimization", {"n_iter", "cv_folds", "random_state", "scoring", "param_distributions"},
             {"n_iter", "cv_folds", "random_state", "scoring", "param_distributions"})
    for key, minimum in (("n_iter", 1), ("cv_folds", 2)):
        if type(optimization[key]) is not int or optimization[key] < minimum:
            raise ValueError(f"optimization.{key} debe ser un entero >= {minimum}.")
    _seed(optimization["random_state"], "optimization.random_state")
    if optimization["scoring"] not in METRIC_NAMES:
        raise ValueError(f"optimization.scoring debe pertenecer a {METRIC_NAMES}.")
    spaces = optimization["param_distributions"]
    _mapping(spaces, "optimization.param_distributions", set(MODEL_NAMES))
    if not spaces:
        raise ValueError("Se necesita al menos un espacio de búsqueda.")
    for model_name, space in spaces.items():
        if not isinstance(space, dict) or not space:
            raise ValueError(f"Espacio de búsqueda vacío o inválido para {model_name}.")
        for key, values in space.items():
            if (not isinstance(key, str) or not key or key in ("random_state", "n_jobs")
                    or any(word in key.lower() for word in ("token", "password", "secret", "apikey", "api_key"))):
                raise ValueError("La búsqueda no puede alterar semillas, hilos o credenciales.")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Los candidatos de {model_name}.{key} deben ser una lista no vacía.")
            if any(type(value) not in (str, int, float, bool, type(None))
                   or (type(value) is float and not math.isfinite(value)) for value in values):
                raise ValueError(f"Candidatos no escalares o no finitos en {model_name}.{key}.")


def _columns(value, label):
    if (not isinstance(value, list) or not all(isinstance(column, str) and column for column in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} debe ser una lista de nombres únicos.")


def _seed(value, label):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{label} debe ser un entero en [0, 2**32).")


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_config(yaml.safe_load(handle))
