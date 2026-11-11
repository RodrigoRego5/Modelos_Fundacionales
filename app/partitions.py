"""Particiones verificables y subconjuntos anidados, independientes del modelo."""

import hashlib
import json
import math

import numpy as np

from .datasets import split_data


def fingerprint(value):
    """SHA-256 estable para estructuras JSON; rechaza valores no finitos."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_train_size(size):
    """Entero >= 2: filas; float en (0, 1]: fracción del pool de train."""
    if type(size) is int and size >= 2:
        return
    if type(size) is float and math.isfinite(size) and 0 < size <= 1:
        return
    raise ValueError("train_sizes admite enteros >= 2 o fracciones float en (0, 1].")


def _nested_order(labels, sampling_seed):
    """Orden determinista con cuotas proporcionales y ambas clases al inicio.

    Cada prefijo es un subconjunto del siguiente. Tras reservar un ejemplo por
    clase, se añade la clase con mayor déficit respecto a su proporción original.
    Dentro de cada clase las filas se permutan con la semilla de muestreo.
    """
    rng = np.random.default_rng(sampling_seed)
    groups = [rng.permutation(labels.index[labels == label]).tolist() for label in (0, 1)]
    counts = np.array([len(group) for group in groups])
    if np.any(counts == 0):
        raise ValueError("El pool de entrenamiento necesita ambas clases.")
    used = np.ones(2, dtype=int)
    order = [groups[label][0] for label in rng.permutation(2)]
    proportions = counts / counts.sum()
    while len(order) < len(labels):
        deficits = (len(order) + 1) * proportions - used
        deficits[used == counts] = -np.inf
        label = int(np.argmax(deficits))
        order.append(groups[label][used[label]])
        used[label] += 1
    return order


def make_partitions(X, y, config, train_sizes, sampling_seed, dataset_sha256):
    """Materializa una sola partición externa y todos sus tamaños de train."""
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X e y necesitan índices únicos y alineados.")
    X_pool, X_test, y_pool, _ = split_data(X, y, config)
    sizes = []
    for requested in train_sizes:
        validate_train_size(requested)
        count = requested if type(requested) is int else math.floor(len(X_pool) * requested)
        if not 2 <= count <= len(X_pool):
            raise ValueError(f"Tamaño {requested!r}: {count} filas; pool disponible: {len(X_pool)}.")
        if count in sizes:
            raise ValueError("Dos tamaños solicitados producen el mismo número de filas.")
        sizes.append(count)
    order = _nested_order(y_pool, sampling_seed)
    partitions = []
    for requested, count in zip(train_sizes, sizes):
        partition = {
            "schema_version": 1,
            "dataset_sha256": dataset_sha256,
            "dataset_rules_sha256": fingerprint(config["dataset"]),
            "split_seed": config["split"]["random_state"],
            "test_size": config["split"]["test_size"],
            "sampling_seed": sampling_seed,
            "train_size_requested": requested,
            "train_pool": X_pool.index.tolist(),
            "train": order[:count],
            "test": X_test.index.tolist(),
        }
        partition["partition_id"] = fingerprint(partition)
        partitions.append(partition)
    return partitions


def apply_partition(X, y, config, partition, dataset_sha256):
    """Verifica integridad, procedencia y ausencia de solapamiento antes del fit."""
    content = {key: value for key, value in partition.items() if key != "partition_id"}
    if partition.get("schema_version") != 1 or fingerprint(content) != partition.get("partition_id"):
        raise ValueError("La partición está alterada o tiene una versión incompatible.")
    if (partition["dataset_sha256"] != dataset_sha256
            or partition["dataset_rules_sha256"] != fingerprint(config["dataset"])
            or partition["split_seed"] != config["split"]["random_state"]
            or partition["test_size"] != config["split"]["test_size"]):
        raise ValueError("La partición no corresponde a estos datos o configuración.")
    train, test, pool = (partition[key] for key in ("train", "test", "train_pool"))
    if (not X.index.is_unique or not X.index.equals(y.index)
            or any(len(rows) != len(set(rows)) for rows in (train, test, pool))
            or set(pool) & set(test) or not set(train) <= set(pool)
            or set(pool) | set(test) != set(X.index) or len(train) < 2):
        raise ValueError("Índices inválidos, solapados o incompletos en la partición.")
    if any(set(y.loc[rows].unique()) != {0, 1} for rows in (train, test)):
        raise ValueError("Train y test deben conservar ambas clases.")
    return X.loc[train], X.loc[test], y.loc[train], y.loc[test]
