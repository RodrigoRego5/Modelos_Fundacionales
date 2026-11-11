"""Selección de variables y degradación reproducible, exclusiva de train."""

import hashlib
import math

import numpy as np
import pandas as pd


def select_features(X, settings):
    """Selección por esquema, sin ajustar estadísticas ni consultar el objetivo."""
    X = X.copy()
    categories = settings.get("categorical_columns", [])
    missing = set(categories).difference(X.columns)
    if missing:
        raise ValueError(f"Variables categóricas declaradas inexistentes: {sorted(missing)}")
    for column in categories:
        X[column] = X[column].astype(object)
    mode = settings.get("mode", "all")
    if mode == "numeric":
        X = X.select_dtypes(include="number")
    elif mode == "categorical":
        X = X.select_dtypes(exclude="number")
    if X.shape[1] == 0:
        raise ValueError(f"La selección {mode!r} no deja ninguna variable.")
    return X


def apply_training_conditions(X, y, settings):
    """Degrada una copia de train. Nunca recibe test ni validación.

    El ruido usa desviaciones calculadas exclusivamente en este train. La misma
    función se aplica dentro de cada fold de búsqueda y en el ajuste final.
    """
    X, y = X.copy(), y.copy()
    original_count = len(y)
    seed = settings.get("seed", 42)
    keep_fraction = settings.get("positive_keep_fraction", 1.0)
    if keep_fraction < 1:
        positives = y.index[y == 1].to_numpy()
        keep = max(1, math.floor(len(positives) * keep_fraction))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        retained = rng.permutation(positives)[:keep]
        mask = (y != 1) | y.index.isin(retained)
        X, y = X.loc[mask].copy(), y.loc[mask].copy()
    if set(y.unique()) != {0, 1}:
        raise ValueError("La condición de entrenamiento necesita ambas clases.")

    noise_std = settings.get("numeric_noise_std", 0.0)
    noise_columns = settings.get("noise_columns", [])
    invalid = set(noise_columns).difference(X.select_dtypes(include="number").columns)
    if invalid:
        raise ValueError(f"noise_columns debe contener variables numéricas seleccionadas: {sorted(invalid)}")
    scales = {}
    for position, column in enumerate(noise_columns):
        scale = float(X[column].std(ddof=0))
        scale = scale if np.isfinite(scale) else 0.0
        scales[column] = scale
        if noise_std:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1, position]))
            X[column] = X[column].astype(float) + rng.normal(0, noise_std * scale, len(X))

    injected = 0
    missing_fraction = settings.get("missing_fraction", 0.0)
    if missing_fraction:
        for position, column in enumerate(X.columns):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 2, position]))
            mask = (rng.random(len(X)) < missing_fraction) & X[column].notna().to_numpy()
            injected += int(mask.sum())
            X.loc[mask, column] = np.nan

    # Permite verificar que ambos modelos han recibido la misma degradación.
    digest = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()
                            + pd.util.hash_pandas_object(y, index=True).values.tobytes()).hexdigest()
    context = {"n_train_before_conditions": original_count, "n_train_actual": len(y),
               "train_rows": X.index.tolist(), "selected_columns": X.columns.tolist(),
               "injected_missing_cells": injected, "noise_scales": scales,
               "training_data_sha256": digest}
    return X, y, context
