"""Construcción de LightGBM y del cliente remoto de TabPFN."""

import os

def build_lightgbm(params=None):
    from lightgbm import LGBMClassifier
    # Evita la detección de núcleos físicos vía WMIC en Windows.
    return LGBMClassifier(**{"random_state": 42, "verbosity": -1, "n_jobs": -1, **(params or {})})


def build_tabpfn(params=None, apikey=None):
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
    access_token = apikey or os.getenv("TABPFN_TOKEN") or os.getenv("TABPFN_API_KEY")
    if access_token:
        tabpfn_client.set_access_token(access_token)
    return TabPFNClassifier(**(params or {}))


MODEL_NAMES = ("lightgbm", "tabpfn")


def build_model(name, config):
    if name not in MODEL_NAMES:
        raise ValueError(f"Modelo desconocido: {name!r}")
    params = {"random_state": config["split"]["random_state"],
              **config.get("models", {}).get(name, {})}
    builder = build_lightgbm if name == "lightgbm" else build_tabpfn
    # TABPFN_TOKEN es el nombre que espera tabpfn-client; se conserva el alias anterior.
    return builder(params)
