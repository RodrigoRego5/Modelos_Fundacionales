"""Imputación y codificación ajustadas exclusivamente con entrenamiento."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer


def _one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_preprocessor(one_hot: bool = True, *, numerical_missing="median", categorical_missing="most_frequent") -> ColumnTransformer:
    """Build a leakage-free transformer for mixed tabular datasets."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numerical_missing, keep_empty_features=True)),
    ])

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_missing, keep_empty_features=True)),
    ]
    if one_hot:
        categorical_steps.append(("encoder", _one_hot_encoder()))
    else:
        categorical_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

    categorical_pipeline = Pipeline(categorical_steps)
    return ColumnTransformer([
        ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
        ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number")),
    ])


def preprocess_data(X_train, X_test, one_hot=True):
    preprocessor = build_preprocessor(one_hot=one_hot)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, preprocessor


def preprocess_framingham(X_train, X_test, one_hot=True):
    """Framingham: imputar ausentes; sin escalado ni ingeniería de variables."""
    return preprocess_data(X_train, X_test, one_hot=one_hot)


def clean_telco(frame):
    """Dataset-specific, row-wise cleanup safe to repeat in each CV fold."""
    frame = frame.drop(columns=["customerID"], errors="ignore").copy()
    frame["TotalCharges"] = pd.to_numeric(
        frame["TotalCharges"].astype(str).str.strip().replace("", float("nan")),
        errors="raise",
    )
    return frame


def preprocess_telco(X_train, X_test, one_hot=True):
    """Telco: eliminar identificadores, convertir importes y codificar texto."""
    return preprocess_data(clean_telco(X_train), clean_telco(X_test), one_hot=one_hot)


def clean_credit(frame):
    return frame.drop(columns=["ID"], errors="ignore")


def preprocess_credit(X_train, X_test, one_hot=True):
    """Crédito: eliminar ID e imputar ausentes; conservar códigos numéricos."""
    return preprocess_data(clean_credit(X_train), clean_credit(X_test), one_hot=one_hot)


def build_dataset_pipeline(dataset, model, one_hot=True):
    """Cloneable pipeline used only by optional cross-validation optimization."""
    cleaning = {
        "framingham": None,
        "telco": clean_telco,
        "credit": clean_credit,
    }
    if dataset not in cleaning:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    steps = []
    if cleaning[dataset] is not None:
        steps.append(("clean", FunctionTransformer(cleaning[dataset], validate=False)))
    steps.extend([("preprocessing", build_preprocessor(one_hot=one_hot)), ("model", model)])
    return Pipeline(steps)


def build_preprocessing(config):
    settings = config["preprocessing"]
    return build_preprocessor(
        one_hot=settings["categorical_encoding"] == "one_hot",
        numerical_missing=settings["numerical_missing"],
        categorical_missing=settings.get("categorical_missing", "most_frequent"),
    )
