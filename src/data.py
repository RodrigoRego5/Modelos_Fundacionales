
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
try:
    from .preprocessing import preprocess_data, preprocess_framingham, preprocess_telco, preprocess_credit
except ImportError:  # imports used by notebooks after appending src/
    from preprocessing import preprocess_data, preprocess_framingham, preprocess_telco, preprocess_credit


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_absolute():
        candidates = [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
        # UCI credit workbook has technical names in row 0 and real names in row 1.
        if "Y" in df.columns and str(df.iloc[0]["Y"]).strip() == "default payment next month":
            df = pd.read_excel(path, header=1)
        return df
    return pd.read_csv(path)


def _normalise_target(y: pd.Series) -> pd.Series:
    if y.dtype == "object" or str(y.dtype).startswith("string"):
        values = y.astype("string").str.strip()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
        lowered = values.str.lower()
        if lowered.dropna().isin(mapping).all():
            return lowered.map(mapping).astype("int64")
    return y


def load_dataset(path=None, target=None, preprocess=True, one_hot=True, test_size=0.2, random_state=42, *, dataset=None, return_preprocessor=False):
    profiles = {
        "framingham": ("framingham.csv", "TenYearCHD"),
        "telco": ("Telco-Customer-Churn.csv", "Churn"),
        "credit": ("default of credit card clients.xls", "default payment next month"),
    }
    if dataset is not None:
        if dataset not in profiles:
            raise ValueError(f"Unknown dataset {dataset!r}; choose {list(profiles)}")
        filename, default_target = profiles[dataset]
        path = path if path is not None else Path(__file__).resolve().parents[1] / "data" / filename
        target = target or default_target
    if path is None or target is None:
        raise ValueError("Provide dataset or both path and target.")
    # Mantener también la selección específica con la API antigua path/target.
    if dataset is None:
        dataset = next((name for name, (filename, _) in profiles.items()
                        if Path(path).name.lower() == filename.lower()), None)
    df = _read_table(path)
    df.columns = [str(column).strip() for column in df.columns]
    if target not in df.columns:
        raise KeyError(f"Target {target!r} not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = _normalise_target(df[target])
    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected a binary target encoded as 0/1 or Yes/No.")
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    transformer = None
    if preprocess:
        preprocessing_functions = {
            "framingham": preprocess_framingham,
            "telco": preprocess_telco,
            "credit": preprocess_credit,
        }
        preprocess_function = preprocessing_functions.get(dataset, preprocess_data)
        X_train, X_test, transformer = preprocess_function(
            X_train, X_test, one_hot=one_hot
        )

    result = (X_train, X_test, y_train, y_test)
    return (*result, transformer) if return_preprocessor else result
