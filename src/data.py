
import pandas as pd

from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data


def load_dataset(path, target, preprocess=False, one_hot=False):
    df = pd.read_csv(path)
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if preprocess:
        X_train, X_test, _ = preprocess_data(
            X_train, X_test, one_hot=one_hot
        )

    return X_train, X_test, y_train, y_test
