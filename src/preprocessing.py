
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(X_train, X_test, one_hot=False):
    numeric_cols = X_train.select_dtypes(
        include="number"
    ).columns

    categorical_cols = X_train.select_dtypes(
        exclude="number"
    ).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]

    if one_hot:
        categorical_steps.append(
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        )

    categorical_pipeline = Pipeline(categorical_steps)

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, preprocessor
