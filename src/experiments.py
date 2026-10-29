import mlflow

try:
    from .evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


def run_experiment(
    model_name,
    dataset,
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    params=None,
    participant=None,
    preprocessor=None,
):
    with mlflow.start_run(run_name=model_name):
        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Evaluar en el conjunto de test
        metrics = evaluate_model(model, X_test, y_test)

        # Registrar parámetros, métricas y etiquetas
        mlflow.log_params(params if params is not None else model.get_params())
        mlflow.log_params({"n_train": len(y_train), "n_test": len(y_test), "n_features": X_train.shape[1]})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model": model_name,
            "dataset": dataset
        })
        if participant:
            mlflow.set_tag("participant", participant)
        if preprocessor is not None:
            mlflow.log_dict({"features": list(preprocessor.get_feature_names_out())}, "features.json")

    return model, metrics
