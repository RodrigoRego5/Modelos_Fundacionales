import mlflow

from evaluation import evaluate_model


def run_experiment(
    model_name,
    dataset,
    X_train,
    X_test,
    y_train,
    y_test,
    model,
    params=None
):
    with mlflow.start_run(run_name=model_name):
        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Evaluar en el conjunto de test
        metrics = evaluate_model(model, X_test, y_test)

        # Registrar parámetros, métricas y etiquetas
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model": model_name,
            "dataset": dataset
        })

    return model, metrics
