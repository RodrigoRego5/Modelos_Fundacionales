# Comparación de modelos tabulares

## Introducción

Pipeline de clasificación binaria para comparar **LightGBM** y **TabPFN** sobre
Framingham, Telco Customer Churn y Default of Credit Card Clients. Los datasets
se configuran mediante YAML y las ejecuciones se registran en MLflow local.

## Estructura del proyecto

```text
app/
  pipeline.py        # Flujo común y punto de entrada
  config.py          # Lectura de configuraciones
  datasets.py        # Carga, limpieza y partición
  preprocessing.py   # Imputación y codificación
  models.py          # LightGBM y TabPFN
  evaluation.py      # Métricas de clasificación
  tracking.py        # Registro en MLflow
  experiments.py     # Entrenamiento desde matrices
  optimization.py    # Optimización de hiperparámetros y umbral
configs/             # Configuraciones de datasets
data/                # Datasets originales
docs/                # Especificación del pipeline
requirements.txt     # Dependencias
```

## Guía de ejecución

Ejecutar desde la raíz del proyecto, con el entorno de Python activado.

### Terminal

```bash
python -m pip install -r requirements.txt
python -m app.pipeline --config configs/telco.yaml --model lightgbm
```

Para cambiar de dataset, seleccionar otro YAML de [configs/](./configs/).
TabPFN requiere conexión y una clave en `TABPFN_API_KEY`. En PowerShell:

```powershell
$env:TABPFN_API_KEY = "TU_API_KEY"
python -m app.pipeline --config configs/framingham.yaml --model tabpfn
```

La consola muestra métricas y duración. Para exportar además el resultado a un
archivo, añadir `--output results/resumen.json`.

Consultar los experimentos locales en http://localhost:5000:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Python

```python
from app.pipeline import run_pipeline

resultado = run_pipeline("configs/telco.yaml", "lightgbm")
metricas = resultado["metrics"]
modelo = resultado["model"]
preprocesador = resultado["preprocessor"]
```

La función acepta una ruta YAML o un diccionario y devuelve los objetos ajustados,
las métricas, los tiempos y los identificadores de MLflow.

## Configuración de datasets

| Configuración | Objetivo | Preparación específica |
| --- | --- | --- |
| [framingham.yaml](./configs/framingham.yaml) | `TenYearCHD` | Imputación de ausentes |
| [telco.yaml](./configs/telco.yaml) | `Churn` | Eliminar `customerID` y convertir `TotalCharges` a número |
| [credit.yaml](./configs/credit.yaml) | `default payment next month` | Leer la cabecera del Excel UCI y eliminar `ID` |

Cada YAML define datos, preprocesamiento, partición, métricas y parámetros de
los modelos. Ejemplo:

```yaml
name: telco_churn
dataset:
  path: data/Telco-Customer-Churn.csv
  target: Churn
  drop_columns: [customerID]
  numeric_columns: [TotalCharges]
preprocessing:
  numerical_missing: median
  categorical_encoding: one_hot
split: {test_size: 0.2, random_state: 42}
evaluation:
  metrics: [roc_auc, f1, accuracy, recall, pr_auc, balanced_accuracy, precision]
models:
  lightgbm: {n_estimators: 200, learning_rate: 0.05, verbosity: -1}
  tabpfn: {}
tracking:
  experiment_name: comparacion_modelos  # Grupo de experimentos en MLflow
  run_name: telco_baseline              # Nombre de la ejecución
```

Las rutas relativas de datos parten de la raíz del proyecto. La imputación y
codificación se ajustan solo con entrenamiento. Para comparar modelos, mantener
los mismos datos, preprocesamiento, semilla y partición.

`numerical_missing` admite `median`, `mean` o `most_frequent`;
`categorical_encoding` admite `one_hot` u `ordinal`. LightGBM usa `n_jobs: -1`
por defecto; se puede limitar el número de hilos en `models.lightgbm`.

`tracking.experiment_name` y `tracking.run_name` son obligatorios y se toman
exclusivamente del YAML. Cada ejecución recibe un identificador único, aunque
se repita el nombre. MLflow guarda parámetros, métricas y artefactos en la
base local `mlflow.db` y en `mlruns/`.
