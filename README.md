# Comparación de modelos tabulares

## Introducción

Pipeline de clasificación binaria para comparar **LightGBM** y **TabPFN** sobre
Framingham, Telco Customer Churn y Default of Credit Card Clients. Los datasets
se configuran mediante YAML y las ejecuciones se registran en MLflow local.

## Estructura del proyecto

```text
app/
  pipeline.py        # Flujo común y punto de entrada
  batteries.py       # Baterías registradas en MLflow
  partitions.py      # Particiones compartidas y tamaños de entrenamiento
  conditions.py      # Selección de variables y perturbaciones
  config.py          # Lectura de configuraciones
  datasets.py        # Carga, limpieza y partición
  preprocessing.py   # Imputación y codificación
  models.py          # LightGBM y TabPFN
  evaluation.py      # Métricas de clasificación
  tracking.py        # Registro en MLflow
  experiments.py     # Entrenamiento desde matrices
  optimization.py    # Optimización de hiperparámetros y umbral
configs/             # Configuraciones de datasets y baterías en batteries/
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
TabPFN requiere conexión y un token en la variable `TABPFN_TOKEN`. En PowerShell:

```powershell
$env:TABPFN_TOKEN = "TU_TOKEN"
python -m app.pipeline --config configs/framingham.yaml --model tabpfn
```

La consola muestra métricas y duración. Para exportar además el resultado a un
archivo, añadir `--output results/resumen.json`.

Consultar los experimentos locales en http://localhost:5000:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Baterías de experimentos

Una batería ejecuta las combinaciones de datasets, modelos, semillas, tamaños y
variantes definidas en un YAML de [configs/batteries/](./configs/batteries/):

```bash
python -m app.batteries --config configs/batteries/baseline.yaml
```

Este ejemplo compara LightGBM y TabPFN en Framingham con tres tamaños y cinco
particiones (30 ejecuciones). Requiere el token de TabPFN indicado arriba.
`train_sizes` admite enteros como número de filas y decimales como fracciones del
entrenamiento disponible: `200` son 200 filas y `1.0` usa todo el entrenamiento.
Los modelos y variantes comparten las mismas particiones; el test se mantiene
fijo al cambiar de tamaño dentro de cada semilla.

Todo se registra en MLflow: una ejecución padre agrupa las ejecuciones hijas y
contiene el plan, el resumen y una tabla de resultados. No hay que indicar un
directorio de salida. Cada invocación crea una batería nueva. El experimento se
define en `tracking.experiment_name` del YAML de la batería; los nombres de sus
ejecuciones se generan automáticamente.

### Optimización del umbral

Además de la optimización de hiperparametros, el pipeline admite un bloque opcional `threshold_optimization`, independiente de `optimization`. Puede ponerse en el YAML del dataset o en una variante de la
batería. En [tuning.yaml](./configs/batteries/tuning.yaml) se activa en `ajustado`:

```yaml
  - name: ajustado
    threshold_optimization:
      scoring: f1
      cv_folds: 3
      random_state: 789
    optimization:
      # Mantener aquí el bloque de hiperparámetros existente.
```

Primero se eligen los hiperparámetros; después se generan probabilidades
out-of-fold del entrenamiento con esos parámetros. Cada fold ajusta su propio
preprocesamiento y aplica las perturbaciones solo a su entrenamiento. Se elige
el umbral que maximiza F1 sobre las probabilidades OOF agrupadas y se entrena
el modelo final con todo el entrenamiento disponible. El test externo se usa
únicamente para la evaluación final. La puntuación OOF es un criterio de selección,
no una estimación independiente del rendimiento tras seleccionar hiperparámetros.
El ajuste de umbral añade `cv_folds` entrenamientos por ejecución que lo active.

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

En las ejecuciones individuales, `tracking.experiment_name` y
`tracking.run_name` son obligatorios y se toman del YAML del dataset. Cada
ejecución recibe un identificador único, aunque se repita el nombre. MLflow
guarda parámetros, métricas y artefactos en la base local `mlflow.db` y en `mlruns/`.
