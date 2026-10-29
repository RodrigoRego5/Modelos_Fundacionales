## Descripción del proyecto

Comparación de LightGBM y TabPFN en tres datasets de clasificación binaria mediante MLflow. Cada integrante crea su notebook en su rama y utiliza los módulos comunes para ejecutar y comparar experimentos.

## Estructura del proyecto

```text
proyecto/
├── data/                 # Datasets originales
├── src/
│   ├── data.py           # Carga, limpieza y división de datos
│   ├── preprocessing.py  # Imputación y codificación
│   ├── models.py         # Construcción de modelos
│   ├── evaluation.py     # Cálculo de métricas
│   ├── optimization.py   # Optimización pendiente de implementar
│   └── experiments.py    # Entrenamiento y registro en MLflow
├── tests/                # Pruebas del flujo completo
├── framhingam.ipynb       # Ejemplo de ejecución
├── requirements.txt
└── README.md
```

## Guía de ejecución

Instalar las dependencias y ejecutar el notebook desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
```

### 1. Importar módulos y configurar MLflow

```python
import mlflow

from src.data import load_dataset
from src.models import build_lightgbm, build_tabpfn
from src.experiments import run_experiment

participant = "TU_NOMBRE"
apikey = "TU_API_KEY"
mlflow.set_experiment("comparacion_modelos")
```

### 2. Cargar el dataset

```python
dataset = "framingham"  # framingham, telco o credit

X_train, X_test, y_train, y_test = load_dataset(
    dataset=dataset,
    random_state=42,
    test_size=0.2,
    one_hot=True,
)
```

| Dataset | Variable objetivo | Limpieza específica |
| --- | --- | --- |
| `framingham` | `TenYearCHD` | Imputación de valores ausentes |
| `telco` | `Churn` | Eliminación de customerID, conversión de TotalCharges y objetivo Yes/No |
| `credit` | `default payment next month` | Lectura de la cabecera del Excel y eliminación de ID |

El preprocesamiento se ajusta solo con entrenamiento. Por defecto, imputa valores ausentes y aplica one-hot encoding; con `one_hot=False` utiliza codificación ordinal. El objetivo positivo es siempre `1`.

Cada dataset tiene su función en [src/preprocessing.py](src/preprocessing.py):
`preprocess_framingham`, `preprocess_telco` y `preprocess_credit`. Cada integrante
puede modificar su función sin alterar las otras; `load_dataset` la selecciona
automáticamente. Solo se realiza limpieza básica, imputación y codificación, sin
escalado, balanceo ni selección de variables. Con `preprocess=False` se omiten
estas funciones y se devuelven los predictores originales.

### 3. Ejecutar un experimento

```python
params_lgbm = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1,
}

model_lgbm, metrics_lgbm = run_experiment(
    model=build_lightgbm(params_lgbm),
    model_name="LightGBM",
    dataset=dataset,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=params_lgbm,
    participant=participant,
)

print(metrics_lgbm)
```

Cada integrante introduce su clave en `apikey` en la primera celda. Para TabPFN, utilizar `build_tabpfn(params={}, apikey=apikey)`, `model_name="TabPFN"` y `params={}`. Requiere conexión al servicio remoto.

### 4. Consultar resultados

Desde la raíz del proyecto:

```bash
mlflow ui
```

Abrir http://localhost:5000 para consultar parámetros, métricas y etiquetas. Para comparar resultados, mantener el mismo dataset, semilla, split y codificación. Los registros locales no se comparten automáticamente entre equipos.


