## Estructura del proyecto

```text
proyecto/
├── data/                 # Datasets
├── src/
│   ├── data.py           # Carga y división de datos
│   ├── preprocessing.py  # Preprocesamiento
│   ├── models.py         # Definición de modelos
│   ├── evaluation.py     # Cálculo de métricas
│   ├── optimization.py   # Optimización de hiperparametros POR IMPLEMENTAR
│   └── experiments.py    # Entrenamiento, evaluación y registro
├── notebook.ipynb        # Ejecución y análisis de experimentos
└── README.md
```

Cada uno de los integrantes crea su notebook en su rama y trabaja desde el

## Guía de ejecución

En un notebook nuevo situado en la raíz del proyecto, ejecutar las siguientes celdas en orden.

**1. Importar módulos y configurar MLflow**

```python
import sys
sys.path.append("src")

import mlflow

from data import load_dataset
from models import build_lightgbm, build_tabpfn
from experiments import run_experiment

#MODIFICAR ESTOS VALORES
apikey = "TU_API_KEY"
mlflow.set_experiment("NOMBRE_EXPERIMENTO")
```

**2. Cargar el dataset**

```python
X_train, X_test, y_train, y_test = load_dataset(
    path="data/NOMBRE_DATASET.csv",
    target="VARIABLE_OBJETIVO"
)
```

Cambiar la ruta y la variable objetivo según el dataset utilizado.

**3. Ejecutar un experimento con parámetros determinados**

```python
params_lgbm = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1
}

model_lgbm, metrics_lgbm = run_experiment(
    model=build_lightgbm(params_lgbm),
    model_name="LightGBM",
    dataset="Framingham",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=params_lgbm
)


```

Para probar otro modelo, sustituir `build_lightgbm()` por `build_tabpfn()`.

**4. Consultar resultados**

Ejecutar en una terminal desde la raíz del proyecto:

```bash
mlflow ui 
```

Abrir http://localhost:5000 para consultar las métricas registradas de los experimentos realizados.
