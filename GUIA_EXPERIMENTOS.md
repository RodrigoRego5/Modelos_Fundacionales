# Guía personal para realizar la práctica

Esta guía organiza las preguntas del [PDF de la práctica](./docs/piia2-20261105-experimentacion-reproducible%20(1).pdf) en seis experimentos. Usa **Framingham en todos salvo el primero**, que compara los tres datasets. Todo se registra en MLflow; no necesitas indicar un directorio de salida.

## 1. Preparar el entorno

Abre una terminal en la raíz del proyecto y activa tu entorno de Python. Si aún no has instalado las dependencias:

```powershell
python -m pip install -r requirements.txt
```

En la terminal desde la que ejecutarás las baterías, configura el token de TabPFN:

```powershell
$env:TABPFN_TOKEN = "TU_TOKEN"
```

TabPFN utiliza el cliente remoto y necesita conexión. En otra terminal, desde la misma raíz y con el entorno activado, inicia MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://localhost:5000. Mantén esa terminal abierta y ejecuta las baterías en la primera, una detrás de otra, para no solapar las mediciones de tiempo.

Como comprobación inicial opcional, ejecuta:

```powershell
python -m app.batteries --config configs/batteries/smoke.yaml
```

En `practica_00_comprobacion` deben aparecer una ejecución padre y dos hijas de LightGBM sobre Framingham: con 200 filas y con todo el entrenamiento. Esta comprobación usa 20 árboles, no necesita TabPFN y no valida su conexión. No la incluyas en las conclusiones.

## 2. Qué vas a ejecutar

| Experimento | Pregunta del PDF | Batería | Experimento en MLflow | Ejecuciones hijas |
| --- | --- | --- | --- | ---: |
| 1 | ¿Cambia la comparación según el dataset? | [datasets.yaml](./configs/batteries/datasets.yaml) | `practica_01_datasets` | 18 |
| 2 | ¿Qué ocurre con pocos datos y distintas particiones? | [baseline.yaml](./configs/batteries/baseline.yaml) | `practica_02_tamano_semillas` | 30 |
| 3 | ¿Cuánto influye el preprocesamiento? | [preprocessing.yaml](./configs/batteries/preprocessing.yaml) | `practica_03_preprocesamiento` | 18 |
| 4 | ¿Cómo influyen los tipos de variables? | [variables.yaml](./configs/batteries/variables.yaml) | `practica_04_variables` | 18 |
| 5 | ¿Compensa ajustar los hiperparámetros? | [tuning.yaml](./configs/batteries/tuning.yaml) | `practica_05_ajuste` | 12 |
| 6 | ¿Qué modelo resiste mejor el deterioro del entrenamiento? | [robustness.yaml](./configs/batteries/robustness.yaml) | `practica_06_robustez` | 30 |

Son **126 ejecuciones finales**, más los ajustes internos del experimento 5. Las preguntas de métricas y costes se responden con estas mismas ejecuciones, sin otra batería. La comprobación inicial añade dos ejecuciones.

Condiciones ya fijadas en los YAML:

- Test estratificado del 20 %. Los modelos y variantes comparten las filas de test y el subconjunto de entrenamiento para cada dataset, semilla y tamaño.
- Semillas de partición `11, 22, 33`; el experimento 2 añade `44, 55`. Semilla de submuestreo `123` y del modelo `42`. Se cambia la partición, no simultáneamente la semilla del modelo.
- Los tamaños `0.1`, `0.3` y `1.0` son fracciones del **80 % reservado para entrenamiento**, no de todo el dataset. Los subconjuntos pequeños están contenidos en los mayores. En los YAML, `200` son 200 filas y `1.0` es todo el entrenamiento.
- Referencia: LightGBM con 400 árboles, learning rate 0.05 y cuatro hilos; TabPFN con `n_estimators: 4`. La búsqueda modifica solo los parámetros indicados en su espacio de ajuste. Son referencias fijadas para esta práctica, no todos los valores por defecto de las librerías.
- La imputación y codificación se aprenden solo del entrenamiento. En la búsqueda se vuelven a ajustar dentro de cada fold. El test externo se reserva para evaluar.

Antes de mirar resultados, fija `average_precision` como métrica principal de Framingham y contrástala con ROC-AUC, recall, precisión, F1 y balanced accuracy. Usa los tiempos como segundo criterio. Así no escoges después la métrica que favorezca a un modelo.

## 3. Consultar cada batería en MLflow

Cada comando crea una **ejecución padre** que agrupa sus ejecuciones hijas. La consola muestra el identificador del padre. Entra en el experimento de la tabla y abre esa ejecución o despliega sus hijas.

El padre contiene estos artefactos:

- `plan.json`: configuraciones, filas de las particiones, semillas, versiones y huellas de datos/código.
- `summary.json`: cuántas ejecuciones terminaron, fallaron o quedaron pendientes.
- `results.csv`: tabla de todas las combinaciones, métricas, tiempos y errores. Puedes verla o descargarla desde MLflow; no tienes que gestionar una carpeta de resultados.

Las hijas contienen sus métricas y tiempos y los artefactos `config.json`, `split.json`, `conditions.json`, `result.json` y `predictions.csv`. Las ajustadas incluyen además `search_results.json` y los resultados de los candidatos en `search/`.

En la tabla de MLflow selecciona las etiquetas `dataset`, `model`, `variant`, `split_seed` y `partition_id`; los parámetros `train_size_requested`, `n_train_selected`, `n_train`, `n_input_features`, `n_features` y `train_positive_rate`; y las métricas que estés comparando. `n_train_selected` cuenta las filas seleccionadas antes de perturbarlas y `n_train` las realmente utilizadas.

Para restringirte a una batería concreta, filtra por su padre:

```text
tags.mlflow.parentRunId = 'ID_DEL_PADRE'
```

**Si algo falla:** se registra el error y la batería continúa con las demás combinaciones. El padre termina en estado `FAILED` si hubo fallos; consulta `results.csv`. Si interrumpes con Ctrl+C, se conserva el resumen parcial. Una nueva invocación ejecuta la batería completa y crea otro padre: no reanuda ni mezcla resultados anteriores. Analiza un padre cada vez y anota las comparaciones incompletas.

## 4. Experimento 1: comparar datasets

```powershell
python -m app.batteries --config configs/batteries/datasets.yaml
```

Ejecuta ambos modelos en Framingham, Telco y Credit con tres particiones y todo el entrenamiento: `3 datasets × 3 semillas × 2 modelos = 18` ejecuciones.

En `practica_01_datasets`, compara los modelos **dentro de cada dataset**, emparejando la misma semilla. Prepara una tabla con media y desviación estándar de ROC-AUC, average precision, F1, recall y tiempos por modelo y dataset. Comprueba si el orden de los modelos cambia entre problemas y si se mantiene en las tres particiones.

No promedies los tres datasets como si fueran repeticiones del mismo problema: tienen tamaños, prevalencias y variables diferentes. Average precision depende especialmente de la prevalencia. Presenta las diferencias entre modelos dentro de cada problema.

**Qué escribir:** qué modelo obtiene mejores resultados en cada dataset, con qué diferencia y estabilidad, y si las conclusiones cambian entre problemas. A partir de aquí trabaja solo con Framingham.

## 5. Experimento 2: tamaño y estabilidad

```powershell
python -m app.batteries --config configs/batteries/baseline.yaml
```

En `practica_02_tamano_semillas` tendrás `3 tamaños × 5 semillas × 2 modelos = 30` ejecuciones. Los tamaños son el 10 %, 30 % y 100 % del entrenamiento disponible.

Agrupa por modelo y `train_size_requested`. Calcula media y desviación entre las cinco particiones y representa average precision frente a `n_train`. Añade curvas de `training_seconds` e `inference_seconds`. Comprueba si cambia el orden de los modelos al disponer de más datos.

Para cada tamaño, calcula también `AP(TabPFN) − AP(LightGBM)` emparejando la semilla. Una media positiva con diferencias que cambian de signo merece una conclusión más prudente que una ventaja en todas las particiones. Son cinco particiones del mismo dataset, no cinco datasets independientes.

**Qué escribir:** cuánto aporta aumentar el entrenamiento, qué modelo rinde mejor con pocas filas y cuánto cambia la conclusión entre particiones. Como pregunta adicional del PDF puedes plantear «¿La ventaja de un modelo desaparece al aumentar los datos?». Esta batería ya permite estudiarla.

## 6. Experimento 3: preprocesamiento

```powershell
python -m app.batteries --config configs/batteries/preprocessing.yaml
```

En `practica_03_preprocesamiento` se comparan tres variantes, dos modelos y tres semillas: 18 ejecuciones.

| Variante | Imputación numérica | Codificación categórica | Comparación |
| --- | --- | --- | --- |
| `minimo` | Mediana | Ordinal | Referencia mínima común |
| `one_hot` | Mediana | One-hot | Frente a `minimo`: efecto de la codificación |
| `media` | Media | Ordinal | Frente a `minimo`: efecto de la imputación |

Los ausentes categóricos se imputan con la moda. Compara cada alternativa con `minimo` dentro del mismo modelo y semilla. Revisa rendimiento, `preprocessing_seconds`, `training_seconds` y `n_features`. Entre `media` y `one_hot` cambian dos factores: no atribuyas su diferencia solo a uno.

En [framingham.yaml](./configs/framingham.yaml) se declaran como categóricas `male`, `education`, `currentSmoker`, `BPMeds`, `prevalentStroke`, `prevalentHyp` y `diabetes`, aunque el CSV las almacene como números. Los otros predictores son numéricos. Para esta práctica, educación se trata como categoría; la variante ordinal codifica sus niveles, sin asumir una escala clínica validada. Todas las baterías de Framingham conservan esta decisión.

**Qué escribir:** si una preparación más elaborada mejora cada modelo y cuánto cuesta. Usa la referencia común para interpretar diferencias, sin cambiar las otras baterías después de observar este test.

## 7. Experimento 4: tipos de variables

```powershell
python -m app.batteries --config configs/batteries/variables.yaml
```

En `practica_04_variables` encontrarás `todas`, `numericas` y `categoricas`, con tres semillas y ambos modelos: 18 ejecuciones. Comparten filas y test; cambia el conjunto de predictores. Usan mediana y one-hot.

Compara cada subconjunto con `todas` dentro del mismo modelo y semilla. `n_input_features` cuenta las variables seleccionadas; `n_features` es la dimensión tras codificar. Comprueba qué información pierdes al retirar cada grupo y si afecta de forma diferente a los modelos. La comparación cambia también el contenido informativo, no solamente el tipo de dato.

Relaciona este experimento con el 3 para comentar la dimensión añadida por one-hot y con el 6 para estudiar los ausentes. **Framingham no permite concluir sobre alta dimensionalidad en general**: describe esa limitación y el cambio de dimensión observado. No hace falta inventar variables o añadir datasets para esta práctica centrada en Framingham.

**Qué escribir:** qué grupos de variables aportan información y si alguno de los modelos depende más de ellos.

## 8. Experimento 5: ajuste acotado y coste

```powershell
python -m app.batteries --config configs/batteries/tuning.yaml
```

En `practica_05_ajuste` se usa el 30 % del entrenamiento, tres semillas, ambos modelos y dos variantes: `referencia` y `ajustado`. Son 12 ejecuciones finales.

El ajuste evalúa cuatro candidatos por modelo mediante validación cruzada estratificada de tres folds dentro del entrenamiento, eligiendo por average precision:

- LightGBM: las cuatro combinaciones de `num_leaves: [15, 31]` y `min_child_samples: [10, 30]`.
- TabPFN: `n_estimators: [2, 4, 8, 16]`.

Son 12 ajustes internos por ejecución ajustada, **72 ajustes internos en total**, además de los 12 entrenamientos finales. Igual número de candidatos y folds no implica igual tiempo ni coste del servicio remoto.

Compara `ajustado − referencia` para el mismo modelo y partición. Anota la mejora en average precision de test, su estabilidad y el incremento de `search_seconds`. Consulta `search_results.json` para los mejores parámetros. `cv_selection_score` es la puntuación interna de selección; no la mezcles con el resultado de test.

Para el coste de búsqueda y entrenamiento compara `search_seconds + training_seconds`; para el flujo completo medido, `duration_seconds`. Los tiempos separados de preprocesamiento, entrenamiento e inferencia corresponden al modelo final. El trabajo de validación cruzada está dentro de `search_seconds`.

**Qué escribir:** si la mejora compensa el tiempo extra dentro de este presupuesto y tamaño. No amplíes la búsqueda mirando el test; si no mejora, ese resultado también responde a la pregunta.

## 9. Experimento 6: ausentes, ruido y desequilibrio

```powershell
python -m app.batteries --config configs/batteries/robustness.yaml
```

En `practica_06_robustez` se ejecutan cinco variantes, tres semillas y ambos modelos: 30 ejecuciones. Las perturbaciones afectan **solo al entrenamiento**. El test conserva sus datos originales, incluidos sus ausentes naturales.

| Variante | Qué hace | Qué comparar |
| --- | --- | --- |
| `referencia` | Conserva el entrenamiento original | Referencia de esta batería |
| `ausentes_10` | Oculta cada celda observada con probabilidad 0.10 | Caída respecto a referencia |
| `ausentes_30` | Oculta cada celda observada con probabilidad 0.30 | Efecto de aumentar los ausentes |
| `ruido_20` | Añade ruido gaussiano a `sysBP`, `diaBP`, `BMI`, `totChol` y `glucose` | Sensibilidad al ruido de medida simulado |
| `menos_positivos` | Conserva la mitad de los positivos del entrenamiento | Menor representación de la clase positiva |

La desviación del ruido es el 20 % de la desviación de cada columna, calculada con entrenamiento. Es una simulación controlada, no un modelo clínico validado de errores de medida. Los porcentajes de ausentes son probabilidades de ocultación adicionales, no el porcentaje final exacto de celdas vacías. Ambos modelos reciben los mismos datos deteriorados; las intensidades de ausentes comparten la máscara aleatoria.

Compara cada perturbación con `referencia` por modelo y semilla, usando sobre todo average precision, recall y precisión. En `menos_positivos`, anota `train_positive_rate` y `n_train`: al retirar positivos también disminuye el tamaño. No atribuyas todo el cambio exclusivamente a la proporción de clases. `conditions.json` conserva las condiciones aplicadas.

**Qué escribir:** qué modelo pierde menos rendimiento en cada perturbación y si la caída crece con los ausentes. Esto estudia la calidad del entrenamiento, no nuevas perturbaciones durante el uso del modelo.

## 10. Responder las preguntas de métricas y tiempos

Reutiliza las ejecuciones anteriores. Presenta average precision y ROC-AUC junto con recall, precisión, F1, balanced accuracy y accuracy. `pr_auc` es el área trapezoidal de la curva precisión-recall y `average_precision` es AP: no son exactamente la misma medida. Las métricas de clasificación usan las predicciones nativas; estas baterías no ajustan umbrales.

Una mejora solo en accuracy no demuestra una mejor detección de positivos. Comprueba si el orden de los modelos se mantiene al cambiar de métrica y señala cualquier intercambio entre recall y precisión.

| Tiempo | Interpretación |
| --- | --- |
| `preprocessing_seconds` | Ajustar/transformar entrenamiento y transformar test |
| `training_seconds` | Llamada al entrenamiento final |
| `inference_seconds` | Suma de las llamadas finales a `predict` y `predict_proba` |
| `inference_seconds_per_row` | Esa suma dividida por las filas de test |
| `search_seconds` | Búsqueda interna completa; cero sin ajuste |
| `training_conditions_seconds` | Aplicar perturbaciones al entrenamiento final |
| `duration_seconds` | Flujo total medido hasta la evaluación, incluidas otras fases |

El preprocesamiento no incluye selección de columnas ni perturbaciones, que tienen tiempos propios. La duración no es simplemente la suma de las tres fases principales ni incluye todo el registro final en MLflow. Puedes consultar además inicialización, carga, partición y evaluación en `result.json`.

TabPFN usa un servicio remoto y LightGBM se ejecuta localmente. Los tiempos del cliente de TabPFN incluyen comunicaciones y esperas; parte del trabajo remoto puede ocurrir en la predicción. Presenta los tiempos observados de las llamadas y del flujo completo, sin interpretarlos como cómputo puro en el mismo hardware. Usa el mismo equipo y evita otras cargas intensivas. Estas baterías no miden memoria.

## 11. Obtener medias y diferencias directamente de MLflow

Puedes comparar ejecuciones en la interfaz. Para calcular el resumen del experimento 2 sin descargar archivos, ejecuta este fragmento desde Python en la raíz, sustituyendo el identificador del padre:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experimento = mlflow.get_experiment_by_name("practica_02_tamano_semillas")
padre = "PEGA_AQUI_EL_ID_DEL_PADRE"
datos = mlflow.search_runs(
    experiment_ids=[experimento.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{padre}'",
)
print(datos["status"].value_counts())  # Anota fallos antes de resumir.
datos = datos.loc[datos["status"].eq("FINISHED")].copy()
datos["params.train_size_requested"] = datos["params.train_size_requested"].astype(float)
metricas = ["metrics.average_precision", "metrics.roc_auc", "metrics.recall",
            "metrics.precision", "metrics.preprocessing_seconds",
            "metrics.training_seconds", "metrics.inference_seconds"]
print(datos.groupby(["tags.model", "params.train_size_requested"])[metricas]
      .agg(["mean", "std", "count"]).to_string())

parejas = datos.pivot(
    index=["params.train_size_requested", "tags.split_seed"],
    columns="tags.model", values="metrics.average_precision",
).reindex(columns=["tabpfn", "lightgbm"])
print("Pares incompletos:", parejas.isna().any(axis=1).sum())
diferencias = (parejas["tabpfn"] - parejas["lightgbm"]).dropna()
print(diferencias.groupby(level=0).agg(["mean", "std", "count"]))
```

Para los experimentos 3–6 agrupa por `tags.model` y `tags.variant` y empareja variantes por `tags.split_seed` dentro de cada modelo. Para el primero añade `tags.dataset` a la agrupación y al emparejamiento. Conserva el número de resultados válidos junto a media y desviación. No rellenes fallos con ceros ni sustituyas una semilla por obtener un resultado desfavorable.

## 12. Preparar la entrega

Por cada experimento incluye:

1. La pregunta, la comparación prevista y la métrica principal.
2. El YAML utilizado, el identificador del padre en MLflow y las condiciones de datos, semillas y tamaños.
3. Una tabla o figura por modelo/variante con media, desviación y número de repeticiones.
4. Las diferencias emparejadas y los costes cuando afecten a la conclusión.
5. Una conclusión limitada a lo observado, incidencias, resultados inconclusos y preguntas abiertas.

No necesitas una batería nueva para cada métrica o gráfico. Usa el experimento 1 para comparar problemas y el resto para profundizar en Framingham. Si una diferencia es pequeña o cambia de signo, indícalo en vez de declarar un ganador general.

## Cambios del pipeline que sustentan esta guía

- [batteries.py](./app/batteries.py) ejecuta los YAML con `--config` y registra plan, progreso y resultados en un padre de MLflow. Se ha eliminado la gestión paralela de directorios, bloqueos y reanudación y las opciones de planificación y parada temprana.
- [partitions.py](./app/partitions.py) conserva particiones compartidas y tamaños anidados. [conditions.py](./app/conditions.py) permite los grupos de variables y perturbaciones necesarios.
- [pipeline.py](./app/pipeline.py) admite ejecuciones hijas y conserva tiempos separados, predicciones y condiciones. [tracking.py](./app/tracking.py) expone tamaños y dimensiones como parámetros consultables en MLflow.
- [optimization.py](./app/optimization.py) realiza el ajuste acotado dentro del entrenamiento. [evaluation.py](./app/evaluation.py) registra AP además del resto de métricas.
- [configs/batteries/](./configs/batteries/) contiene una batería por comparación. Solo [datasets.yaml](./configs/batteries/datasets.yaml) incluye otros datasets; el resto usa Framingham y sus tipos de variables declarados.

Las comprobaciones del código se ejecutan con `python -m unittest discover -s tests -v`. Incluyen particiones, ausencia de fuga en preprocesamiento y ajuste, perturbaciones, registro en MLflow y tiempos. El cliente remoto se sustituye por un modelo local en las pruebas; los resultados de tu práctica se obtienen ejecutando las baterías anteriores.

