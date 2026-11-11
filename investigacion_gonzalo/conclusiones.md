# Investigación reproducible de Gonzalo Landeira

153 ejecuciones nuevas completadas. Semillas previstas: [17, 42, 73].

La batería reutiliza carga, preparación, partición, preprocesamiento, modelos y métricas de app. La extensión personal añade muestreo, perturbaciones, búsqueda interna y registro de condiciones. El código compartido permanece en el commit de referencia recogido en estado_git.txt.

## Diseño y controles

Test estratificado del YAML (20% en las configuraciones iniciales). Las fracciones de muestra se aplican solo al entrenamiento. El test se mantiene idéntico para cada dataset y semilla. Imputación y codificación se ajustan solo con entrenamiento. Ajustes de hiperparámetros y umbral usan una validación interna estratificada del 20% del entrenamiento. Tras elegirlos se reajusta con todo el entrenamiento y se evalúa una vez en test. Se conserva el valor de semilla del modelo.

Comparación de referencia: parámetros del YAML y umbral 0.5. El ajuste de parámetros evalúa tres candidatos, incluido el original, y elige por PR AUC de validación. Para LightGBM se varían num_leaves y n_estimators; para TabPFN n_estimators. El umbral se selecciona maximizando F1 de validación sobre una rejilla de 0.05 a 0.95. Son comparaciones exploratorias, con presupuestos acotados.

PR AUC usa el área trapezoidal de app.evaluation; no es Average Precision. Brier evalúa error de las probabilidades (menor es mejor), combinando discriminación y calibración.

## 1. Dataset: ¿se mantienen las conclusiones en los tres problemas?

| dataset | model | roc_auc | pr_auc | f1 | recall | precision | accuracy | balanced_accuracy | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit | lightgbm | 0.7825 | 0.5551 | 0.4737 | 0.367 | 0.668 | 0.8197 | 0.6576 | 0.1345 |
| credit | tabpfn | 0.792 | 0.5694 | 0.4776 | 0.372 | 0.6672 | 0.8201 | 0.6597 | 0.1328 |
| framingham | lightgbm | 0.6391 | 0.249 | 0.129 | 0.0775 | 0.3845 | 0.8404 | 0.5274 | 0.1314 |
| framingham | tabpfn | 0.7175 | 0.3284 | 0.0959 | 0.0517 | 0.7222 | 0.8518 | 0.5235 | 0.1181 |
| telco | lightgbm | 0.8402 | 0.6545 | 0.5972 | 0.5481 | 0.6562 | 0.8039 | 0.7222 | 0.1379 |
| telco | tabpfn | 0.8537 | 0.6753 | 0.5906 | 0.5205 | 0.6829 | 0.8086 | 0.7166 | 0.1318 |

Las métricas se comparan entre modelos dentro de cada dataset. La prevalencia y la dificultad cambian entre datasets, por lo que un PR AUC mayor entre problemas no demuestra por sí solo superioridad.

## Cobertura real de la bateria

| dataset | model | ejecuciones |
| --- | --- | --- |
| credit | lightgbm | 42 |
| credit | tabpfn | 9 |
| framingham | lightgbm | 42 |
| framingham | tabpfn | 9 |
| telco | lightgbm | 42 |
| telco | tabpfn | 9 |

Plan por modelo: 126 ejecuciones. Cada combinacion dataset/variante requiere tres semillas. El archivo cobertura.csv permite identificar condiciones incompletas. La presencia de un modelo en la tabla no implica que haya completado todas las condiciones.

| dataset | seed | lightgbm | tabpfn | delta_tabpfn_menos_lightgbm |
| --- | --- | --- | --- | --- |
| credit | 17 | 0.5555 | 0.5682 | 0.0127 |
| credit | 42 | 0.5503 | 0.5701 | 0.0198 |
| credit | 73 | 0.5594 | 0.57 | 0.0106 |
| framingham | 17 | 0.271 | 0.3476 | 0.0766 |
| framingham | 42 | 0.2076 | 0.2957 | 0.0881 |
| framingham | 73 | 0.2683 | 0.3418 | 0.0736 |
| telco | 17 | 0.6735 | 0.6911 | 0.0175 |
| telco | 42 | 0.6361 | 0.6692 | 0.0331 |
| telco | 73 | 0.6539 | 0.6655 | 0.0116 |

Un delta positivo favorece TabPFN en PR AUC para esa partición; se revisa su consistencia entre semillas.

## 2. Tamaño de muestra

Se comparan 10%, 50% y 100% del entrenamiento; la referencia es el 100%. La selección es estratificada y usa la misma semilla para ambos modelos. Los tamaños absolutos se muestran porque las fracciones no representan el mismo número de filas entre datasets.

| dataset | model | variant | n_train | delta_pr_auc | delta_f1 | delta_recall | delta_precision | delta_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit | lightgbm | medium_50pct | 12000.0 | -0.011 | -0.0124 | -0.0106 | -0.0138 | 0.0018 |
| credit | lightgbm | small_10pct | 2400.0 | -0.053 | -0.0355 | -0.0239 | -0.0611 | 0.0145 |
| credit | tabpfn | medium_50pct | 12000.0 | -0.0026 | -0.0108 | -0.0163 | 0.0123 | 0.0004 |
| credit | tabpfn | small_10pct | 2400.0 | -0.0151 | -0.0111 | -0.0128 | -0.0017 | 0.0028 |
| framingham | lightgbm | medium_50pct | 1696.0 | 0.001 | 0.005 | 0.0078 | -0.0686 | 0.0027 |
| framingham | lightgbm | small_10pct | 339.0 | -0.0218 | 0.0283 | 0.0284 | -0.0764 | 0.0137 |
| framingham | tabpfn | medium_50pct | 1696.0 | -0.0082 | -0.0252 | -0.0129 | -0.0556 | 0.0006 |
| framingham | tabpfn | small_10pct | 339.0 | -0.0406 | -0.0858 | -0.0465 | -0.2778 | 0.0028 |
| telco | lightgbm | medium_50pct | 2817.0 | -0.0223 | -0.0042 | -0.0027 | -0.0064 | 0.0042 |
| telco | lightgbm | small_10pct | 563.0 | -0.0799 | -0.0699 | -0.0615 | -0.0807 | 0.0307 |
| telco | tabpfn | medium_50pct | 2817.0 | -0.0044 | 0.0084 | 0.0143 | -0.0019 | 0.0008 |
| telco | tabpfn | small_10pct | 563.0 | -0.0234 | -0.0235 | -0.0241 | -0.0215 | 0.0049 |

Conclusión sobre el tamaño de muestra
En los tres conjuntos de datos, usar menos datos tendió a reducir el PR AUC de TabPFN. La caída fue pequeña con el 50 % de los datos: entre 0,003 y 0,008 puntos, según el conjunto. Con solo el 10 %, la bajada fue mayor: entre 0,015 y 0,041 puntos. La diferencia más marcada apareció en Framingham.

LightGBM mostró una tendencia parecida, aunque la pérdida con el 10 % fue especialmente grande en crédito y Telco. En Framingham, en cambio, el PR AUC de LightGBM bajó poco. Esto sugiere que el efecto de reducir los datos depende tanto del modelo como del conjunto de datos.

Para TabPFN, el 50 % mantuvo resultados cercanos a los obtenidos con el 100 %. Con el 10 %, el descenso fue más claro, sobre todo en Framingham. Por tanto, en estas pruebas TabPFN parece conservar mejor el PR AUC cuando se reduce la muestra, especialmente en crédito y Telco. Aun así, no se puede concluir que sea mejor en general: solo se probaron tres semillas y observamos variación entre ellas.

Las demás métricas no siempre siguieron la misma tendencia. Por ejemplo, el F1 de LightGBM en Framingham subió ligeramente con el 10 %, aunque su PR AUC bajó. Esto puede ocurrir porque F1 depende del umbral usado para clasificar, mientras que PR AUC evalúa el ordenamiento de las probabilidades a distintos umbrales. Por eso conviene interpretar las métricas juntas y elegir la que mejor responda al objetivo del problema.

TabPFN muestra más resistencia al reducir los datos en estos resultados, pero no se puede afirmar que sea siempre el mejor con pocos datos ni que gane en general.

## 3. Semilla y estabilidad: crédito

Comparamos el PR AUC en las mismas diez semillas para ambos modelos.

| modelo | semillas | PR AUC mínimo | PR AUC máximo | rango (máx.  mín.) |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 10 | 0.5465 | 0.5789 | 0.0324 |
| TabPFN | 10 | 0.5557 | 0.5922 | 0.0365 |

En crédito, LightGBM varió entre 0,5465 y 0,5789, con un rango de 0,0324. TabPFN varió entre 0,5557 y 0,5922, con un rango de 0,0365. Los rangos se solapan ampliamente, así que el PR AUC depende de la semilla y estas pruebas no muestran un modelo consistentemente superior. La variación fue parecida, aunque el rango de TabPFN fue algo mayor. Al haber probado 10 semillas distintas y ver que a penas hay cambio podemos asumir que en ambos casos es bastante estable y no depende muchísimo de la partición.

## 4. Preprocesamiento: minimo frente a avanzado, incluye tambiénn pregunta 5

Comparamos dos niveles en credito, con LightGBM y TabPFN y las mismas tres semillas. El preprocesamiento minimo conserva las 23 columnas numericas tal como vienen; la tabla no contiene valores ausentes, asi que la imputacion no llega a intervenir. El avanzado interpreta SEX, EDUCATION y MARRIAGE como categorias y aplica one-hot, pasando de 23 a 33 columnas.

| Modelo | PR AUC minimo | PR AUC avanzado | Cambio medio | Cambios por semilla (minimo a maximo) | Cambio medio de F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 0.5551 | 0.5543 | -0.00073 | -0.00339 a +0.00292 | -0.00036 |
| tabpfn | 0.5694 | 0.5697 | +0.00022 | -0.00076 a +0.00147 | -0.00323 |

Conclusion: en este dataset no se observa una mejora util por el preprocesamiento avanzado. En LightGBM el PR AUC medio bajo 0.00073; en TabPFN subio 0.00022. Ambos cambios son diminutos y las diferencias cambian de signo segun la semilla. F1 tampoco mejora de forma consistente. Para credito, el preprocesamiento minimo parece suficiente para estos dos modelos y estas particiones.

Esta prueba no demuestra que todo preprocesamiento sea innecesario: credito no tiene ausentes que imputar y solo se probo una transformacion avanzada de tres variables. La conclusion se limita a estas condiciones.


## 6. Ajuste de hiperparámetros: crédito

Para cada modelo comparamos la configuración inicial con tres candidatos, seleccionados por PR AUC en una validación interna. Después se reajustó el modelo seleccionado con todos los datos de entrenamiento y se evaluó en el mismo test. Las tres semillas son emparejadas.

Se probaron tres configuraciones por modelo: LightGBM comparó la inicial (`n_estimators=200`), una con `n_estimators=100` y `num_leaves=15`, y otra con `n_estimators=300` y `num_leaves=63`; TabPFN comparó sus valores predeterminados con `n_estimators=4` y `n_estimators=8`. Se eligió la configuración con mayor PR AUC en la validación interna.

| Modelo | PR AUC inicial | PR AUC ajustado | Cambio medio | Cambio por semilla (mínimo a máximo) | Cambio medio de F1 | Búsqueda media | Tiempo total medio ajustado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lightgbm | 0.5551 | 0.5580 | +0.00290 | -0.00408 a +0.00653 | -0.00204 | 4.2 s | 12.3 s |
| tabpfn | 0.5694 | 0.5695 | +0.00007 | +0.00000 a +0.00022 | -0.00067 | 17.5 s | 26.6 s |

En LightGBM, el ajuste subio el PR AUC medio en 0,00290, pero el cambio fue desigual: mejoro 0,00653 y 0,00624 en dos semillas, y bajo 0,00408 en la tercera. El F1 medio bajo 0,00204. La busqueda tardo 4,2 segundos de media; contando el pipeline completo, cada ejecucion ajustada tardo 12,3 segundos de media.

En TabPFN, el PR AUC medio apenas cambio (+0,00007): en una semilla subio 0,00022 y en las otras dos quedo practicamente igual. El F1 medio bajo 0,00067. La busqueda tardo 17,5 segundos de media y el tiempo total fue 26,6 segundos por ejecucion ajustada.

Conclusion: para credito, la busqueda probada no aporta una mejora clara. LightGBM puede ganar algo en algunas particiones, pero no de forma estable y con una ligera bajada de F1; TabPFN practicamente no mejora. En ambos casos se paga tiempo adicional por probar candidatos. Con este presupuesto de tres candidatos, no parece compensar activar el ajuste si se prioriza tiempo y un resultado reproducible. Los tiempos remotos de TabPFN pueden variar por la red y el servicio; el coste de busqueda es una media de solo tres semillas.

## 7. ¿Cuentan todas las métricas la misma historia?

| dataset | model | accuracy | accuracy_majority | balanced_accuracy | pr_auc | f1 | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| credit | lightgbm | 0.8197 | 0.7788 | 0.6576 | 0.5551 | 0.4737 | 0.367 | 0.668 |
| credit | tabpfn | 0.8201 | 0.7788 | 0.6597 | 0.5694 | 0.4776 | 0.372 | 0.6672 |
| framingham | lightgbm | 0.8404 | 0.8479 | 0.5274 | 0.249 | 0.129 | 0.0775 | 0.3845 |
| framingham | tabpfn | 0.8518 | 0.8479 | 0.5235 | 0.3284 | 0.0959 | 0.0517 | 0.7222 |
| telco | lightgbm | 0.8039 | 0.7346 | 0.7222 | 0.6545 | 0.5972 | 0.5481 | 0.6562 |
| telco | tabpfn | 0.8086 | 0.7346 | 0.7166 | 0.6753 | 0.5906 | 0.5205 | 0.6829 |

accuracy_majority es la exactitud de predecir siempre la clase mayoritaria en ese test. Las diferencias de recall y precision permiten valorar falsos negativos y falsas alarmas. No se decide el mejor modelo por accuracy aislada ni se presupone que PR AUC y ROC AUC ordenen igual.

## 8. Coste de ejecución

| Dataset | Modelo | Repeticiones | Ajuste mediano (s) | Predicción mediana (s) | Predicción (ms/fila) | Tiempo total mediano (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| credit | lightgbm | 13 | 0.781 | 0.027 | 0.004 | 5.586 |
| credit | tabpfn | 13 | 2.452 | 4.186 | 0.698 | 12.007 |
| framingham | lightgbm | 6 | 0.456 | 0.006 | 0.007 | 0.522 |
| framingham | tabpfn | 6 | 2.127 | 0.788 | 0.929 | 3.002 |
| telco | lightgbm | 6 | 0.700 | 0.011 | 0.008 | 0.870 |
| telco | tabpfn | 6 | 2.220 | 1.471 | 1.044 | 3.901 |

La tabla muestra medianas de 13 ejecuciones en crédito y 6 en cada uno de Telco y Framingham; en todos los casos usamos la misma cantidad de filas de prueba al comparar los dos modelos.

TabPFN fue más lento en ajuste y predicción en los tres datasets. La diferencia de predicción es especialmente clara: LightGBM tarda milésimas de segundo por fila, mientras que TabPFN requiere alrededor de 0,7 a 1,0 ms por fila. En este proyecto, TabPFN usa un servicio remoto; sus tiempos incluyen comunicación y espera del servicio, además del cálculo. LightGBM se ejecuta localmente y su predicción es muy rápida.

La demora de TabPFN se repitió en las medianas, pero no todos los llamados tardan igual. En crédito hubo una ejecución de ajuste de 87 segundos, frente a una mediana de 2,45 segundos; ese pico muestra que puede haber esperas excepcionales y que la media de la tabla original exageraba el tiempo habitual. Por tanto, TabPFN suele tardar más en estas mediciones, aunque no siempre tarda lo mismo.

Los tiempos se midieron en este equipo y servicio. El ajuste incluye inicialización; la predicción mide una llamada predict_proba para todo el test. No se midió RAM ni coste monetario.

## 9. Calidad, ruido y desequilibrio

Pregunta: ¿Cómo afectan al rendimiento los valores ausentes, el ruido en las etiquetas y el desequilibrio de clases? Estas pruebas se hicieron con LightGBM; se modificó solo el entrenamiento y se mantuvo limpio el test.

| Dataset | 10% etiquetas cambiadas: cambio PR AUC | Balanceo: cambio PR AUC | Control de igual tamaño: cambio PR AUC | 10% celdas ausentes: cambio PR AUC |
| --- | ---: | ---: | ---: | ---: |
| Crédito | -0,0072 | -0,0124 | -0,0102 | -0,0002 |
| Framingham | +0,0058 | +0,0112 | +0,0013 | +0,0078 |
| Telco | -0,0255 | -0,0153 | -0,0237 | -0,0054 |

Conclusión: cambiar etiquetas perjudicó sobre todo a Telco, mientras que en Framingham el pequeño aumento no demuestra una mejora real. El balanceo aumentó el recall, pero redujo la precisión; por eso no mejora todas las medidas a la vez. En crédito y Telco, el PR AUC bajó tanto al balancear como al reducir el entrenamiento al mismo tamaño sin balancearlo, así que perder datos también influye. El 10% de ausencias aleatorias tuvo un efecto pequeño en PR AUC. Son resultados exploratorios de tres semillas y no cubren todos los tipos de ruido o datos faltantes.

## 10. Preguntas adicionales

**Pregunta 1: ¿Cambiar el umbral de clasificación mejora las predicciones?**

En LightGBM se seleccionó el umbral que maximizaba F1 en validación interna y se comparó con 0,5 en el test. F1 aumentó en promedio 0,067 en crédito, 0,186 en Framingham y 0,022 en Telco; el recall subió, pero la precisión bajó. ROC AUC y PR AUC no cambiaron, porque el umbral modifica las etiquetas predichas, no las probabilidades. Conclusión: el umbral permite priorizar recall o precisión según la necesidad, pero no hace que el modelo ordene mejor los casos.

**Pregunta 2: ¿Qué modelo produce probabilidades con menor error?**

El Brier medio fue menor para TabPFN que para LightGBM en los tres datasets: crédito, 0,1328 frente a 0,1345; Framingham, 0,1181 frente a 0,1314; y Telco, 0,1318 frente a 0,1379. Menor Brier significa menor error cuadrático de las probabilidades. Conclusión: TabPFN tuvo menor Brier en estas pruebas, pero esto por sí solo no demuestra mejor calibración; para afirmarlo harían falta curvas de fiabilidad.

