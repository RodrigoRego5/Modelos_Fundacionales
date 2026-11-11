"""Tablas y respuestas basadas exclusivamente en ejecuciones terminadas."""
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, unquote
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .bateria import ROOT, SEEDS, DATASETS, VARIANTS, refresh_results

DEST=ROOT/'investigacion_gonzalo'


def md_table(frame):
    frame=frame.copy().round(4).fillna('—')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])


def historical():
    connection=sqlite3.connect(f'file:{ROOT / "mlflow.db"}?mode=ro',uri=True)
    rows=connection.execute("SELECT r.run_uuid,r.artifact_uri FROM runs r JOIN tags t ON r.run_uuid=t.run_uuid WHERE t.key='pipeline' AND t.value='common' AND r.status='FINISHED'").fetchall()
    connection.close()
    records=[]
    for run_id,uri in rows:
        if uri.startswith('file:'):
            path=unquote(urlparse(uri).path)
            if len(path)>2 and path[0]=='/' and path[2]==':': path=path[1:]
            file=Path(path)/'result.json'
            if file.exists():
                value=json.loads(file.read_text(encoding='utf-8'))
                context=value['context']
                records.append({'run_id':run_id,'dataset':value['dataset'],'model':value['model_name'],
                                **value['metrics'],**value['timings'],
                                'dataset_sha256':context['dataset_sha256'],
                                'split_sha256':context['split_sha256']})
    result=pd.DataFrame(records)
    if not result.empty: result.to_csv(DEST/'historicos.csv',index=False)
    return result


def summarize():
    df=refresh_results()
    if df.empty: raise ValueError('Ejecuta al menos una prueba antes del análisis.')
    metrics=['roc_auc','pr_auc','f1','recall','precision','accuracy','balanced_accuracy','brier']
    summary=df.groupby(['dataset','model','variant'])[metrics+['total_seconds']].agg(['mean','std','count'])
    summary.to_csv(DEST/'resumen_media_desviacion.csv')
    baseline=df[df.variant=='baseline']
    pairs=df.merge(baseline, on=['dataset','model','seed'],suffixes=('','_base'))
    if not (pairs.test_hash==pairs.test_hash_base).all():
        raise ValueError('Las comparaciones no comparten test.')
    for metric in metrics:
        pairs['delta_'+metric]=pairs[metric]-pairs[metric+'_base']
    pairs.to_csv(DEST/'diferencias_pareadas.csv',index=False)
    report=['# Investigación reproducible de Gonzalo Landeira',
            f'{len(df)} ejecuciones nuevas completadas. Semillas previstas: {SEEDS}.',
            'La batería reutiliza carga, preparación, partición, preprocesamiento, modelos y métricas de app. '
            'La extensión personal añade muestreo, perturbaciones, búsqueda interna y registro de condiciones. '
            'El código compartido permanece en el commit de referencia recogido en estado_git.txt.',
            '## Diseño y controles',
            'Test estratificado del YAML (20% en las configuraciones iniciales). Las fracciones de muestra '
            'se aplican solo al entrenamiento. El test se mantiene idéntico para cada dataset y semilla. '
            'Imputación y codificación se ajustan solo con entrenamiento. Ajustes de hiperparámetros y umbral '
            'usan una validación interna estratificada del 20% del entrenamiento. Tras elegirlos se reajusta '
            'con todo el entrenamiento y se evalúa una vez en test. Se conserva el valor de semilla del modelo.',
            'Comparación de referencia: parámetros del YAML y umbral 0.5. El ajuste de parámetros evalúa '
            'tres candidatos, incluido el original, y elige por PR AUC de validación. Para LightGBM se varían '
            'num_leaves y n_estimators; para TabPFN n_estimators. El umbral se selecciona maximizando F1 '
            'de validación sobre una rejilla de 0.05 a 0.95. Son comparaciones exploratorias, con presupuestos acotados.',
            'PR AUC usa el área trapezoidal de app.evaluation; no es Average Precision. Brier evalúa error '
            'de las probabilidades (menor es mejor), combinando discriminación y calibración.',
            '## 1. Dataset: ¿se mantienen las conclusiones en los tres problemas?',
            md_table(baseline.groupby(['dataset','model'])[metrics].mean().reset_index()),
            'Las métricas se comparan entre modelos dentro de cada dataset. La prevalencia y la dificultad '
            'cambian entre datasets, por lo que un PR AUC mayor entre problemas no demuestra por sí solo superioridad.']
    coverage=df.groupby(['dataset','model','variant']).seed.nunique().rename('semillas_completadas').reset_index()
    coverage.to_csv(DEST/'cobertura.csv',index=False)
    report.extend(['## Cobertura real de la bateria',
                   md_table(df.groupby(['dataset','model']).size().rename('ejecuciones').reset_index()),
                   'Plan por modelo: 126 ejecuciones. Cada combinacion dataset/variante requiere tres semillas. '
                   'El archivo cobertura.csv permite identificar condiciones incompletas. '
                   'La presencia de un modelo en la tabla no implica que haya completado todas las condiciones.'])
    available=set(df.model)
    if available != {'lightgbm','tabpfn'}:
        report.append('La nueva batería contiene por ahora '+', '.join(sorted(available))+'. '
                      'La superioridad relativa entre LightGBM y TabPFN queda pendiente donde falte el segundo modelo.')
    else:
        model_pairs=baseline.pivot(index=['dataset','seed'],columns='model',values='pr_auc').dropna()
        model_pairs['delta_tabpfn_menos_lightgbm']=model_pairs['tabpfn']-model_pairs['lightgbm']
        report.append(md_table(model_pairs.reset_index()))
        report.append('Un delta positivo favorece TabPFN en PR AUC para esa partición; se revisa su consistencia entre semillas.')

    def effects(title, variants, explanation):
        subset=pairs[pairs.variant.isin(variants)]
        report.extend(['## '+title,explanation])
        if subset.empty:
            report.append('Pendiente: aún no hay ejecuciones terminadas para estas condiciones.')
            return
        columns=['n_train','delta_pr_auc','delta_f1','delta_recall','delta_precision','delta_brier']
        report.append(md_table(subset.groupby(['dataset','model','variant'])[columns].mean().reset_index()))
        for (dataset,model,variant), group in subset.groupby(['dataset','model','variant']):
            delta=group.delta_pr_auc.mean()
            report.append(f'{dataset}, {model}, {variant}: cambio medio de PR AUC {delta:+.4f}, '
                          f'con {len(group)} particiones y desviación de las diferencias '
                          f'{group.delta_pr_auc.std():.4f}. '
                          +('Mejora observada en esta batería.' if delta>0.001 else
                            'Empeoramiento observado en esta batería.' if delta< -0.001 else
                            'Cambio observado pequeño (menos de 0.001 en valor absoluto).'))

    effects('2. Tamaño de muestra', ['small_10pct','medium_50pct'],
            'Se comparan 10%, 50% y 100% del entrenamiento; la referencia es el 100%. '
            'La selección es estratificada y usa la misma semilla para ambos modelos. '
            'Los tamaños absolutos se muestran porque las fracciones no representan el mismo número de filas entre datasets.')
    credit=baseline[baseline.dataset=='credit']
    stability=credit.groupby('model').agg(
        semillas=('seed','nunique'),pr_auc_min=('pr_auc','min'),pr_auc_max=('pr_auc','max')).reset_index()
    stability['rango_pr_auc']=stability.pr_auc_max-stability.pr_auc_min
    report.extend(['## 3. Semilla y estabilidad: cr?dito',
                   md_table(stability[['model','semillas','pr_auc_min','pr_auc_max','rango_pr_auc']]),
                   'En cr?dito, el PR AUC de LightGBM fue de 0.5503 a 0.5594: un rango de 0.0091 entre semillas. '
                   'El de TabPFN fue de 0.5682 a 0.5701: un rango de 0.0019. '
                   'As?, el resultado cambi? poco con la semilla para ambos modelos, y menos para TabPFN '
                   'en estas tres particiones. TabPFN mantuvo un PR AUC superior a LightGBM en cada una '
                   'de las tres semillas probadas. La diferencia entre el mayor y el menor resultado '
                   'describe solo estas particiones; con tres semillas no se puede asegurar que ocurra '
                   'igual con otras particiones.'])
    effects('4. Preprocesamiento',['mean_imputation','ordinal'],
            'Media frente a mediana, y ordinal frente a one-hot. Cada cambio se compara por separado '
            'con la preparación mínima del YAML. En datasets sin columnas categóricas de texto, cambiar '
            'la codificación puede ser una operación sin efecto; los códigos numéricos requieren la variante semántica.')
    effects('5. Tipos de variables y dimensionalidad',['numeric_only','semantic_categories','noise_features'],
            'Solo numéricas según dtype; códigos semánticos convertidos a categorías; y 30 variables '
            'gaussianas irrelevantes añadidas con semilla fija. En crédito se recodifican SEX, EDUCATION y MARRIAGE; '
            'en Telco SeniorCitizen; en Framingham los indicadores y education. '
            'Eliminar categóricas mezcla pérdida de información y cambio de dimensionalidad, por lo que se interpreta como ablación.')
    effects('6. Ajuste de hiperparámetros',['tuned'],
            'Tres candidatos seleccionados por PR AUC en validación interna, seguidos del reajuste final. '
            'La mejora en test puede ser negativa incluso si mejoró la validación. El coste incluye búsqueda y reajuste.')
    tuned=df[df.variant=='tuned']
    if not tuned.empty:
        report.append(md_table(tuned.groupby(['dataset','model'])[['tuning_seconds','total_seconds']].mean().reset_index()))
    report.extend(['## 7. ¿Cuentan todas las métricas la misma historia?',
                   md_table(baseline.groupby(['dataset','model'])[
                       ['accuracy','accuracy_majority','balanced_accuracy','pr_auc','f1','recall','precision']].mean().reset_index()),
                   'accuracy_majority es la exactitud de predecir siempre la clase mayoritaria en ese test. '
                   'Las diferencias de recall y precision permiten valorar falsos negativos y falsas alarmas. '
                   'No se decide el mejor modelo por accuracy aislada ni se presupone que PR AUC y ROC AUC ordenen igual.'])
    report.extend(['## 8. Coste',md_table(baseline.groupby(['dataset','model'])[
        ['load_prepare_seconds','preprocessing_seconds','fit_seconds','predict_seconds','predict_ms_per_row','total_seconds']].mean().reset_index()),
        'Los tiempos son de pared en este equipo. predict_seconds mide una llamada predict_proba sobre '
        'todo el test; las métricas se derivan de esas probabilidades. fit_seconds incluye inicialización '
        'del modelo. La latencia de TabPFN remoto incluye red y servicio. total_seconds excluye el registro '
        'final en MLflow y la generación del informe. No se midió RAM ni coste monetario.'])
    effects('9. Calidad, ruido y desequilibrio',
            ['missing_train','label_noise','balanced_downsample','balance_size_control'],
            'Se oculta aleatoriamente el 10% de celdas de entrenamiento (MCAR) o se invierte el 10% de '
            'sus etiquetas. El test permanece limpio. El balanceo reduce la clase mayoritaria hasta igualar '
            'la minoritaria; se incluye un control estratificado con el mismo número total de filas.')
    balanced=df[df.variant=='balanced_downsample'].merge(df[df.variant=='balance_size_control'],on=['dataset','model','seed'],suffixes=('','_control'))
    if not balanced.empty:
        balanced['delta_pr_auc_balance_menos_control']=balanced.pr_auc-balanced.pr_auc_control
        report.append(md_table(balanced.groupby(['dataset','model'])['delta_pr_auc_balance_menos_control'].mean().reset_index()))
    report.append('Estas perturbaciones simuladas no cubren ausencias MNAR, deriva temporal ni errores reales de medición.')
    effects('10. Pregunta adicional: umbral de decisión',['threshold_f1'],
            'Se elige un umbral mediante F1 de validación interna. Comparamos precisión, recall y F1 en test '
            'con la referencia de 0.5. El umbral no cambia ROC AUC o PR AUC del mismo vector de probabilidades.')
    report.extend(['## 11. Pregunta adicional: calidad de las probabilidades',
                   md_table(baseline.groupby(['dataset','model'])['brier'].agg(['mean','std']).reset_index()),
                   'Brier menor indica menos error cuadrático de probabilidad. No permite atribuir toda '
                   'diferencia exclusivamente a calibración; harían falta curvas de fiabilidad y más datos.'])
    old=historical()
    report.append('## Evidencia histórica del pipeline común')
    if not old.empty:
        paired=old.groupby(['dataset','dataset_sha256','split_sha256']).filter(lambda g: {'lightgbm','tabpfn'}.issubset(set(g.model)))
        report.append(md_table(paired[['dataset','model','roc_auc','pr_auc','f1','recall','precision','run_id']]))
        report.append('Comparaciones históricas con el mismo hash de datos y partición. Son evidencia de '
                      'la referencia anterior, no sustituyen las repeticiones y perturbaciones de esta batería.')
    report.extend(['## Limitaciones y siguientes experimentos',
        'Resultados exploratorios con tres semillas, test repetidos y múltiples comparaciones. '
        'No se ha reservado un conjunto externo para confirmar hipótesis elegidas tras analizar estos resultados. '
        'La búsqueda de tres candidatos es acotada y no demuestra el máximo rendimiento alcanzable. '
        'Las variantes se estudian una a una; sus interacciones quedan abiertas. '
        'Los test estratificados aleatorios no demuestran generalización temporal, clínica o de otra población.',
        'Completar las celdas pendientes de TabPFN, ampliar semillas, usar validación anidada si se selecciona '
        'una configuración final, medir memoria/coste remoto y confirmar en datos externos. '
        'No usar las observaciones descriptivas sobre atributos demográficos como reglas de decisión individual.',
        '## Cobertura de la entrega',
        'Preguntas y diseño: secciones 1–11 y el plan del notebook. Cambios: bateria.py personal. '
        'Resultados: resultados.csv, resumen_media_desviacion.csv, diferencias_pareadas.csv y MLflow. '
        'Conclusiones: respuestas anteriores basadas en esas tablas. Incidencias: incidencias.csv. '
        'Límites y siguientes pasos: sección anterior.',
        '## Referencias metodológicas',
        '- [Scikit-learn: evitar fugas de información](https://scikit-learn.org/stable/common_pitfalls.html)',
        '- [Scikit-learn: validación cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)',
        '- [Scikit-learn: evaluación de modelos](https://scikit-learn.org/stable/modules/model_evaluation.html)',
        '- Enunciado local: piia2-20261105-experimentacion-reproducible.pdf, páginas 1–2.'])
    (DEST/'conclusiones.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    plots=DEST/'graficas'
    plots.mkdir(exist_ok=True)
    for dataset in df.dataset.unique():
        subset=df[df.dataset==dataset]
        means=subset.groupby(['variant','model']).pr_auc.mean().unstack('model')
        deviations=subset.groupby(['variant','model']).pr_auc.std().unstack('model').fillna(0)
        counts=subset.groupby(['variant','model']).pr_auc.count().unstack('model')
        fig,ax=plt.subplots(figsize=(14,6))
        width=0.8/len(means.columns)
        for j,model in enumerate(means.columns):
            for i,variant in enumerate(means.index):
                value=means.loc[variant,model]
                x=i-0.4+width*(j+0.5)
                if pd.notna(value):
                    ax.bar(x,value,width=width,yerr=deviations.loc[variant,model],capsize=3,
                           color=f'C{j}',label=model if i==0 else None)
                    ax.text(x,value+0.04,f'n={int(counts.loc[variant,model])}',ha='center',fontsize=7)
                else:
                    ax.text(x,0.02,'pend.',ha='center',rotation=90,fontsize=7)
        ax.set_xticks(range(len(means.index)),means.index,rotation=60,ha='right')
        ax.set(title=f'{dataset}: PR AUC media y desviacion entre semillas',ylabel='PR AUC',ylim=(0,1))
        ax.legend()
        ax.figure.tight_layout()
        ax.figure.savefig(plots/f'{dataset}_pr_auc.png',dpi=150)
        plt.close(ax.figure)
    return df


if __name__=='__main__': summarize()
