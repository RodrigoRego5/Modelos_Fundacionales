"""Batería personal que reutiliza app, sin modificar el pipeline común."""
import hashlib
import json
import os
import platform
import re
import subprocess
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split

from app.config import load_config
from app.datasets import load_data, prepare_data, split_data, resolve_data_path
from app.preprocessing import build_preprocessing
from app.models import build_model
from app.evaluation import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'investigacion_gonzalo'
EXPERIMENT = 'investigacion_gonzalo_20261105'
SEEDS = [17, 42, 73]
DATASETS = ['credit', 'telco', 'framingham']
VARIANTS = ['baseline', 'small_10pct', 'medium_50pct', 'mean_imputation',
            'ordinal', 'numeric_only', 'semantic_categories', 'noise_features',
            'missing_train', 'label_noise', 'balanced_downsample', 'balance_size_control', 'tuned', 'threshold_f1']
VERSIONS = {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'mlflow', 'tabpfn-client']}
VERSIONS['python'] = platform.python_version()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def config_for(dataset, seed):
    config = load_config(ROOT / 'configs' / f'{dataset}.yaml')
    config['split']['random_state'] = seed
    return config


def scores(y, probabilities, threshold=0.5):
    prediction = (probabilities >= threshold).astype(int)
    return {**binary_metrics(y, prediction, probabilities),
            'accuracy': float(accuracy_score(y, prediction)),
            'brier': float(brier_score_loss(y, probabilities))}


def fit_predict(config, model_name, X, y, test):
    start = perf_counter()
    prep = build_preprocessing(config)
    train_matrix = prep.fit_transform(X)
    test_matrix = prep.transform(test)
    prep_seconds = perf_counter() - start
    start = perf_counter()
    model = build_model(model_name, config)
    model.fit(train_matrix, y)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probability = model.predict_proba(test_matrix)[:, list(model.classes_).index(1)]
    predict_seconds = perf_counter() - start
    return probability, {'preprocessing_seconds': prep_seconds,
                         'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                         'predict_ms_per_row': predict_seconds * 1000 / len(test),
                         'n_features': train_matrix.shape[1]}


def prepare_variant(config, variant):
    start = perf_counter()
    frame = load_data(config)
    X, y = prepare_data(frame, config)
    train, test, labels, target = split_data(X, y, config)
    train, test, labels = train.copy(), test.copy(), labels.copy()
    original_train_ids = train.index.tolist()
    seed = config['split']['random_state']
    rng = np.random.default_rng(seed)
    if variant in ['small_10pct', 'medium_50pct']:
        fraction = 0.1 if variant == 'small_10pct' else 0.5
        train, _, labels, _ = train_test_split(train, labels, train_size=fraction,
                                              stratify=labels, random_state=seed)
    if variant in ['balanced_downsample', 'balance_size_control']:
        minority_count = int(labels.value_counts().min())
        if variant == 'balanced_downsample':
            ids = np.concatenate([rng.choice(labels.index[labels == label], minority_count, replace=False)
                                  for label in [0, 1]])
            train, labels = train.loc[ids].copy(), labels.loc[ids].copy()
        elif 2 * minority_count < len(labels):
            train, _, labels, _ = train_test_split(train, labels, train_size=2 * minority_count,
                                                  stratify=labels, random_state=seed)
    if variant == 'mean_imputation':
        config['preprocessing']['numerical_missing'] = 'mean'
    if variant == 'ordinal':
        config['preprocessing']['categorical_encoding'] = 'ordinal'
    if variant == 'numeric_only':
        columns = train.select_dtypes(include='number').columns
        train, test = train[columns].copy(), test[columns].copy()
    if variant == 'semantic_categories':
        columns = {'credit': ['SEX', 'EDUCATION', 'MARRIAGE'],
                   'framingham': ['male', 'education', 'currentSmoker', 'BPMeds',
                                  'prevalentStroke', 'prevalentHyp', 'diabetes'],
                   'telco_churn': ['SeniorCitizen']}[config['name']]
        for col in columns:
            if col in train.columns:
                for part in [train, test]:
                    part[col] = part[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
    if variant == 'noise_features':
        for i in range(30):
            train[f'noise_{i}'] = rng.normal(size=len(train))
            test[f'noise_{i}'] = rng.normal(size=len(test))
    if variant == 'missing_train':
        train = train.mask(rng.random(train.shape) < 0.1)
    if variant == 'label_noise':
        flip = rng.choice(len(labels), size=int(0.1 * len(labels)), replace=False)
        labels.iloc[flip] = 1 - labels.iloc[flip]
    context = {'n_train': len(train), 'n_test': len(test), 'raw_features': train.shape[1],
               'train_positive_rate': float(labels.mean()), 'test_positive_rate': float(target.mean()),
               'train_missing_rate': float(train.isna().mean().mean()),
               'train_ids': train.index.tolist(), 'test_ids': test.index.tolist(),
               'original_train_ids': original_train_ids,
               'test_hash': digest(test.index.tolist()), 'train_hash': digest(train.index.tolist()),
               'load_prepare_seconds': perf_counter() - start}
    if set(train.index) & set(test.index):
        raise ValueError('Solapamiento entre entrenamiento y test')
    return train, test, labels, target, context


def execute(dataset, model_name, seed, variant):
    config = config_for(dataset, seed)
    identity = {'dataset': dataset, 'model': model_name, 'seed': seed, 'variant': variant,
                'config': config, 'versions': VERSIONS,
                'dataset_sha256': hashlib.sha256(resolve_data_path(config).read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'app_sha256': digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((ROOT / 'app').glob('*.py'))})}
    key = digest(identity)
    folder = OUTPUT / key
    completed = folder / 'result.json'
    if completed.exists():
        return json.loads(completed.read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///' + (ROOT / 'mlflow.db').as_posix())
    mlflow.set_experiment(EXPERIMENT)
    result = {'key': key, 'dataset': dataset, 'model': model_name, 'seed': seed, 'variant': variant}
    with mlflow.start_run(run_name=f'gonzalo_{dataset}_{model_name}_{variant}_s{seed}') as run:
        result['run_id'] = run.info.run_id
        mlflow.set_tags({'participant': 'Gonzalo Landeira', 'model': model_name,
                         'dataset': dataset, 'variant': variant, 'research_key': key})
        mlflow.log_dict(identity, 'reproducibilidad.json')
        mlflow.log_artifact(str(Path(__file__)), 'codigo')
        mlflow.log_dict(config, 'config_original.json')
        try:
            start = perf_counter()
            train, test, labels, target, context = prepare_variant(config, variant)
            threshold = 0.5
            tuning_seconds = 0.0
            selections = []
            if variant in ['tuned', 'threshold_f1']:
                search_start = perf_counter()
                inner, valid, y_inner, y_valid = train_test_split(
                    train, labels, test_size=0.2, stratify=labels, random_state=seed)
                candidates = [{}]
                if variant == 'tuned':
                    candidates += ([{'num_leaves': 15, 'n_estimators': 100},
                                    {'num_leaves': 63, 'n_estimators': 300}]
                                   if model_name == 'lightgbm' else [{'n_estimators': 4}, {'n_estimators': 8}])
                best_score = -1
                for params in candidates:
                    trial = deepcopy(config)
                    trial.setdefault('models', {}).setdefault(model_name, {}).update(params)
                    prob, _ = fit_predict(trial, model_name, inner, y_inner, valid)
                    score = scores(y_valid, prob)['pr_auc']
                    selections.append({'params': params, 'validation_pr_auc': score})
                    if score > best_score:
                        best_score, best_config, best_prob = score, trial, prob
                config = best_config
                if variant == 'threshold_f1':
                    thresholds = np.linspace(0.05, 0.95, 19)
                    values = [f1_score(y_valid, best_prob >= t, zero_division=0) for t in thresholds]
                    threshold = float(thresholds[int(np.argmax(values))])
                tuning_seconds = perf_counter() - search_start
                mlflow.log_dict({'train': inner.index.tolist(), 'validation': valid.index.tolist(),
                                 'candidates': selections, 'threshold': threshold}, 'selection.json')
            probabilities, costs = fit_predict(config, model_name, train, labels, test)
            metrics = scores(target, probabilities, threshold)
            result.update(status='FINISHED', **metrics, **costs, threshold=threshold,
                          tuning_seconds=tuning_seconds, total_seconds=perf_counter()-start)
            result.update({k: v for k, v in context.items() if not k.endswith('_ids')})
            result['dataset_sha256'] = identity['dataset_sha256']
            result['runner_sha256'] = identity['runner_sha256']
            result['selected_params'] = json.dumps(config['models'].get(model_name, {}), sort_keys=True)
            mlflow.log_dict(context, 'particion.json')
            mlflow.log_dict(config, 'config_efectiva.json')
            mlflow.log_params({'seed': seed, 'variant': variant, 'threshold': threshold,
                              'n_train': len(train), 'n_test': len(test),
                              'model_params': result['selected_params']})
            mlflow.log_metrics({k: float(v) for k,v in result.items()
                                if isinstance(v,(int,float)) and k != 'seed'})
            predictions = pd.DataFrame({'row_id': test.index, 'y_true': target.values,
                                        'probability': probabilities, 'prediction': probabilities >= threshold})
            predictions.to_csv(folder/'predicciones.csv', index=False)
            mlflow.log_artifact(str(folder/'predicciones.csv'))
            result['accuracy_majority'] = float(max(target.mean(), 1-target.mean()))
            mlflow.log_metric('accuracy_majority', result['accuracy_majority'])
            mlflow.log_dict(result, 'resultado.json')
            temporary = folder/'result.tmp'
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(completed)
            return result
        except Exception as error:
            message = re.sub(r'tabpfn_sk_[\w.-]+', '[TOKEN]', str(error))[:1500]
            result.update(status='FAILED', error_type=type(error).__name__, error=message)
            (folder/'error.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            mlflow.log_dict(result, 'incidencia.json')
            mlflow.end_run(status='FAILED')
            return result


def run_battery(models=('lightgbm',), datasets=DATASETS, seeds=SEEDS, variants=VARIANTS):
    if 'tabpfn' in models and not (os.getenv('TABPFN_TOKEN') or os.getenv('TABPFN_API_KEY')):
        raise ValueError('Introduce TABPFN_TOKEN en la celda oculta antes de lanzar la batería remota.')
    rows = []
    for dataset in datasets:
        for seed in seeds:
            for variant in variants:
                for model in models:
                    row = execute(dataset, model, seed, variant)
                    rows.append(row)
                    print(dataset, seed, variant, model, row['status'], flush=True)
    refresh_results()
    return pd.DataFrame(rows)


def refresh_results():
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT.glob('*/result.json')]
    errors = [json.loads(p.read_text(encoding='utf-8')) for p in OUTPUT.glob('*/error.json')
              if not (p.parent/'result.json').exists()]
    current_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = [r for r in rows if r.get('runner_sha256') == current_hash]
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(['dataset','model','variant','seed'])
        table.to_csv(ROOT/'investigacion_gonzalo'/'resultados.csv', index=False)
    pd.DataFrame(errors).to_csv(ROOT/'investigacion_gonzalo'/'incidencias.csv',index=False)
    return table


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['lightgbm','tabpfn'], default=['lightgbm'])
    args=parser.parse_args()
    run_battery(models=args.models)
