
# import numpy as np

# from sklearn.base import clone
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import (
#     GridSearchCV,
#     StratifiedKFold,
#     cross_val_predict
# )

# from preprocessing import build_preprocessor


# def optimize_model(
#     model,
#     X_train,
#     y_train,
#     param_grid,
#     one_hot=False
# ):
#     cv = StratifiedKFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=42
#     )

#     pipeline = Pipeline([
#         ("preprocessing", build_preprocessor(one_hot=one_hot)),
#         ("model", model)
#     ])

#     search = GridSearchCV(
#         pipeline,
#         param_grid={
#             f"model__{key}": values
#             for key, values in param_grid.items()
#         },
#         scoring="average_precision",
#         cv=cv,
#         n_jobs=-1,
#         refit=True
#     )

#     search.fit(X_train, y_train)

#     return search.best_estimator_, search.best_params_, search.best_score_


# def optimize_threshold(
#     model,
#     X_train,
#     y_train,
#     min_precision=0.30
# ):
#     cv = StratifiedKFold(
#         n_splits=5,
#         shuffle=True,
#         random_state=42
#     )

#     # Probabilidades out-of-fold: cada observación se predice
#     # con un modelo que no ha sido entrenado con ella.
#     probabilities = cross_val_predict(
#         clone(model),
#         X_train,
#         y_train,
#         cv=cv,
#         method="predict_proba",
#         n_jobs=-1
#     )

#     classes = list(model.classes_)
#     positive_idx = classes.index(1)
#     probabilities = probabilities[:, positive_idx]

#     from sklearn.metrics import precision_score, recall_score

#     thresholds = np.unique(probabilities)
#     candidates = []

#     for threshold in thresholds:
#         predictions = (probabilities >= threshold).astype(int)

#         precision = precision_score(
#             y_train, predictions, zero_division=0
#         )
#         recall = recall_score(
#             y_train, predictions, zero_division=0
#         )

#         if precision >= min_precision:
#             candidates.append((recall, threshold, precision))

#     if not candidates:
#         raise ValueError(
#             "Ningún umbral alcanza la precision mínima solicitada."
#         )

#     recall, threshold, precision = max(
#         candidates, key=lambda item: (item[0], item[2])
#     )

#     return {
#         "threshold": float(threshold),
#         "recall_cv": float(recall),
#         "precision_cv": float(precision)
#     }
