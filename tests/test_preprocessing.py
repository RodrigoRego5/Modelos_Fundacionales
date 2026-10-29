"""Verificación de los tres preprocesadores con los datos originales."""
import unittest

import numpy as np

from src.data import load_dataset
from src.evaluation import evaluate_model
from src.models import build_lightgbm


class PreprocessingTests(unittest.TestCase):
    def test_datasets(self):
        for dataset in ("framingham", "telco", "credit"):
            for one_hot in (True, False):
                with self.subTest(dataset=dataset, one_hot=one_hot):
                    train, test, y_train, y_test = load_dataset(dataset=dataset, one_hot=one_hot)
                    self.assertTrue(np.isfinite(train).all())
                    self.assertTrue(np.isfinite(test).all())
                    self.assertEqual(train.shape[1], test.shape[1])
                    model = build_lightgbm({"n_estimators": 3, "n_jobs": 2})
                    model.fit(train, y_train)
                    self.assertTrue(all(np.isfinite(v) for v in evaluate_model(model, test, y_test).values()))


if __name__ == "__main__":
    unittest.main()
