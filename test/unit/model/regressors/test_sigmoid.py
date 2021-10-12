import unittest

import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.estimator_checks import check_estimator

from openstf.model.regressors.sigmoid import (
    SigmoidOpenstfRegressor,
    PREOLEOpenstfRegressor,
)
from test.utils import BaseTestCase, TestData

train_input = TestData.load("reference_sets/307-train-data.csv")


class TestPREOLEOpenstfRegressor(BaseTestCase):
    # Test the the mother class because PREOLE perform positivity check on the inputs and the targets
    # These positivity constraints break the check_estimator
    def test_sklearn_compliant(self):
        # Use sklearn build in check, this will raise an exception if some check fails
        # During these tests the fit and predict methods are elaborately tested
        # More info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
        check_estimator(SigmoidOpenstfRegressor())

    def test_value_error_raised(self):
        model = PREOLEOpenstfRegressor()
        # check if Value Error  is raised when X[:, 0] < 0
        with self.assertRaises(ValueError):
            model.fit(-np.ones((10, 3)), np.zeros(10))

        # check if Value Error  is raised when y < 0
        with self.assertRaises(ValueError):
            model.fit(np.ones((10, 3)), -np.ones(10))

        # check if the model is fitted even if X[:, 1:] < 1
        model.fit(np.array([[1, -5], [2, 9]]), np.ones(2))
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

    def test_fit(self):
        X = train_input["windspeed_100m"].to_numpy()[:, np.newaxis]
        y = train_input["load"] + train_input["load"].min()
        y = y / y.max()

        model = PREOLEOpenstfRegressor().fit(X, y)

        # check if the model was fitted (raises NotFittedError when not fitted)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        # check if model is sklearn compatible
        self.assertTrue(isinstance(model, sklearn.base.BaseEstimator))

        # check if model output positive prediction
        self.assertTrue((model.predict(X) >= 0).all())
