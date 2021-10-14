# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union
import numpy as np
import numbers

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_non_negative, check_array, check_is_fitted
from scipy.optimize import least_squares

from openstf.model.regressors.regressor import OpenstfRegressor


class SigmoidRobustRegressor(BaseEstimator, RegressorMixin):
    """Class for tangent hyperbolic robust regression Models,

    Similar to a perceptron with tanh as activation function.
    The robusness mechanism is an iterative process:
        1. Fits the parameters (with least_square solver)
        2. Estimates MAE
        3. Filters-out outliers with a MAE above a threshold
        4. Refits parameters with inliers

    Parameters
    ----------
    scale: float, default=1.
        Scaling factor to map the prediction from [0, 1] to the original [0, scale].

    max_iter: int, default=5
        The maximum number of iterations for the robusness mechanism.

    lambda_thr: int, default=6
        The multiplier of the threshold for outlier detection.
        Data point for which residuals > lambda_thr * threshold are considered as outliers.

    Attributes
    ----------
    init_coef_: ndarray of shape (n_feature + 1,)
        Initial parameters for the first iteration of least squared optimization

    n_iter_: int
        The actual number of iterations performed.

    intercept_: float
        Constant in tanh function.

    coef_: ndarray of shape (n_feature,)
        Weights assigned to the features.

    params_: ndarray of shape (n_feature + 1,)
        The concatenation of the intercept and the coefficients.

    feature_importance_: ndarray of shape (n_feature,)
        The absolute values of the coefficients.
    """

    def __init__(
        self,
        scale=1.0,
        max_iter=5,
        lambda_thr=6,
        init_intercept: float = 0,
        init_coef: Union[float, np.array] = 0,
        bounds=(-np.inf, np.inf),
    ):
        self.scale = scale
        self.max_iter = max_iter
        self.lambda_thr = lambda_thr
        self.init_intercept = init_intercept
        self.init_coef = init_coef
        self.bounds = bounds

    def _more_tags(self):
        # Poor score because of the robustness mechanism
        return {"requires_positive_y": True, "poor_score": True}

    @staticmethod
    def _scaled_tanh(x, scale, p):
        a = p[0]
        coef = p[1:]
        return scale * (np.tanh(a + x @ coef) - np.tanh(a)) / (1 - np.tanh(a) + 1e-15)

    @staticmethod
    def _check_init_coef(init_coef, X, dtype=None, copy=False):
        """Validate initial coefficient for least squared based methods.

        Parameters
        ----------
        init_coef : {ndarray, Number}, shape (n_features,)
           Input  initial coefficients.
        X : {ndarray, list, sparse matrix}
            Input data.
        dtype : dtype, default=None
           dtype of the validated `init_coef`.
           If None, and the input `init_coef` is an array, the dtype of the
           input is preserved; otherwise an array with the default numpy dtype
           is be allocated.  If `dtype` is not one of `float32`, `float64`,
           `None`, the output will be of dtype `float64`.
        copy : bool, default=False
            If True, a copy of init_coef will be created.

        Returns
        -------
        init_coef : ndarray of shape (n_features,)
           Validated initial coefficients. It is guaranteed to be "C" contiguous.
        """
        n_features = X.shape[1]

        if dtype is not None and dtype not in [np.float32, np.float64]:
            dtype = np.float64

        if isinstance(init_coef, numbers.Number):
            init_coef = np.full(n_features, init_coef, dtype=dtype)
        else:
            if dtype is None:
                dtype = [np.float64, np.float32]
            init_coef = check_array(
                init_coef,
                accept_sparse=False,
                ensure_2d=False,
                dtype=dtype,
                order="C",
                copy=copy,
            )
            if init_coef.ndim != 1:
                raise ValueError("Initial coefficients must be 1D array or scalar")

            if init_coef.shape != (n_features,):
                raise ValueError(
                    "init_coef.shape == {}, but input data has {} features!".format(
                        init_coef.shape, (n_features + 1,)
                    )
                )

        return init_coef

    def fit(self, X, y, **kwargs):

        # Check data
        X, y = self._validate_data(X, y, y_numeric=True)
        init_coef = self._check_init_coef(self.init_coef, X, copy=True)

        # Initialization
        p0 = np.concatenate([[self.init_intercept], init_coef])
        popt = p0 + 0.1
        indexes = np.arange(len(X))

        # Iterative fitting for robustness
        for i in range(self.max_iter):
            Xi = X[indexes]
            yi = y[indexes]

            res = least_squares(
                lambda p: yi / self.scale - self._scaled_tanh(Xi, 1, p),
                p0,
                bounds=self.bounds,
            )

            popt = res.x

            dist = np.sum((p0 - popt) ** 2)

            if dist <= 0.001:
                break
            p0 = popt

            # Filter-out outliers
            yhat = self._scaled_tanh(X, self.scale, popt)
            residual = yhat - y
            threshold = np.median(np.abs(residual - np.median(residual)))
            select = np.abs(residual) <= self.lambda_thr * threshold
            indexes = np.where(select)[0]

        self.n_iter_ = i
        self.intercept_ = popt[0]
        self.coef_ = popt[1:]
        self.params_ = popt

        self.feature_importance_ = np.abs(self.coef_)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"X_{i + 1}" for i in range(X.shape[1])]

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        return self._scaled_tanh(X, self.scale, self.params_)

    @property
    def feature_names(self):
        return self.feature_names_


class PREOLE(SigmoidRobustRegressor):
    """Class for the PREOLE model, a forcast model for wind turbine plants,

    The PREOLE model is a robust tanh regression model to predict generated power according the wind force
    (and eventually other features). The first feature is considered as the wind force.
    Data from under maintenance plants/turbines are filtered-out (low generated power with high wind force).

     See Also
    --------
    SigmoidRobustRegressor : Class for tangent hyperbolic robust regression Models.
    """

    def __init__(self, scale=1.0, max_iter=5, lambda_thr=6):
        super().__init__(scale, max_iter, lambda_thr, init_intercept=-2, init_coef=2)

    def _more_tags(self):
        return {"requires_positive_X": True, "requires_positive_y": True}

    def fit(self, X, y, **kwargs):

        # Check data positivity
        X, y = self._validate_data(X, y, y_numeric=True)
        check_non_negative(X[:, 0], "PREOLE (input first features X[:, 0])")
        check_non_negative(y, "PREOLE (output power y)")

        # Filter under maintenance wind pannels
        select = (y >= 0.001) | (X[:, 0] <= 3)

        return super().fit(X[select], y[select])

    def predict(self, X):
        X = self._validate_data(X)
        check_non_negative(X[:, 0], "PREOLE (input first features X[:, 0])")
        return super().predict(X)


class SigmoidOpenstfRegressor(SigmoidRobustRegressor, OpenstfRegressor):
    gain_importance_name = "total_gain"
    weight_importance_name = "weight"


class PREOLEOpenstfRegressor(PREOLE, OpenstfRegressor):
    gain_importance_name = "total_gain"
    weight_importance_name = "weight"
