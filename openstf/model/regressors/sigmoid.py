import numpy as np
import pandas as pd
import numbers

from sklearn.utils.validation import check_non_negative, check_array, check_is_fitted
from scipy.optimize import least_squares

from openstf.model.regressors.regressor import OpenstfRegressor


class TanhOpenstfRegressor(OpenstfRegressor):
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

    Attributes
    ----------
    init_params_: ndarray of shape (n_feature + 1,)
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

    gain_importance_name = "total_gain"
    weight_importance_name = "weight"

    def __init__(self, scale=1.0, max_iter=5):
        self.scale = scale
        self.max_iter = max_iter

    def _more_tags(self):
        return {"requires_positive_y": True, "poor_score": True}

    @staticmethod
    def _scaled_tanh(x, scale, p):
        a = p[0]
        coef = p[1:]
        return scale * (np.tanh(a + x @ coef) - np.tanh(a)) / (1 - np.tanh(a) + 1e-15)

    @staticmethod
    def _check_init_params(init_params, X, dtype=None, copy=False):
        """Validate initial parameters for least squared based methods.
        Note that passing init_params=None will output an array of zeros.
        Therefore, in some cases, you may want to protect the call with:
        if init_params is not None:
            init_params = _check_init_params(...)

        Parameters
        ----------
        init_params : {ndarray, Number or None}, shape (n_samples,)
           Input  initial parameters.
        X : {ndarray, list, sparse matrix}
            Input data.
        dtype : dtype, default=None
           dtype of the validated `init_params`.
           If None, and the input `init_params` is an array, the dtype of the
           input is preserved; otherwise an array with the default numpy dtype
           is be allocated.  If `dtype` is not one of `float32`, `float64`,
           `None`, the output will be of dtype `float64`.
        copy : bool, default=False
            If True, a copy of init_params will be created.

        Returns
        -------
        init_params : ndarray of shape (n_samples,)
           Validated initial parameters. It is guaranteed to be "C" contiguous.
        """
        n_samples = X.shape[1]

        if dtype is not None and dtype not in [np.float32, np.float64]:
            dtype = np.float64

        if init_params is None:
            init_params = np.zeros(n_samples + 1, dtype=dtype)
        elif isinstance(init_params, numbers.Number):
            init_params = np.full(n_samples + 1, init_params, dtype=dtype)
        else:
            if dtype is None:
                dtype = [np.float64, np.float32]
            init_params = check_array(
                init_params,
                accept_sparse=False,
                ensure_2d=False,
                dtype=dtype,
                order="C",
                copy=copy,
            )
            if init_params.ndim != 1:
                raise ValueError("Initial parameters must be 1D array or scalar")

            if init_params.shape != (n_samples + 1,):
                raise ValueError(
                    "init_params.shape == {}, expected {}!".format(
                        init_params.shape, (n_samples + 1,)
                    )
                )

        return init_params

    def fit(self, X, y, init_params=None, bounds=(-np.inf, np.inf), **kwargs):

        # Check data
        X, y = self._validate_data(X, y, y_numeric=True)
        init_params = self._check_init_params(init_params, X)

        # Initialization
        p0 = init_params
        popt = p0 + 0.1
        indexes = np.arange(len(X))

        # Iterative fitting for robustness
        for i in range(self.max_iter):
            Xi = X[indexes]
            yi = y[indexes]

            res = least_squares(
                lambda p: yi / self.scale - self._scaled_tanh(Xi, 1, p),
                p0,
                bounds=bounds,
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
            select = np.abs(residual) <= 6 * threshold
            indexes = np.where(select)[0]

        self.init_params_ = init_params
        self.n_iter_ = i
        self.intercept_ = popt[0]
        self.coef_ = popt[1:]
        self.params_ = popt

        self.feature_importance_ = np.abs(self.coef_)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        return self._scaled_tanh(X, self.scale, self.params_)


class PREOLEOpenstfRegressor(TanhOpenstfRegressor):
    """Class for the PREOLE model, a forcast model for wind turbine plants,

    The PREOLE model is a robust tanh regression model to predict generated power according the wind force
    (and eventually other features). The first feature is considered as the wind force.
    Data from under maintenance plants/turbines are filtered-out (low generated power with high wind force).

     See Also
    --------
    TanhOpenstfRegressor : Class for tangent hyperbolic robust regression Models.
    """

    def __init__(self, scale=1.0, max_iter=5):
        super().__init__(scale, max_iter)

    def _more_tags(self):
        return {"requires_positive_X": True, "requires_positive_y": True}

    def fit(self, X, y, **kwargs):

        # Check data positivity
        X, y = self._validate_data(X, y, y_numeric=True)
        check_non_negative(X[:, 0], "PREOLE (input first features X[:, 0])")
        check_non_negative(y, "PREOLE (output power y)")

        # Filter under maintenance wind pannels
        select = (y >= 0.001) | (X[:, 0] <= 3)

        # Bounds and init_params for LS solver
        n_feat = X.shape[1]
        bounds = (
            [-10, 0] + (n_feat - 1) * [-np.inf],
            [0, 10] + (n_feat - 1) * [+np.inf],
        )
        init_params = [-2.0, 2.0] + (n_feat - 1) * [0.0]

        return super().fit(X[select], y[select], init_params=init_params, bounds=bounds)

    def predict(self, X):
        X = self._validate_data(X)
        check_non_negative(X[:, 0], "PREOLE (input first features X[:, 0])")
        return super().predict(X)
