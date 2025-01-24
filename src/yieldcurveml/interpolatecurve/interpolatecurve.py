import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Literal, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    max_error
)
from ..utils.utils import swap_cashflows_matrix
from ..utils.kernels import generate_kernel
from ..stripcurve.bootstrapcurve import RateCurveBootstrapper
from ..stripcurve.stripcurve import CurveStripper
from tabulate import tabulate
from scipy.optimize import newton
from scipy.interpolate import interp1d
from ..utils.datastructures import YieldsContainer, CurveRates, RegressionDiagnostics


class CurveInterpolator(RateCurveBootstrapper, CurveStripper):
    """Yield curve interpolator.
    
    Parameters
    ----------
    estimator : sklearn estimator, default=None
        Scikit-learn estimator to use for fitting. If None, uses Ridge.
    lambda1 : float, default=2.5
        First lambda parameter for NSS function
    lambda2 : float, default=4.5
        Second lambda parameter for NSS function
    type_regressors : str, default="laguerre"
        Type of basis functions, one of "laguerre", "cubic"
    """
    def __init__(
        self,
        estimator=None,
        lambda1: float = 2.5,
        lambda2: float = 4.5,
        type_regressors: Optional[Literal["laguerre", "cubic", "kernel"]] = "cubic",
        kernel_type: Optional[Literal['matern', 'rbf', 'rationalquadratic', 'smithwilson']] = None,
        interpolation: Literal['linear', 'cubic'] = 'linear',
        **kwargs
    ):
        self.estimator = estimator
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.type_regressors = type_regressors
        self.kernel_type = kernel_type
        self.interpolation = interpolation
        self.maturities = None
        if self.type_regressors != "kernel":
            self.kernel_type = None
        self.maturities = None
        self.swap_rates = None
        self.tenor_swaps = None
        self.T_UFR = None
        self.kernel_params_ = kwargs  # Store kernel parameters
        self.coef_ = None
        self.cashflows_ = None
        self.maturities = None
        self.curve_rates_ = None

    def fit(
        self, 
        maturities: np.ndarray, 
        yields: np.ndarray,
        T_UFR: Optional[float] = None,
        **kwargs
    ) -> "CurveInterpolator":
        """Fit the curve Interpolator model.
        
        Parameters
        ----------
        maturities : np.ndarray
            Maturities of the swap rates
        yields: np.ndarray
            Yields to interpolate
        T_UFR : float, default=None
            UFR to use for the Smith-Wilson method

        Returns
        -------
        self : CurveInterpolator
            Fitted curve Interpolator model
        """
        assert len(maturities) == len(yields), "Maturities and yields must have the same length"
        self.maturities = np.asarray(maturities)
        self.yields = np.asarray(yields)
        self.T_UFR = T_UFR        
        # Store inputs
        self.rates_ = YieldsContainer(maturities=self.maturities, yields=self.yields)        
        # Calculate discount factors from actual yields
        V = np.exp(-self.maturities * self.yields)
        # Get basis functions
        X = self._get_basis_functions(self.maturities)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Use actual discount factors to calculate target values
        y = (1 - V) / self.maturities
        # Fit the model
        self.estimator.fit(X=X, y=y, **kwargs)
        predictions = self.estimator.predict(X)
        self.residuals_ = predictions - y
        return self

    def predict(self, maturities: np.ndarray) -> np.ndarray:
        """Predict interpolated rates at given maturities."""
        check_is_fitted(self)
        X = self._get_basis_functions(maturities)
        if X.ndim == 1:
            X = X.reshape(-1, 1)        
        # Model predicts (1-V)/T values
        y_pred = self.estimator.predict(X)        
        # Convert back to spot rates
        spot_rates = -np.log(1 - y_pred * maturities) / maturities
        discount_factors = np.exp(-maturities * spot_rates)
        forward_rates = self._calculate_forward_rates(maturities, spot_rates, discount_factors)
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )