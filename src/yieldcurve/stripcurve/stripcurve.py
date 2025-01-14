import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.validation import check_X_y, check_array
from typing import Literal, Optional, Union
from dataclasses import dataclass
from ..utils.utils import swap_cashflows_matrix

@dataclass
class CurveRates:
    """Container for curve stripping results"""
    maturities: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    forward_rates: Optional[np.ndarray] = None
    min_cv_error: Optional[float] = None

class CurveStripper(BaseEstimator, RegressorMixin):
    """Yield curve stripping estimator.
    
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
        type_regressors: Literal["laguerre", "cubic"] = "laguerre"
    ):
        self.estimator = Ridge() if estimator is None else estimator
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.type_regressors = type_regressors
    
    def _get_basis_functions(self, maturities: np.ndarray) -> np.ndarray:
        """Generate basis functions for the regression."""
        if self.type_regressors == "laguerre":
            temp1 = maturities / self.lambda1
            temp = np.exp(-temp1)
            temp2 = maturities / self.lambda2
            X = np.column_stack([
                np.ones_like(maturities),
                temp,
                temp1 * temp,
                temp2 * np.exp(-temp2)
            ])
        else:  # cubic
            X = np.column_stack([
                maturities,
                maturities**2,
                maturities**3
            ])
        return X
    
    def fit(
        self, 
        maturities: np.ndarray, 
        swap_rates: np.ndarray,
        tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m"
    ) -> "CurveStripper":
        """Fit the curve stripper.
        
        Parameters
        ----------
        maturities : array-like of shape (n_samples,)
            The maturities
        swap_rates : array-like of shape (n_samples,)
            The swap rates
        tenor_swaps : str, default="6m"
            Tenor for the swaps, one of "1m", "3m", "6m", "1y"
            
        Returns
        -------
        self : object
            Returns self
        """
        maturities, swap_rates = check_X_y(
            maturities.reshape(-1, 1), swap_rates, ensure_2d=False
        )
        maturities = maturities.ravel()
        swap_rates = swap_rates.ravel()
        
        # Get cashflows
        cashflows = swap_cashflows_matrix(
            swap_rates=swap_rates,
            maturities=maturities,
            tenor_swaps=tenor_swaps
        )
        
        # Prepare regression inputs
        X = self._get_basis_functions(maturities)
        V = np.ones_like(maturities)  # swap notional
        y = (cashflows.cashflow_matrix.sum(axis=1) - V) / maturities
        
        # Fit the model with weights
        weights = (1 / maturities) / (1 / maturities).sum()
        try:
            self.estimator.fit(X, y, sample_weight=weights)
        except Exception as e:
            self.estimator.fit(X, y) 
        
        # Get coefficients
        if hasattr(self.estimator, 'coef_'):
            self.coef_ = self.estimator.coef_
        else:
            self.coef_ = None
            
        # Calculate rates
        self.rates_ = self._calculate_rates(maturities)
        
        return self
    
    def _calculate_rates(self, maturities: np.ndarray) -> CurveRates:
        """Calculate spot, discount and forward rates."""
        X = self._get_basis_functions(maturities)
        
        if hasattr(self.estimator, 'predict'):
            spot_rates = self.estimator.predict(X)
        else:
            spot_rates = X @ self.coef_
            
        discount_factors = np.exp(-maturities * spot_rates)
        
        # Calculate forward rates if using Laguerre
        forward_rates = None
        if self.type_regressors == "laguerre" and self.coef_ is not None:
            temp1 = maturities / self.lambda1
            temp = np.exp(-temp1)
            temp2 = maturities / self.lambda2
            X_forward = np.column_stack([
                np.ones_like(maturities),
                temp,
                temp1 * temp,
                temp2 * np.exp(-temp2)
            ])
            forward_rates = X_forward @ self.coef_
            
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            discount_factors=discount_factors,
            forward_rates=forward_rates,
            min_cv_error=getattr(self.estimator, 'best_score_', None)
        )
    
    def predict(self, maturities: np.ndarray) -> CurveRates:
        """Predict rates for given maturities.
        
        Parameters
        ----------
        maturities : array-like of shape (n_samples,)
            The maturities to predict for
            
        Returns
        -------
        CurveRates
            Container with spot rates, discount factors, and forward rates
        """
        check_array(maturities.reshape(-1, 1))
        return self._calculate_rates(maturities.ravel())
