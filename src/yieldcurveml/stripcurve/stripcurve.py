import numpy as np
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

import pandas as pd
from tabulate import tabulate
from scipy.optimize import newton
from scipy.interpolate import interp1d

@dataclass
class RatesContainer:
    """Container for input rates data"""
    maturities: np.ndarray
    swap_rates: np.ndarray

@dataclass
class CurveRates:
    """Container for curve stripping results"""
    maturities: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    forward_rates: Optional[np.ndarray] = None
    min_cv_error: Optional[float] = None

@dataclass
class RegressionDiagnostics:
    """Container for regression diagnostics
    
    Parameters
    ----------
    r2_score : float
        R-squared score (coefficient of determination)
    rmse : float
        Root Mean Square Error
    mae : float
        Mean Absolute Error
    max_error : float
        Maximum absolute error
    min_error : float
        Minimum absolute error
    residuals : np.ndarray
        Model residuals (actual - predicted)
    fitted_values : np.ndarray
        Model predictions
    actual_values : np.ndarray
        Actual values
    n_samples : int
        Number of samples
    residuals_summary : dict
        Summary statistics of residuals including mean, std, 
        median, and various percentiles
    """
    r2_score: float
    rmse: float
    mae: float
    max_error: float
    min_error: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    actual_values: np.ndarray
    n_samples: int
    residuals_summary: dict

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
        type_regressors: Literal["laguerre", "cubic", "kernel"] = "laguerre",
        kernel_type: Literal['matern', 'rbf', 'rationalquadratic', 'smithwilson'] = 'matern',
        **kwargs
    ):
        self.estimator = estimator
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.type_regressors = type_regressors
        self.kernel_type = kernel_type
        self.kernel_params_ = kwargs  # Store kernel parameters
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
        elif self.type_regressors == "cubic":  # cubic
            X = np.column_stack([
                maturities,
                maturities**2,
                maturities**3
            ])
        elif self.type_regressors == "kernel":
            X = generate_kernel(maturities, kernel_type=self.kernel_type, 
                              **self.kernel_params_)
        return X
    
    def fit(
        self, 
        maturities: np.ndarray, 
        swap_rates: np.ndarray,
        tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m"
    ) -> "CurveStripper":
        """Fit the curve stripper model."""
        # Store inputs
        self.rates_ = RatesContainer(
            maturities=np.asarray(maturities),
            swap_rates=np.asarray(swap_rates)
        )
        
        # For kernel regressors, store nodal points
        if self.type_regressors == "kernel":
            self.kernel_params_['nodal_points'] = maturities
            # Generate training features
            X = generate_kernel(
                maturities.reshape(-1, 1),
                kernel_type=self.kernel_type,
                **self.kernel_params_
            )
        else:
            X = self._get_basis_functions(maturities)
            
        # Get cashflows and store them
        self.cashflows_ = swap_cashflows_matrix(
            swap_rates=swap_rates,
            maturities=maturities,
            tenor_swaps=tenor_swaps
        )
        
        if self.estimator is None:
            # Bootstrap method
            n_maturities = len(maturities)
            spot_rates = np.zeros(n_maturities)
            discount_factors = np.zeros(n_maturities)
            
            # Initial guess: spot rates = swap rates
            spot_rates[0] = swap_rates[0]
            discount_factors[0] = np.exp(-maturities[0] * spot_rates[0])
            
            # Bootstrap iteratively
            for i in range(1, n_maturities):
                def swap_value(rate):
                    # Calculate discount factor for current rate
                    df = np.exp(-maturities[i] * rate)
                    
                    # Get cashflows and dates for current swap
                    cashflows = self.cashflows_.cashflow_matrix[i, :i+1]
                    payment_times = self.cashflows_.cashflow_dates[i, :i+1]
                    
                    # Interpolate spot rates
                    spot_rates_interp = interp1d(
                        maturities[:i+1],
                        np.append(spot_rates[:i], rate),
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    
                    # Calculate discount factors from interpolated spot rates
                    interpolated_spots = spot_rates_interp(payment_times)
                    disc_factors = np.exp(-payment_times * interpolated_spots)
                    
                    # Return swap value
                    return np.sum(cashflows * disc_factors) - 1
                
                try:
                    # Use scipy.optimize.newton with more relaxed parameters
                    spot_rates[i] = newton(
                        swap_value,
                        x0=swap_rates[i],  # Initial guess
                        tol=1e-6,          # Relaxed tolerance
                        maxiter=1000,      # More iterations
                        rtol=1e-6,         # Relative tolerance
                        full_output=False,
                        disp=False
                    )
                except RuntimeError:
                    # Fallback: use previous rate if Newton fails
                    print(f"Warning: Newton failed to converge for maturity {maturities[i]}. Using fallback.")
                    spot_rates[i] = spot_rates[i-1]
                
                discount_factors[i] = np.exp(-maturities[i] * spot_rates[i])

            # Calculate forward rates using numerical differentiation
            h = 1e-6  # Small step size
            forward_rates = np.zeros_like(maturities)
            
            # First point: forward derivative
            spot_plus_h = -np.log(np.interp(maturities[0] + h, maturities, discount_factors)) / (maturities[0] + h)
            forward_rates[0] = spot_rates[0] + maturities[0] * (spot_plus_h - spot_rates[0]) / h
            
            # Middle points: centered difference
            for i in range(1, n_maturities - 1):
                spot_plus_h = -np.log(np.interp(maturities[i] + h, maturities, discount_factors)) / (maturities[i] + h)
                spot_minus_h = -np.log(np.interp(maturities[i] - h, maturities, discount_factors)) / (maturities[i] - h)
                derivative = (spot_plus_h - spot_minus_h) / (2 * h)
                forward_rates[i] = spot_rates[i] + maturities[i] * derivative
            
            # Last point: backward derivative
            spot_minus_h = -np.log(np.interp(maturities[-1] - h, maturities, discount_factors)) / (maturities[-1] - h)
            forward_rates[-1] = spot_rates[-1] + maturities[-1] * (spot_rates[-1] - spot_minus_h) / h

            # Store results
            self.curve_rates_ = CurveRates(
                maturities=maturities,
                spot_rates=spot_rates,
                discount_factors=discount_factors,
                forward_rates=forward_rates
            )
            
        else:
            # Original regression-based method
            X = self._get_basis_functions(maturities)
            y = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / maturities
            self.estimator.fit(X, y)
            self.curve_rates_ = self._calculate_rates(maturities)
        
        return self
    
    def _calculate_rates(self, maturities: np.ndarray, X: Optional[np.ndarray] = None) -> CurveRates:
        """Calculate spot rates, forward rates, and discount factors."""
        if self.estimator is None:
            # For bootstrap method, interpolate the stored rates
            from scipy.interpolate import interp1d
            
            # Create interpolation function for spot rates
            spot_rate_interp = interp1d(
                self.rates_.maturities,
                self.curve_rates_.spot_rates,
                kind='linear',
                fill_value='extrapolate'
            )
            
            # Interpolate spot rates at requested maturities
            spot_rates = spot_rate_interp(maturities)
        else:
            # For regression methods, predict using features
            if X is None:
                X = self._get_basis_functions(maturities)
            spot_rates = self.estimator.predict(X)
        
        # Calculate discount factors
        discount_factors = np.exp(-maturities * spot_rates)
        
        # Calculate forward rates (excluding the last point)
        forward_rates = np.zeros_like(spot_rates)
        forward_rates[:-1] = -(np.log(discount_factors[1:]) - np.log(discount_factors[:-1])) / (
            maturities[1:] - maturities[:-1]
        )
        # Set the last forward rate equal to the last spot rate
        forward_rates[-1] = spot_rates[-1]
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )
    
    def predict(self, maturities: np.ndarray) -> CurveRates:
        """Predict rates for given maturities."""
        # Check if fitted
        check_is_fitted(self)
        
        # Convert to numpy array
        maturities = np.asarray(maturities)
        
        # For kernel regressors, we need to generate the kernel features
        if self.type_regressors == "kernel":
            # Generate kernel features relative to training points
            X = generate_kernel(
                maturities.reshape(-1, 1),
                kernel_type=self.kernel_type,
                nodal_points=self.rates_.maturities,  # Use training points as nodes
                **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}  # Exclude nodal_points
            )
        else:
            # For other regressors, use the standard basis functions
            X = self._get_basis_functions(maturities.ravel())
        
        # Calculate rates using the appropriate features
        return self._calculate_rates(maturities.ravel(), X)
    
    def get_diagnostics(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Union[RegressionDiagnostics, tuple[RegressionDiagnostics, RegressionDiagnostics]]:
        """Calculate detailed regression diagnostics."""
        def calculate_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionDiagnostics:
            residuals = y_true - y_pred
            abs_residuals = np.abs(residuals)
            
            residuals_summary = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'median': np.median(residuals),
                'mad': np.median(abs_residuals),
                'skewness': float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3)),
                'kurtosis': float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 4) - 3),
                'percentiles': {
                    '1%': np.percentile(residuals, 1),
                    '5%': np.percentile(residuals, 5),
                    '25%': np.percentile(residuals, 25),
                    '75%': np.percentile(residuals, 75),
                    '95%': np.percentile(residuals, 95),
                    '99%': np.percentile(residuals, 99)
                }
            }
            
            return RegressionDiagnostics(
                r2_score=r2_score(y_true, y_pred),
                rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
                mae=mean_absolute_error(y_true, y_pred),
                max_error=max_error(y_true, y_pred),
                min_error=float(np.min(abs_residuals)),
                residuals=residuals,
                fitted_values=y_pred,
                actual_values=y_true,
                n_samples=len(y_true),
                residuals_summary=residuals_summary
            )
        
        # Training set diagnostics
        if self.estimator is None:
            # For bootstrap method, compare original swap rates with reconstructed rates
            y_train = self.rates_.swap_rates
            y_train_pred = self.predict(self.rates_.maturities).spot_rates
        else:
            # For regression methods
            X_train = self._get_basis_functions(self.rates_.maturities)
            y_train = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / self.rates_.maturities
            y_train_pred = self.estimator.predict(X_train)
        
        train_diagnostics = calculate_diagnostics(y_train, y_train_pred)
        
        # Test set diagnostics if provided
        if X_test is not None and y_test is not None:
            if self.estimator is None:
                y_test_pred = self.predict(X_test).spot_rates
            else:
                X_test = self._get_basis_functions(X_test)
                y_test_pred = self.estimator.predict(X_test)
            test_diagnostics = calculate_diagnostics(y_test, y_test_pred)
            return train_diagnostics, test_diagnostics
        
        return train_diagnostics