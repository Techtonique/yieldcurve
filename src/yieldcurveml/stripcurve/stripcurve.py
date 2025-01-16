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
from .bootstrapcurve import RateCurveBootstrapper

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
        type_regressors: Optional[Literal["laguerre", "cubic", "kernel"]] = None,
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
        self.cashflow_dates_ = None
        if self.type_regressors != "kernel":
            self.kernel_type = None
        self.maturities = None
        self.kernel_params_ = kwargs  # Store kernel parameters
        self.coef_ = None

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
        tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m",
        T_UFR: Optional[float] = None
    ) -> "CurveStripper":
        """Fit the curve stripper model."""
        self.maturities = maturities
        # Store inputs
        self.rates_ = RatesContainer(
            maturities=np.asarray(maturities),
            swap_rates=np.asarray(swap_rates)
        )
        print("self.rates_.maturities: ", self.rates_.maturities)
        print("self.rates_.swap_rates: ", self.rates_.swap_rates)
        # Get cashflows and store them
        self.cashflows_ = swap_cashflows_matrix(
            swap_rates=swap_rates,
            maturities=maturities,
            tenor_swaps=tenor_swaps
        )
        self.cashflow_dates_ = self.cashflows_.cashflow_dates[-1]        
        if self.type_regressors == None and self.estimator is None:
            # Bootstrap method
            bootstrapper = RateCurveBootstrapper()
            self.curve_rates_ = bootstrapper.bootstrap_curve(
                maturities=self.rates_.maturities,
                swap_rates=self.rates_.swap_rates
            )
            #print("self.curve_rates_.spot_rates: ", self.curve_rates_.spot_rates)
            #print("self.curve_rates_.discount_factors: ", self.curve_rates_.discount_factors)
            #print("self.curve_rates_.forward_rates: ", self.curve_rates_.forward_rates)
            #print("self.curve_rates_.maturities: ", self.curve_rates_.maturities)
            
        if self.type_regressors == "kernel":
            if self.estimator is None:
                # Direct kernel solver (kernel inversion)
                lambda_reg = self.kernel_params_.get('lambda_reg', 1e-6)
                print("self.kernel_params_: ", self.kernel_params_)
                # Generate kernel matrix using cashflow dates
                K = generate_kernel(
                    self.cashflow_dates_,
                    kernel_type=self.kernel_type,
                    **self.kernel_params_
                )
                print("K shape: ", K.shape)
                # Add regularization
                K_reg = K + lambda_reg * np.eye(len(K))                
                # Calculate coefficients (solving the system)
                C = self.cashflows_.cashflow_matrix
                print("C shape: ", C.shape)
                V = np.ones_like(self.maturities)
                print("V shape: ", V.shape)

                if self.kernel_type == "smithwilson":
                    # For Smith-Wilson, include UFR adjustment
                    ufr = self.kernel_params_.get('ufr', 0.03)
                    mu = np.exp(-ufr * self.cashflow_dates_)
                    target = V - C @ mu
                    print("Target shape: ", target.shape)
                else:
                    target = V
                # Solve the system
                A = C @ K_reg @ C.T
                #A = (A + A.T) / 2  # Ensure symmetry
                print("A shape: ", A.shape)
                print("Target shape: ", target.shape)
                self.coef_ = np.linalg.solve(A, target)
                print("Coef shape: ", self.coef_.shape)
            else:
                # Kernel regression (using kernel as features)
                X = generate_kernel(
                    maturities.reshape(-1, 1),
                    kernel_type=self.kernel_type,
                    **self.kernel_params_
                )
                y = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / maturities
                self.estimator.fit(X, y)        
        # Calculate initial rates
        self.curve_rates_ = self._calculate_rates(self.maturities)
        return self
    
    def _calculate_rates(self, maturities: np.ndarray, X: Optional[np.ndarray] = None) -> CurveRates:
        """Calculate spot rates, forward rates, and discount factors."""
        print("\n\n self.kernel_type: ", self.kernel_type)
        if self.estimator is None:            
            if self.coef_ is None:                
                K_interp = generate_kernel(
                    self.maturities.reshape(-1, 1),
                    kernel_type=self.kernel_type,
                    **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}
                )
            else: 
                # Direct kernel prediction
                K_interp = generate_kernel(
                    maturities.reshape(-1, 1),
                    kernel_type=self.kernel_type,
                    nodal_points=self.maturities.reshape(-1, 1),
                    **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}
                )            
            # Calculate discount factors
            if self.kernel_type == "smithwilson":
                ufr = self.kernel_params_.get('ufr', 0.03)
                mu_interp = np.exp(-ufr * maturities)
                discount_factors = mu_interp + K_interp @ self.coef_
            else:
                discount_factors = K_interp @ self.coef_            
            # Calculate spot rates from discount factors
            spot_rates = -np.log(discount_factors) / maturities            
        elif self.estimator is None:
            # Bootstrap method
            # Calculate initial spot rates from swap rates
            spot_rates = self.rates_.swap_rates.copy()  # Start with swap rates as initial guess
            discount_factors = np.exp(-maturities * spot_rates)
            
        else: # Regression prediction
            if X is None:
                X = self._get_basis_functions(maturities)
            spot_rates = self.estimator.predict(X)
            discount_factors = np.exp(-maturities * spot_rates)
        
        # Calculate forward rates
        forward_rates = np.zeros_like(spot_rates)
        forward_rates[:-1] = -(np.log(discount_factors[1:]) - np.log(discount_factors[:-1])) / (
            maturities[1:] - maturities[:-1]
        )
        forward_rates[-1] = spot_rates[-1]
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )
    
    def predict(self, maturities: np.ndarray) -> CurveRates:
        """Predict rates for given maturities."""
        check_is_fitted(self)
        # Ensure maturities is 2D for kernel methods
        maturities = np.asarray(maturities).reshape(-1)  # First ensure 1D
        
        if self.type_regressors == None and self.estimator is None:
            # Interpolate bootstrap results using scipy's interp1d
            spot_rate_interpolator = interp1d(
                self.curve_rates_.maturities,
                self.curve_rates_.spot_rates,
                kind=self.interpolation,
                fill_value='extrapolate'
            )
            spot_rates = spot_rate_interpolator(maturities)
            discount_factors = np.exp(-maturities * spot_rates)
            
            # Calculate forward rates
            forward_rates = np.zeros_like(spot_rates)
            forward_rates[:-1] = -(np.log(discount_factors[1:]) - np.log(discount_factors[:-1])) / (
                maturities[1:] - maturities[:-1]
            )
            forward_rates[-1] = spot_rates[-1]
            
            return CurveRates(
                maturities=maturities,
                spot_rates=spot_rates,
                forward_rates=forward_rates,
                discount_factors=discount_factors
            )
        
        if self.type_regressors == "kernel":
            if self.estimator is None:
                # Direct kernel prediction
                return self._calculate_rates(maturities)
            else:
                # Ensure both inputs are 2D for kernel generation
                maturities_2d = maturities.reshape(-1, 1)
                nodal_points_2d = self.maturities.reshape(-1, 1)
                
                X = generate_kernel(
                    maturities_2d,
                    kernel_type=self.kernel_type,
                    nodal_points=nodal_points_2d,
                    **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}
                )
                return self._calculate_rates(maturities, X)
        else:
            # Standard basis functions
            X = self._get_basis_functions(maturities.ravel())
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