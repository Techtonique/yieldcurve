import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Literal, Optional, Union, Any
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
from tabulate import tabulate
from scipy.optimize import newton
from scipy.interpolate import interp1d
from ..utils.datastructures import RatesContainer, CurveRates, RegressionDiagnostics


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
    type_regressors : str, default=None
        Type of basis functions, one of "laguerre", "cubic", "kernel", or None for bootstrap
    kernel_type : str, default='matern'
        Type of kernel to use if type_regressors is "kernel"
    interpolation : str, default='linear'
        Interpolation method for bootstrap
    **kwargs : dict
        Additional parameters for kernel generation
    """
    
    def __init__(
        self,
        estimator: Optional[Any] = None,
        lambda1: float = 2.5,
        lambda2: float = 4.5,
        type_regressors: Optional[str] = None,
        kernel_type: str = 'matern',
        interpolation: str = 'linear',
        **kwargs
    ):
        self.estimator = estimator
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.type_regressors = type_regressors
        self.kernel_type = kernel_type
        self.interpolation = interpolation
        self.kernel_params_ = kwargs  # Store kernel parameters
        
        # Initialize attributes that will be set during fit
        self.cashflow_dates_ = None
        self.maturities = None
        self.swap_rates = None
        self.tenor_swaps = None
        self.T_UFR = None
        self.coef_ = None
        self.cashflows_ = None
        self.curve_rates_ = None
        self.rates_ = None

    def _get_basis_functions(self, maturities: np.ndarray) -> np.ndarray:
        """Generate basis functions for the regression."""
        maturities = np.asarray(maturities).reshape(-1)
        
        if self.type_regressors == "laguerre":
            temp1 = maturities / self.lambda1
            temp = np.exp(-temp1)
            temp2 = maturities / self.lambda2
            return np.column_stack([
                np.ones_like(maturities),
                temp,
                temp1 * temp,
                temp2 * np.exp(-temp2)
            ])
        elif self.type_regressors == "cubic":
            return np.column_stack([
                maturities,
                maturities**2,
                maturities**3
            ])
        elif self.type_regressors == "kernel":
            return generate_kernel(
                maturities, 
                kernel_type=self.kernel_type, 
                **self.kernel_params_
            )
        else:
            raise ValueError(f"Unsupported type_regressors: {self.type_regressors}")

    def _validate_inputs(self, maturities: np.ndarray, swap_rates: np.ndarray) -> None:
        """Validate input arrays."""
        maturities = np.asarray(maturities)
        swap_rates = np.asarray(swap_rates)
        
        if maturities.ndim != 1:
            raise ValueError("maturities must be a 1D array")
        if swap_rates.ndim != 1:
            raise ValueError("swap_rates must be a 1D array")
        if len(maturities) != len(swap_rates):
            raise ValueError("maturities and swap_rates must have the same length")
        if np.any(maturities <= 0):
            raise ValueError("maturities must be positive")
        if np.any(swap_rates < 0):
            warnings.warn("Negative swap rates detected")

    def fit(
        self, 
        maturities: np.ndarray, 
        swap_rates: np.ndarray,
        tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m",
        T_UFR: Optional[float] = None
    ) -> "CurveStripper":
        """Fit the curve stripper model.
        
        Parameters
        ----------
        maturities : np.ndarray
            Maturities of the swap rates
        swap_rates : np.ndarray
            Swap rates
        tenor_swaps : Literal["1m", "3m", "6m", "1y"], default="6m"
            Tenor of the swaps to use for the bootstrap
        T_UFR : float, default=None
            UFR to use for the Smith-Wilson method

        Returns
        -------
        self : CurveStripper
            Fitted curve stripper model
        """
        # Validate inputs
        self._validate_inputs(maturities, swap_rates)
        
        self.maturities = np.asarray(maturities).reshape(-1)
        self.swap_rates = np.asarray(swap_rates).reshape(-1)
        self.tenor_swaps = tenor_swaps
        self.T_UFR = T_UFR
        
        # Store inputs
        self.rates_ = RatesContainer(
            maturities=self.maturities,
            swap_rates=self.swap_rates
        )
        
        # Get cashflows and store them
        self.cashflows_ = swap_cashflows_matrix(
            swap_rates=self.swap_rates,
            maturities=self.maturities,
            tenor_swaps=self.tenor_swaps
        )
        self.cashflow_dates_ = self.cashflows_.cashflow_dates[-1]
        
        # Handle different fitting methods
        if self.type_regressors is None and self.estimator is None:
            # Bootstrap method
            bootstrapper = RateCurveBootstrapper(interpolation=self.interpolation)
            self.curve_rates_ = bootstrapper.fit(
                maturities=self.rates_.maturities,
                swap_rates=self.rates_.swap_rates,
                tenor_swaps=self.tenor_swaps
            )
            
        elif self.type_regressors == "kernel":
            self._fit_kernel_method()
        else:
            self._fit_basis_regression()
            
        return self

    def _fit_kernel_method(self) -> None:
        """Fit using kernel methods."""
        if self.estimator is None:
            # Direct kernel solver (kernel inversion)
            lambda_reg = self.kernel_params_.get('lambda_reg', 1e-6)
            
            # Generate kernel matrix using cashflow dates
            K = generate_kernel(
                self.cashflow_dates_,
                kernel_type=self.kernel_type,
                **self.kernel_params_
            )
            
            # Calculate coefficients
            C = self.cashflows_.cashflow_matrix
            V = np.ones_like(self.maturities)
            
            if self.kernel_type == "smithwilson":
                # For Smith-Wilson, include UFR adjustment
                ufr = self.kernel_params_.get('ufr', 0.03)
                mu = np.exp(-ufr * self.cashflow_dates_)
                target = V - C @ mu
                
                # Solve the system
                A = C @ K @ C.T + lambda_reg * np.eye(len(C))
                self.coef_ = np.linalg.solve(A, target)
            else:
                A = C @ K @ C.T + lambda_reg * np.eye(len(C))
                A = (A + A.T) / 2  # Ensure symmetry
                self.coef_ = np.linalg.solve(A, V)
        else:
            # Kernel regression (using kernel as features)
            X = generate_kernel(
                self.maturities.reshape(-1, 1),
                kernel_type=self.kernel_type,
                **self.kernel_params_
            )
            y = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / self.maturities
            self.estimator.fit(X, y)

    def _fit_basis_regression(self) -> None:
        """Fit using basis function regression."""
        if self.estimator is None:
            # Use Ridge as default estimator
            self.estimator = Ridge(alpha=1.0)
            
        X_train = self._get_basis_functions(self.maturities)
        y_train = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / self.maturities
        
        # Ensure numpy arrays and correct shapes
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).reshape(-1)
        
        # Fit the estimator
        self.estimator.fit(X_train, y_train)

    def _calculate_rates(self, maturities: np.ndarray, X: Optional[np.ndarray] = None) -> CurveRates:
        """Calculate spot rates, forward rates, and discount factors."""
        maturities = np.asarray(maturities).reshape(-1)
        
        # Handle bootstrap case
        if self.type_regressors is None and self.estimator is None:
            if self.curve_rates_ is None:
                raise ValueError("Curve rates not available. Model may not be fitted properly.")
            return self._interpolate_bootstrap_rates(maturities)

        # Handle kernel methods
        if self.type_regressors == "kernel":
            return self._calculate_kernel_rates(maturities)
        
        # Handle basis regression methods
        return self._calculate_basis_rates(maturities, X)

    def _interpolate_bootstrap_rates(self, maturities: np.ndarray) -> CurveRates:
        """Interpolate bootstrap rates for requested maturities."""
        if np.array_equal(maturities, self.curve_rates_.maturities):
            return self.curve_rates_
            
        # Interpolate spot rates
        spot_rate_interpolator = interp1d(
            self.curve_rates_.maturities,
            self.curve_rates_.spot_rates,
            kind=self.interpolation,
            fill_value='extrapolate',
            bounds_error=False
        )
        spot_rates = spot_rate_interpolator(maturities)
        
        # Calculate discount factors and forward rates
        discount_factors = np.exp(-maturities * spot_rates)
        forward_rates = self._calculate_forward_rates(maturities, discount_factors)
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )

    def _calculate_kernel_rates(self, maturities: np.ndarray) -> CurveRates:
        """Calculate rates using kernel methods."""
        if self.estimator is None:
            # Direct kernel prediction
            K_interp = generate_kernel(
                maturities,
                kernel_type=self.kernel_type,
                **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}
            )
            
            if self.kernel_type == "smithwilson":
                ufr = self.kernel_params_.get('ufr', 0.03)
                mu_interp = np.exp(-ufr * maturities)
                C = self.cashflows_.cashflow_matrix
                discount_factors = mu_interp + K_interp @ C.T @ self.coef_
            else:
                discount_factors = K_interp @ self.coef_
        else:
            # Kernel regression prediction
            nodal_points_2d = self.maturities.reshape(-1, 1)
            maturities_2d = maturities.reshape(-1, 1)
            
            X = generate_kernel(
                maturities_2d,
                kernel_type=self.kernel_type,
                nodal_points=nodal_points_2d,
                **{k: v for k, v in self.kernel_params_.items() if k != 'nodal_points'}
            )
            spot_rates = self.estimator.predict(X)
            discount_factors = np.exp(-maturities * spot_rates)
            forward_rates = self._calculate_forward_rates(maturities, discount_factors)
            
            return CurveRates(
                maturities=maturities,
                spot_rates=spot_rates,
                forward_rates=forward_rates,
                discount_factors=discount_factors
            )
        
        # Calculate rates from discount factors
        spot_rates = -np.log(discount_factors) / maturities
        forward_rates = self._calculate_forward_rates(maturities, discount_factors)
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )

    def _calculate_basis_rates(self, maturities: np.ndarray, X: Optional[np.ndarray] = None) -> CurveRates:
        """Calculate rates using basis function regression."""
        if X is None:
            X = self._get_basis_functions(maturities)
            
        if self.estimator is None:
            raise ValueError("Estimator is required for basis regression")
            
        spot_rates = self.estimator.predict(X)
        discount_factors = np.exp(-maturities * spot_rates)
        forward_rates = self._calculate_forward_rates(maturities, discount_factors)
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            forward_rates=forward_rates,
            discount_factors=discount_factors
        )

    def _calculate_forward_rates(self, maturities: np.ndarray, discount_factors: np.ndarray) -> np.ndarray:
        """Calculate forward rates from discount factors."""
        forward_rates = np.zeros_like(maturities)
        if len(maturities) > 1:
            forward_rates[:-1] = -(np.log(discount_factors[1:]) - np.log(discount_factors[:-1])) / (
                maturities[1:] - maturities[:-1]
            )
            forward_rates[-1] = forward_rates[-2] if len(forward_rates) > 1 else 0.0
        return forward_rates

    def predict(self, maturities: np.ndarray) -> CurveRates:
        """Predict rates for given maturities."""
        check_is_fitted(self, ['rates_'])
        maturities = np.asarray(maturities).reshape(-1)
        
        return self._calculate_rates(maturities)

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
        if self.type_regressors is None and self.estimator is None:
            # For bootstrap method, compare original swap rates with reconstructed rates
            y_train = self.rates_.swap_rates
            y_train_pred = self.predict(self.rates_.maturities).spot_rates
        elif self.estimator is not None:
            # For regression methods
            if self.type_regressors == "kernel" and self.estimator is not None:
                X_train = generate_kernel(
                    self.maturities.reshape(-1, 1),
                    kernel_type=self.kernel_type,
                    **self.kernel_params_
                )
            else:
                X_train = self._get_basis_functions(self.maturities)
            y_train = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / self.maturities
            y_train_pred = self.estimator.predict(X_train)
        else:
            raise ValueError("Cannot calculate diagnostics for current configuration")
        
        train_diagnostics = calculate_diagnostics(y_train, y_train_pred)
        
        # Test set diagnostics if provided
        if X_test is not None and y_test is not None:
            if self.type_regressors is None and self.estimator is None:
                y_test_pred = self.predict(X_test).spot_rates
            else:
                if self.type_regressors == "kernel" and self.estimator is not None:
                    X_test_basis = generate_kernel(
                        X_test.reshape(-1, 1),
                        kernel_type=self.kernel_type,
                        **self.kernel_params_
                    )
                else:
                    X_test_basis = self._get_basis_functions(X_test)
                y_test_pred = self.estimator.predict(X_test_basis)
            test_diagnostics = calculate_diagnostics(y_test, y_test_pred)
            return train_diagnostics, test_diagnostics
        
        return train_diagnostics