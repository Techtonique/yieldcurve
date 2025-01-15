import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_X_y, check_array
from typing import Literal, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    max_error
)
from ..utils.utils import swap_cashflows_matrix
import pandas as pd
from tabulate import tabulate

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
        type_regressors: Literal["laguerre", "cubic"] = "laguerre"
    ):
        self.estimator = ExtraTreesRegressor(n_estimators=100, random_state=42) if estimator is None else estimator
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
        """Fit the curve stripper model.
        
        Parameters
        ----------
        maturities : array-like of shape (n_samples,)
            Maturities of the swap rates
        swap_rates : array-like of shape (n_samples,)
            Swap rates
        tenor_swaps : {"1m", "3m", "6m", "1y"}, default="6m"
            Tenor of the swap rates
            
        Returns
        -------
        self : CurveStripper
            Fitted estimator
        """
        # Store inputs
        self.rates_ = RatesContainer(
            maturities=np.asarray(maturities),
            swap_rates=np.asarray(swap_rates)
        )
        
        # Get cashflows and store them
        self.cashflows_ = swap_cashflows_matrix(
            swap_rates=swap_rates,
            maturities=maturities,
            tenor_swaps=tenor_swaps
        )
        
        # Get basis functions
        X = self._get_basis_functions(maturities)
        y = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / maturities
        
        # Fit the model
        self.estimator.fit(X, y)
        
        # Calculate rates using the input maturities and store them
        self.curve_rates_ = self._calculate_rates(maturities)
        
        return self
    
    def _calculate_rates(self, maturities: np.ndarray) -> CurveRates:
        """Calculate spot, discount and forward rates."""
        X = self._get_basis_functions(maturities)
        spot_rates = self.estimator.predict(X)
        discount_factors = np.exp(-maturities * spot_rates)
        
        # Calculate forward rates if using Laguerre
        forward_rates = None
        if self.type_regressors == "laguerre":
            temp1 = maturities / self.lambda1
            temp = np.exp(-temp1)
            temp2 = maturities / self.lambda2
            
            # Base Laguerre functions
            X_base = np.column_stack([
                np.ones_like(maturities),
                temp,
                temp1 * temp,
                temp2 * np.exp(-temp2)
            ])
            
            # Get coefficients from the estimator if possible
            if hasattr(self.estimator, 'coef_') and self.estimator.coef_ is not None:
                print("Using linear coefficients method")
                # For nnetsauce, we need to use predict instead of direct multiplication
                spot_rates = self.estimator.predict(X_base)
                
                # Calculate derivative using numerical approximation
                h = 1e-6
                t_plus_h = maturities + h
                t_minus_h = maturities - h
                
                # Get basis functions for t+h and t-h
                temp1_plus = t_plus_h / self.lambda1
                temp_plus = np.exp(-temp1_plus)
                temp2_plus = t_plus_h / self.lambda2
                X_plus_h = np.column_stack([
                    np.ones_like(maturities),
                    temp_plus,
                    temp1_plus * temp_plus,
                    temp2_plus * np.exp(-temp2_plus)
                ])
                
                temp1_minus = t_minus_h / self.lambda1
                temp_minus = np.exp(-temp1_minus)
                temp2_minus = t_minus_h / self.lambda2
                X_minus_h = np.column_stack([
                    np.ones_like(maturities),
                    temp_minus,
                    temp1_minus * temp_minus,
                    temp2_minus * np.exp(-temp2_minus)
                ])
                
                # Calculate forward rates using centered difference
                spot_plus_h = self.estimator.predict(X_plus_h)
                spot_minus_h = self.estimator.predict(X_minus_h)
                derivative = (spot_plus_h - spot_minus_h) / (2 * h)
                forward_rates = spot_rates + maturities * derivative
                
                # Validation check
                if np.allclose(forward_rates, spot_rates):
                    print("Warning: Forward rates are equal to spot rates!")
                    print(f"First few derivatives: {derivative[:5]}")
                    print(f"First few maturities: {maturities[:5]}")
            else:
                print("Using numerical approximation method")
                try:
                    # Try numerical approximation for non-linear models using centered difference
                    h = 1e-6  # Small step size
                    t_plus_h = maturities + h
                    t_minus_h = maturities - h
                    X_plus_h = self._get_basis_functions(t_plus_h)
                    X_minus_h = self._get_basis_functions(t_minus_h)
                    
                    # Calculate forward rates using centered finite difference
                    spot_plus_h = self.estimator.predict(X_plus_h)
                    spot_minus_h = self.estimator.predict(X_minus_h)
                    derivative = (spot_plus_h - spot_minus_h) / (2 * h)
                    forward_rates = spot_rates + maturities * derivative
                except Exception as e:
                    print(f"Error in forward rates calculation: {str(e)}")
                    print(f"Exception type: {type(e)}")
                    forward_rates = None

            print(f"Final forward_rates is None: {forward_rates is None}")
        
        if self.type_regressors == "cubic":
            if hasattr(self.estimator, 'coef_') and self.estimator.coef_ is not None:
                coef = self.estimator.coef_
                # For cubic spline: R(t) = c₁t + c₂t² + c₃t³
                spot = (coef[0] * maturities + 
                       coef[1] * maturities**2 + 
                       coef[2] * maturities**3)
                # Derivative: dR/dt = c₁ + 2c₂t + 3c₃t²
                derivative = (coef[0] + 
                            2 * coef[1] * maturities + 
                            3 * coef[2] * maturities**2)
                # Instantaneous forward rate: f(t) = R(t) + t * dR/dt
                forward_rates = spot + maturities * derivative
                
                # Validation check
                if np.allclose(forward_rates, spot):
                    print("Warning: Forward rates are equal to spot rates!")
                    print(f"First few derivatives: {derivative[:5]}")
                    print(f"First few maturities: {maturities[:5]}")
            else:
                try:
                    h = 1e-6
                    t_plus_h = maturities + h
                    t_minus_h = maturities - h
                    X_plus_h = self._get_basis_functions(t_plus_h)
                    X_minus_h = self._get_basis_functions(t_minus_h)
                    
                    spot_plus_h = self.estimator.predict(X_plus_h)
                    spot_minus_h = self.estimator.predict(X_minus_h)
                    derivative = (spot_plus_h - spot_minus_h) / (2 * h)
                    forward_rates = spot_rates + maturities * derivative
                    
                    # Validation check
                    if np.allclose(forward_rates, spot_rates):
                        print("Warning: Forward rates are equal to spot rates in numerical method!")
                        print(f"First few derivatives: {derivative[:5]}")
                        print(f"First few maturities: {maturities[:5]}")
                except Exception as e:
                    print(f"Error in cubic forward rates calculation: {str(e)}")
                    forward_rates = None
        
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
    
    def get_diagnostics(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Union[RegressionDiagnostics, tuple[RegressionDiagnostics, RegressionDiagnostics]]:
        """Calculate detailed regression diagnostics.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples,), optional
            Test set maturities
        y_test : array-like of shape (n_samples,), optional
            Test set rates
            
        Returns
        -------
        train_diagnostics : RegressionDiagnostics
            Diagnostics for training set
        test_diagnostics : RegressionDiagnostics, optional
            Diagnostics for test set if provided
        """        
        
        def calculate_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionDiagnostics:
            residuals = y_true - y_pred
            abs_residuals = np.abs(residuals)
            
            # Calculate residuals summary statistics
            residuals_summary = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'median': np.median(residuals),
                'mad': np.median(abs_residuals),  # Median Absolute Deviation
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
        X_train = self._get_basis_functions(self.rates_.maturities)
        y_train_pred = self.estimator.predict(X_train)
        y_train = (self.cashflows_.cashflow_matrix.sum(axis=1) - 1) / self.rates_.maturities
        
        train_diagnostics = calculate_diagnostics(y_train, y_train_pred)
        
        # Test set diagnostics if provided
        if X_test is not None and y_test is not None:
            X_test = self._get_basis_functions(X_test)
            y_test_pred = self.estimator.predict(X_test)
            test_diagnostics = calculate_diagnostics(y_test, y_test_pred)
            return train_diagnostics, test_diagnostics
        
        return train_diagnostics