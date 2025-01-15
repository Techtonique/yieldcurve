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
        
        if hasattr(self.estimator, 'predict'):
            spot_rates = self.estimator.predict(X)
        else:
            spot_rates = X @ self.coef_
            
        discount_factors = np.exp(-maturities * spot_rates)
        
        # Calculate forward rates if using Laguerre
        forward_rates = None
        if self.type_regressors == "laguerre":
            temp1 = maturities / self.lambda1
            temp = np.exp(-temp1)
            temp2 = maturities / self.lambda2
            X_forward = np.column_stack([
                np.ones_like(maturities),
                temp,
                temp1 * temp,
                temp2 * np.exp(-temp2)
            ])
            forward_rates = self.estimator.predict(X_forward)
            
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

    @staticmethod
    def regression_report(
        diagnostics: Union[RegressionDiagnostics, tuple[RegressionDiagnostics, RegressionDiagnostics]],
        names: Union[str, tuple[str, str]] = ("Train", "Test")
    ) -> str:
        """Create a formatted report of regression diagnostics.
        
        Parameters
        ----------
        diagnostics : RegressionDiagnostics or tuple[RegressionDiagnostics, RegressionDiagnostics]
            Diagnostics object(s) to format
        names : str or tuple[str, str], default=("Train", "Test")
            Names to use for the model(s) in the report
            
        Returns
        -------
        str
            Formatted report string
        """
        def _create_tables(diag: RegressionDiagnostics, name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            metrics_data = {
                'Metric': ['Samples', 'R²', 'RMSE', 'MAE', 'Min Error', 'Max Error'],
                name: [
                    diag.n_samples,
                    f"{diag.r2_score:.4f}",
                    f"{diag.rmse:.4f}",
                    f"{diag.mae:.4f}",
                    f"{diag.min_error:.4f}",
                    f"{diag.max_error:.4f}"
                ]
            }
            
            summary_data = {
                'Statistic': ['Mean', 'Std Dev', 'Median', 'MAD', 'Skewness', 'Kurtosis'],
                name: [
                    f"{diag.residuals_summary['mean']:.4f}",
                    f"{diag.residuals_summary['std']:.4f}",
                    f"{diag.residuals_summary['median']:.4f}",
                    f"{diag.residuals_summary['mad']:.4f}",
                    f"{diag.residuals_summary['skewness']:.4f}",
                    f"{diag.residuals_summary['kurtosis']:.4f}"
                ]
            }
            
            percentiles_data = {
                'Percentile': ['1%', '5%', '25%', '75%', '95%', '99%'],
                name: [
                    f"{diag.residuals_summary['percentiles']['1%']:.4f}",
                    f"{diag.residuals_summary['percentiles']['5%']:.4f}",
                    f"{diag.residuals_summary['percentiles']['25%']:.4f}",
                    f"{diag.residuals_summary['percentiles']['75%']:.4f}",
                    f"{diag.residuals_summary['percentiles']['95%']:.4f}",
                    f"{diag.residuals_summary['percentiles']['99%']:.4f}"
                ]
            }
            
            return (
                pd.DataFrame(metrics_data),
                pd.DataFrame(summary_data),
                pd.DataFrame(percentiles_data)
            )
        
        if isinstance(diagnostics, tuple):
            # Handle train/test diagnostics
            train_diag, test_diag = diagnostics
            train_name, test_name = names if isinstance(names, tuple) else ("Train", "Test")
            
            # Create tables for both train and test
            train_tables = _create_tables(train_diag, train_name)
            test_tables = _create_tables(test_diag, test_name)
            
            # Merge corresponding tables
            metrics_df = pd.merge(train_tables[0], test_tables[0], on='Metric')
            summary_df = pd.merge(train_tables[1], test_tables[1], on='Statistic')
            percentiles_df = pd.merge(train_tables[2], test_tables[2], on='Percentile')
        else:
            # Handle single diagnostics
            name = names if isinstance(names, str) else "Model"
            metrics_df, summary_df, percentiles_df = _create_tables(diagnostics, name)
        
        # Format the report
        report = (
            f"\nModel Performance Metrics:\n"
            f"{tabulate(metrics_df, headers='keys', tablefmt='pipe', showindex=False)}\n\n"
            f"Residuals Summary Statistics:\n"
            f"{tabulate(summary_df, headers='keys', tablefmt='pipe', showindex=False)}\n\n"
            f"Residuals Percentiles:\n"
            f"{tabulate(percentiles_df, headers='keys', tablefmt='pipe', showindex=False)}"
        )
        
        return report
