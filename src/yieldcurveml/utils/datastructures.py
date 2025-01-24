from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RatesContainer:
    """Container for input rates data"""
    maturities: np.ndarray
    swap_rates: np.ndarray

@dataclass
class YieldsContainer:
    """Container for input yields data"""
    maturities: np.ndarray
    yields: np.ndarray

@dataclass
class CurveRates:
    """Container for curve stripping results"""
    maturities: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    forward_rates: Optional[np.ndarray] = None

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