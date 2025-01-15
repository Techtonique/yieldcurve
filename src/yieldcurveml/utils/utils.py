import numpy as np
from typing import List, Dict, Union, Literal, NamedTuple
from dataclasses import dataclass
import pandas as pd
from tabulate import tabulate

@dataclass
class SwapCashflows:
    nb_swaps: int
    swaps_maturities: np.ndarray
    nb_swap_dates: np.ndarray
    swap_rates: np.ndarray
    cashflow_dates: np.ndarray
    cashflow_matrix: np.ndarray

class SwapRatesData(NamedTuple):
    maturity: np.ndarray
    rate: np.ndarray

def swap_cashflows_matrix(
    swap_rates: Union[List[float], np.ndarray],
    maturities: Union[List[float], np.ndarray],
    tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m"
) -> SwapCashflows:
    """
    Creates a matrix of swap cashflows.

    Args:
        swap_rates: Vector of swap rates
        maturities: Vector of maturities (in years)
        tenor_swaps: Tenor for the swaps, one of "1m", "3m", "6m", "1y"

    Returns:
        SwapCashflows object containing:
            - nb_swaps: number of swaps
            - swaps_maturities: original maturities
            - nb_swap_dates: number of cashflow dates per swap
            - swap_rates: original swap rates
            - cashflow_dates: matrix of cashflow dates
            - cashflow_matrix: matrix of cashflows

    Example:
        >>> rates = [0.02, 0.025, 0.03]
        >>> maturities = [1, 2, 3]
        >>> result = swap_cashflows_matrix(rates, maturities, "6m")
    """
    # Convert inputs to numpy arrays if they aren't already
    swap_rates = np.array(swap_rates)
    maturities = np.array(maturities)
    
    nb_swaps = len(swap_rates)
    if nb_swaps != len(maturities):
        raise ValueError("There must be as many swap rates as maturities")
    
    # Define frequency mapping
    freq_map = {
        "1m": 1/12,
        "3m": 1/4,
        "6m": 1/2,
        "1y": 1
    }
    
    if tenor_swaps not in freq_map:
        raise ValueError("tenor_swaps must be one of: '1m', '3m', '6m', '1y'")
    
    freq = freq_map[tenor_swaps]
    
    # Create cashflow dates
    cashflow_dates = np.arange(freq, max(maturities) + freq, freq)
    nb_cashflow_dates = len(cashflow_dates)
    nb_cashflow_dates_swaps = (1 / freq) * maturities
    
    # Initialize matrices
    swap_cashflows_matrix = np.zeros((nb_swaps, nb_cashflow_dates))
    cashflow_dates_matrix = np.zeros((nb_swaps, nb_cashflow_dates))
    nb_swap_dates = np.zeros(nb_swaps)
    
    # Fill matrices
    for i in range(nb_swaps):
        nb_cashflow_dates_swaps_i = int(nb_cashflow_dates_swaps[i])
        swap_rate_i_times_freq = swap_rates[i] * freq
        
        # Set regular cashflows
        swap_cashflows_matrix[i, :nb_cashflow_dates_swaps_i] = swap_rate_i_times_freq
        # Add principal repayment at maturity
        swap_cashflows_matrix[i, nb_cashflow_dates_swaps_i - 1] += 1
        
        nb_swap_dates[i] = np.sum(swap_cashflows_matrix[i, :] > 0)
        cashflow_dates_matrix[i, :nb_cashflow_dates_swaps_i] = cashflow_dates[:nb_cashflow_dates_swaps_i]
    
    # Add row and column names
    swap_names = [f"swap{i+1}" for i in range(nb_swaps)]
    date_names = [f"{d}y" for d in cashflow_dates]
    
    return SwapCashflows(
        nb_swaps=nb_swaps,
        swaps_maturities=maturities,
        nb_swap_dates=nb_swap_dates,
        swap_rates=swap_rates,
        cashflow_dates=cashflow_dates_matrix,
        cashflow_matrix=swap_cashflows_matrix
    )

def get_swap_rates(dataset: Literal["ap10", "and07", "ab13e6m", "ab13ois", "negativerates"]) -> SwapRatesData:
    """
    Get example swap rates datasets.

    Args:
        dataset: String specifying the dataset to use,
                one of "ap10", "and07", "ab13e6m", "ab13ois", "negativerates"

    Returns:
        SwapRatesData containing maturity and rate vectors

    Examples:
        >>> print(get_swap_rates("ap10"))
        SwapRatesData(maturity=array([ 1,  2,  3,  5,  7, 10, 12, 15, 20, 25]), 
                     rate=array([0.042, 0.043, 0.047, 0.054, 0.057, 0.06 , 0.061, 0.059, 0.056, 0.0555]))
    """
    # Define all datasets
    swap_rates: Dict[str, SwapRatesData] = {
        "ap10": SwapRatesData(
            maturity=np.array([1, 2, 3, 5, 7, 10, 12, 15, 20, 25]),
            rate=np.array([4.2, 4.3, 4.7, 5.4, 5.7, 6, 6.1, 5.9, 5.6, 5.55]) / 100
        ),
        
        "and07": SwapRatesData(
            maturity=np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 12, 15, 20, 30]),
            rate=np.array([2.75, 3.10, 3.30, 3.43, 3.53, 3.30, 3.78, 3.95, 
                          4.25, 4.50, 4.65, 4.78, 4.88, 4.85]) / 100
        ),
        
        "ab13e6m": SwapRatesData(
            maturity=np.array(list(range(1, 31)) + [35, 40, 50, 60]),
            rate=np.array([
                0.286, 0.324, 0.424, 0.576, 0.762, 0.954, 1.135, 1.303, 1.452, 1.584,
                1.703, 1.809, 1.901, 1.976, 2.037, 2.086, 2.123, 2.150, 2.171, 2.187,
                2.200, 2.211, 2.220, 2.228, 2.234, 2.239, 2.243, 2.247, 2.251, 2.256,
                2.295, 2.348, 2.421, 2.463
            ]) / 100
        ),
        
        "ab13ois": SwapRatesData(
            maturity=np.array(list(range(1, 13)) + [15, 20, 25, 30]),
            rate=np.array([
                0.000, 0.036, 0.127, 0.274, 0.456, 0.647, 0.827, 0.996, 1.147, 1.280,
                1.404, 1.516, 1.764, 1.939, 2.003, 2.038
            ]) / 100
        ),
        
        "negativerates": SwapRatesData(
            maturity=np.array([0.5, 1, 2, 3, 5]),
            rate=np.array([-0.00337, -0.003610, -0.003647, -0.003413, -0.003047])
        )
    }
    
    if dataset not in swap_rates:
        raise ValueError(
            f"Dataset {dataset} not found. Available datasets: {', '.join(swap_rates.keys())}"
        )
    
    return swap_rates[dataset]

def regression_report(
    model: 'CurveStripper',
    name: str = "Model"
) -> str:
    """Create a formatted report of regression diagnostics for a fitted CurveStripper model.
    
    Parameters
    ----------
    model : CurveStripper
        Fitted CurveStripper model
    name : str, default="Model"
        Name to use for the model in the report
        
    Returns
    -------
    str
        Formatted report string
        
    Examples
    --------
    >>> from yieldcurveml.stripcurve import CurveStripper
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = CurveStripper(RandomForestRegressor())
    >>> model.fit(X, y)
    >>> print(regression_report(model, "RandomForest"))
    """
    if not hasattr(model, 'curve_rates_'):
        raise ValueError("Model must be fitted before generating report")
        
    diagnostics = model.get_diagnostics()
    
    metrics_data = {
        'Metric': ['Samples', 'R²', 'RMSE', 'MAE', 'Min Error', 'Max Error'],
        name: [
            diagnostics.n_samples,
            f"{diagnostics.r2_score:.4f}",
            f"{diagnostics.rmse:.4f}",
            f"{diagnostics.mae:.4f}",
            f"{diagnostics.min_error:.4f}",
            f"{diagnostics.max_error:.4f}"
        ]
    }
    
    summary_data = {
        'Statistic': ['Mean', 'Std Dev', 'Median', 'MAD', 'Skewness', 'Kurtosis'],
        name: [
            f"{diagnostics.residuals_summary['mean']:.4f}",
            f"{diagnostics.residuals_summary['std']:.4f}",
            f"{diagnostics.residuals_summary['median']:.4f}",
            f"{diagnostics.residuals_summary['mad']:.4f}",
            f"{diagnostics.residuals_summary['skewness']:.4f}",
            f"{diagnostics.residuals_summary['kurtosis']:.4f}"
        ]
    }
    
    percentiles_data = {
        'Percentile': ['1%', '5%', '25%', '75%', '95%', '99%'],
        name: [
            f"{diagnostics.residuals_summary['percentiles']['1%']:.4f}",
            f"{diagnostics.residuals_summary['percentiles']['5%']:.4f}",
            f"{diagnostics.residuals_summary['percentiles']['25%']:.4f}",
            f"{diagnostics.residuals_summary['percentiles']['75%']:.4f}",
            f"{diagnostics.residuals_summary['percentiles']['95%']:.4f}",
            f"{diagnostics.residuals_summary['percentiles']['99%']:.4f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    summary_df = pd.DataFrame(summary_data)
    percentiles_df = pd.DataFrame(percentiles_data)
    
    report = (
        f"\nModel Performance Metrics:\n"
        f"{tabulate(metrics_df, headers='keys', tablefmt='pipe', showindex=False)}\n\n"
        f"Residuals Summary Statistics:\n"
        f"{tabulate(summary_df, headers='keys', tablefmt='pipe', showindex=False)}\n\n"
        f"Residuals Percentiles:\n"
        f"{tabulate(percentiles_df, headers='keys', tablefmt='pipe', showindex=False)}"
    )
    
    return report