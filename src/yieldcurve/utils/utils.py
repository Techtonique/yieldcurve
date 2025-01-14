import numpy as np
from typing import List, Dict, Union, Literal
from dataclasses import dataclass

@dataclass
class SwapCashflows:
    nb_swaps: int
    swaps_maturities: np.ndarray
    nb_swap_dates: np.ndarray
    swap_rates: np.ndarray
    cashflow_dates: np.ndarray
    cashflow_matrix: np.ndarray

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