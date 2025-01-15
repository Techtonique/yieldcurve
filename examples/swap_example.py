import numpy as np
from yieldcurveml.utils.utils import swap_cashflows_matrix, get_swap_rates
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

def main():
    # Get example data from ap10 dataset
    data = get_swap_rates("and07")
    
    # Calculate swap cashflows using the example data
    result = swap_cashflows_matrix(
        swap_rates=data.rate,
        maturities=data.maturity,
        tenor_swaps="6m"
    )
    print("Input swap rates", result)
    
    # Print results
    print("Using AP10 dataset:")
    print(f"Number of swaps: {result.nb_swaps}")
    print("\nSwap Cashflow Matrix:")
    print(result.cashflow_matrix)
    print("\nCashflow Dates:")
    print(result.cashflow_dates)

if __name__ == "__main__":
    main()