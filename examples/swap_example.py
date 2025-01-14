import numpy as np
from yieldcurve.utils.utils import swap_cashflows_matrix

def main():
    # Example data
    rates = [0.02, 0.025, 0.03]
    maturities = [1, 2, 3]
    
    # Calculate swap cashflows
    result = swap_cashflows_matrix(rates, maturities, "6m")
    
    # Print results
    print("Swap Cashflow Matrix:")
    print(result.cashflow_matrix)
    print("\nCashflow Dates:")
    print(result.cashflow_dates)

if __name__ == "__main__":
    main()