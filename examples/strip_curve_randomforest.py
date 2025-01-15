import numpy as np
from yieldcurveml.utils import get_swap_rates, regression_report
from yieldcurveml.stripcurve import CurveStripper
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

def main():
    # Get example data
    data = get_swap_rates("and07")
    
    # Create and fit both models
    stripper_laguerre = CurveStripper(
        estimator=RandomForestRegressor(n_estimators=1000, random_state=42),
        lambda1=2.5,
        lambda2=4.5,
        type_regressors="laguerre"
    )
    
    stripper_cubic = CurveStripper(
        estimator=RandomForestRegressor(n_estimators=1000, random_state=42),
        type_regressors="cubic"
    )
    
    stripper_laguerre.fit(data.maturity, data.rate, tenor_swaps="6m")
    stripper_cubic.fit(data.maturity, data.rate, tenor_swaps="6m")
    
    # Print diagnostics
    print("\nLaguerre Model:")
    print(regression_report(stripper_laguerre, "Laguerre"))
    
    print("\nCubic Model:")
    print(regression_report(stripper_cubic, "Cubic"))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot discount factors
    axes[0, 0].plot(data.maturity, stripper_laguerre.curve_rates_.discount_factors, 'o-', label='Laguerre')
    axes[0, 0].plot(data.maturity, stripper_cubic.curve_rates_.discount_factors, 's--', label='Cubic')
    axes[0, 0].set_title('Discount Factors')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot spot rates
    axes[0, 1].plot(data.maturity, stripper_laguerre.curve_rates_.spot_rates, 'o-', label='Laguerre')
    axes[0, 1].plot(data.maturity, stripper_cubic.curve_rates_.spot_rates, 's--', label='Cubic')
    axes[0, 1].plot(data.maturity, data.rate, 'kx', label='Original')
    axes[0, 1].set_title('Spot Rates')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot forward rates (Laguerre only)
    if stripper_laguerre.curve_rates_.forward_rates is not None:
        axes[1, 0].plot(data.maturity, stripper_laguerre.curve_rates_.forward_rates, 'o-', label='Laguerre')
        axes[1, 0].set_title('Forward Rates (Laguerre)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
