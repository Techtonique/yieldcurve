from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

from yieldcurveml.utils import get_swap_rates, regression_report
from yieldcurveml.stripcurve import CurveStripper


def main():
    # Get example data
    data = get_swap_rates("and07")
    
    # Create and fit both models
    stripper_laguerre = CurveStripper(
        estimator=ExtraTreesRegressor(n_estimators=100, random_state=42),
        lambda1=2.5,
        lambda2=4.5,
        type_regressors="laguerre"
    )
    
    stripper_cubic = CurveStripper(
        estimator=ExtraTreesRegressor(n_estimators=100, random_state=42),
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot discount factors
    axes[0].plot(data.maturity, stripper_laguerre.curve_rates_.discount_factors, 'o-', label='Laguerre')
    axes[0].plot(data.maturity, stripper_cubic.curve_rates_.discount_factors, 's--', label='Cubic')
    axes[0].set_title('Discount Factors')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot spot rates
    axes[1].plot(data.maturity, stripper_laguerre.curve_rates_.spot_rates, 'o-', label='Laguerre')
    axes[1].plot(data.maturity, stripper_cubic.curve_rates_.spot_rates, 's--', label='Cubic')
    axes[1].plot(data.maturity, data.rate, 'kx', label='Original')
    axes[1].set_title('Spot Rates')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot forward rates (both models)
    if (stripper_laguerre.curve_rates_.forward_rates is not None and 
        stripper_cubic.curve_rates_.forward_rates is not None):
        axes[2].plot(data.maturity, stripper_laguerre.curve_rates_.forward_rates, 'o-', label='Laguerre')
        axes[2].plot(data.maturity, stripper_cubic.curve_rates_.forward_rates, 's--', label='Cubic')
        axes[2].set_title('Forward Rates')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

    print(stripper_cubic.curve_rates_.forward_rates == stripper_cubic.curve_rates_.spot_rates)
    print(stripper_laguerre.curve_rates_.forward_rates == stripper_laguerre.curve_rates_.spot_rates)


if __name__ == "__main__":
    main()
