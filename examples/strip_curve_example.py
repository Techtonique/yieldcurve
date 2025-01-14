import numpy as np
from yieldcurve.utils.utils import get_swap_rates
from yieldcurve.stripcurve.stripcurve import CurveStripper
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

def main():
    # Get example data
    data = get_swap_rates("and07")
    
    # Create and fit both models
    stripper_laguerre = CurveStripper(
        estimator=ExtraTreesRegressor(n_estimators=100, 
                                      random_state=42),
        lambda1=2.5,
        lambda2=4.5,
        type_regressors="laguerre"
    )
    
    stripper_cubic = CurveStripper(
        estimator=ExtraTreesRegressor(n_estimators=100, 
                                      random_state=42),
        type_regressors="cubic"
    )
    
    stripper_laguerre.fit(data.maturity, data.rate, tenor_swaps="6m")
    stripper_cubic.fit(data.maturity, data.rate, tenor_swaps="6m")
    
    # Create interpolation points for smoother curves
    maturities_fine = np.linspace(min(data.maturity), max(data.maturity), 100)
    
    # Get predictions for interpolated points
    spot_rates_laguerre = stripper_laguerre.predict(maturities_fine)
    spot_rates_cubic = stripper_cubic.predict(maturities_fine)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Laguerre vs Cubic Basis Functions')
    
    # Plot Discount Factors
    axes[0, 0].plot(stripper_laguerre.rates_.maturities, 
                    stripper_laguerre.rates_.discount_factors, 
                    'o-', label='Original points')
    axes[0, 0].plot(maturities_fine, 
                    np.exp(-maturities_fine * spot_rates_laguerre),
                    '--', label='Interpolated')
    axes[0, 0].set_title('Discount Factors (Laguerre)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(stripper_cubic.rates_.maturities, 
                    stripper_cubic.rates_.discount_factors,
                    'o-', label='Original points')
    axes[0, 1].plot(maturities_fine,
                    np.exp(-maturities_fine * spot_rates_cubic),
                    '--', label='Interpolated')
    axes[0, 1].set_title('Discount Factors (Cubic)')
    axes[0, 1].legend()
    
    # Plot Spot Rates
    axes[1, 0].plot(stripper_laguerre.rates_.maturities,
                    stripper_laguerre.rates_.spot_rates,
                    'o-', color='black', label='Original points')
    axes[1, 0].plot(maturities_fine,
                    spot_rates_laguerre,
                    '--', color='black', label='Interpolated')
    axes[1, 0].set_title('Spot Rates (Laguerre)')
    axes[1, 0].legend()
    
    axes[1, 1].plot(stripper_cubic.rates_.maturities,
                    stripper_cubic.rates_.spot_rates,
                    'o-', color='black', label='Original points')
    axes[1, 1].plot(maturities_fine,
                    spot_rates_cubic,
                    '--', color='black', label='Interpolated')
    axes[1, 1].set_title('Spot Rates (Cubic)')
    axes[1, 1].legend()
    
    # Plot Forward Rates (only for Laguerre)
    if stripper_laguerre.rates_.forward_rates is not None:
        axes[2, 0].plot(stripper_laguerre.rates_.maturities,
                       stripper_laguerre.rates_.forward_rates,
                       'o-', color='blue', label='Original points')
        # For forward rates, we still need to use _calculate_rates as it's not part of the standard predict interface
        forward_rates = stripper_laguerre._calculate_rates(maturities_fine).forward_rates
        if forward_rates is not None:
            axes[2, 0].plot(maturities_fine,
                           forward_rates,
                           '--', color='blue', label='Interpolated')
        axes[2, 0].set_title('Forward Rates (Laguerre)')
        axes[2, 0].legend()
    
    axes[2, 1].set_visible(False)  # Hide the last plot since cubic doesn't have forward rates
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
