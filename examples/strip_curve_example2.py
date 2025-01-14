import numpy as np
from yieldcurve.utils.utils import get_swap_rates
from yieldcurve.stripcurve.stripcurve import CurveStripper
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

def main():
    # Get example data
    data = get_swap_rates("ap10")
    
    # Create and fit both models
    stripper_laguerre = CurveStripper(
        estimator=KernelRidge(),
        lambda1=2.5,
        lambda2=4.5,
        type_regressors="laguerre"
    )
    
    stripper_cubic = CurveStripper(
        estimator=KernelRidge(),
        type_regressors="cubic"
    )
    
    stripper_laguerre.fit(data.maturity, data.rate, tenor_swaps="6m")
    stripper_cubic.fit(data.maturity, data.rate, tenor_swaps="6m")
    
    # Create interpolation points for smoother curves
    maturities_fine = np.linspace(min(data.maturity), max(data.maturity), 100)
    
    # Get predictions
    pred_laguerre = stripper_laguerre.predict(maturities_fine)
    pred_cubic = stripper_cubic.predict(maturities_fine)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Laguerre vs Cubic Basis Functions with ExtraTrees')
    
    # Plot Discount Factors
    axes[0, 0].plot(data.maturity, stripper_laguerre.rates_.discount_factors, 'o-', label='Original points')
    axes[0, 0].plot(maturities_fine, pred_laguerre.discount_factors, '--', label='Interpolated')
    axes[0, 0].set_title('Discount Factors (Laguerre)')
    axes[0, 0].legend()
    
    axes[0, 1].plot(data.maturity, stripper_cubic.rates_.discount_factors, 'o-', label='Original points')
    axes[0, 1].plot(maturities_fine, pred_cubic.discount_factors, '--', label='Interpolated')
    axes[0, 1].set_title('Discount Factors (Cubic)')
    axes[0, 1].legend()
    
    # Plot Spot Rates
    axes[1, 0].plot(data.maturity, stripper_laguerre.rates_.spot_rates, 'o-', color='blue', label='Original points')
    axes[1, 0].plot(maturities_fine, pred_laguerre.spot_rates, '--', color='red', label='Interpolated')
    axes[1, 0].set_title('Spot Rates (Laguerre)')
    axes[1, 0].legend()
    
    axes[1, 1].plot(data.maturity, stripper_cubic.rates_.spot_rates, 'o-', color='blue', label='Original points')
    axes[1, 1].plot(maturities_fine, pred_cubic.spot_rates, '--', color='red', label='Interpolated')
    axes[1, 1].set_title('Spot Rates (Cubic)')
    axes[1, 1].legend()
    
    # Plot Forward Rates (only for Laguerre)
    if stripper_laguerre.rates_.forward_rates is not None:
        axes[2, 0].plot(data.maturity, stripper_laguerre.rates_.forward_rates, 'o-', color='blue', label='Original points')
        if pred_laguerre.forward_rates is not None:
            axes[2, 0].plot(maturities_fine, pred_laguerre.forward_rates, '--', color='blue', label='Interpolated')
        axes[2, 0].set_title('Forward Rates (Laguerre)')
        axes[2, 0].legend()
    
    axes[2, 1].set_visible(False)  # Hide the last plot since cubic doesn't have forward rates
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
