import numpy as np
import matplotlib.pyplot as plt
from yieldcurveml.stripcurve import CurveStripper
from yieldcurveml.utils import get_swap_rates
from sklearn.linear_model import Ridge

def main():
    # Get example data
    data = get_swap_rates("ap10")
    
    # Initialize different kernel strippers
    stripper_rbf = CurveStripper(
        estimator=Ridge(alpha=1e-6),
        type_regressors="kernel",
        kernel_type="rbf",
        length_scale=2.0
    )
    
    stripper_matern = CurveStripper(
        estimator=Ridge(alpha=1e-6),
        type_regressors="kernel",
        kernel_type="matern",
        length_scale=2.0,
        nu=1.5
    )
    
    stripper_sw = CurveStripper(
        estimator=Ridge(alpha=1e-6),
        type_regressors="kernel",
        kernel_type="smithwilson",
        alpha=0.1,
        ufr=0.03
    )
    
    stripper_sw_no_ufr = CurveStripper(
        estimator=Ridge(alpha=1e-6),
        type_regressors="kernel",
        kernel_type="smithwilson",
        alpha=0.1,
        ufr=None
    )
    
    # Add bootstrapped stripper
    stripper_bootstrap = CurveStripper(
        estimator=None  # None means use bootstrap method
    )

    # Smith-Wilson direct
    stripper_sw_direct = CurveStripper(
        estimator=None,
        type_regressors="kernel",
        kernel_type="smithwilson",
        alpha=0.1,
        ufr=0.03,
        lambda_reg=1e-6
    )
    
    # Fit all strippers
    strippers = {
        'RBF': stripper_rbf,
        'Matern': stripper_matern,
        'Smith-Wilson (UFR=3%)': stripper_sw,
        'Smith-Wilson (no UFR)': stripper_sw_no_ufr,
        'Bootstrap': stripper_bootstrap,
        'Smith-Wilson Direct': stripper_sw_direct
    }
    
    for name, stripper in strippers.items():
        stripper.fit(data.maturity, data.rate)
        print("Coeffs: ", stripper.coef_)
    
    # Create extended maturity grid for extrapolation
    t_extended = np.linspace(0, max(data.maturity) * 1.5, 100)
    
    # Plot results with symlog scale for negative rates
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (name, stripper) in enumerate(strippers.items()):
        if idx < len(axes):  # Ensure we don't exceed available subplots
            # Get predictions on extended grid
            predictions = stripper.predict(t_extended)            
            # Plot spot rates
            axes[idx].plot(t_extended, predictions.spot_rates, '-', label='Spot Rates')
            axes[idx].plot(t_extended, predictions.forward_rates, '--', label='Forward Rates')
            axes[idx].plot(data.maturity, data.rate, 'ko', label='Market Rates')            
            # If using Smith-Wilson with UFR, plot the UFR level
            if name == 'Smith-Wilson (UFR=3%)':
                axes[idx].axhline(0.03, color='r', linestyle=':', label='UFR')
            
            axes[idx].set_title(f'{name} Method')
            axes[idx].set_xlabel('Maturity')
            axes[idx].set_ylabel('Rate')
            axes[idx].grid(True)
            axes[idx].legend()
            axes[idx].set_yscale('symlog')  # Use symlog scale for negative rates
    
    plt.tight_layout()
    plt.show()
    
    # Plot discount factors comparison
    plt.figure(figsize=(8, 6))
    for name, stripper in strippers.items():
        predictions = stripper.predict(t_extended)
        plt.plot(t_extended, predictions.discount_factors, '-', label=name)
    
    plt.title('Discount Factors Comparison')
    plt.xlabel('Maturity')
    plt.ylabel('Discount Factor')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Print diagnostics
    print("\nModel Diagnostics:")
    print("-" * 50)
    for name, stripper in strippers.items():
        diag = stripper.get_diagnostics()
        print(f"\n{name}:")
        print(f"R² Score: {diag.r2_score:.4f}")
        print(f"RMSE: {diag.rmse:.6f}")
        print(f"MAE: {diag.mae:.6f}")
        print(f"Max Error: {diag.max_error:.6f}")

if __name__ == "__main__":
    main() 