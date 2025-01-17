import numpy as np
import matplotlib.pyplot as plt
from yieldcurveml.stripcurve import CurveStripper
from yieldcurveml.utils import get_swap_rates
from sklearn.linear_model import Ridge

def main():
    # Get example data
    data = get_swap_rates("ab13ois")
    
 
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
    

    # Smith-Wilson direct
    stripper_sw_direct = CurveStripper(
        estimator=None,
        type_regressors="kernel",
        kernel_type="smithwilson",
        alpha=0.1,
        ufr=0.055,
        lambda_reg=1e-4
    )

    # Add bootstrapped stripper
    stripper_bootstrap = CurveStripper(
        estimator=None  # None means use bootstrap method
    )
    
    # Fit all strippers
    strippers = {
        'Bootstrap': stripper_bootstrap,
        'Smith-Wilson (UFR=3%)': stripper_sw,
        'Smith-Wilson (no UFR)': stripper_sw_no_ufr,        
        'Smith-Wilson Direct': stripper_sw_direct,        
    }
    
    for name, stripper in strippers.items():
        stripper.fit(data.maturity, data.rate)
        print("Coeffs: ", stripper.coef_)
    
    # Create extended maturity grid for extrapolation
    t_extended = np.linspace(1, max(data.maturity) * 1.5, 100)
    
    # Plot results with symlog scale for negative rates
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define what to plot in each subplot
    plot_configs = [
        {'title': 'Spot Rates', 'attr': 'spot_rates', 'style': '-'},
        {'title': 'Forward Rates', 'attr': 'forward_rates', 'style': '--'},
        {'title': 'Discount Factors', 'attr': 'discount_factors', 'style': '-'}
    ]
    
    # Plot each type of curve
    for ax, config in zip(axes, plot_configs):
        for name, stripper in strippers.items():
            predictions = stripper.predict(t_extended)
            values = getattr(predictions, config['attr'])
            ax.plot(t_extended, values, config['style'], label=name)
            
            # Add dots for bootstrap points
            if name == 'Bootstrap':
                bootstrap_values = getattr(stripper.curve_rates_, config['attr'])
                ax.plot(stripper.curve_rates_.maturities, bootstrap_values, 'o', color='blue')
                        
        # Add UFR level to relevant plots
        if config['title'] in ['Spot Rates', 'Forward Rates']:
            ax.axhline(0.03, color='r', linestyle=':', label='UFR')
        
        ax.set_title(config['title'])
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Rate' if 'Rates' in config['title'] else 'Factor')
        ax.grid(True)
        
        if 'Rates' in config['title']:
            ax.set_yscale('symlog')  # Use symlog scale for rates (handles negatives)
            
        # Adjust legend to prevent overlap
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    
    plt.tight_layout()
    plt.show()
    
    # Plot discount factors comparison
    plt.figure(figsize=(6, 6))
    for name, stripper in strippers.items():
        predictions = stripper.predict(t_extended)
        plt.plot(t_extended, predictions.discount_factors, '-', label=name)
    
    plt.title('Discount Factors Comparison')
    plt.xlabel('Maturity')
    plt.ylabel('Discount Factor')
    plt.grid(True)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main() 