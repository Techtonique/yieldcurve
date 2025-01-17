import matplotlib.pyplot as plt

from yieldcurveml.utils import get_swap_rates, regression_report
from yieldcurveml.stripcurve import CurveStripper


datasets = ["and07", "ab13ois", "ap10", "ab13e6m", "negativerates"]

def main():

    for dataset in datasets:
        # Get example data
        data = get_swap_rates(dataset)        
        
        stripper_bootstrap = CurveStripper()
        
        stripper_bootstrap.fit(data.maturity, 
                               data.rate, 
                               tenor_swaps="6m")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(stripper_bootstrap.rates_.maturities, stripper_bootstrap.curve_rates_.spot_rates * 100, 
                label='Zero Rates', linewidth=2)
        plt.plot(data.maturity, data.rate * 100, 'o', 
                label='Market Swap Rates', markersize=8)
        plt.title(f'Bootstrapped Zero Curve - {dataset}')
        plt.xlabel('Time to Maturity (years)')
        plt.ylabel('Rate (%)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    main()