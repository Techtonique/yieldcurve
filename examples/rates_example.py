from yieldcurve.utils.utils import get_swap_rates

def main():
    # Get example data
    data = get_swap_rates("ap10")
    
    print("Maturities:", data.maturity)
    print("Rates:", data.rate)

if __name__ == "__main__":
    main() 