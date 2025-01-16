import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import warnings

@dataclass
class CurveRates:
    maturities: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    forward_rates: np.ndarray

class RateCurveBootstrapper:
    def bootstrap_curve(
        self, 
        maturities: np.ndarray, 
        swap_rates: np.ndarray,
        interpolation: Literal['linear', 'cubic'] = 'cubic'
    ) -> CurveRates:
        """
        Bootstrap interest rate curve from swap rates using an improved procedure.
        
        Args:
            maturities: Array of swap maturities
            swap_rates: Array of corresponding swap rates
            interpolation: Type of interpolation to use ('linear' or 'cubic')
            
        Returns:
            CurveRates object containing bootstrapped curves
        """
        if interpolation not in ['linear', 'cubic']:
            raise ValueError("interpolation must be either 'linear' or 'cubic'")
            
        n_maturities = len(maturities)
        spot_rates = np.zeros(n_maturities)
        discount_factors = np.zeros(n_maturities)
        
        # Validate inputs
        self._validate_inputs(maturities, swap_rates)
        
        # Initialize first point
        spot_rates[0] = swap_rates[0]
        discount_factors[0] = self._compute_discount_factor(maturities[0], spot_rates[0])
        
        # Bootstrap iteratively
        for i in range(1, n_maturities):
            spot_rates[i], discount_factors[i] = self._bootstrap_rate(
                i, maturities, spot_rates, swap_rates, interpolation
            )
            
        # Calculate forward rates using improved method
        forward_rates = self._calculate_forward_rates(maturities, spot_rates, discount_factors)
        
        return CurveRates(
            maturities=maturities,
            spot_rates=spot_rates,
            discount_factors=discount_factors,
            forward_rates=forward_rates
        )
    
    def _validate_inputs(self, maturities: np.ndarray, swap_rates: np.ndarray) -> None:
        """Validate input arrays."""
        if len(maturities) != len(swap_rates):
            raise ValueError("Maturities and swap rates must have same length")
        if not np.all(np.diff(maturities) > 0):
            raise ValueError("Maturities must be strictly increasing")
        if np.any(maturities <= 0) or np.any(swap_rates < 0):
            raise ValueError("Maturities must be positive and swap rates non-negative")
    
    def _compute_discount_factor(self, maturity: float, rate: float) -> float:
        """Compute discount factor from spot rate."""
        return np.exp(-maturity * rate)
    
    def _bootstrap_rate(
        self, 
        index: int, 
        maturities: np.ndarray, 
        spot_rates: np.ndarray, 
        swap_rates: np.ndarray,
        interpolation: str
    ) -> Tuple[float, float]:
        """Bootstrap single spot rate using improved Newton method with fallbacks."""
        
        def swap_value(rate: float) -> float:
            """Compute swap value for given rate."""
            df = self._compute_discount_factor(maturities[index], rate)
            
            # Get cashflows and dates
            cashflows = self.cashflows_.cashflow_matrix[index, :index+1]
            payment_times = self.cashflows_.cashflow_dates[index, :index+1]
            
            # Create interpolator based on specified method
            spot_rates_interp = interp1d(
                maturities[:index+1],
                np.append(spot_rates[:index], rate),
                kind=interpolation,
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            interpolated_spots = spot_rates_interp(payment_times)
            disc_factors = np.exp(-payment_times * interpolated_spots)
            
            return np.sum(cashflows * disc_factors) - 1
        
        # Try Newton method with multiple initial guesses
        initial_guesses = [
            swap_rates[index],           # Current swap rate
            spot_rates[index-1],         # Previous spot rate
            (swap_rates[index] + spot_rates[index-1]) / 2  # Average
        ]
        
        for guess in initial_guesses:
            try:
                rate = newton(
                    swap_value,
                    x0=guess,
                    tol=1e-8,
                    maxiter=1000,
                    rtol=1e-8,
                    full_output=False,
                    disp=False
                )
                # Validate result
                if rate > 0 and np.isfinite(rate):
                    discount_factor = self._compute_discount_factor(maturities[index], rate)
                    return rate, discount_factor
            except:
                continue
        
        # If all attempts fail, use interpolation as final fallback
        warnings.warn(f"Newton method failed for maturity {maturities[index]}. Using linear interpolation.")
        
        # Always use linear interpolation for fallback (more stable than cubic)
        rate = interp1d(
            maturities[:index], 
            spot_rates[:index], 
            kind='linear',  # Force linear interpolation for fallback
            bounds_error=False,
            fill_value='extrapolate'
        )(maturities[index])
        
        discount_factor = self._compute_discount_factor(maturities[index], rate)
        return rate, discount_factor
    
    def _calculate_forward_rates(
        self,
        maturities: np.ndarray,
        spot_rates: np.ndarray,
        discount_factors: np.ndarray
    ) -> np.ndarray:
        """Calculate forward rates using improved method."""
        n_maturities = len(maturities)
        forward_rates = np.zeros(n_maturities)
        
        # First point: use spot rate
        forward_rates[0] = spot_rates[0]
        
        # Remaining points: use discrete formula for better numerical stability
        for i in range(1, n_maturities):
            t1, t2 = maturities[i-1], maturities[i]
            df1, df2 = discount_factors[i-1], discount_factors[i]
            forward_rates[i] = -np.log(df2/df1) / (t2 - t1)
        
        return forward_rates