import numpy as np
import warnings
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional, Tuple, Literal
from ..utils.utils import swap_cashflows_matrix
from ..utils.datastructures import RatesContainer, CurveRates, RegressionDiagnostics


class RateCurveBootstrapper(BaseEstimator, RegressorMixin):
    """
    Bootstrap interest rate curve from swap rates.

    Parameters
    ----------

    interpolation: Literal['linear', 'cubic'] = 'linear'
    """
    def __init__(
        self,
        interpolation: Literal['linear', 'cubic'] = 'linear'
    ):
        self.interpolation = interpolation
        self.maturities = None
        self.swap_rates = None
        self.tenor_swaps = None
        self.n_maturities_ = None
        self.cashflows_ = None
        self.cashflow_dates_ = None
        self.spot_rates_ = None
        self.discount_factors_ = None 
        self.forward_rates_ = None

    def fit(
        self, 
        maturities: np.ndarray, 
        swap_rates: np.ndarray,
        tenor_swaps: Literal["1m", "3m", "6m", "1y"] = "6m",
    ) -> CurveRates:
        """
        Bootstrap interest rate curve from swap rates using an improved procedure.
        
        Args:
            maturities: Array of swap maturities
            swap_rates: Array of corresponding swap rates
            tenor_swaps: Tenor of the swaps to use for the bootstrap
            
        Returns:
            CurveRates object containing bootstrapped curves
        """
        if self.interpolation not in ['linear', 'cubic']:
            raise ValueError("interpolation must be either 'linear' or 'cubic'")
        self.maturities = maturities        
        self.swap_rates = swap_rates
        self.tenor_swaps = tenor_swaps
        self.spot_rates_ = self.swap_rates
        self._validate_inputs()
        if self.tenor_swaps == "1m":
            self.freq_payments_ = 12
        elif self.tenor_swaps == "3m":
            self.freq_payments_ = 4
        elif self.tenor_swaps == "6m":
            self.freq_payments_ = 2
        elif self.tenor_swaps == "1y":
            self.freq_payments_ = 1
        self.n_maturities_ = len(self.maturities)                
        cashflow_data = swap_cashflows_matrix(
            swap_rates=self.swap_rates,
            maturities=self.maturities,
            tenor_swaps=self.tenor_swaps)
        self.cashflows_ = cashflow_data.cashflow_matrix
        self.cashflow_dates_ = cashflow_data.cashflow_dates[-1,:]  
        self.discount_factors_ = np.zeros(self.n_maturities_)
        self.spot_rates_ = np.zeros(self.n_maturities_)
        self.forward_rates_ = np.zeros(self.n_maturities_)

        def objective_function(x):
            current_maturity = self.maturities[i]
            mask_maturities = (self.spot_rates_ > 0)
            mask_cashflow_dates = (self.cashflow_dates_ <= current_maturity) # mask for cashflow dates before maturity
            cashflow_dates = self.cashflow_dates_[mask_cashflow_dates] # cashflow dates before maturity
            cashflows = self.cashflows_[i, mask_cashflow_dates] # cashflows before maturity 
            if (not np.all(self.spot_rates_ <= 0)): # at least one spot rate is positive
                spot_rates = np.interp(x=cashflow_dates, 
                                    xp=self.maturities[mask_maturities], 
                                    fp=self.spot_rates_[mask_maturities]) if self.interpolation == 'linear' else interp1d(self.maturities[mask_maturities], self.spot_rates_[mask_maturities], kind=self.interpolation)
            else: # first guess for spot rate is the swap rate
                spot_rates = self.swap_rates[0]
            dfs = np.exp(-cashflow_dates*spot_rates)
            dfs[-1:] = np.exp(-current_maturity*x) # last cashflow is the maturity, we are solving for x 
            drs_ = -np.log(dfs)/cashflow_dates # discount rates for cashflows, excluding the first date = 0
            dfs_ = self._compute_discount_factor(cashflow_dates, drs_) # discount factors for cashflow dates after the first date = 0
            return np.sum(cashflows*dfs_) - 1
        
        # Bootstrap iteratively
        for i in range(self.n_maturities_):
            result = root_scalar(f=objective_function, x0=self.swap_rates[i])
            self.spot_rates_[i] = result.root
            self.discount_factors_[i] = self._compute_discount_factor(self.maturities[i], 
                                                                      self.spot_rates_[i])
        # Calculate forward rates using improved method
        self.forward_rates_ = self._calculate_forward_rates(self.maturities, 
                                                           self.spot_rates_, 
                                                           self.discount_factors_)
        
        return CurveRates(
            maturities=self.maturities,
            spot_rates=self.spot_rates_,
            discount_factors=self.discount_factors_,
            forward_rates=self.forward_rates_
        )
    
    def _validate_inputs(self) -> None:
        """Validate input arrays."""
        if len(self.maturities) != len(self.swap_rates):
            raise ValueError("Maturities and swap rates must have same length")
        if not np.all(np.diff(self.maturities) > 1e-10):
            raise ValueError("Maturities must be strictly increasing with minimum spacing of 1e-10")
        if np.any(self.maturities <= 0):
            raise ValueError("Maturities must be positive")
        if np.any(~np.isfinite(self.maturities)) or np.any(~np.isfinite(self.swap_rates)):
            raise ValueError("Maturities and swap rates must be finite numbers")
    
    def _compute_discount_factor(self, maturity: float, rate: float) -> float:
        """Compute discount factor from spot rate.

        Args:
            maturity: Maturity of the discount factor
            rate: Spot rate
            
        Returns:
            Discount factor
        """
        return np.exp(-maturity * rate)
    
    def _calculate_forward_rates(
        self,
        maturities: np.ndarray,
        spot_rates: np.ndarray,
        discount_factors: np.ndarray
    ) -> np.ndarray:
        """Calculate forward rates using improved method.
        
        Args:
            maturities: Array of maturities
            spot_rates: Array of spot rates
            discount_factors: Array of discount factors
            
        Returns:
            Array of forward rates
        """
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