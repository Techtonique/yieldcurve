import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings

@dataclass
class PricingResult:
    """Container for pricing results with confidence intervals"""
    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_simulations: int
    
    def __repr__(self):
        return (f"PricingResult(price={self.price:.6f}, "
                f"std_error={self.std_error:.6f}, "
                f"95% CI=[{self.ci_lower:.6f}, {self.ci_upper:.6f}], "
                f"N={self.n_simulations})")

@dataclass
class CapletResult:
    """Container for individual caplet pricing"""
    reset_time: float
    payment_time: float
    value: float
    std_error: float
    strike: float
    forward_rate_mean: float

class ArbitrageFreeShortRate:
    """
    Production-grade implementation of Moudiki (2025) framework for arbitrage-free 
    short rate simulation.
    
    This class implements:
    - Three methods for short rate construction (Section 3)
    - Deterministic shift adjustment for arbitrage-free pricing (Proposition 1)
    - Monte Carlo pricing with confidence intervals
    - Interest rate derivatives (caps, swaptions)
    
    Key Features:
    - Correct integral bounds [t,T] in forward rate calculation (Eq. 5)
    - Proper cumulative shift integration (Eq. 6)
    - Robust numerical methods with trapezoidal integration
    - Confidence interval support for uncertainty quantification
    - Production-ready error handling and validation
    
    References:
        Moudiki, T. (2025). New Short Rate Models and their Arbitrage-Free Extension:
        A Flexible Framework for Historical and Market-Consistent Simulation.
        
    Example:
        >>> model = ArbitrageFreeShortRate(lambda_param=0.7)
        >>> short_rates = model.method1_ns_extrapolation(yield_data, maturities)
        >>> paths = model.simulate_paths(n_paths=1000, n_periods=60)
        >>> adjusted_prices, shift = model.deterministic_shift_adjustment(paths, market_prices)
        >>> cap_price, cap_se, details = model.price_cap(paths, time_grid, strike=0.03)
    """
    
    def __init__(self, lambda_param: float = 0.0609, dt: float = 1/12):
        """
        Initialize the arbitrage-free short rate model.
        
        Parameters:
        -----------
        lambda_param : float, default=0.0609
            Nelson-Siegel decay parameter (standard value from Diebold-Li)
        dt : float, default=1/12
            Time step for discrete simulation (monthly by default)
        """
        self.lambda_param = lambda_param
        self.dt = dt
        self.short_rates = None
        self.ns_factors = None
        self.shift_function = None
        
    def nelson_siegel_curve(self, maturity: np.ndarray, beta1: float, 
                           beta2: float, beta3: float) -> np.ndarray:
        """
        Vectorized Nelson-Siegel yield curve formula.
        
        R(τ) = β₁ + β₂ * (1-exp(-λτ))/(λτ) + β₃ * [(1-exp(-λτ))/(λτ) - exp(-λτ)]
        
        Parameters:
        -----------
        maturity : array-like
            Time to maturity (in years)
        beta1, beta2, beta3 : float
            Nelson-Siegel factors (level, slope, curvature)
            
        Returns:
        --------
        yields : ndarray
            Predicted yields for given maturities
        """
        maturity = np.asarray(maturity, dtype=float)
        decay = self.lambda_param * maturity
        factor1 = np.where(decay == 0.0, 1.0, (1 - np.exp(-decay)) / decay)
        factor2 = factor1 - np.exp(-decay)
        return beta1 + beta2 * factor1 + beta3 * factor2
    
    def method1_ns_extrapolation(self, yield_data: np.ndarray, 
                                 maturities: np.ndarray) -> np.ndarray:
        """
        Method 1: Nelson-Siegel extrapolation to zero maturity (Equation 8).
        
        Extracts instantaneous short rate: r(t) = lim_{τ→0⁺} R_t(τ) = β₁ + β₂
        
        Parameters:
        -----------
        yield_data : ndarray, shape (n_dates, n_maturities)
            Historical yield curve observations
        maturities : ndarray, shape (n_maturities,)
            Maturity points (in years)
            
        Returns:
        --------
        short_rates : ndarray, shape (n_dates,)
            Extracted instantaneous short rates for each date
        """
        n_dates = yield_data.shape[0]
        short_rates = np.zeros(n_dates)
        ns_factors = np.zeros((n_dates, 3))
        
        def ns_residual(params, tau, y):
            return self.nelson_siegel_curve(tau, *params) - y
        
        for i in range(n_dates):
            try:
                init = [np.mean(yield_data[i]), 
                       yield_data[i, 0] - yield_data[i, -1], 
                       0.0]
                
                result = least_squares(
                    ns_residual, init, 
                    args=(maturities, yield_data[i]),
                    bounds=([-0.1, -0.1, -0.1], [0.2, 0.1, 0.1]),
                    ftol=1e-8, xtol=1e-8
                )
                
                if result.success:
                    ns_factors[i] = result.x
                    short_rates[i] = result.x[0] + result.x[1]  # r(t) = β₁ + β₂
                else:
                    short_mats = maturities[maturities <= 1]
                    short_yields = yield_data[i, maturities <= 1]
                    if len(short_mats) >= 2:
                        coeffs = np.polyfit(short_mats, short_yields, 1)
                        short_rates[i] = coeffs[1]
                    else:
                        short_rates[i] = yield_data[i, 0]
                    ns_factors[i] = [short_rates[i], 0, 0]
                    
            except Exception as e:
                warnings.warn(f"NS fit failed at date {i}: {str(e)}, using fallback")
                short_rates[i] = yield_data[i, 0]
                ns_factors[i] = [short_rates[i], 0, 0]
        
        self.short_rates = short_rates
        self.ns_factors = ns_factors
        return short_rates
    
    def method2_ml_features(self, yield_data: np.ndarray, 
                           maturities: np.ndarray,
                           model_type: str = 'rf') -> np.ndarray:
        """
        Method 2: NS features with machine learning (Definition 2).
        
        Trains ML model on NS feature representation, predicts at limiting values:
        r(t) = M(1, 1, 0)
        
        Parameters:
        -----------
        yield_data : ndarray, shape (n_dates, n_maturities)
            Historical yield curve observations
        maturities : ndarray, shape (n_maturities,)
            Maturity points (in years)
        model_type : str, default='rf'
            ML model type: 'rf' (Random Forest)
            
        Returns:
        --------
        short_rates : ndarray, shape (n_dates,)
            ML-predicted instantaneous short rates
        """
        n_dates = yield_data.shape[0]
        short_rates = np.zeros(n_dates)
        
        X_maturities = []
        for tau in maturities:
            decay = self.lambda_param * tau
            if decay > 0:
                level = 1.0
                slope = (1 - np.exp(-decay)) / decay
                curvature = slope - np.exp(-decay)
            else:
                level, slope, curvature = 1.0, 1.0, 0.0
            X_maturities.append([level, slope, curvature])
        X_maturities = np.array(X_maturities)
        
        X_short = np.array([[1.0, 1.0, 0.0]])
        
        for i in range(n_dates):
            try:
                if model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=50, random_state=42, 
                                                 max_depth=5, min_samples_leaf=2)
                else:
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=0.01)
                
                model.fit(X_maturities, yield_data[i])
                short_rates[i] = model.predict(X_short)[0]
                
            except Exception as e:
                warnings.warn(f"ML fit failed at date {i}: {str(e)}")
                short_rates[i] = np.mean(yield_data[i, :min(3, len(yield_data[i]))])
        
        self.short_rates = short_rates
        return short_rates
    
    def method3_direct_regression(self, yield_data: np.ndarray, 
                                  maturities: np.ndarray,
                                  method: str = 'spline') -> np.ndarray:
        """
        Method 3: Direct regression to zero maturity (Definition 3).
        
        Non-parametric: Fit M_t: τ → R_t(τ), then r(t) = M_t(0)
        
        Parameters:
        -----------
        yield_data : ndarray, shape (n_dates, n_maturities)
            Historical yield curve observations
        maturities : ndarray, shape (n_maturities,)
            Maturity points (in years)
        method : str, default='spline'
            Extrapolation method: 'spline', 'linear', or 'polynomial'
            
        Returns:
        --------
        short_rates : ndarray, shape (n_dates,)
            Extrapolated instantaneous short rates
        """
        n_dates = yield_data.shape[0]
        short_rates = np.zeros(n_dates)
        
        for i in range(n_dates):
            try:
                if method == 'spline' and len(maturities) >= 4:
                    spline = CubicSpline(maturities, yield_data[i], 
                                        bc_type='natural', extrapolate=True)
                    short_rates[i] = spline(0.0)
                elif method == 'polynomial' and len(maturities) >= 3:
                    coeffs = np.polyfit(maturities, yield_data[i], 2)
                    short_rates[i] = coeffs[2]
                else:
                    coeffs = np.polyfit(maturities[:2], yield_data[i, :2], 1)
                    short_rates[i] = coeffs[1]
            except Exception as e:
                warnings.warn(f"Extrapolation failed at date {i}: {str(e)}")
                short_rates[i] = yield_data[i, 0]
        
        self.short_rates = short_rates
        return short_rates
    
    def simulate_paths(self, n_paths: int = 1000, n_periods: int = 60, 
                      model_type: str = 'AR1', 
                      custom_params: Optional[Dict] = None) -> np.ndarray:
        """
        Simulate future short rate paths under physical measure.
        
        Parameters:
        -----------
        n_paths : int, default=1000
            Number of Monte Carlo paths
        n_periods : int, default=60
            Number of time periods
        model_type : str, default='AR1'
            Dynamics model: 'AR1', 'Vasicek', or 'constant'
        custom_params : dict, optional
            Custom parameters: {'kappa': 0.3, 'theta': 0.05, 'sigma': 0.02}
            
        Returns:
        --------
        paths : ndarray, shape (n_paths, n_periods)
            Simulated short rate paths
        """
        if self.short_rates is None:
            raise ValueError("Must estimate short rates first using method1/2/3")
        
        current_rate = self.short_rates[-1]
        paths = np.zeros((n_paths, n_periods))
        paths[:, 0] = current_rate
        
        if model_type == 'constant':
            paths[:] = current_rate
            return paths
        
        if model_type == 'AR1' and len(self.short_rates) > 10:
            mu = np.mean(self.short_rates)
            demeaned = self.short_rates - mu
            returns = demeaned[1:]
            lag_returns = demeaned[:-1]
            
            if len(returns) > 1 and np.var(lag_returns) > 1e-10:
                phi = np.cov(returns, lag_returns)[0, 1] / np.var(lag_returns)
                phi = np.clip(phi, -0.99, 0.99)
                residuals = returns - phi * lag_returns
                sigma = np.std(residuals)
            else:
                phi, sigma = 0.9, 0.01
                
        elif model_type == 'Vasicek' or custom_params is not None:
            if custom_params:
                kappa = custom_params.get('kappa', 0.3)
                theta = custom_params.get('theta', current_rate)
                sigma = custom_params.get('sigma', 0.02)
            else:
                kappa = 0.3
                theta = np.mean(self.short_rates) if len(self.short_rates) > 10 else current_rate
                sigma = np.std(self.short_rates) * 0.5 if len(self.short_rates) > 10 else 0.02
            
            phi = np.exp(-kappa * self.dt)
            mu = theta * (1 - phi)
        else:
            mu = np.mean(self.short_rates) if len(self.short_rates) > 1 else current_rate
            phi, sigma = 0.9, 0.01
        
        for t in range(1, n_periods):
            innovation = np.random.normal(0, sigma, n_paths)
            if model_type == 'Vasicek':
                paths[:, t] = mu + phi * paths[:, t-1] + innovation
            else:
                paths[:, t] = mu + phi * (paths[:, t-1] - mu) + innovation
            paths[:, t] = np.clip(paths[:, t], -0.02, 0.25)
        
        return paths
    
    def _compute_forward_rates(self, sim_paths: np.ndarray, 
                              time_grid: np.ndarray, 
                              t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        CORRECTED: Compute simulated bond prices and forward rates (Equation 5).
        
        Key fix: Uses proper integral bounds [t,T] not [0,T].
        
        Implements: f̂_t(T) = E[r(T) · exp(-∫_t^T r(s)ds)] / P̂_t(T)
        """
        n_paths, n_periods = sim_paths.shape
        idx_t = max(0, np.argmin(np.abs(time_grid - t)))
        
        mc_prices = np.ones(n_periods)
        mc_forward_rates = np.zeros(n_periods)
        
        for T_idx in range(idx_t, n_periods):
            if T_idx == idx_t:
                mc_forward_rates[T_idx] = np.mean(sim_paths[:, idx_t])
                continue
            
            # CORRECTED: Integrate from idx_t to T_idx
            integrals_t_to_T = np.trapz(
                sim_paths[:, idx_t:T_idx+1], 
                time_grid[idx_t:T_idx+1], 
                axis=1
            )
            
            discount_factors = np.exp(-integrals_t_to_T)
            mc_prices[T_idx] = np.mean(discount_factors)
            
            r_T = sim_paths[:, T_idx]
            numerator = np.mean(r_T * discount_factors)
            
            if mc_prices[T_idx] > 1e-12:
                mc_forward_rates[T_idx] = numerator / mc_prices[T_idx]
            else:
                mc_forward_rates[T_idx] = (mc_forward_rates[T_idx-1] 
                                          if T_idx > idx_t + 1 
                                          else np.mean(r_T))
        
        return mc_prices, mc_forward_rates
    
    def _compute_market_forward_rates(self, market_prices: np.ndarray, 
                                     time_grid: np.ndarray) -> np.ndarray:
        """
        Compute market instantaneous forward rates.
        
        f^M_t(T) = -∂/∂T log P^M_t(T)
        """
        n_periods = len(market_prices)
        market_forwards = np.zeros(n_periods)
        log_prices = np.log(np.maximum(market_prices, 1e-10))
        
        for i in range(n_periods):
            if i == 0:
                if time_grid[i] > 0:
                    market_forwards[i] = -log_prices[i] / time_grid[i]
                else:
                    market_forwards[i] = -(log_prices[1] - log_prices[0]) / (time_grid[1] - time_grid[0])
            elif i == n_periods - 1:
                dT = time_grid[i] - time_grid[i-1]
                market_forwards[i] = -(log_prices[i] - log_prices[i-1]) / dT
            else:
                dT = time_grid[i+1] - time_grid[i-1]
                market_forwards[i] = -(log_prices[i+1] - log_prices[i-1]) / dT
        
        if n_periods >= 5:
            market_forwards = gaussian_filter1d(market_forwards, sigma=0.5)
        
        return market_forwards
    
    def deterministic_shift_adjustment(self, sim_paths: np.ndarray, 
                                      market_prices: np.ndarray,
                                      time_grid: Optional[np.ndarray] = None,
                                      smooth_shift: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        CORRECTED: Deterministic shift adjustment (Proposition 1).
        
        Computes: φ(T) = f^M_t(T) - f̂_t(T)
        Applies: P̃_t(T) = exp(-∫_t^T φ(s)ds) · P̂_t(T)
        
        Parameters:
        -----------
        sim_paths : ndarray, shape (n_paths, n_periods)
            Simulated short rate paths
        market_prices : ndarray, shape (n_periods,)
            Market zero-coupon bond prices
        time_grid : ndarray, optional
            Time points (if None, uses uniform grid)
        smooth_shift : bool, default=True
            Apply Gaussian smoothing to shift function
            
        Returns:
        --------
        adjusted_prices : ndarray
            Adjusted bond prices satisfying FTAP
        shift : ndarray
            Deterministic shift function φ(T)
        """
        n_periods = sim_paths.shape[1]
        
        if time_grid is None:
            time_grid = np.arange(n_periods) * self.dt
        
        mc_prices, mc_forward_rates = self._compute_forward_rates(sim_paths, time_grid, t=0.0)
        market_forward_rates = self._compute_market_forward_rates(market_prices, time_grid)
        
        # Equation 4: φ(T) = f^M_t(T) - f̂_t(T)
        shift = market_forward_rates - mc_forward_rates
        
        if smooth_shift and n_periods >= 5:
            shift = gaussian_filter1d(shift, sigma=1.0)
        
        # CORRECTED: Equation 6 with proper integration
        adjusted_prices = np.zeros(n_periods)
        for T_idx in range(n_periods):
            cum_shift_integral = np.trapz(shift[:T_idx+1], time_grid[:T_idx+1])
            adjusted_prices[T_idx] = np.exp(-cum_shift_integral) * mc_prices[T_idx]
        
        self.shift_function = shift
        return adjusted_prices, shift
    
    def get_adjusted_paths(self, sim_paths: np.ndarray) -> np.ndarray:
        """Get risk-neutral adjusted paths: r̃(s) = r(s) + φ(s)"""
        if self.shift_function is None:
            raise ValueError("Must run deterministic_shift_adjustment first")
        return sim_paths + self.shift_function[:sim_paths.shape[1]]
    
    def monte_carlo_price_with_ci(self, sim_paths: np.ndarray, 
                                  time_grid: np.ndarray,
                                  T: float, 
                                  alpha: float = 0.05,
                                  use_adjusted: bool = False) -> PricingResult:
        """
        Compute Monte Carlo ZCB price with confidence intervals.
        
        Returns PricingResult with price, std_error, CI bounds, n_simulations
        """
        if use_adjusted and self.shift_function is not None:
            paths = self.get_adjusted_paths(sim_paths)
        else:
            paths = sim_paths
        
        idx_T = np.argmin(np.abs(time_grid - T))
        n_sims = paths.shape[0]
        
        integrals = np.trapz(paths[:, :idx_T+1], time_grid[:idx_T+1], axis=1)
        discount_factors = np.exp(-integrals)
        
        price = np.mean(discount_factors)
        std_error = np.std(discount_factors, ddof=1) / np.sqrt(n_sims)
        
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error
        
        return PricingResult(
            price=price,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_simulations=n_sims
        )
    
    def price_cap(self, sim_paths: np.ndarray, 
                 time_grid: np.ndarray,
                 strike: float = 0.03, 
                 cap_maturity: float = 5.0, 
                 payment_freq: float = 0.25,
                 notional: float = 100000,
                 use_adjusted_rates: bool = True) -> Tuple[float, float, List[CapletResult]]:
        """
        CORRECTED: Price interest rate cap (Equations 11-13).
        
        Cap = Sum of caplets with quarterly resets
        
        Returns:
        --------
        cap_value : float
            Total cap value
        std_error : float
            Standard error (computed across paths)
        caplet_details : list of CapletResult
            Details for each caplet
        """
        if use_adjusted_rates and self.shift_function is not None:
            paths = self.get_adjusted_paths(sim_paths)
        else:
            paths = sim_paths
        
        n_paths = paths.shape[0]
        n_periods = paths.shape[1]
        
        reset_times = np.arange(payment_freq, 
                               min(cap_maturity, time_grid[-1]) + payment_freq/2, 
                               payment_freq)
        payment_times = reset_times + payment_freq
        
        caplet_details = []
        total_payoffs = np.zeros(n_paths)
        
        for T_reset, T_payment in zip(reset_times, payment_times):
            idx_reset = np.argmin(np.abs(time_grid - T_reset))
            idx_payment = np.argmin(np.abs(time_grid - T_payment))
            
            if idx_payment >= n_periods:
                continue
            
            caplet_payoffs = np.zeros(n_paths)
            forward_rates = np.zeros(n_paths)
            
            for i in range(n_paths):
                integral_reset = np.trapz(paths[i, :idx_reset+1], 
                                         time_grid[:idx_reset+1])
                P_0_to_reset = np.exp(-integral_reset) if idx_reset > 0 else 1.0
                
                integral_payment = np.trapz(paths[i, :idx_payment+1], 
                                            time_grid[:idx_payment+1])
                P_0_to_payment = np.exp(-integral_payment)
                
                if P_0_to_payment > 1e-12:
                    forward_rate = (P_0_to_reset / P_0_to_payment - 1) / payment_freq
                else:
                    forward_rate = 0.0
                
                forward_rates[i] = forward_rate
                intrinsic = max(forward_rate - strike, 0)
                payoff = notional * payment_freq * intrinsic
                caplet_payoffs[i] = payoff * P_0_to_payment
            
            caplet_value = np.mean(caplet_payoffs)
            caplet_std_error = np.std(caplet_payoffs, ddof=1) / np.sqrt(n_paths)
            
            total_payoffs += caplet_payoffs
            
            caplet_details.append(CapletResult(
                reset_time=T_reset,
                payment_time=T_payment,
                value=caplet_value,
                std_error=caplet_std_error,
                strike=strike,
                forward_rate_mean=np.mean(forward_rates)
            ))
        
        cap_value = np.mean(total_payoffs)
        cap_std_error = np.std(total_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return cap_value, cap_std_error, caplet_details
    
    def price_swaption(self, sim_paths: np.ndarray,
                      time_grid: np.ndarray,
                      T_option: float,
                      swap_maturity: float,
                      strike: float,
                      payment_freq: float = 0.5,
                      notional: float = 100000,
                      is_payer: bool = True,
                      use_adjusted_rates: bool = True) -> PricingResult:
        """
        Price payer/receiver swaption (Equation 17).
        
        Payoff at T_option:
        - Payer: max(S - K, 0) × A
        - Receiver: max(K - S, 0) × A
        
        Where S = swap rate, A = annuity
        """
        if use_adjusted_rates and self.shift_function is not None:
            paths = self.get_adjusted_paths(sim_paths)
        else:
            paths = sim_paths
        
        n_paths = paths.shape[0]
        idx_option = np.argmin(np.abs(time_grid - T_option))
        
        swap_end = T_option + swap_maturity
        payment_dates = np.arange(T_option + payment_freq, swap_end + payment_freq/2, payment_freq)
        
        payoffs = np.zeros(n_paths)
        
        for i in range(n_paths):
            zcb_prices = []
            for T_j in payment_dates:
                if T_j > time_grid[-1]:
                    break
                idx_j = np.argmin(np.abs(time_grid - T_j))
                integral = np.trapz(paths[i, idx_option:idx_j+1], 
                                   time_grid[idx_option:idx_j+1])
                zcb_prices.append(np.exp(-integral))
            
            if len(zcb_prices) == 0:
                continue
            
            annuity = payment_freq * sum(zcb_prices)
            
            if annuity > 1e-10:
                swap_rate = (1.0 - zcb_prices[-1]) / annuity
                
                if is_payer:
                    intrinsic = max(swap_rate - strike, 0)
                else:
                    intrinsic = max(strike - swap_rate, 0)
                
                payoffs[i] = notional * annuity * intrinsic
            
            integral_to_present = np.trapz(paths[i, :idx_option+1], 
                                          time_grid[:idx_option+1])
            payoffs[i] *= np.exp(-integral_to_present)
        
        price = np.mean(payoffs)
        std_error = np.std(payoffs, ddof=1) / np.sqrt(n_paths)
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return PricingResult(
            price=price,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_simulations=n_paths
        )
    
    def validate_arbitrage_free(self, adjusted_prices: np.ndarray, 
                               market_prices: np.ndarray, 
                               tolerance: float = 0.001) -> bool:
        """Validate that adjusted prices match market within tolerance"""
        errors = np.abs(adjusted_prices - market_prices) / market_prices
        max_error = np.max(errors)
        avg_error = np.mean(errors)
        
        print(f"\nArbitrage-Free Validation:")
        print(f"  Max relative error: {max_error:.4%}")
        print(f"  Avg relative error: {avg_error:.4%}")
        print(f"  Tolerance: {tolerance:.2%}")
        print(f"  Status: {'✓ PASS' if max_error <= tolerance else '✗ FAIL'}")
        
        return max_error <= tolerance


