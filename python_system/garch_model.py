#!/usr/bin/env python3
"""
GARCH(1,1) Model Fitting Module
Fits GARCH(1,1) with Student-t distribution for fat tails
Extracts AIC, BIC, and distribution parameters
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


def fit_garch_model(returns, dist='t'):
    """
    Fit GARCH(1,1) model with Student-t distribution
    
    Args:
        returns: Array of log returns
        dist: Distribution ('normal', 't', 'skewt', 'ged')
    
    Returns:
        dict with GARCH parameters and fit statistics
    """
    try:
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 30:
            return get_fallback_garch()
        
        # Scale returns to percentage (arch expects percentage returns)
        returns_pct = returns * 100
        
        # Fit GARCH(1,1) model with Student-t distribution
        model = arch_model(
            returns_pct,
            vol='Garch',
            p=1,  # GARCH order
            q=1,  # ARCH order
            dist=dist,  # Student-t for fat tails
            rescale=False
        )
        
        # Fit with MLE
        result = model.fit(disp='off', show_warning=False)
        
        # Extract parameters
        params = result.params
        
        # Get conditional volatility (annualized)
        conditional_vol = result.conditional_volatility
        if hasattr(conditional_vol, 'iloc'):
            current_vol = float(conditional_vol.iloc[-1] / 100)
        else:
            current_vol = float(conditional_vol[-1] / 100)  # numpy array
        
        # Annualize volatility (assuming daily data)
        current_vol_annual = current_vol * np.sqrt(252)
        
        # Extract GARCH parameters
        omega = float(params.get('omega', 0))
        alpha = float(params.get('alpha[1]', 0))
        beta = float(params.get('beta[1]', 0))
        
        # Extract distribution parameter (degrees of freedom for Student-t)
        nu = float(params.get('nu', 5.0))  # nu is degrees of freedom
        
        # Model fit statistics
        aic = float(result.aic)
        bic = float(result.bic)
        log_likelihood = float(result.loglikelihood)
        
        # Persistence (alpha + beta)
        persistence = alpha + beta
        
        # Unconditional volatility
        uncond_vol = np.sqrt(omega / (1 - alpha - beta)) if (alpha + beta) < 1 else current_vol_annual
        
        return {
            'model': 'GARCH(1,1)',
            'distribution': 'Student-t',
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'fat_tail_df': nu,  # Degrees of freedom (lower = fatter tails)
            'current_volatility': current_vol_annual,
            'unconditional_volatility': uncond_vol,
            'persistence': persistence,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'converged': result.convergence_flag == 0,
            'n_obs': len(returns)
        }
        
    except Exception as e:
        print(f"GARCH fitting error: {e}")
        return get_fallback_garch()


def get_fallback_garch():
    """
    Return fallback GARCH parameters when fitting fails
    """
    return {
        'model': 'GARCH(1,1)',
        'distribution': 'Student-t',
        'omega': 0.0,
        'alpha': 0.1,
        'beta': 0.85,
        'fat_tail_df': 5.0,
        'current_volatility': 0.25,
        'unconditional_volatility': 0.25,
        'persistence': 0.95,
        'aic': None,
        'bic': None,
        'log_likelihood': None,
        'converged': False,
        'n_obs': 0
    }


def forecast_garch_volatility(returns, horizon=30, dist='t'):
    """
    Forecast volatility using GARCH(1,1) model
    
    Args:
        returns: Array of log returns
        horizon: Forecast horizon in days
        dist: Distribution type
    
    Returns:
        dict with volatility forecast
    """
    try:
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 30:
            return None
        
        returns_pct = returns * 100
        
        model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist=dist, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        # Forecast volatility
        forecast = result.forecast(horizon=horizon, reindex=False)
        
        # Get variance forecast
        variance_forecast = forecast.variance.values[-1, :]
        
        # Convert to volatility (annualized)
        vol_forecast = np.sqrt(variance_forecast) / 100 * np.sqrt(252)
        
        return {
            'horizon': horizon,
            'mean_forecast_vol': float(np.mean(vol_forecast)),
            'min_forecast_vol': float(np.min(vol_forecast)),
            'max_forecast_vol': float(np.max(vol_forecast)),
            'forecast_path': vol_forecast.tolist()
        }
        
    except Exception as e:
        print(f"GARCH forecast error: {e}")
        return None


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=252) * 0.02  # Simulated returns with fat tails
    
    print("Testing GARCH(1,1) model fitting...")
    result = fit_garch_model(returns)
    
    print("\nGARCH Model Results:")
    print(f"Model: {result['model']}")
    print(f"Distribution: {result['distribution']}")
    print(f"Omega (ω): {result['omega']:.6f}")
    print(f"Alpha (α): {result['alpha']:.6f}")
    print(f"Beta (β): {result['beta']:.6f}")
    print(f"Fat-tail DF (ν): {result['fat_tail_df']:.2f}")
    print(f"Current Volatility: {result['current_volatility']:.4f}")
    print(f"Persistence (α+β): {result['persistence']:.4f}")
    if result['aic'] is not None:
        print(f"AIC: {result['aic']:.2f}")
        print(f"BIC: {result['bic']:.2f}")
    else:
        print("AIC: N/A (fallback)")
        print("BIC: N/A (fallback)")
    print(f"Converged: {result['converged']}")
    
    print("\nTesting GARCH volatility forecast...")
    forecast = forecast_garch_volatility(returns, horizon=30)
    if forecast:
        print(f"30-day mean forecast vol: {forecast['mean_forecast_vol']:.4f}")
        print(f"Vol range: [{forecast['min_forecast_vol']:.4f}, {forecast['max_forecast_vol']:.4f}]")
