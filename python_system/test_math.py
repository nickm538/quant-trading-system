#!/usr/bin/env python3
"""Test advanced math calculations for production accuracy."""

import numpy as np
from scipy.stats import norm

def test_black_scholes():
    """Test Black-Scholes formula."""
    S = 100  # Stock price
    K = 100  # Strike price
    T = 1.0  # Time to expiration (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.20  # Volatility
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    print('Black-Scholes Calculation Test:')
    print(f'  Call Price: ${call_price:.2f}')
    print(f'  Expected Call: ~$10.45')
    passed = 10 <= call_price <= 11
    print(f'  {"PASS" if passed else "FAIL"}')
    return passed

def test_monte_carlo():
    """Test Monte Carlo simulation."""
    np.random.seed(42)
    S0 = 100
    mu = 0.10
    sigma = 0.20
    T = 1.0
    n_sims = 10000
    
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    mean_price = np.mean(ST)
    expected_mean = S0 * np.exp(mu * T)
    
    print(f'\nMonte Carlo Simulation Test:')
    print(f'  Mean Final Price: ${mean_price:.2f}')
    print(f'  Expected Mean: ${expected_mean:.2f}')
    
    error = abs(mean_price - expected_mean) / expected_mean
    passed = error < 0.05
    print(f'  {"PASS" if passed else "FAIL"}')
    return passed

if __name__ == '__main__':
    all_pass = test_black_scholes() and test_monte_carlo()
    print(f'\n=== {"ALL ADVANCED MATH TESTS PASSED" if all_pass else "SOME TESTS FAILED"} ===')
