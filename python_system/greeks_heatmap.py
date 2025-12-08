#!/usr/bin/env python3
"""
Options Greeks Heatmap Generator
Calculates Delta, Gamma, Theta, Vega, Rho for multiple strikes and expirations
"""

import sys
import json
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import yfinance as yf

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all Greeks using Black-Scholes model
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
    
    Returns:
        dict with Delta, Gamma, Theta, Vega, Rho
    """
    if T <= 0:
        # At expiration
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {
            'delta': delta,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (same for calls and puts, per 1% change in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta (per day)
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Rho (per 1% change in interest rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega),
        'rho': float(rho)
    }

def get_risk_free_rate():
    """Fetch current 10-year Treasury yield"""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100  # Convert to decimal
    except:
        pass
    return 0.04  # Fallback to 4%

def generate_greeks_heatmap(symbol, num_strikes=15, num_expirations=6):
    """
    Generate Greeks heatmap data for a stock
    
    Args:
        symbol: Stock ticker
        num_strikes: Number of strike prices to analyze
        num_expirations: Number of expiration dates to analyze
    
    Returns:
        dict with heatmap data
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Get options chain
        expirations = stock.options[:num_expirations] if len(stock.options) >= num_expirations else stock.options
        
        if not expirations:
            return {
                'success': False,
                'error': f'No options data available for {symbol}'
            }
        
        # Get risk-free rate
        r = get_risk_free_rate()
        
        # Generate strike prices around current price
        strike_min = current_price * 0.85
        strike_max = current_price * 1.15
        strikes = np.linspace(strike_min, strike_max, num_strikes)
        
        # Calculate Greeks for each strike/expiration combination
        heatmap_data = {
            'symbol': symbol,
            'current_price': float(current_price),
            'risk_free_rate': float(r),
            'strikes': [float(k) for k in strikes],
            'expirations': [],
            'calls': {
                'delta': [],
                'gamma': [],
                'theta': [],
                'vega': [],
                'rho': []
            },
            'puts': {
                'delta': [],
                'gamma': [],
                'theta': [],
                'vega': [],
                'rho': []
            }
        }
        
        for exp_date in expirations:
            # Calculate time to expiration
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_exp = (exp_datetime - datetime.now()).days
            T = days_to_exp / 365.0
            
            heatmap_data['expirations'].append({
                'date': exp_date,
                'days': days_to_exp
            })
            
            # Get options chain for this expiration
            try:
                opt_chain = stock.option_chain(exp_date)
                
                # Calculate average IV from the chain
                calls_iv = opt_chain.calls['impliedVolatility'].median()
                puts_iv = opt_chain.puts['impliedVolatility'].median()
                
                if np.isnan(calls_iv):
                    calls_iv = 0.3  # Default 30%
                if np.isnan(puts_iv):
                    puts_iv = 0.3
                
            except:
                calls_iv = 0.3
                puts_iv = 0.3
            
            # Calculate Greeks for each strike
            call_row = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
            put_row = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
            
            for K in strikes:
                # Call Greeks
                call_greeks = black_scholes_greeks(current_price, K, T, r, calls_iv, 'call')
                call_row['delta'].append(call_greeks['delta'])
                call_row['gamma'].append(call_greeks['gamma'])
                call_row['theta'].append(call_greeks['theta'])
                call_row['vega'].append(call_greeks['vega'])
                call_row['rho'].append(call_greeks['rho'])
                
                # Put Greeks
                put_greeks = black_scholes_greeks(current_price, K, T, r, puts_iv, 'put')
                put_row['delta'].append(put_greeks['delta'])
                put_row['gamma'].append(put_greeks['gamma'])
                put_row['theta'].append(put_greeks['theta'])
                put_row['vega'].append(put_greeks['vega'])
                put_row['rho'].append(put_greeks['rho'])
            
            # Add rows to heatmap
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                heatmap_data['calls'][greek].append(call_row[greek])
                heatmap_data['puts'][greek].append(put_row[greek])
        
        heatmap_data['success'] = True
        return heatmap_data
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol
        }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'Symbol required'}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    num_strikes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    num_expirations = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    
    result = generate_greeks_heatmap(symbol, num_strikes, num_expirations)
    print(json.dumps(result, indent=2))
