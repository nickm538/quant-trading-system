"""Debug options data structure"""
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

from data.enhanced_data_ingestion import EnhancedDataIngestion

# Get data
data_ingestion = EnhancedDataIngestion()
complete_data = data_ingestion.get_complete_stock_data('AAPL')
price_data = complete_data['price_data']
current_price = price_data['close'].iloc[-1]

print(f"Current price: ${current_price:.2f}")

# Get options
options_data = complete_data.get('options', {})
print(f"\nOptions data type: {type(options_data)}")
print(f"Options data keys: {list(options_data.keys())[:3]}")

# Check first expiration
if options_data:
    first_key = list(options_data.keys())[0]
    print(f"\nFirst key: {first_key} (type: {type(first_key)})")
    
    first_data = options_data[first_key]
    print(f"First data type: {type(first_data)}")
    print(f"First data keys: {list(first_data.keys()) if isinstance(first_data, dict) else 'N/A'}")
    
    # Try to parse expiration
    if isinstance(first_key, str):
        try:
            exp_date = datetime.strptime(first_key, '%Y-%m-%d')
            print(f"Parsed expiration: {exp_date}")
        except:
            print(f"Could not parse as date string")
    
    # Check calls structure
    calls = first_data.get('calls', pd.DataFrame())
    print(f"\nCalls type: {type(calls)}")
    print(f"Calls shape: {calls.shape if isinstance(calls, pd.DataFrame) else 'N/A'}")
    
    if isinstance(calls, pd.DataFrame) and not calls.empty:
        print(f"\nCalls columns: {list(calls.columns)}")
        print(f"\nFirst call:")
        print(calls.iloc[0])
        
        # Test delta calculation
        row = calls.iloc[0]
        strike = row['strike']
        last_price = row.get('lastPrice', 0)
        iv_raw = row.get('impliedVolatility', None)
        
        print(f"\n=== Testing Delta Calculation ===")
        print(f"Strike: ${strike:.2f}")
        print(f"Last price: ${last_price:.2f}")
        print(f"IV (raw): {iv_raw}")
        print(f"IV is None: {iv_raw is None}")
        print(f"IV is NaN: {pd.isna(iv_raw) if iv_raw is not None else 'N/A'}")
        
        # Calculate historical vol
        returns = price_data['close'].pct_change().dropna()
        hist_vol = returns.tail(30).std() * np.sqrt(252)
        min_vol = max(hist_vol, 0.15)
        
        print(f"Historical vol: {hist_vol:.1%}")
        print(f"Min vol: {min_vol:.1%}")
        
        # Determine IV to use
        if iv_raw is None or pd.isna(iv_raw) or iv_raw <= 0:
            iv = min_vol
            print(f"Using fallback IV: {iv:.1%}")
        else:
            iv = max(iv_raw, min_vol)
            print(f"Using IV: {iv:.1%}")
        
        # Calculate delta
        # Parse expiration from key
        if isinstance(first_key, str):
            try:
                exp_date = datetime.strptime(first_key, '%Y-%m-%d')
                days_to_expiry = (exp_date - datetime.now()).days
                T = days_to_expiry / 365.0
                
                print(f"\nExpiration: {exp_date.strftime('%Y-%m-%d')}")
                print(f"Days to expiry: {days_to_expiry}")
                print(f"T (years): {T:.4f}")
                
                r = 0.045
                d1 = (np.log(current_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                delta = norm.cdf(d1)
                
                print(f"\nd1: {d1:.4f}")
                print(f"Delta: {delta:.4f}")
                print(f"In range 0.3-0.6? {0.3 <= delta <= 0.6}")
                
            except Exception as e:
                print(f"Error calculating delta: {e}")
