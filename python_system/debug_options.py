"""Debug options analyzer to find why 0 candidates"""
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
print(f"Found {len(options_data)} expiration dates")

# Historical vol
returns = price_data['close'].pct_change().dropna()
hist_vol = returns.tail(30).std() * np.sqrt(252)
min_vol = max(hist_vol, 0.15)

print(f"Historical vol: {hist_vol:.1%}")
print(f"Min vol: {min_vol:.1%}")

# Check first expiration
if options_data:
    first_exp = list(options_data.keys())[0]
    exp_date = datetime.fromtimestamp(first_exp)
    days_to_expiry = (exp_date - datetime.now()).days
    T = days_to_expiry / 365.0
    
    print(f"\nFirst expiration: {exp_date.strftime('%Y-%m-%d')} ({days_to_expiry} days)")
    print(f"T = {T:.4f} years")
    
    calls = options_data[first_exp].get('calls', pd.DataFrame())
    
    if not calls.empty:
        print(f"\nFound {len(calls)} calls")
        print("\nFirst 5 calls:")
        print(calls[['strike', 'lastPrice', 'impliedVolatility', 'volume', 'openInterest']].head())
        
        # Test delta calculation on first call
        row = calls.iloc[0]
        strike = row['strike']
        last_price = row.get('lastPrice', 0)
        iv_raw = row.get('impliedVolatility', None)
        
        print(f"\nTesting first call:")
        print(f"  Strike: ${strike:.2f}")
        print(f"  Last price: ${last_price:.2f}")
        print(f"  IV (raw): {iv_raw}")
        print(f"  IV (is NaN): {pd.isna(iv_raw)}")
        
        if iv_raw is None or pd.isna(iv_raw) or iv_raw <= 0:
            iv = min_vol
            print(f"  Using fallback IV: {iv:.1%}")
        else:
            iv = max(iv_raw, min_vol)
            print(f"  Using IV: {iv:.1%}")
        
        # Calculate delta
        r = 0.045
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        delta = norm.cdf(d1)
        
        print(f"  Calculated delta: {delta:.4f}")
        print(f"  In range 0.3-0.6? {0.3 <= delta <= 0.6}")
        
        # Check all deltas
        print(f"\nChecking all {len(calls)} calls for delta range 0.3-0.6:")
        in_range = 0
        for idx, row in calls.iterrows():
            strike = row['strike']
            last_price = row.get('lastPrice', 0)
            
            if last_price == 0:
                continue
            
            iv_raw = row.get('impliedVolatility', None)
            if iv_raw is None or pd.isna(iv_raw) or iv_raw <= 0:
                iv = min_vol
            else:
                iv = max(iv_raw, min_vol)
            
            d1 = (np.log(current_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            delta = norm.cdf(d1)
            
            if 0.3 <= delta <= 0.6:
                in_range += 1
                print(f"  ✓ Strike ${strike:.2f}: delta={delta:.4f}, price=${last_price:.2f}, iv={iv:.1%}")
        
        print(f"\nTotal in range: {in_range}")
