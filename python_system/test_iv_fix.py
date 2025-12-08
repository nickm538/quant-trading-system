"""Test IV fix and find options in delta range"""
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

# Historical vol
returns = price_data['close'].pct_change().dropna()
hist_vol = returns.tail(30).std() * np.sqrt(252)
min_vol = max(hist_vol, 0.15)

print(f"Historical vol: {hist_vol:.1%}")
print(f"Min vol: {min_vol:.1%}\n")

# Get options
options_data = complete_data.get('options', {})

# Check all expirations
r = 0.045
total_in_range = 0

for exp_key, exp_data in list(options_data.items())[:3]:  # First 3 expirations
    exp_date = datetime.strptime(exp_key, '%Y-%m-%d')
    days_to_expiry = (exp_date - datetime.now()).days
    
    if days_to_expiry < 7:
        continue
    
    T = days_to_expiry / 365.0
    
    print(f"=== Expiration: {exp_key} ({days_to_expiry} days) ===")
    
    calls = exp_data.get('calls', pd.DataFrame())
    
    if calls.empty:
        continue
    
    print(f"Total calls: {len(calls)}")
    
    # Test all calls
    in_range_count = 0
    for idx, row in calls.iterrows():
        strike = row['strike']
        last_price = row.get('lastPrice', 0)
        
        if last_price == 0:
            continue
        
        # Apply IV fix
        iv_raw = row.get('impliedVolatility', None)
        if iv_raw is None or pd.isna(iv_raw) or iv_raw <= 0:
            iv = min_vol
        else:
            if iv_raw > 1:
                iv_raw = iv_raw / 100
            iv = max(iv_raw, min_vol)
        
        # Calculate delta
        d1 = (np.log(current_price / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        delta = norm.cdf(d1)
        
        if 0.3 <= delta <= 0.6:
            in_range_count += 1
            total_in_range += 1
            print(f"  ✓ Strike ${strike:.2f}: delta={delta:.3f}, IV={iv:.1%}, price=${last_price:.2f}")
            
            if in_range_count >= 3:  # Show first 3 per expiration
                break
    
    print(f"Found {in_range_count} calls in delta range 0.3-0.6\n")

print(f"\n=== TOTAL: {total_in_range} options in delta range 0.3-0.6 ===")
