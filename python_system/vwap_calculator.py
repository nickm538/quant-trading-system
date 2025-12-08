#!/usr/bin/env python3
"""
VWAP (Volume Weighted Average Price) Calculator
Uses Yahoo Finance intraday data (FREE, no API key required)

VWAP Formula: Σ(Price × Volume) / Σ(Volume)
Where Price = (High + Low + Close) / 3 (typical price)
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json
from datetime import datetime

def calculate_vwap(symbol, interval='5m', range_='1d'):
    """
    Calculate VWAP from intraday data
    
    Args:
        symbol: Stock symbol
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m)
        range_: Time range (1d, 5d)
    
    Returns:
        dict: VWAP analysis
    """
    client = ApiClient()
    
    try:
        # Fetch intraday data from Yahoo Finance
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': symbol,
            'region': 'US',
            'interval': interval,
            'range': range_,
            'includeAdjustedClose': False
        })
        
        if not response or 'chart' not in response:
            return {'error': 'No data received from API'}
        
        result = response['chart']['result'][0]
        meta = result['meta']
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Calculate VWAP
        cumulative_pv = 0  # Price × Volume
        cumulative_volume = 0
        
        vwap_values = []
        
        for i in range(len(timestamps)):
            high = quotes['high'][i]
            low = quotes['low'][i]
            close = quotes['close'][i]
            volume = quotes['volume'][i]
            
            # Skip if any value is None
            if None in [high, low, close, volume]:
                continue
            
            # Typical price = (High + Low + Close) / 3
            typical_price = (high + low + close) / 3
            
            # Cumulative calculations
            cumulative_pv += typical_price * volume
            cumulative_volume += volume
            
            # VWAP at this point
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                vwap_values.append({
                    'timestamp': timestamps[i],
                    'vwap': vwap,
                    'price': close,
                    'volume': volume
                })
        
        if not vwap_values:
            return {'error': 'No valid data points for VWAP calculation'}
        
        # Current VWAP (latest value)
        current_vwap = vwap_values[-1]['vwap']
        current_price = meta['regularMarketPrice']
        
        # Calculate deviation from VWAP
        deviation = ((current_price - current_vwap) / current_vwap) * 100
        
        # Determine signal
        if deviation > 1.0:
            signal = 'ABOVE VWAP'
            interpretation = 'Bullish - Price trading above average, buyers in control'
        elif deviation < -1.0:
            signal = 'BELOW VWAP'
            interpretation = 'Bearish - Price trading below average, sellers in control'
        else:
            signal = 'AT VWAP'
            interpretation = 'Neutral - Price near fair value, balanced market'
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'vwap': current_vwap,
            'deviation_percent': deviation,
            'signal': signal,
            'interpretation': interpretation,
            'data_points': len(vwap_values),
            'interval': interval,
            'range': range_,
            'total_volume': cumulative_volume,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'symbol': symbol
        }

def get_vwap_bands(symbol, interval='5m', range_='1d', std_dev=2):
    """
    Calculate VWAP with standard deviation bands
    
    Args:
        symbol: Stock symbol
        interval: Data interval
        range_: Time range
        std_dev: Number of standard deviations for bands
    
    Returns:
        dict: VWAP with bands
    """
    client = ApiClient()
    
    try:
        response = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': symbol,
            'region': 'US',
            'interval': interval,
            'range': range_,
            'includeAdjustedClose': False
        })
        
        if not response or 'chart' not in response:
            return {'error': 'No data received from API'}
        
        result = response['chart']['result'][0]
        meta = result['meta']
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Calculate VWAP and variance
        cumulative_pv = 0
        cumulative_volume = 0
        cumulative_pv2 = 0  # For variance calculation
        
        vwap_data = []
        
        for i in range(len(timestamps)):
            high = quotes['high'][i]
            low = quotes['low'][i]
            close = quotes['close'][i]
            volume = quotes['volume'][i]
            
            if None in [high, low, close, volume]:
                continue
            
            typical_price = (high + low + close) / 3
            
            cumulative_pv += typical_price * volume
            cumulative_pv2 += (typical_price ** 2) * volume
            cumulative_volume += volume
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                
                # Calculate variance and standard deviation
                variance = (cumulative_pv2 / cumulative_volume) - (vwap ** 2)
                std = variance ** 0.5 if variance > 0 else 0
                
                vwap_data.append({
                    'vwap': vwap,
                    'upper_band': vwap + (std_dev * std),
                    'lower_band': vwap - (std_dev * std),
                    'price': close
                })
        
        if not vwap_data:
            return {'error': 'No valid data points'}
        
        latest = vwap_data[-1]
        current_price = meta['regularMarketPrice']
        
        # Determine position relative to bands
        if current_price > latest['upper_band']:
            position = 'ABOVE UPPER BAND'
            signal = 'Overbought - Consider taking profits or waiting for pullback'
        elif current_price < latest['lower_band']:
            position = 'BELOW LOWER BAND'
            signal = 'Oversold - Potential buying opportunity if trend is bullish'
        elif current_price > latest['vwap']:
            position = 'ABOVE VWAP'
            signal = 'Bullish zone - Price above fair value'
        else:
            position = 'BELOW VWAP'
            signal = 'Bearish zone - Price below fair value'
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'vwap': latest['vwap'],
            'upper_band': latest['upper_band'],
            'lower_band': latest['lower_band'],
            'position': position,
            'signal': signal,
            'std_dev_used': std_dev,
            'data_points': len(vwap_data)
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'symbol': symbol
        }

def main():
    """Test VWAP calculator"""
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = 'AAPL'
    
    print(f"\n=== VWAP Analysis for {symbol} ===\n")
    
    # Basic VWAP
    vwap_result = calculate_vwap(symbol, interval='5m', range_='1d')
    print("Basic VWAP:")
    print(json.dumps(vwap_result, indent=2))
    
    # VWAP with bands
    print(f"\n=== VWAP Bands for {symbol} ===\n")
    bands_result = get_vwap_bands(symbol, interval='5m', range_='1d', std_dev=2)
    print("VWAP with Standard Deviation Bands:")
    print(json.dumps(bands_result, indent=2))

if __name__ == "__main__":
    main()
