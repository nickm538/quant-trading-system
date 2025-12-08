#!/usr/bin/env python3
"""
Finnhub Earnings Calendar Integration
FREE tier: 1 month historical + new updates
Endpoint: /calendar/earnings
"""

import sys
import requests
from datetime import datetime, timedelta
import json

# Finnhub FREE API key (60 API calls/minute)
FINNHUB_API_KEY = "ct2oo9hr01qr6bvnlmtgct2oo9hr01qr6bvnlmu0"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

def fetch_earnings_calendar(symbol=None, days_ahead=45):
    """
    Fetch earnings calendar from Finnhub
    
    Args:
        symbol: Stock symbol (optional, if None returns all earnings)
        days_ahead: Number of days to look ahead (default 45)
    
    Returns:
        dict: Earnings calendar data
    """
    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    
    params = {
        'from': today.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }
    
    if symbol:
        params['symbol'] = symbol.upper()
    
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/calendar/earnings",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            'success': True,
            'data': data.get('earningsCalendar', []),
            'count': len(data.get('earningsCalendar', []))
        }
    
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'data': []
        }

def get_next_earnings(symbol):
    """
    Get next earnings date for a specific stock
    
    Args:
        symbol: Stock symbol
    
    Returns:
        dict: Next earnings information
    """
    result = fetch_earnings_calendar(symbol=symbol, days_ahead=90)
    
    if not result['success'] or not result['data']:
        return {
            'has_earnings': False,
            'symbol': symbol,
            'message': 'No upcoming earnings in next 90 days'
        }
    
    # Get the earliest earnings date
    earnings = sorted(result['data'], key=lambda x: x['date'])
    next_earnings = earnings[0]
    
    earnings_date = datetime.strptime(next_earnings['date'], '%Y-%m-%d')
    days_until = (earnings_date - datetime.now()).days
    
    return {
        'has_earnings': True,
        'symbol': symbol,
        'date': next_earnings['date'],
        'days_until': days_until,
        'quarter': next_earnings.get('quarter'),
        'year': next_earnings.get('year'),
        'eps_estimate': next_earnings.get('epsEstimate'),
        'revenue_estimate': next_earnings.get('revenueEstimate'),
        'hour': next_earnings.get('hour', 'unknown'),  # 'bmo' (before market), 'amc' (after close), 'dmh' (during hours)
        'timing': 'Before Market Open' if next_earnings.get('hour') == 'bmo' else 
                 'After Market Close' if next_earnings.get('hour') == 'amc' else
                 'During Market Hours' if next_earnings.get('hour') == 'dmh' else
                 'Unknown'
    }

def analyze_earnings_impact(symbol):
    """
    Analyze potential earnings impact for options trading
    
    Args:
        symbol: Stock symbol
    
    Returns:
        dict: Earnings impact analysis
    """
    earnings_info = get_next_earnings(symbol)
    
    if not earnings_info['has_earnings']:
        return {
            'iv_crush_risk': 'LOW',
            'recommendation': 'Normal options trading - no imminent earnings',
            'details': earnings_info
        }
    
    days_until = earnings_info['days_until']
    
    if days_until <= 7:
        return {
            'iv_crush_risk': 'VERY HIGH',
            'recommendation': 'AVOID buying options - IV crush imminent. Consider selling premium or waiting until after earnings.',
            'warning': f'Earnings in {days_until} days - IV will collapse post-earnings',
            'details': earnings_info
        }
    elif days_until <= 14:
        return {
            'iv_crush_risk': 'HIGH',
            'recommendation': 'CAUTION - IV starting to inflate. Consider shorter-dated trades or wait until after earnings.',
            'warning': f'Earnings in {days_until} days - IV inflation beginning',
            'details': earnings_info
        }
    elif days_until <= 30:
        return {
            'iv_crush_risk': 'MODERATE',
            'recommendation': 'Monitor IV levels. Consider earnings play strategies (straddles, strangles) if IV is still reasonable.',
            'details': earnings_info
        }
    else:
        return {
            'iv_crush_risk': 'LOW',
            'recommendation': 'Normal options trading window - earnings not immediate concern',
            'details': earnings_info
        }

def main():
    """Test the Finnhub earnings calendar"""
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        print(f"\n=== Earnings Analysis for {symbol} ===\n")
        
        analysis = analyze_earnings_impact(symbol)
        print(json.dumps(analysis, indent=2))
    else:
        print("\n=== Next Week's Earnings (All Stocks) ===\n")
        result = fetch_earnings_calendar(days_ahead=7)
        
        if result['success']:
            print(f"Found {result['count']} earnings reports in next 7 days:\n")
            for earning in result['data'][:10]:  # Show first 10
                print(f"{earning['symbol']:6} - {earning['date']} ({earning.get('hour', 'N/A')})")
                print(f"         EPS Est: ${earning.get('epsEstimate', 'N/A')}")
                print()
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
