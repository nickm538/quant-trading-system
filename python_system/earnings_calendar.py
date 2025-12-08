#!/usr/bin/env python3
"""
Earnings Calendar Module
Fetches upcoming earnings dates from yfinance
Stores in market_events table for event-driven trading
"""

import sys
import yfinance as yf
from datetime import datetime, timedelta
import json

def fetch_earnings_date(symbol):
    """
    Fetch earnings date from yfinance
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dict with earnings info or None
    """
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is None or calendar.empty:
            return None
        
        # Extract earnings date
        if 'Earnings Date' in calendar.index:
            earnings_dates = calendar.loc['Earnings Date']
            if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                earnings_date = earnings_dates[0]
            else:
                earnings_date = earnings_dates
            
            # Convert to datetime if needed
            if hasattr(earnings_date, 'date'):
                earnings_date = earnings_date.date()
            elif isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
            
            return {
                'symbol': symbol,
                'reportDate': str(earnings_date),
                'estimate': None  # yfinance doesn't provide EPS estimate in calendar
            }
        
        return None
        
    except Exception as e:
        print(f"Error fetching earnings date for {symbol}: {e}")
        return None

def get_upcoming_earnings(symbol, days_ahead=45):
    """
    Get upcoming earnings for a specific symbol
    
    Args:
        symbol: Stock symbol
        days_ahead: Number of days to look ahead
    
    Returns:
        Dict with earnings info or None
    """
    try:
        earnings = fetch_earnings_date(symbol)
        
        if not earnings:
            return None
        
        # Check if earnings is within days_ahead
        today = datetime.now().date()
        report_date = datetime.strptime(earnings['reportDate'], '%Y-%m-%d').date()
        
        if report_date < today:
            return None  # Earnings already happened
        
        days_until = (report_date - today).days
        
        if days_until > days_ahead:
            return None  # Earnings too far away
        
        earnings['days_until_earnings'] = days_until
        return earnings
        
    except Exception as e:
        print(f"Error getting upcoming earnings: {e}")
        return None

def analyze_earnings_impact(symbol):
    """
    Analyze potential earnings impact on options and stock
    
    Returns:
        Dict with earnings analysis
    """
    earnings = get_upcoming_earnings(symbol, days_ahead=45)
    
    if not earnings:
        return {
            'has_earnings': False,
            'days_until': None,
            'date': None,
            'estimate': None,
            'iv_crush_risk': 'LOW',
            'recommendation': 'No earnings in next 45 days - normal options trading'
        }
    
    days_until = earnings['days_until_earnings']
    
    # Determine IV crush risk based on days until earnings
    if days_until <= 7:
        iv_crush_risk = 'EXTREME'
        recommendation = 'AVOID buying options - IV crush imminent. Consider selling premium or waiting until after earnings.'
    elif days_until <= 14:
        iv_crush_risk = 'HIGH'
        recommendation = 'CAUTION with long options - IV will decay rapidly. Consider shorter-dated spreads.'
    elif days_until <= 30:
        iv_crush_risk = 'MODERATE'
        recommendation = 'Monitor IV levels - may see gradual increase. Consider calendar spreads.'
    else:
        iv_crush_risk = 'LOW'
        recommendation = 'Normal options trading - earnings far enough away.'
    
    return {
        'has_earnings': True,
        'days_until': days_until,
        'date': earnings['reportDate'],
        'estimate': earnings.get('estimate'),
        'fiscal_period': earnings.get('fiscalDateEnding'),
        'iv_crush_risk': iv_crush_risk,
        'recommendation': recommendation
    }

if __name__ == '__main__':
    # Test with AAPL
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    
    print(f"\n=== Earnings Calendar for {symbol} ===\n")
    
    analysis = analyze_earnings_impact(symbol)
    
    print(f"Has Upcoming Earnings: {analysis['has_earnings']}")
    if analysis['has_earnings']:
        print(f"Days Until Earnings: {analysis['days_until']}")
        print(f"Earnings Date: {analysis['date']}")
        print(f"EPS Estimate: ${analysis['estimate']}" if analysis['estimate'] else "EPS Estimate: N/A")
        print(f"Fiscal Period: {analysis['fiscal_period']}")
        print(f"\nIV Crush Risk: {analysis['iv_crush_risk']}")
        print(f"Recommendation: {analysis['recommendation']}")
    else:
        print(f"Recommendation: {analysis['recommendation']}")
    
    print("\n" + "="*50)
