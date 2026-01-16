#!/usr/bin/env python3
"""
UNIFIED SCANNER RUNNER
======================
Runs the 4 scanner modules from financial-analysis-system:
1. Dark Pool Scanner (Insider Movement/Oracle)
2. TTM Squeeze Scanner
3. Options Flow (Bear to Bull)
4. Breakout Detector

Usage:
    python run_scanners.py dark_pool AAPL
    python run_scanners.py ttm_squeeze AAPL
    python run_scanners.py options_flow AAPL
    python run_scanners.py breakout AAPL
"""

import sys
import json
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_dark_pool(symbol: str) -> dict:
    """Run Dark Pool Scanner"""
    try:
        from dark_pool_scanner import DarkPoolScanner
        scanner = DarkPoolScanner()
        result = scanner.get_dark_pool_analysis(symbol)
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def run_ttm_squeeze(symbol: str) -> dict:
    """Run TTM Squeeze Scanner"""
    try:
        # Try the v2 version first (from financial-analysis-system)
        try:
            from ttm_squeeze_v2 import TTMSqueeze
            # Get TwelveData API key from environment
            api_key = os.environ.get('TWELVEDATA_API_KEY', '5e7a5daaf41d46a8966963106ebef210')
            scanner = TTMSqueeze(api_key)
            result = scanner.calculate_squeeze(symbol)
            return result
        except ImportError:
            # Fall back to the indicators version
            from indicators.ttm_squeeze import TTMSqueeze
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo')
            
            if hist.empty:
                return {"error": f"No data for {symbol}", "symbol": symbol}
            
            # Standardize columns
            hist.columns = [c.lower() for c in hist.columns]
            
            scanner = TTMSqueeze()
            result = scanner.calculate(hist)
            
            # Get current values
            current_squeeze = result['squeeze_state'].iloc[-1] if len(result['squeeze_state']) > 0 else False
            current_momentum = result['momentum'].iloc[-1] if len(result['momentum']) > 0 else 0
            current_signal = result['signal'].iloc[-1] if len(result['signal']) > 0 else 'none'
            
            return {
                "status": "success",
                "symbol": symbol,
                "squeeze_on": bool(current_squeeze),
                "momentum": float(current_momentum),
                "signal": current_signal,
                "score_contrib": result.get('score_contrib', 0)
            }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def run_options_flow(symbol: str) -> dict:
    """Run Options Flow (Bear to Bull) Scanner"""
    try:
        from options_pressure import OptionsPressure
        scanner = OptionsPressure()
        result = scanner.get_pressure_analysis(symbol)
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def run_breakout(symbol: str) -> dict:
    """Run Breakout Detector"""
    try:
        from breakout_detector import BreakoutDetector
        scanner = BreakoutDetector()
        result = scanner.analyze_breakout(symbol)
        return result
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python run_scanners.py <scanner_type> <symbol>"}))
        sys.exit(1)
    
    scanner_type = sys.argv[1].lower()
    symbol = sys.argv[2].upper()
    
    if scanner_type == "dark_pool":
        result = run_dark_pool(symbol)
    elif scanner_type == "ttm_squeeze":
        result = run_ttm_squeeze(symbol)
    elif scanner_type == "options_flow":
        result = run_options_flow(symbol)
    elif scanner_type == "breakout":
        result = run_breakout(symbol)
    else:
        result = {"error": f"Unknown scanner type: {scanner_type}"}
    
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
