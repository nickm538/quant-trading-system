#!/usr/bin/env python3
"""
UNIFIED SCANNER RUNNER
======================
Runs the 4 scanner modules from financial-analysis-system:
1. Dark Pool Scanner (Insider Movement/Oracle)
2. TTM Squeeze Scanner
3. Options Flow (Bear to Bull)
4. Breakout Detector
5. Market-Wide TTM Squeeze Scanner (NEW)
6. Market-Wide Breakout Scanner (NEW)

Usage:
    python run_scanners.py dark_pool AAPL
    python run_scanners.py ttm_squeeze AAPL
    python run_scanners.py options_flow AAPL
    python run_scanners.py breakout AAPL
    python run_scanners.py market_ttm_squeeze 100
    python run_scanners.py market_breakout 100
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


def run_market_ttm_squeeze(max_stocks: int = 100) -> dict:
    """Run Market-Wide TTM Squeeze Scanner"""
    try:
        from market_wide_scanner import MarketWideScanner
        scanner = MarketWideScanner()
        result = scanner.scan_ttm_squeeze(max_stocks=max_stocks, min_score=30)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


def run_market_breakout(max_stocks: int = 100) -> dict:
    """Run Market-Wide Breakout Scanner"""
    try:
        from market_wide_scanner import MarketWideScanner
        scanner = MarketWideScanner()
        result = scanner.scan_breakouts(max_stocks=max_stocks, min_score=30)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


def run_market_intelligence() -> dict:
    """Run Market Intelligence - VIX, regime, sentiment, catalysts"""
    try:
        from market_intelligence import MarketIntelligence
        mi = MarketIntelligence()
        result = mi.get_full_market_intelligence()
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python run_scanners.py <scanner_type> <symbol_or_count>"}))
        sys.exit(1)
    
    scanner_type = sys.argv[1].lower()
    arg = sys.argv[2]
    
    if scanner_type == "dark_pool":
        result = run_dark_pool(arg.upper())
    elif scanner_type == "ttm_squeeze":
        result = run_ttm_squeeze(arg.upper())
    elif scanner_type == "options_flow":
        result = run_options_flow(arg.upper())
    elif scanner_type == "breakout":
        result = run_breakout(arg.upper())
    elif scanner_type == "market_ttm_squeeze":
        result = run_market_ttm_squeeze(int(arg))
    elif scanner_type == "market_breakout":
        result = run_market_breakout(int(arg))
    elif scanner_type == "market_intelligence":
        result = run_market_intelligence()
    else:
        result = {"error": f"Unknown scanner type: {scanner_type}"}
    
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
