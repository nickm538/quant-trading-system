#!/usr/bin/env python3.11
"""
Wrapper script to run trading system analysis with clean environment
"""
import sys
import json
import os

# Ensure we're using the correct Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_trading_system import InstitutionalTradingSystem

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "analyze_stock":
        symbol = sys.argv[2]
        monte_carlo_sims = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
        forecast_days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        bankroll = float(sys.argv[5]) if len(sys.argv) > 5 else 1000
        
        system = InstitutionalTradingSystem()
        result = system.analyze_stock_comprehensive(
            symbol,
            monte_carlo_sims=monte_carlo_sims,
            forecast_days=forecast_days,
            bankroll=bankroll
        )
        print(json.dumps(result, default=str))
    
    elif command == "analyze_options":
        from options_analyzer import OptionsAnalyzer
        
        symbol = sys.argv[2]
        min_delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        max_delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
        min_days = int(sys.argv[5]) if len(sys.argv) > 5 else 7
        
        analyzer = OptionsAnalyzer()
        result = analyzer.analyze_options_chain(
            symbol=symbol,
            min_delta=min_delta,
            max_delta=max_delta,
            min_days_to_expiry=min_days
        )
        print(json.dumps(result, default=str))
    
    elif command == "scan_market":
        from market_scanner import MarketScanner
        
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        
        scanner = MarketScanner()
        result = scanner.scan_market(top_n=top_n)
        print(json.dumps(result, default=str))
    
    elif command == "health_check":
        print("OK")
    
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
