#!/usr/bin/env python3
"""
Production Analysis Runner
Integrates the production analyzer into the main trading system
Replaces ALL placeholder data with real AlphaVantage + yfinance data
"""

import sys
import json
from production_stock_analyzer import ProductionStockAnalyzer

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_production_analysis.py <symbol>"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    try:
        analyzer = ProductionStockAnalyzer()
        result = analyzer.analyze_stock(symbol)
        
        # Convert to JSON-serializable format
        output = {
            'symbol': result['symbol'],
            'timestamp': result['timestamp'],
            'current_price': result['current_price'],
            'price_change_pct': result['price_change_pct'],
            'fundamental_score': result['fundamental_score'],
            'technical_score': result['technical_score'],
            'sentiment_score': result['sentiment_score'],
            'overall_score': result['overall_score'],
            'recommendation': result['recommendation'],
            'confidence': result['confidence'],
            'fundamentals': result['fundamentals'],
            'technical_indicators': result['technical_indicators'],
            'news_count': result['news_count'],
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
