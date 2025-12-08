#!/usr/bin/env python3
"""
Final Analysis Runner - Uses Manus API Hub (No Rate Limits)
Wrapper for the UI to call the final production analyzer
"""

import sys
import json
from final_production_analyzer import FinalProductionAnalyzer

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_final_analysis.py <symbol>"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    try:
        analyzer = FinalProductionAnalyzer()
        result = analyzer.analyze_stock(symbol)
        
        # Convert to format expected by UI
        output = {
            'symbol': result['symbol'],
            'timestamp': result['timestamp'],
            'current_price': result['current_price'],
            'price_change_pct': result['price_change_pct'],
            'signal': result['recommendation'],  # Map recommendation to signal
            'confidence': result['confidence'],
            'fundamental_score': result['fundamental_score'],
            'technical_score': result['technical_score'],
            'sentiment_score': result['sentiment_score'],
            'overall_score': result['overall_score'],
            'recommendation': result['recommendation'],
            'fundamentals': result['fundamentals'],
            'technical_indicators': result['technical_indicators'],
            'news_count': result['news_count'],
            # Add placeholder fields for UI compatibility
            'target_price': result['current_price'] * 1.05,  # 5% above current
            'stop_loss': result['current_price'] * 0.95,  # 5% below current
            'position_size': 0,
            'technical_analysis': {
                'overall_score': result['technical_score'],
                'momentum_score': result['technical_score'],
                'trend_score': result['technical_score'],
                'volatility_score': 50.0,
                'rsi': result['technical_indicators'].get('rsi', 50),
                'adx': 25.0,
                'volatility': 0.25
            },
            'stochastic_analysis': {
                'expected_price': result['current_price'],
                'expected_return': 0.0,
                'confidence_interval_lower': result['current_price'] * 0.9,
                'confidence_interval_upper': result['current_price'] * 1.1,
                'var_95': result['current_price'] * 0.05,
                'cvar_95': result['current_price'] * 0.07,
                'max_drawdown': 0.15,
                'fat_tail_df': 5.0
            },
            'options_analysis': {
                'recommended_option': None,
                'total_options_analyzed': 0
            },
            'news_sentiment': {
                'sentiment_score': result['sentiment_score'],
                'total_articles': result['news_count'],
                'recent_headlines': []
            },
            'risk_assessment': {
                'risk_reward_ratio': 2.0,
                'potential_gain_pct': 5.0,
                'potential_loss_pct': 5.0
            }
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
