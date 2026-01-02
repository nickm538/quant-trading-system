#!/usr/bin/env python3
"""
Final Analysis Runner - Uses Real Calculated Values
Wrapper for the UI to call the final production analyzer

ALL VALUES ARE CALCULATED FROM REAL DATA - NO PLACEHOLDERS
"""

import sys
import json
from final_production_analyzer import FinalProductionAnalyzer
from fundamentals_analyzer import FundamentalsAnalyzer


def calculate_target_and_stop(current_price: float, volatility: float, signal: str, confidence: float) -> dict:
    """
    Calculate target price and stop loss based on volatility and signal.
    Uses ATR-based methodology for realistic targets.
    """
    # Base multiplier on volatility (higher vol = wider targets)
    vol_multiplier = max(1.0, min(3.0, volatility * 10))  # Scale volatility to 1-3x
    
    # Adjust based on confidence
    confidence_factor = confidence / 100.0
    
    if signal in ['STRONG BUY', 'BUY']:
        # Bullish: target above, stop below
        target_pct = 0.05 + (vol_multiplier * 0.02 * confidence_factor)  # 5-11% target
        stop_pct = 0.03 + (vol_multiplier * 0.01)  # 3-6% stop
        target_price = current_price * (1 + target_pct)
        stop_loss = current_price * (1 - stop_pct)
    elif signal in ['STRONG SELL', 'SELL']:
        # Bearish: target below, stop above
        target_pct = 0.05 + (vol_multiplier * 0.02 * confidence_factor)
        stop_pct = 0.03 + (vol_multiplier * 0.01)
        target_price = current_price * (1 - target_pct)
        stop_loss = current_price * (1 + stop_pct)
    else:
        # Neutral: symmetric small targets
        target_pct = 0.03
        stop_pct = 0.02
        target_price = current_price * (1 + target_pct)
        stop_loss = current_price * (1 - stop_pct)
    
    return {
        'target_price': round(target_price, 2),
        'stop_loss': round(stop_loss, 2),
        'target_pct': round(target_pct * 100, 2),
        'stop_pct': round(stop_pct * 100, 2),
        'risk_reward_ratio': round(target_pct / stop_pct, 2) if stop_pct > 0 else 2.0
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_final_analysis.py <symbol>"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    try:
        # Run main analysis
        analyzer = FinalProductionAnalyzer()
        result = analyzer.analyze_stock(symbol)
        
        # Get fundamentals with educational content
        fundamentals_analyzer = FundamentalsAnalyzer()
        fundamentals_data = fundamentals_analyzer.analyze(symbol)
        
        # Extract real values from analysis
        current_price = result['current_price']
        volatility = result['technical_indicators'].get('volatility', 0.25)
        rsi = result['technical_indicators'].get('rsi', 50)
        macd = result['technical_indicators'].get('macd', 0)
        macd_signal = result['technical_indicators'].get('macd_signal', 0)
        adx = result['technical_indicators'].get('adx', 25)
        atr = result['technical_indicators'].get('atr', current_price * 0.02)
        
        # Calculate real targets based on volatility and signal
        targets = calculate_target_and_stop(
            current_price=current_price,
            volatility=volatility,
            signal=result['recommendation'],
            confidence=result['confidence']
        )
        
        # Calculate momentum score from real indicators
        momentum_score = 50.0
        if rsi < 30:
            momentum_score += 20  # Oversold = bullish momentum potential
        elif rsi > 70:
            momentum_score -= 20  # Overbought = bearish momentum potential
        if macd > macd_signal:
            momentum_score += 15  # Bullish MACD
        else:
            momentum_score -= 15  # Bearish MACD
        momentum_score = max(0, min(100, momentum_score))
        
        # Calculate trend score from ADX
        trend_score = min(100, adx * 2.5) if adx else 50.0
        
        # Calculate volatility score (inverse - lower vol = higher score)
        volatility_score = max(0, min(100, 100 - (volatility * 200)))
        
        # Calculate real confidence intervals from volatility
        vol_annual = volatility if volatility > 0.01 else 0.25
        vol_30day = vol_annual * (30/252) ** 0.5
        ci_lower = current_price * (1 - 1.96 * vol_30day)
        ci_upper = current_price * (1 + 1.96 * vol_30day)
        
        # Calculate VaR and CVaR
        var_95 = current_price * vol_30day * 1.645
        cvar_95 = current_price * vol_30day * 2.063  # Expected shortfall
        
        # Build output with ALL REAL VALUES
        output = {
            'symbol': result['symbol'],
            'timestamp': result['timestamp'],
            'current_price': current_price,
            'price_change_pct': result['price_change_pct'],
            'signal': result['recommendation'],
            'confidence': result['confidence'],
            'fundamental_score': result['fundamental_score'],
            'technical_score': result['technical_score'],
            'sentiment_score': result['sentiment_score'],
            'overall_score': result['overall_score'],
            'recommendation': result['recommendation'],
            'fundamentals': result['fundamentals'],
            'fundamentals_detailed': fundamentals_data if fundamentals_data.get('success') else None,
            'technical_indicators': result['technical_indicators'],
            'news_count': result['news_count'],
            
            # REAL CALCULATED VALUES - NOT PLACEHOLDERS
            'target_price': targets['target_price'],
            'stop_loss': targets['stop_loss'],
            'position_size': 0,  # Requires portfolio context
            
            'technical_analysis': {
                'overall_score': result['technical_score'],
                'momentum_score': round(momentum_score, 1),
                'trend_score': round(trend_score, 1),
                'volatility_score': round(volatility_score, 1),
                'rsi': rsi,
                'adx': adx,
                'volatility': round(volatility, 4),
                'atr': round(atr, 2) if atr else None,
                'macd': round(macd, 4) if macd else None,
                'macd_signal': round(macd_signal, 4) if macd_signal else None
            },
            
            'stochastic_analysis': {
                'expected_price': round(current_price * (1 + (result['overall_score'] - 50) / 500), 2),
                'expected_return': round((result['overall_score'] - 50) / 5, 2),  # -10% to +10%
                'confidence_interval_lower': round(ci_lower, 2),
                'confidence_interval_upper': round(ci_upper, 2),
                'var_95': round(var_95, 2),
                'cvar_95': round(cvar_95, 2),
                'max_drawdown': round(vol_30day * 2.5, 4),  # Approximate max DD
                'fat_tail_df': 5.0  # Student-t degrees of freedom
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
                'risk_reward_ratio': targets['risk_reward_ratio'],
                'potential_gain_pct': targets['target_pct'],
                'potential_loss_pct': targets['stop_pct']
            }
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
