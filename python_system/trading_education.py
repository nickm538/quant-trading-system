"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TRADING EDUCATION MODULE v1.0                                      ║
║                                                                              ║
║  Comprehensive definitions, pro tips, and educational content for:          ║
║  - Technical Indicators (RSI, MACD, Bollinger Bands, etc.)                  ║
║  - Options Greeks (Delta, Gamma, Theta, Vega, etc.)                         ║
║  - Chart Patterns and Signals                                               ║
║  - Risk Management Concepts                                                 ║
║  - Market Terminology                                                       ║
║                                                                              ║
║  Designed for beginners and as a reference for experienced traders          ║
║  Copyright © 2026 SadieAI - All Rights Reserved                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict, Any


class TradingEducation:
    """
    Comprehensive trading education with definitions and pro tips.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    TECHNICAL_INDICATORS = {
        'rsi': {
            'name': 'Relative Strength Index (RSI)',
            'category': 'Momentum',
            'definition': 'Measures the speed and magnitude of price changes on a scale of 0-100.',
            'interpretation': {
                'overbought': 'RSI > 70 suggests overbought conditions - potential reversal down',
                'oversold': 'RSI < 30 suggests oversold conditions - potential reversal up',
                'neutral': 'RSI between 30-70 indicates neutral momentum'
            },
            'pro_tips': [
                'Look for RSI divergence - when price makes new highs but RSI doesn\'t, it signals weakness',
                'In strong trends, RSI can stay overbought/oversold for extended periods',
                'Use RSI with other indicators for confirmation, not as a standalone signal',
                'The 50 level acts as support/resistance in trends'
            ],
            'formula': 'RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss over N periods',
            'default_period': 14
        },
        
        'macd': {
            'name': 'Moving Average Convergence Divergence (MACD)',
            'category': 'Trend/Momentum',
            'definition': 'Shows the relationship between two moving averages of price.',
            'interpretation': {
                'bullish_crossover': 'MACD line crosses above signal line - buy signal',
                'bearish_crossover': 'MACD line crosses below signal line - sell signal',
                'histogram': 'Positive histogram = bullish momentum; Negative = bearish momentum'
            },
            'pro_tips': [
                'MACD works best in trending markets, avoid in choppy/sideways markets',
                'Divergence between MACD and price is a powerful reversal signal',
                'The zero line crossover confirms trend direction change',
                'Histogram shrinking often precedes a crossover'
            ],
            'formula': 'MACD Line = 12-day EMA - 26-day EMA; Signal Line = 9-day EMA of MACD Line',
            'default_periods': {'fast': 12, 'slow': 26, 'signal': 9}
        },
        
        'bollinger_bands': {
            'name': 'Bollinger Bands',
            'category': 'Volatility',
            'definition': 'Price envelope with bands at standard deviations above/below a moving average.',
            'interpretation': {
                'squeeze': 'Bands narrowing = low volatility, often precedes big move',
                'expansion': 'Bands widening = high volatility, trend in progress',
                'touch_upper': 'Price touching upper band = overbought (in range) or strong (in trend)',
                'touch_lower': 'Price touching lower band = oversold (in range) or weak (in trend)'
            },
            'pro_tips': [
                'Bollinger Squeeze (narrow bands) is a powerful setup for breakout trades',
                'In trends, price can "walk the band" - don\'t fade strong moves',
                'Mean reversion works best when bands are wide',
                'Use %B indicator to quantify position within bands'
            ],
            'formula': 'Middle Band = 20-day SMA; Upper/Lower = Middle ± (2 × 20-day Std Dev)',
            'default_periods': {'sma': 20, 'std_dev': 2}
        },
        
        'ttm_squeeze': {
            'name': 'TTM Squeeze',
            'category': 'Volatility/Momentum',
            'definition': 'Identifies periods of low volatility (squeeze) that often precede explosive moves.',
            'interpretation': {
                'squeeze_on': 'Red dots = Bollinger Bands inside Keltner Channels = compression',
                'squeeze_off': 'Green dots = Bands outside Keltner = expansion beginning',
                'momentum_positive': 'Histogram above zero = bullish momentum',
                'momentum_negative': 'Histogram below zero = bearish momentum'
            },
            'pro_tips': [
                'The longer the squeeze, the more powerful the breakout',
                'First green dot after red dots = entry signal in direction of momentum',
                'Combine with volume confirmation for higher probability trades',
                'Works exceptionally well on daily and weekly timeframes'
            ],
            'formula': 'Squeeze = BB inside KC; Momentum = Linear Regression of (Close - Average of Highest High and Lowest Low)',
            'creator': 'John Carter'
        },
        
        'atr': {
            'name': 'Average True Range (ATR)',
            'category': 'Volatility',
            'definition': 'Measures market volatility by calculating the average range of price movement.',
            'interpretation': {
                'high_atr': 'High ATR = high volatility, wider stops needed',
                'low_atr': 'Low ATR = low volatility, tighter stops possible',
                'increasing': 'Rising ATR often accompanies trend moves',
                'decreasing': 'Falling ATR may signal consolidation or trend exhaustion'
            },
            'pro_tips': [
                'Use ATR for position sizing - risk same dollar amount regardless of volatility',
                'Set stops at 1.5-2x ATR from entry for swing trades',
                'ATR expansion at breakout confirms the move',
                'Compare current ATR to historical average for context'
            ],
            'formula': 'ATR = Moving Average of True Range over N periods',
            'default_period': 14
        },
        
        'vwap': {
            'name': 'Volume Weighted Average Price (VWAP)',
            'category': 'Volume/Price',
            'definition': 'Average price weighted by volume - shows the "fair" price for the day.',
            'interpretation': {
                'above_vwap': 'Price above VWAP = bullish, buyers in control',
                'below_vwap': 'Price below VWAP = bearish, sellers in control',
                'at_vwap': 'Price at VWAP = equilibrium, potential support/resistance'
            },
            'pro_tips': [
                'Institutional traders use VWAP as a benchmark - they often buy below and sell above',
                'VWAP acts as dynamic support/resistance throughout the day',
                'Anchored VWAP from significant events provides key levels',
                'Resets at market open - most relevant for intraday trading'
            ],
            'formula': 'VWAP = Cumulative(Price × Volume) / Cumulative(Volume)',
            'timeframe': 'Intraday'
        },
        
        'stochastic': {
            'name': 'Stochastic Oscillator',
            'category': 'Momentum',
            'definition': 'Compares closing price to price range over a period, scaled 0-100.',
            'interpretation': {
                'overbought': '%K > 80 = overbought',
                'oversold': '%K < 20 = oversold',
                'bullish_crossover': '%K crosses above %D in oversold zone = buy',
                'bearish_crossover': '%K crosses below %D in overbought zone = sell'
            },
            'pro_tips': [
                'Use slow stochastic (smoothed) to reduce false signals',
                'Divergence between stochastic and price is a strong reversal signal',
                'In strong trends, stochastic can stay overbought/oversold - don\'t fight the trend',
                'Best used in ranging markets, less effective in strong trends'
            ],
            'formula': '%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low); %D = 3-day SMA of %K',
            'default_periods': {'k': 14, 'd': 3}
        },
        
        'adx': {
            'name': 'Average Directional Index (ADX)',
            'category': 'Trend Strength',
            'definition': 'Measures trend strength regardless of direction, scaled 0-100.',
            'interpretation': {
                'strong_trend': 'ADX > 25 = strong trend in place',
                'weak_trend': 'ADX < 20 = weak or no trend (ranging market)',
                'rising': 'Rising ADX = trend strengthening',
                'falling': 'Falling ADX = trend weakening'
            },
            'pro_tips': [
                'ADX doesn\'t tell you direction - use +DI/-DI for that',
                'ADX > 40 indicates very strong trend - ride it, don\'t fade it',
                'ADX rising from below 20 to above 25 = new trend starting',
                'Combine with other indicators for entry timing'
            ],
            'formula': 'ADX = 100 × Smoothed Moving Average of |+DI - -DI| / (+DI + -DI)',
            'default_period': 14
        },
        
        'obv': {
            'name': 'On-Balance Volume (OBV)',
            'category': 'Volume',
            'definition': 'Cumulative volume indicator that adds volume on up days and subtracts on down days.',
            'interpretation': {
                'rising': 'Rising OBV = accumulation, bullish',
                'falling': 'Falling OBV = distribution, bearish',
                'divergence': 'OBV diverging from price = potential reversal'
            },
            'pro_tips': [
                'OBV breakouts often precede price breakouts - leading indicator',
                'Look for OBV making new highs before price does = bullish',
                'OBV failing to confirm new price highs = bearish divergence',
                'Use trend lines on OBV just like on price charts'
            ],
            'formula': 'OBV = Previous OBV + (Volume if Close > Previous Close, -Volume if Close < Previous Close)',
            'creator': 'Joe Granville'
        }
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIONS GREEKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    OPTIONS_GREEKS = {
        'delta': {
            'name': 'Delta (Δ)',
            'category': 'First-Order Greek',
            'definition': 'Measures how much the option price changes for a $1 move in the underlying.',
            'range': 'Calls: 0 to 1; Puts: -1 to 0',
            'interpretation': {
                'atm': 'ATM options have delta around ±0.50',
                'itm': 'Deep ITM options approach ±1.00 (move like stock)',
                'otm': 'Deep OTM options approach 0 (minimal price sensitivity)'
            },
            'pro_tips': [
                'Delta approximates probability of expiring ITM (0.30 delta ≈ 30% chance)',
                'Use delta to calculate equivalent stock position (100 shares = 1.00 delta)',
                'Delta changes as price moves - this is gamma',
                'Selling high delta options = more premium but more risk'
            ],
            'formula': 'Δ = ∂V/∂S (partial derivative of option value with respect to stock price)'
        },
        
        'gamma': {
            'name': 'Gamma (Γ)',
            'category': 'Second-Order Greek',
            'definition': 'Measures how much delta changes for a $1 move in the underlying.',
            'interpretation': {
                'high_gamma': 'High gamma = delta changes rapidly, more volatile P&L',
                'low_gamma': 'Low gamma = delta stable, more predictable P&L',
                'atm_peak': 'Gamma is highest for ATM options near expiration'
            },
            'pro_tips': [
                'Long gamma = profit from big moves in either direction',
                'Short gamma = profit from small moves, lose from big moves',
                'Gamma risk increases dramatically near expiration for ATM options',
                'Market makers are often short gamma - they hedge by buying/selling stock'
            ],
            'formula': 'Γ = ∂²V/∂S² = ∂Δ/∂S'
        },
        
        'theta': {
            'name': 'Theta (Θ)',
            'category': 'First-Order Greek',
            'definition': 'Measures how much the option price decays per day due to time.',
            'interpretation': {
                'negative': 'Long options have negative theta (lose value over time)',
                'positive': 'Short options have positive theta (gain value over time)',
                'acceleration': 'Theta decay accelerates as expiration approaches'
            },
            'pro_tips': [
                'Theta is the "rent" you pay for holding long options',
                'Sell options to collect theta - but beware of gamma risk',
                'ATM options have highest theta decay',
                'Weekend theta: options lose 2-3 days of value over weekends'
            ],
            'formula': 'Θ = ∂V/∂t (typically expressed as daily decay)'
        },
        
        'vega': {
            'name': 'Vega (ν)',
            'category': 'First-Order Greek',
            'definition': 'Measures how much the option price changes for a 1% change in implied volatility.',
            'interpretation': {
                'long_vega': 'Long options benefit from rising IV',
                'short_vega': 'Short options benefit from falling IV',
                'atm_highest': 'ATM options have highest vega'
            },
            'pro_tips': [
                'Buy options when IV is low, sell when IV is high',
                'IV crush after earnings can devastate long option positions',
                'Vega decreases as expiration approaches',
                'Long-dated options have more vega exposure'
            ],
            'formula': 'ν = ∂V/∂σ (per 1% change in IV)'
        },
        
        'rho': {
            'name': 'Rho (ρ)',
            'category': 'First-Order Greek',
            'definition': 'Measures how much the option price changes for a 1% change in interest rates.',
            'interpretation': {
                'calls': 'Calls have positive rho (benefit from rising rates)',
                'puts': 'Puts have negative rho (hurt by rising rates)',
                'magnitude': 'Rho is typically small for short-dated options'
            },
            'pro_tips': [
                'Rho matters more for LEAPS and long-dated options',
                'In low-rate environments, rho is often ignored',
                'Rising rates increase call values, decrease put values',
                'Consider rho when trading options on rate-sensitive assets'
            ],
            'formula': 'ρ = ∂V/∂r (per 1% change in risk-free rate)'
        },
        
        'vanna': {
            'name': 'Vanna',
            'category': 'Second-Order Greek',
            'definition': 'Measures how delta changes with volatility, or how vega changes with price.',
            'interpretation': {
                'positive': 'Positive vanna = delta increases as IV rises',
                'negative': 'Negative vanna = delta decreases as IV rises'
            },
            'pro_tips': [
                'Vanna flows can drive market moves - dealers hedging vanna exposure',
                'Important for understanding how your position behaves in vol spikes',
                'OTM calls have positive vanna; OTM puts have negative vanna',
                'Vanna is highest for OTM options'
            ],
            'formula': 'Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S'
        },
        
        'charm': {
            'name': 'Charm (Delta Decay)',
            'category': 'Second-Order Greek',
            'definition': 'Measures how delta changes with time.',
            'interpretation': {
                'effect': 'Shows how your delta exposure changes day-to-day',
                'near_expiry': 'Charm effects are strongest near expiration'
            },
            'pro_tips': [
                'Charm tells you how much to adjust hedges over time',
                'OTM options have positive charm (delta moves toward 0)',
                'ITM options have negative charm (delta moves toward ±1)',
                'Important for managing positions over weekends'
            ],
            'formula': 'Charm = ∂²V/∂S∂t = ∂Δ/∂t'
        }
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK MANAGEMENT CONCEPTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    RISK_MANAGEMENT = {
        'kelly_criterion': {
            'name': 'Kelly Criterion',
            'definition': 'Mathematical formula for optimal bet sizing to maximize long-term growth.',
            'formula': 'Kelly % = (Win Probability × Win/Loss Ratio - Loss Probability) / Win/Loss Ratio',
            'interpretation': 'Tells you what percentage of capital to risk on each trade.',
            'pro_tips': [
                'Full Kelly is aggressive - most traders use Half Kelly or Quarter Kelly',
                'Kelly assumes you know exact probabilities - you don\'t, so be conservative',
                'Never risk more than Kelly suggests, but often risk less',
                'Kelly optimizes for growth, not for minimizing drawdowns'
            ]
        },
        
        'position_sizing': {
            'name': 'Position Sizing',
            'definition': 'Determining how much capital to allocate to each trade.',
            'methods': {
                'fixed_dollar': 'Risk same dollar amount per trade',
                'fixed_percent': 'Risk same percentage of portfolio per trade (e.g., 1-2%)',
                'volatility_based': 'Adjust size based on asset volatility (ATR)'
            },
            'pro_tips': [
                'Never risk more than 2% of portfolio on a single trade',
                'Reduce position size during drawdowns',
                'Increase size gradually as account grows',
                'Consider correlation - don\'t have too many similar positions'
            ]
        },
        
        'risk_reward_ratio': {
            'name': 'Risk/Reward Ratio',
            'definition': 'Comparison of potential loss to potential gain on a trade.',
            'interpretation': {
                'good': '1:2 or better (risk $1 to make $2)',
                'minimum': '1:1.5 for most strategies',
                'scalping': 'Scalpers may accept 1:1 with high win rate'
            },
            'pro_tips': [
                'You can be profitable with 40% win rate if R:R is 1:2',
                'Always define your stop loss BEFORE entering',
                'Don\'t move stops to avoid losses - that\'s how accounts blow up',
                'Let winners run, cut losers quickly'
            ]
        },
        
        'max_drawdown': {
            'name': 'Maximum Drawdown',
            'definition': 'Largest peak-to-trough decline in portfolio value.',
            'interpretation': 'Measures worst-case scenario for your strategy.',
            'pro_tips': [
                'Expect 2x your backtest drawdown in live trading',
                'A 50% drawdown requires 100% gain to recover',
                'Set a max drawdown limit and stop trading if hit',
                'Drawdowns are psychological killers - plan for them'
            ]
        },
        
        'var': {
            'name': 'Value at Risk (VaR)',
            'definition': 'Maximum expected loss over a time period at a given confidence level.',
            'example': '95% VaR of $10,000 means 95% confident you won\'t lose more than $10k.',
            'pro_tips': [
                'VaR underestimates tail risk - use CVaR for better risk assessment',
                'Historical VaR assumes future looks like past - dangerous assumption',
                'VaR is a minimum loss in bad scenarios, not maximum',
                'Complement VaR with stress testing'
            ]
        },
        
        'sharpe_ratio': {
            'name': 'Sharpe Ratio',
            'definition': 'Risk-adjusted return: excess return per unit of volatility.',
            'formula': 'Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev',
            'interpretation': {
                'excellent': '> 2.0',
                'good': '1.0 - 2.0',
                'acceptable': '0.5 - 1.0',
                'poor': '< 0.5'
            },
            'pro_tips': [
                'Higher Sharpe = better risk-adjusted returns',
                'Compare Sharpe ratios across strategies, not absolute returns',
                'Sharpe can be gamed by leverage - look at underlying strategy',
                'Sortino ratio is better if you only care about downside risk'
            ]
        }
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET TERMINOLOGY
    # ═══════════════════════════════════════════════════════════════════════════
    
    MARKET_TERMS = {
        'iv_rank': {
            'name': 'IV Rank',
            'definition': 'Where current IV sits relative to its 52-week range (0-100%).',
            'interpretation': {
                'high': 'IV Rank > 50% = IV is elevated, good for selling options',
                'low': 'IV Rank < 30% = IV is low, good for buying options'
            },
            'pro_tips': [
                'Sell premium when IV Rank is high, buy when low',
                'IV Rank is more useful than raw IV for comparing across stocks',
                'High IV Rank often occurs before earnings or events',
                'IV Rank reversion is a tradeable edge'
            ]
        },
        
        'iv_percentile': {
            'name': 'IV Percentile',
            'definition': 'Percentage of days in past year when IV was lower than today.',
            'interpretation': '80% IV Percentile = IV was lower 80% of the time.',
            'pro_tips': [
                'IV Percentile accounts for time spent at each level',
                'More accurate than IV Rank for skewed distributions',
                'Use both IV Rank and Percentile for complete picture'
            ]
        },
        
        'open_interest': {
            'name': 'Open Interest',
            'definition': 'Total number of outstanding option contracts.',
            'interpretation': {
                'high_oi': 'High OI = liquid market, tighter spreads',
                'low_oi': 'Low OI = illiquid, wider spreads, harder to exit'
            },
            'pro_tips': [
                'Trade options with high open interest for better fills',
                'Rising OI with rising price = bullish confirmation',
                'OI at round strikes often acts as support/resistance',
                'Max pain theory: price gravitates toward max OI strike at expiry'
            ]
        },
        
        'bid_ask_spread': {
            'name': 'Bid-Ask Spread',
            'definition': 'Difference between highest buy price and lowest sell price.',
            'interpretation': {
                'tight': '< 5% of option price = liquid, good to trade',
                'wide': '> 10% of option price = illiquid, avoid or use limits'
            },
            'pro_tips': [
                'Never use market orders on options - always use limits',
                'Wide spreads eat into profits significantly',
                'Trade during market hours for tightest spreads',
                'Bid-ask is a hidden cost - factor it into your analysis'
            ]
        },
        
        'implied_move': {
            'name': 'Implied Move',
            'definition': 'Expected price range based on option prices.',
            'formula': 'Implied Move = ATM Straddle Price / Stock Price',
            'interpretation': 'If straddle costs $10 on $100 stock, market expects ±10% move.',
            'pro_tips': [
                'Compare implied move to historical moves for edge',
                'Implied move is often overstated before earnings',
                'Sell straddles if you think implied move is too high',
                'Buy straddles if you think implied move is too low'
            ]
        }
    }
    
    @classmethod
    def get_indicator_info(cls, indicator: str) -> Dict[str, Any]:
        """Get educational info for a technical indicator."""
        return cls.TECHNICAL_INDICATORS.get(indicator.lower(), {})
    
    @classmethod
    def get_greek_info(cls, greek: str) -> Dict[str, Any]:
        """Get educational info for an options Greek."""
        return cls.OPTIONS_GREEKS.get(greek.lower(), {})
    
    @classmethod
    def get_risk_concept(cls, concept: str) -> Dict[str, Any]:
        """Get educational info for a risk management concept."""
        return cls.RISK_MANAGEMENT.get(concept.lower(), {})
    
    @classmethod
    def get_market_term(cls, term: str) -> Dict[str, Any]:
        """Get educational info for a market term."""
        return cls.MARKET_TERMS.get(term.lower(), {})
    
    @classmethod
    def get_all_education(cls) -> Dict[str, Any]:
        """Get all educational content."""
        return {
            'technical_indicators': cls.TECHNICAL_INDICATORS,
            'options_greeks': cls.OPTIONS_GREEKS,
            'risk_management': cls.RISK_MANAGEMENT,
            'market_terms': cls.MARKET_TERMS
        }
    
    @classmethod
    def search_education(cls, query: str) -> Dict[str, Any]:
        """Search all educational content for a term."""
        query = query.lower()
        results = []
        
        for category, items in [
            ('technical_indicators', cls.TECHNICAL_INDICATORS),
            ('options_greeks', cls.OPTIONS_GREEKS),
            ('risk_management', cls.RISK_MANAGEMENT),
            ('market_terms', cls.MARKET_TERMS)
        ]:
            for key, value in items.items():
                if query in key or query in value.get('name', '').lower():
                    results.append({
                        'category': category,
                        'key': key,
                        **value
                    })
        
        return {'query': query, 'results': results}


# Export for easy import
def get_education(topic: str = None) -> Dict[str, Any]:
    """Get educational content."""
    if topic:
        return TradingEducation.search_education(topic)
    return TradingEducation.get_all_education()


if __name__ == "__main__":
    # Test
    import json
    
    print("=== RSI Education ===")
    print(json.dumps(TradingEducation.get_indicator_info('rsi'), indent=2))
    
    print("\n=== Delta Education ===")
    print(json.dumps(TradingEducation.get_greek_info('delta'), indent=2))
