"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              USER EDUCATION MODULE v2.0 - BEGINNER FRIENDLY                  ║
║                                                                              ║
║  Comprehensive explanations for all trading concepts, indicators, and        ║
║  metrics used throughout the Sadie AI system.                                ║
║                                                                              ║
║  Features:                                                                   ║
║  ✓ Plain English explanations for beginners                                  ║
║  ✓ Real-world analogies                                                      ║
║  ✓ What it means for YOUR trade                                             ║
║  ✓ Common mistakes to avoid                                                  ║
║  ✓ Position sizing education                                                 ║
║  ✓ Risk management fundamentals                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Dict, Any, Optional


class UserEducation:
    """
    Comprehensive education module for trading concepts.
    Provides beginner-friendly explanations for all indicators and metrics.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════
    
    INDICATOR_EXPLANATIONS = {
        'RSI': {
            'name': 'Relative Strength Index (RSI)',
            'simple': 'Measures how fast and how much a stock price has moved. Think of it like a speedometer for stock momentum.',
            'range': '0 to 100',
            'interpretation': {
                'above_70': 'OVERBOUGHT - The stock has risen quickly and may be due for a pullback. Like a rubber band stretched too far.',
                'below_30': 'OVERSOLD - The stock has fallen quickly and may be due for a bounce. The rubber band is stretched the other way.',
                '40_to_60': 'NEUTRAL - No extreme momentum either way. Stock is trading normally.',
            },
            'for_your_trade': 'If RSI is above 70, consider waiting for a pullback before buying. If below 30, it might be a good entry point, but confirm with other indicators.',
            'common_mistake': "Don't buy just because RSI is oversold - stocks can stay oversold for weeks in a downtrend. Always check the overall trend first.",
            'analogy': "RSI is like checking if a car is speeding. Just because it's going fast doesn't mean it will crash, but you should be more careful.",
        },
        
        'MACD': {
            'name': 'Moving Average Convergence Divergence (MACD)',
            'simple': 'Shows the relationship between two moving averages. When they cross, it often signals a trend change.',
            'interpretation': {
                'bullish_cross': 'MACD line crosses ABOVE signal line = Bullish signal (potential buy)',
                'bearish_cross': 'MACD line crosses BELOW signal line = Bearish signal (potential sell)',
                'histogram_positive': 'Green histogram = Bullish momentum increasing',
                'histogram_negative': 'Red histogram = Bearish momentum increasing',
            },
            'for_your_trade': 'Look for MACD crossovers that align with the overall trend. A bullish cross in an uptrend is more reliable than in a downtrend.',
            'common_mistake': 'MACD is a lagging indicator - by the time it signals, part of the move may be over. Use it for confirmation, not prediction.',
            'analogy': "MACD is like watching two runners. When the faster runner (MACD line) passes the slower one (signal line), momentum is shifting.",
        },
        
        'TTM_SQUEEZE': {
            'name': 'TTM Squeeze',
            'simple': 'Detects when a stock is "coiling up" like a spring, ready to make a big move. Created by John Carter.',
            'interpretation': {
                'squeeze_on': 'RED DOTS = Squeeze is ON. Volatility is compressing. A big move is building up.',
                'squeeze_off': 'GREEN DOTS = Squeeze has FIRED. The move is happening now.',
                'momentum_positive': 'Histogram above zero = Bullish momentum when squeeze fires',
                'momentum_negative': 'Histogram below zero = Bearish momentum when squeeze fires',
            },
            'for_your_trade': 'When you see red dots (squeeze on), prepare for a trade. When dots turn green, enter in the direction of the momentum histogram.',
            'common_mistake': "Don't enter during the squeeze (red dots). Wait for the squeeze to fire (green dots) and confirm direction with momentum.",
            'analogy': "TTM Squeeze is like watching a coiled spring. Red dots = spring is compressing. Green dots = spring has released. The histogram tells you which direction it's going.",
        },
        
        'BOLLINGER_BANDS': {
            'name': 'Bollinger Bands',
            'simple': 'Creates a "channel" around the price. When price touches the edges, it often bounces back.',
            'interpretation': {
                'upper_band': 'Price at upper band = Potentially overbought, may pull back',
                'lower_band': 'Price at lower band = Potentially oversold, may bounce',
                'squeeze': 'Bands narrowing = Low volatility, big move coming',
                'expansion': 'Bands widening = High volatility, trend in progress',
            },
            'for_your_trade': 'In a range-bound market, sell near upper band, buy near lower band. In a trend, bands can "walk" along the upper or lower band.',
            'common_mistake': "Don't automatically sell at upper band in an uptrend - strong trends can ride the upper band for extended periods.",
            'analogy': "Bollinger Bands are like guardrails on a highway. Price usually stays within them, but in strong trends, it can hug one side.",
        },
        
        'VWAP': {
            'name': 'Volume Weighted Average Price (VWAP)',
            'simple': 'The average price weighted by volume. Shows where most trading activity occurred.',
            'interpretation': {
                'above_vwap': 'Price above VWAP = Bullish. Buyers are in control.',
                'below_vwap': 'Price below VWAP = Bearish. Sellers are in control.',
            },
            'for_your_trade': 'Institutions often use VWAP as a benchmark. If you buy below VWAP, you got a "good" price relative to the day.',
            'common_mistake': 'VWAP resets daily. It is most useful for day trading, not swing trading.',
            'analogy': "VWAP is like the average grade in a class. If you score above average (buy below VWAP), you did well.",
        },
        
        'ADX': {
            'name': 'Average Directional Index (ADX)',
            'simple': 'Measures trend STRENGTH, not direction. Tells you if a trend is strong or weak.',
            'range': '0 to 100',
            'interpretation': {
                'below_20': 'WEAK/NO TREND - Market is choppy, range-bound. Avoid trend-following strategies.',
                '20_to_40': 'DEVELOPING TREND - Trend is forming. Good time to enter.',
                'above_40': 'STRONG TREND - Powerful trend in place. Stay with the trend.',
                'above_60': 'EXTREME TREND - Very strong, but may be exhausting. Watch for reversal.',
            },
            'for_your_trade': 'Only use trend-following strategies when ADX > 25. In low ADX environments, use range-trading strategies.',
            'common_mistake': 'ADX measures strength, not direction. A rising ADX in a downtrend means the downtrend is getting stronger.',
            'analogy': "ADX is like a wind speed meter. It tells you how strong the wind is, not which direction it's blowing.",
        },
        
        'ATR': {
            'name': 'Average True Range (ATR)',
            'simple': 'Measures how much a stock typically moves in a day. Used for setting stop losses.',
            'interpretation': {
                'high_atr': 'High ATR = Volatile stock. Needs wider stops.',
                'low_atr': 'Low ATR = Calm stock. Can use tighter stops.',
            },
            'for_your_trade': 'Set your stop loss 1.5-2x ATR away from entry. This gives the trade room to breathe without getting stopped out by normal noise.',
            'common_mistake': 'Using the same dollar stop for all stocks. A $2 stop on a $10 stock with $3 ATR will get stopped out constantly.',
            'analogy': "ATR is like knowing how much a hyperactive kid moves around. You need to give them more space than a calm kid.",
        },
        
        'OBV': {
            'name': 'On-Balance Volume (OBV)',
            'simple': 'Tracks whether volume is flowing into or out of a stock. Rising OBV = accumulation, Falling OBV = distribution.',
            'interpretation': {
                'rising_obv': 'Rising OBV = Smart money accumulating. Bullish.',
                'falling_obv': 'Falling OBV = Smart money distributing. Bearish.',
                'divergence': 'Price up but OBV down = Warning sign. Rally may fail.',
            },
            'for_your_trade': 'Look for OBV to confirm price moves. If price makes new highs but OBV does not, be cautious.',
            'common_mistake': 'Ignoring OBV divergences. They often precede major reversals.',
            'analogy': "OBV is like tracking money flow into a store. If customers are leaving (falling OBV) but prices are rising, something's wrong.",
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FUNDAMENTAL METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    FUNDAMENTAL_EXPLANATIONS = {
        'PE_RATIO': {
            'name': 'Price-to-Earnings Ratio (P/E)',
            'simple': 'How much you pay for each dollar of earnings. Lower P/E = cheaper stock (usually).',
            'formula': 'Stock Price ÷ Earnings Per Share',
            'interpretation': {
                'low_pe': 'P/E < 15 = Potentially undervalued or slow growth',
                'moderate_pe': 'P/E 15-25 = Fair value for growth company',
                'high_pe': 'P/E > 30 = Expensive, high growth expected',
            },
            'for_your_trade': 'Compare P/E to industry average and historical P/E. A stock with P/E of 20 might be cheap for tech but expensive for utilities.',
            'common_mistake': "Don't compare P/E across different industries. Tech stocks naturally have higher P/E than banks.",
            'analogy': "P/E is like price per square foot for a house. A $500/sqft house might be cheap in Manhattan but expensive in rural areas.",
        },
        
        'PEG_RATIO': {
            'name': 'Price/Earnings to Growth Ratio (PEG)',
            'simple': 'P/E adjusted for growth. Helps compare fast-growing vs slow-growing companies fairly.',
            'formula': 'P/E Ratio ÷ Earnings Growth Rate',
            'interpretation': {
                'below_1': 'PEG < 1 = Potentially undervalued relative to growth',
                'around_1': 'PEG ≈ 1 = Fairly valued',
                'above_2': 'PEG > 2 = Potentially overvalued relative to growth',
            },
            'for_your_trade': 'PEG < 1 with strong fundamentals is often a good buy signal. But verify the growth rate is sustainable.',
            'common_mistake': 'Using unrealistic growth projections. A PEG of 0.5 means nothing if the growth rate is fantasy.',
            'analogy': "PEG is like comparing car prices adjusted for horsepower. A $50K car with 500HP is better value than a $40K car with 200HP.",
        },
        
        'EV_EBITDA': {
            'name': 'Enterprise Value to EBITDA',
            'simple': 'How much you pay for the entire company relative to its operating profits. Better than P/E for comparing companies with different debt levels.',
            'formula': 'Enterprise Value ÷ EBITDA',
            'interpretation': {
                'low': 'EV/EBITDA < 10 = Potentially undervalued',
                'moderate': 'EV/EBITDA 10-15 = Fair value',
                'high': 'EV/EBITDA > 20 = Expensive',
            },
            'for_your_trade': 'Use EV/EBITDA when comparing companies in the same industry, especially if they have different debt levels.',
            'common_mistake': 'Ignoring that EV/EBITDA varies by industry. Capital-intensive industries naturally have lower multiples.',
            'analogy': "EV/EBITDA is like comparing house prices to rental income, accounting for the mortgage. It gives a fuller picture than just price.",
        },
        
        'ROIC': {
            'name': 'Return on Invested Capital (ROIC)',
            'simple': 'How efficiently a company uses its money to generate profits. Higher = better.',
            'formula': 'Net Operating Profit After Tax ÷ Invested Capital',
            'interpretation': {
                'excellent': 'ROIC > 20% = Excellent capital allocation',
                'good': 'ROIC 10-20% = Good business',
                'poor': 'ROIC < 10% = May be destroying value',
            },
            'for_your_trade': 'Look for companies with ROIC consistently above their cost of capital (usually 8-10%). These create shareholder value.',
            'common_mistake': 'Ignoring ROIC trends. A declining ROIC may signal competitive pressure.',
            'analogy': "ROIC is like the interest rate on your savings account. If a company earns 20% on every dollar invested, that's like a 20% savings rate.",
        },
        
        'FREE_CASH_FLOW': {
            'name': 'Free Cash Flow (FCF)',
            'simple': 'Cash left over after paying for operations and investments. This is real money the company can use.',
            'formula': 'Operating Cash Flow - Capital Expenditures',
            'interpretation': {
                'positive': 'Positive FCF = Company generates real cash',
                'negative': 'Negative FCF = Company burns cash (may need to raise money)',
                'growing': 'Growing FCF = Business improving',
            },
            'for_your_trade': 'Companies with strong, growing FCF can pay dividends, buy back stock, or invest in growth. This is the real measure of financial health.',
            'common_mistake': 'Confusing earnings with cash flow. A company can report profits but still run out of cash.',
            'analogy': "FCF is like your take-home pay after bills. Earnings are your gross salary, but FCF is what you actually have to spend.",
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    RISK_EXPLANATIONS = {
        'SHARPE_RATIO': {
            'name': 'Sharpe Ratio',
            'simple': 'Measures return per unit of risk. Higher = better risk-adjusted returns.',
            'formula': '(Return - Risk-Free Rate) ÷ Standard Deviation',
            'interpretation': {
                'below_0': 'Sharpe < 0 = Losing money',
                '0_to_1': 'Sharpe 0-1 = Subpar risk-adjusted returns',
                '1_to_2': 'Sharpe 1-2 = Good risk-adjusted returns',
                'above_2': 'Sharpe > 2 = Excellent risk-adjusted returns',
            },
            'for_your_trade': 'Compare Sharpe ratios when choosing between strategies or stocks. A 10% return with Sharpe 2 is better than 15% return with Sharpe 0.5.',
            'common_mistake': 'Chasing high returns without considering risk. A strategy with 50% returns but Sharpe 0.3 will eventually blow up.',
            'analogy': "Sharpe Ratio is like miles per gallon for investments. It tells you how efficiently you're converting risk into returns.",
        },
        
        'MAX_DRAWDOWN': {
            'name': 'Maximum Drawdown',
            'simple': 'The largest peak-to-trough decline. Shows the worst-case scenario you would have experienced.',
            'interpretation': {
                'small': 'Max DD < 10% = Low risk, conservative',
                'moderate': 'Max DD 10-20% = Moderate risk',
                'large': 'Max DD 20-40% = High risk',
                'extreme': 'Max DD > 40% = Very high risk',
            },
            'for_your_trade': 'Ask yourself: "Can I stomach a X% decline?" If your max drawdown is 30%, you need to be prepared to see your account drop 30% at some point.',
            'common_mistake': 'Underestimating drawdowns. If you panic sell during a 20% drawdown, you lock in losses and miss the recovery.',
            'analogy': "Max Drawdown is like the deepest pothole on a road. Even if the road is generally smooth, that one pothole can damage your car.",
        },
        
        'BETA': {
            'name': 'Beta',
            'simple': 'Measures how much a stock moves relative to the market. Beta 1 = moves with market, Beta 2 = moves twice as much.',
            'interpretation': {
                'low': 'Beta < 0.8 = Defensive, less volatile than market',
                'neutral': 'Beta 0.8-1.2 = Moves with the market',
                'high': 'Beta > 1.2 = Aggressive, more volatile than market',
                'negative': 'Beta < 0 = Moves opposite to market (rare)',
            },
            'for_your_trade': 'In bull markets, high beta stocks outperform. In bear markets, low beta stocks protect you. Adjust your beta exposure based on market outlook.',
            'common_mistake': 'Ignoring beta when sizing positions. A $10K position in a beta 2 stock has the same market risk as $20K in a beta 1 stock.',
            'analogy': "Beta is like a boat's sensitivity to waves. A high-beta stock is like a small boat - it moves a lot. A low-beta stock is like a cruise ship - steadier.",
        },
        
        'VAR': {
            'name': 'Value at Risk (VaR)',
            'simple': 'The maximum loss you can expect on a bad day (with 95% confidence).',
            'interpretation': {
                'example': 'VaR 95% = -2% means on 95% of days, you won\'t lose more than 2%',
            },
            'for_your_trade': 'Use VaR to size positions. If your VaR is -3% and you can only afford to lose 1%, you need to reduce position size by 2/3.',
            'common_mistake': 'Forgetting that VaR is a 95% confidence level. 5% of the time, losses will be WORSE than VaR.',
            'analogy': "VaR is like a weather forecast saying '95% chance of rain less than 1 inch.' But 5% of the time, you might get a flood.",
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════════════
    
    POSITION_SIZING = {
        'PERCENT_RISK': {
            'name': 'Percent Risk Method',
            'simple': 'Risk a fixed percentage of your account on each trade (usually 1-2%).',
            'formula': 'Position Size = (Account × Risk%) ÷ (Entry - Stop Loss)',
            'example': 'Account: $10,000, Risk: 2% ($200), Entry: $50, Stop: $48. Position = $200 ÷ $2 = 100 shares ($5,000)',
            'for_your_trade': 'Never risk more than 2% per trade. This ensures you can survive a string of losses.',
            'common_mistake': 'Risking too much per trade. 10 losing trades at 2% risk = 18% drawdown. 10 losing trades at 10% risk = 65% drawdown.',
        },
        
        'KELLY_CRITERION': {
            'name': 'Kelly Criterion',
            'simple': 'Mathematically optimal position size based on your win rate and average win/loss.',
            'formula': 'Kelly % = Win Rate - (Loss Rate ÷ Win/Loss Ratio)',
            'example': 'Win Rate: 60%, Avg Win: $2, Avg Loss: $1. Kelly = 0.6 - (0.4 ÷ 2) = 40%',
            'for_your_trade': 'Full Kelly is too aggressive. Use Half Kelly (divide by 2) or Quarter Kelly for smoother returns.',
            'common_mistake': 'Using full Kelly sizing. It maximizes long-term growth but creates huge drawdowns. Always use fractional Kelly.',
        },
        
        'RISK_REWARD': {
            'name': 'Risk/Reward Ratio',
            'simple': 'How much you stand to gain vs how much you risk. 3:1 means you can make $3 for every $1 risked.',
            'interpretation': {
                'minimum': 'Never take trades with R/R below 2:1',
                'good': 'R/R of 3:1 to 5:1 is ideal',
                'excellent': 'R/R above 5:1 is excellent (Tim Bohen standard)',
            },
            'for_your_trade': 'With 3:1 R/R, you only need to win 25% of trades to break even. With 5:1 R/R, you only need 17%.',
            'common_mistake': 'Taking trades with poor R/R because you "feel confident." Even a 70% win rate loses money with 1:2 R/R.',
            'analogy': "R/R is like a casino game. You want to be the house, not the gambler. The house has better odds on every bet.",
        },
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CATALYST EDUCATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    CATALYST_EXPLANATIONS = {
        'EARNINGS': {
            'name': 'Earnings Report',
            'simple': 'Quarterly report showing company profits. Can cause big moves.',
            'what_to_watch': 'EPS vs expectations, Revenue vs expectations, Guidance for next quarter',
            'risk': 'HIGH - Stocks can move 5-20% on earnings',
            'strategy': 'Either trade before (risky) or wait for reaction and trade the follow-through',
        },
        
        'FDA_APPROVAL': {
            'name': 'FDA Decision',
            'simple': 'Government approval for drugs/medical devices. Binary outcome.',
            'what_to_watch': 'PDUFA date (deadline for FDA decision)',
            'risk': 'EXTREME - Biotech stocks can move 50%+ on FDA decisions',
            'strategy': 'Very risky to hold through. Consider playing the run-up or aftermath instead.',
        },
        
        'FOMC': {
            'name': 'Federal Reserve Meeting',
            'simple': 'Fed decides on interest rates. Affects entire market.',
            'what_to_watch': 'Rate decision, Dot plot, Powell press conference',
            'risk': 'MODERATE - Market can swing 1-3% on Fed days',
            'strategy': 'Reduce position sizes before FOMC. Volatility is high.',
        },
        
        'EX_DIVIDEND': {
            'name': 'Ex-Dividend Date',
            'simple': 'Last day to buy stock and receive the dividend.',
            'what_to_watch': 'Stock typically drops by dividend amount on ex-date',
            'risk': 'LOW - Predictable, small move',
            'strategy': 'Don\'t buy just for dividend. The price drop offsets it.',
        },
    }
    
    @classmethod
    def get_indicator_explanation(cls, indicator: str) -> Dict[str, Any]:
        """Get explanation for a technical indicator."""
        return cls.INDICATOR_EXPLANATIONS.get(indicator.upper(), {
            'name': indicator,
            'simple': 'Explanation not available',
        })
    
    @classmethod
    def get_fundamental_explanation(cls, metric: str) -> Dict[str, Any]:
        """Get explanation for a fundamental metric."""
        return cls.FUNDAMENTAL_EXPLANATIONS.get(metric.upper(), {
            'name': metric,
            'simple': 'Explanation not available',
        })
    
    @classmethod
    def get_risk_explanation(cls, metric: str) -> Dict[str, Any]:
        """Get explanation for a risk metric."""
        return cls.RISK_EXPLANATIONS.get(metric.upper(), {
            'name': metric,
            'simple': 'Explanation not available',
        })
    
    @classmethod
    def get_position_sizing_guide(cls, method: str = None) -> Dict[str, Any]:
        """Get position sizing education."""
        if method:
            return cls.POSITION_SIZING.get(method.upper(), {})
        return cls.POSITION_SIZING
    
    @classmethod
    def get_catalyst_explanation(cls, catalyst: str) -> Dict[str, Any]:
        """Get explanation for a catalyst type."""
        return cls.CATALYST_EXPLANATIONS.get(catalyst.upper(), {
            'name': catalyst,
            'simple': 'Explanation not available',
        })
    
    @classmethod
    def explain_value(cls, metric: str, value: float) -> str:
        """
        Generate a plain English explanation for a specific metric value.
        
        Args:
            metric: The metric name (e.g., 'RSI', 'PE_RATIO')
            value: The current value
        
        Returns:
            Plain English explanation
        """
        metric = metric.upper()
        
        if metric == 'RSI':
            if value >= 70:
                return f"RSI is {value:.1f} (OVERBOUGHT). The stock has risen quickly and may be due for a pullback. Consider waiting for a better entry or taking profits if you're long."
            elif value <= 30:
                return f"RSI is {value:.1f} (OVERSOLD). The stock has fallen quickly and may bounce. This could be a buying opportunity, but confirm with other indicators."
            else:
                return f"RSI is {value:.1f} (NEUTRAL). No extreme momentum. The stock is trading normally."
        
        elif metric == 'PE_RATIO' or metric == 'PE':
            if value < 0:
                return f"P/E is negative (company is losing money). Be cautious - unprofitable companies are higher risk."
            elif value < 15:
                return f"P/E is {value:.1f} (LOW). Stock may be undervalued or a slow-growth company. Compare to industry average."
            elif value < 25:
                return f"P/E is {value:.1f} (MODERATE). Fair valuation for a growth company."
            else:
                return f"P/E is {value:.1f} (HIGH). Stock is expensive - high growth is expected. Make sure the growth justifies the price."
        
        elif metric == 'SHARPE_RATIO' or metric == 'SHARPE':
            if value < 0:
                return f"Sharpe Ratio is {value:.2f} (NEGATIVE). This investment is losing money on a risk-adjusted basis."
            elif value < 1:
                return f"Sharpe Ratio is {value:.2f} (SUBPAR). Risk-adjusted returns are below average."
            elif value < 2:
                return f"Sharpe Ratio is {value:.2f} (GOOD). Solid risk-adjusted returns."
            else:
                return f"Sharpe Ratio is {value:.2f} (EXCELLENT). Outstanding risk-adjusted returns."
        
        elif metric == 'BETA':
            if value < 0.8:
                return f"Beta is {value:.2f} (DEFENSIVE). Less volatile than the market. Good for conservative portfolios."
            elif value < 1.2:
                return f"Beta is {value:.2f} (NEUTRAL). Moves roughly with the market."
            else:
                return f"Beta is {value:.2f} (AGGRESSIVE). More volatile than the market. Higher risk and potential reward."
        
        elif metric == 'MAX_DRAWDOWN':
            value = abs(value)
            if value < 10:
                return f"Max Drawdown is {value:.1f}% (LOW RISK). Conservative investment with small historical declines."
            elif value < 20:
                return f"Max Drawdown is {value:.1f}% (MODERATE RISK). Typical for diversified portfolios."
            elif value < 40:
                return f"Max Drawdown is {value:.1f}% (HIGH RISK). Significant declines possible. Size positions accordingly."
            else:
                return f"Max Drawdown is {value:.1f}% (EXTREME RISK). Very volatile. Only for aggressive investors."
        
        return f"{metric}: {value}"
    
    @classmethod
    def get_all_explanations(cls) -> Dict[str, Dict]:
        """Get all explanations in one dictionary."""
        return {
            'indicators': cls.INDICATOR_EXPLANATIONS,
            'fundamentals': cls.FUNDAMENTAL_EXPLANATIONS,
            'risk': cls.RISK_EXPLANATIONS,
            'position_sizing': cls.POSITION_SIZING,
            'catalysts': cls.CATALYST_EXPLANATIONS,
        }


# Convenience function for quick access
def explain(metric: str, value: float = None) -> str:
    """Quick function to explain a metric."""
    if value is not None:
        return UserEducation.explain_value(metric, value)
    
    # Try to find in any category
    exp = UserEducation.get_indicator_explanation(metric)
    if exp.get('simple') != 'Explanation not available':
        return exp['simple']
    
    exp = UserEducation.get_fundamental_explanation(metric)
    if exp.get('simple') != 'Explanation not available':
        return exp['simple']
    
    exp = UserEducation.get_risk_explanation(metric)
    if exp.get('simple') != 'Explanation not available':
        return exp['simple']
    
    return f"No explanation available for {metric}"


if __name__ == "__main__":
    # Demo
    print("=== USER EDUCATION MODULE ===\n")
    
    print("RSI Explanation:")
    print(UserEducation.explain_value('RSI', 75))
    print()
    
    print("P/E Explanation:")
    print(UserEducation.explain_value('PE_RATIO', 35))
    print()
    
    print("Sharpe Ratio Explanation:")
    print(UserEducation.explain_value('SHARPE_RATIO', 1.5))
    print()
    
    print("TTM Squeeze Full Explanation:")
    ttm = UserEducation.get_indicator_explanation('TTM_SQUEEZE')
    print(f"Name: {ttm['name']}")
    print(f"Simple: {ttm['simple']}")
    print(f"Analogy: {ttm['analogy']}")
