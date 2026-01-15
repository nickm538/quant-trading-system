"""
FACTOR SCORING ENGINE v1.0
==========================

Comprehensive factor weighting system for legendary-grade analysis.
Implements the 40/40/20 weighting system:
- Macro Factors: 40%
- Micro Factors: 40%  
- Non-Traditional Factors: 20%

Each factor is scored 0-100 and weighted according to its importance.
The final composite score drives conviction levels and position sizing.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import math


class FactorScoringEngine:
    """
    Comprehensive factor scoring system for unbiased, weighted analysis.
    
    Scoring Philosophy:
    - 0-20: Very Bearish
    - 21-40: Bearish
    - 41-60: Neutral
    - 61-80: Bullish
    - 81-100: Very Bullish
    """
    
    # Factor weights (must sum to 1.0)
    WEIGHTS = {
        # MACRO FACTORS (40% total)
        "monetary_policy": 0.12,
        "economic_cycle": 0.10,
        "market_regime": 0.08,
        "sector_rotation": 0.05,
        "global_macro": 0.05,
        # MICRO FACTORS (40% total)
        "price_action": 0.12,
        "fundamentals": 0.10,
        "technicals": 0.08,
        "catalysts": 0.05,
        "smart_money": 0.05,
        # NON-TRADITIONAL FACTORS (20% total)
        "sentiment": 0.06,
        "positioning": 0.05,
        "seasonality": 0.04,
        "liquidity": 0.03,
        "correlation": 0.02
    }
    
    def __init__(self):
        """Initialize the factor scoring engine."""
        self.scores = {}
        self.explanations = {}
        
    def score_all_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score all factors and compute weighted composite score.
        
        Args:
            data: Dictionary containing all market and stock data
            
        Returns:
            Dictionary with individual scores, composite score, and analysis
        """
        # Score each factor category
        macro_scores = self._score_macro_factors(data)
        micro_scores = self._score_micro_factors(data)
        nontraditional_scores = self._score_nontraditional_factors(data)
        
        # Combine all scores
        all_scores = {**macro_scores, **micro_scores, **nontraditional_scores}
        
        # Calculate weighted composite score
        composite_score = 0.0
        for factor, score in all_scores.items():
            weight = self.WEIGHTS.get(factor, 0)
            composite_score += score * weight
        
        # Determine overall signal
        if composite_score >= 70:
            signal = "STRONG_BUY"
            conviction = min(10, int((composite_score - 60) / 4) + 7)
        elif composite_score >= 60:
            signal = "BUY"
            conviction = int((composite_score - 50) / 5) + 5
        elif composite_score >= 40:
            signal = "NEUTRAL"
            conviction = 5
        elif composite_score >= 30:
            signal = "SELL"
            conviction = int((50 - composite_score) / 5) + 5
        else:
            signal = "STRONG_SELL"
            conviction = min(10, int((40 - composite_score) / 4) + 7)
        
        # Calculate category scores
        macro_composite = sum(macro_scores.get(f, 50) * self.WEIGHTS.get(f, 0) 
                            for f in ["monetary_policy", "economic_cycle", "market_regime", 
                                     "sector_rotation", "global_macro"]) / 0.40 * 100
        
        micro_composite = sum(micro_scores.get(f, 50) * self.WEIGHTS.get(f, 0)
                            for f in ["price_action", "fundamentals", "technicals",
                                     "catalysts", "smart_money"]) / 0.40 * 100
        
        nontraditional_composite = sum(nontraditional_scores.get(f, 50) * self.WEIGHTS.get(f, 0)
                                      for f in ["sentiment", "positioning", "seasonality",
                                               "liquidity", "correlation"]) / 0.20 * 100
        
        # Check for macro/micro alignment
        alignment = "ALIGNED" if abs(macro_composite - micro_composite) < 20 else "CONFLICTING"
        
        return {
            "composite_score": round(composite_score, 1),
            "signal": signal,
            "conviction": conviction,
            "alignment": alignment,
            "category_scores": {
                "macro": round(macro_composite, 1),
                "micro": round(micro_composite, 1),
                "nontraditional": round(nontraditional_composite, 1)
            },
            "individual_scores": all_scores,
            "explanations": self.explanations,
            "bull_probability": self._calculate_bull_probability(composite_score),
            "bear_probability": self._calculate_bear_probability(composite_score),
            "confidence_interval": self._calculate_confidence_interval(composite_score, all_scores)
        }
    
    def _score_macro_factors(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Score all macro factors (40% weight total)."""
        scores = {}
        macro = data.get("macro", {})
        
        # 1. Monetary Policy (12% weight)
        mp_score = 50  # Neutral default
        mp_explanation = []
        
        vix = macro.get("vix", {})
        vix_level = vix.get("level", 20)
        
        # VIX-based Fed policy proxy
        if vix_level < 15:
            mp_score += 15  # Low fear = accommodative environment
            mp_explanation.append("Low VIX suggests accommodative conditions")
        elif vix_level > 25:
            mp_score -= 15  # High fear = tightening concerns
            mp_explanation.append("Elevated VIX suggests policy uncertainty")
        
        # Bond market signal (TLT)
        bonds = macro.get("bonds", {})
        if bonds.get("trend") == "YIELDS_FALLING":
            mp_score += 10  # Falling yields = dovish
            mp_explanation.append("Falling yields suggest dovish Fed expectations")
        elif bonds.get("trend") == "YIELDS_RISING":
            mp_score -= 10  # Rising yields = hawkish
            mp_explanation.append("Rising yields suggest hawkish Fed expectations")
        
        scores["monetary_policy"] = max(0, min(100, mp_score))
        self.explanations["monetary_policy"] = "; ".join(mp_explanation) if mp_explanation else "Neutral monetary conditions"
        
        # 2. Economic Cycle (10% weight)
        ec_score = 50
        ec_explanation = []
        
        spy = macro.get("spy", {})
        spy_change = spy.get("change_1m", 0)
        
        if spy_change > 5:
            ec_score += 20
            ec_explanation.append(f"Strong market momentum (+{spy_change:.1f}% 1M)")
        elif spy_change > 2:
            ec_score += 10
            ec_explanation.append(f"Positive market momentum (+{spy_change:.1f}% 1M)")
        elif spy_change < -5:
            ec_score -= 20
            ec_explanation.append(f"Weak market momentum ({spy_change:.1f}% 1M)")
        elif spy_change < -2:
            ec_score -= 10
            ec_explanation.append(f"Negative market momentum ({spy_change:.1f}% 1M)")
        
        scores["economic_cycle"] = max(0, min(100, ec_score))
        self.explanations["economic_cycle"] = "; ".join(ec_explanation) if ec_explanation else "Neutral economic conditions"
        
        # 3. Market Regime (8% weight)
        mr_score = 50
        mr_explanation = []
        
        regime = macro.get("market_regime", "NEUTRAL")
        if regime == "RISK_ON_BULL":
            mr_score = 80
            mr_explanation.append("Risk-on bull market regime")
        elif regime == "CAUTIOUS_BULL":
            mr_score = 65
            mr_explanation.append("Cautious bull market regime")
        elif regime == "CORRECTION":
            mr_score = 35
            mr_explanation.append("Market in correction mode")
        elif regime == "RISK_OFF_BEAR":
            mr_score = 20
            mr_explanation.append("Risk-off bear market regime")
        
        scores["market_regime"] = mr_score
        self.explanations["market_regime"] = "; ".join(mr_explanation) if mr_explanation else "Neutral market regime"
        
        # 4. Sector Rotation (5% weight)
        sr_score = 50
        sr_explanation = []
        
        sector_rotation = macro.get("sector_rotation", {})
        leading = sector_rotation.get("leading", [])
        lagging = sector_rotation.get("lagging", [])
        
        # Check if cyclical sectors are leading (bullish) or defensive (bearish)
        cyclical_sectors = ["Technology", "Consumer Discretionary", "Financials", "Industrials"]
        defensive_sectors = ["Utilities", "Consumer Staples", "Healthcare"]
        
        leading_names = [s[0] for s in leading] if leading else []
        
        cyclical_leading = sum(1 for s in leading_names if s in cyclical_sectors)
        defensive_leading = sum(1 for s in leading_names if s in defensive_sectors)
        
        if cyclical_leading > defensive_leading:
            sr_score += 15
            sr_explanation.append("Cyclical sectors leading (risk-on)")
        elif defensive_leading > cyclical_leading:
            sr_score -= 15
            sr_explanation.append("Defensive sectors leading (risk-off)")
        
        scores["sector_rotation"] = max(0, min(100, sr_score))
        self.explanations["sector_rotation"] = "; ".join(sr_explanation) if sr_explanation else "Balanced sector rotation"
        
        # 5. Global Macro (5% weight)
        gm_score = 50
        gm_explanation = []
        
        dxy = macro.get("dxy", {})
        dxy_strength = dxy.get("strength", "NEUTRAL")
        
        if dxy_strength == "WEAK":
            gm_score += 10  # Weak dollar generally bullish for stocks
            gm_explanation.append("Weak dollar supportive for equities")
        elif dxy_strength == "STRONG":
            gm_score -= 10  # Strong dollar can be headwind
            gm_explanation.append("Strong dollar may pressure equities")
        
        scores["global_macro"] = max(0, min(100, gm_score))
        self.explanations["global_macro"] = "; ".join(gm_explanation) if gm_explanation else "Neutral global conditions"
        
        return scores
    
    def _score_micro_factors(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Score all micro factors (40% weight total)."""
        scores = {}
        
        price_data = data.get("price_data", {})
        technicals = data.get("technicals", {})
        fundamentals = data.get("fundamentals", {})
        smart_money = data.get("smart_money", {})
        catalysts = data.get("catalysts", [])
        
        # 1. Price Action (12% weight)
        pa_score = 50
        pa_explanation = []
        
        change_pct = price_data.get("change_pct", 0)
        current = price_data.get("current", 0)
        high_52w = price_data.get("52w_high", current)
        low_52w = price_data.get("52w_low", current)
        
        # Position in 52-week range
        if high_52w > low_52w:
            range_position = (current - low_52w) / (high_52w - low_52w)
            if range_position > 0.8:
                pa_score += 15  # Near highs = strength
                pa_explanation.append("Trading near 52-week highs (strength)")
            elif range_position < 0.2:
                pa_score -= 15  # Near lows = weakness
                pa_explanation.append("Trading near 52-week lows (weakness)")
        
        # Recent momentum
        if change_pct > 3:
            pa_score += 10
            pa_explanation.append(f"Strong recent momentum (+{change_pct:.1f}%)")
        elif change_pct < -3:
            pa_score -= 10
            pa_explanation.append(f"Weak recent momentum ({change_pct:.1f}%)")
        
        scores["price_action"] = max(0, min(100, pa_score))
        self.explanations["price_action"] = "; ".join(pa_explanation) if pa_explanation else "Neutral price action"
        
        # 2. Fundamentals (10% weight)
        fund_score = 50
        fund_explanation = []
        
        pe_ratio = fundamentals.get("pe_ratio")
        peg_ratio = fundamentals.get("peg_ratio")
        profit_margin = fundamentals.get("profit_margin")
        roe = fundamentals.get("roe")
        revenue_growth = fundamentals.get("revenue_growth")
        
        # P/E analysis
        if pe_ratio:
            if pe_ratio < 15:
                fund_score += 10
                fund_explanation.append(f"Attractive P/E ({pe_ratio:.1f})")
            elif pe_ratio > 35:
                fund_score -= 10
                fund_explanation.append(f"Expensive P/E ({pe_ratio:.1f})")
        
        # Profitability
        if profit_margin and profit_margin > 0.15:
            fund_score += 10
            fund_explanation.append(f"Strong margins ({profit_margin*100:.1f}%)")
        elif profit_margin and profit_margin < 0.05:
            fund_score -= 10
            fund_explanation.append(f"Weak margins ({profit_margin*100:.1f}%)")
        
        # Growth
        if revenue_growth and revenue_growth > 0.15:
            fund_score += 10
            fund_explanation.append(f"Strong growth ({revenue_growth*100:.1f}%)")
        elif revenue_growth and revenue_growth < 0:
            fund_score -= 10
            fund_explanation.append(f"Declining revenue ({revenue_growth*100:.1f}%)")
        
        scores["fundamentals"] = max(0, min(100, fund_score))
        self.explanations["fundamentals"] = "; ".join(fund_explanation) if fund_explanation else "Neutral fundamentals"
        
        # 3. Technicals (8% weight)
        tech_score = 50
        tech_explanation = []
        
        rsi = technicals.get("rsi_14")
        trend = technicals.get("trend")
        above_sma20 = technicals.get("above_sma_20")
        above_sma50 = technicals.get("above_sma_50")
        
        # RSI
        if rsi:
            if rsi > 70:
                tech_score -= 10  # Overbought
                tech_explanation.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 30:
                tech_score += 10  # Oversold (potential bounce)
                tech_explanation.append(f"RSI oversold ({rsi:.1f})")
            elif 40 < rsi < 60:
                tech_explanation.append(f"RSI neutral ({rsi:.1f})")
        
        # Trend
        if trend == "BULLISH":
            tech_score += 15
            tech_explanation.append("Bullish trend")
        elif trend == "BEARISH":
            tech_score -= 15
            tech_explanation.append("Bearish trend")
        
        # Moving averages
        if above_sma20 and above_sma50:
            tech_score += 10
            tech_explanation.append("Above key moving averages")
        elif not above_sma20 and not above_sma50:
            tech_score -= 10
            tech_explanation.append("Below key moving averages")
        
        scores["technicals"] = max(0, min(100, tech_score))
        self.explanations["technicals"] = "; ".join(tech_explanation) if tech_explanation else "Neutral technicals"
        
        # 4. Catalysts (5% weight)
        cat_score = 50
        cat_explanation = []
        
        if catalysts:
            for cat in catalysts:
                if cat.get("type") == "EARNINGS":
                    cat_score += 5  # Upcoming earnings can be catalyst
                    cat_explanation.append(f"Earnings upcoming: {cat.get('date', 'TBD')}")
                elif cat.get("type") == "EX_DIVIDEND":
                    cat_score += 3
                    cat_explanation.append(f"Ex-dividend: {cat.get('date', 'TBD')}")
        
        scores["catalysts"] = max(0, min(100, cat_score))
        self.explanations["catalysts"] = "; ".join(cat_explanation) if cat_explanation else "No major catalysts identified"
        
        # 5. Smart Money (5% weight)
        sm_score = 50
        sm_explanation = []
        
        if smart_money:
            signal = smart_money.get("signal", "NEUTRAL")
            sm_score_raw = smart_money.get("smart_money_score", 50)
            
            if signal == "ACCUMULATION":
                sm_score = max(60, sm_score_raw)
                sm_explanation.append("Smart money accumulating")
            elif signal == "DISTRIBUTION":
                sm_score = min(40, sm_score_raw)
                sm_explanation.append("Smart money distributing")
            else:
                sm_score = sm_score_raw
                sm_explanation.append("Neutral smart money activity")
        
        scores["smart_money"] = max(0, min(100, sm_score))
        self.explanations["smart_money"] = "; ".join(sm_explanation) if sm_explanation else "No clear smart money signal"
        
        return scores
    
    def _score_nontraditional_factors(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Score all non-traditional factors (20% weight total)."""
        scores = {}
        
        smart_money = data.get("smart_money", {})
        macro = data.get("macro", {})
        
        # 1. Sentiment (6% weight)
        sent_score = 50
        sent_explanation = []
        
        vix = macro.get("vix", {})
        vix_regime = vix.get("regime", "NEUTRAL")
        
        # VIX as sentiment proxy (contrarian)
        if vix_regime == "EXTREME_FEAR":
            sent_score = 75  # Contrarian bullish
            sent_explanation.append("Extreme fear = contrarian buy signal")
        elif vix_regime == "HIGH_FEAR":
            sent_score = 60
            sent_explanation.append("Elevated fear = potential opportunity")
        elif vix_regime == "LOW_FEAR":
            sent_score = 40  # Complacency warning
            sent_explanation.append("Low fear = complacency risk")
        
        scores["sentiment"] = sent_score
        self.explanations["sentiment"] = "; ".join(sent_explanation) if sent_explanation else "Neutral sentiment"
        
        # 2. Positioning (5% weight)
        pos_score = 50
        pos_explanation = []
        
        short_interest = smart_money.get("short_interest", {})
        short_pct = short_interest.get("short_percent_float", 0)
        
        if short_pct > 20:
            pos_score = 65  # High short interest = squeeze potential
            pos_explanation.append(f"High short interest ({short_pct:.1f}%) - squeeze potential")
        elif short_pct > 10:
            pos_score = 55
            pos_explanation.append(f"Elevated short interest ({short_pct:.1f}%)")
        elif short_pct < 3:
            pos_score = 45  # Low shorts = less fuel for rally
            pos_explanation.append(f"Low short interest ({short_pct:.1f}%)")
        
        scores["positioning"] = pos_score
        self.explanations["positioning"] = "; ".join(pos_explanation) if pos_explanation else "Neutral positioning"
        
        # 3. Seasonality (4% weight)
        seas_score = 50
        seas_explanation = []
        
        now = datetime.now()
        month = now.month
        
        # Historical seasonal patterns
        bullish_months = [11, 12, 1, 4]  # Nov, Dec, Jan, Apr
        bearish_months = [9]  # September
        
        if month in bullish_months:
            seas_score = 60
            seas_explanation.append(f"Historically bullish month ({now.strftime('%B')})")
        elif month in bearish_months:
            seas_score = 40
            seas_explanation.append(f"Historically weak month ({now.strftime('%B')})")
        
        scores["seasonality"] = seas_score
        self.explanations["seasonality"] = "; ".join(seas_explanation) if seas_explanation else "Neutral seasonality"
        
        # 4. Liquidity (3% weight)
        liq_score = 50
        liq_explanation = []
        
        volume_analysis = smart_money.get("volume_analysis", {})
        volume_ratio = volume_analysis.get("volume_ratio_20d", 1.0)
        
        if volume_ratio > 2.0:
            liq_score = 65  # High liquidity = institutional interest
            liq_explanation.append(f"High volume ({volume_ratio:.1f}x avg) - institutional interest")
        elif volume_ratio < 0.5:
            liq_score = 35  # Low liquidity = lack of interest
            liq_explanation.append(f"Low volume ({volume_ratio:.1f}x avg) - limited interest")
        
        scores["liquidity"] = liq_score
        self.explanations["liquidity"] = "; ".join(liq_explanation) if liq_explanation else "Normal liquidity"
        
        # 5. Correlation (2% weight)
        corr_score = 50
        corr_explanation = []
        
        # Check if stock is diverging from market (can be bullish or bearish)
        spy = macro.get("spy", {})
        spy_trend = spy.get("trend", "NEUTRAL")
        
        # Simple correlation check - would need more data for full analysis
        corr_explanation.append("Correlation analysis requires more data")
        
        scores["correlation"] = corr_score
        self.explanations["correlation"] = "; ".join(corr_explanation) if corr_explanation else "Normal correlation"
        
        return scores
    
    def _calculate_bull_probability(self, composite_score: float) -> float:
        """Calculate probability of bullish outcome based on composite score."""
        # Sigmoid-like transformation
        if composite_score >= 70:
            return min(85, 50 + (composite_score - 50) * 0.7)
        elif composite_score <= 30:
            return max(15, 50 + (composite_score - 50) * 0.7)
        else:
            return 50 + (composite_score - 50) * 0.5
    
    def _calculate_bear_probability(self, composite_score: float) -> float:
        """Calculate probability of bearish outcome."""
        return 100 - self._calculate_bull_probability(composite_score)
    
    def _calculate_confidence_interval(self, composite_score: float, 
                                       all_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate confidence interval based on score dispersion."""
        # Calculate standard deviation of scores
        scores_list = list(all_scores.values())
        mean = sum(scores_list) / len(scores_list)
        variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
        std_dev = math.sqrt(variance)
        
        # Higher dispersion = wider confidence interval
        margin = std_dev * 0.5
        
        return {
            "lower": max(0, composite_score - margin),
            "upper": min(100, composite_score + margin),
            "dispersion": round(std_dev, 1)
        }
    
    def get_factor_summary(self, scores: Dict[str, Any]) -> str:
        """Generate a human-readable summary of factor scores."""
        summary_parts = []
        
        summary_parts.append(f"📊 **FACTOR SCORING SUMMARY**")
        summary_parts.append(f"Composite Score: {scores['composite_score']}/100")
        summary_parts.append(f"Signal: {scores['signal']} | Conviction: {scores['conviction']}/10")
        summary_parts.append(f"Macro/Micro Alignment: {scores['alignment']}")
        summary_parts.append("")
        
        cat_scores = scores['category_scores']
        summary_parts.append(f"**Category Breakdown:**")
        summary_parts.append(f"- Macro (40% weight): {cat_scores['macro']:.1f}/100")
        summary_parts.append(f"- Micro (40% weight): {cat_scores['micro']:.1f}/100")
        summary_parts.append(f"- Non-Traditional (20% weight): {cat_scores['nontraditional']:.1f}/100")
        summary_parts.append("")
        
        summary_parts.append(f"**Probability Assessment:**")
        summary_parts.append(f"- Bull Probability: {scores['bull_probability']:.1f}%")
        summary_parts.append(f"- Bear Probability: {scores['bear_probability']:.1f}%")
        
        ci = scores['confidence_interval']
        summary_parts.append(f"- Confidence Interval: {ci['lower']:.1f} - {ci['upper']:.1f}")
        summary_parts.append(f"- Score Dispersion: {ci['dispersion']:.1f} (lower = more aligned factors)")
        
        return "\n".join(summary_parts)


# Standalone test
if __name__ == "__main__":
    # Test with sample data
    engine = FactorScoringEngine()
    
    test_data = {
        "macro": {
            "vix": {"level": 18, "regime": "LOW_FEAR"},
            "spy": {"price": 590, "change_1m": 3.5, "trend": "BULLISH"},
            "bonds": {"trend": "YIELDS_FALLING"},
            "dxy": {"strength": "NEUTRAL"},
            "market_regime": "RISK_ON_BULL",
            "sector_rotation": {
                "leading": [("Technology", 5.2), ("Financials", 4.1)],
                "lagging": [("Utilities", -1.2)]
            }
        },
        "price_data": {
            "current": 185.50,
            "change_pct": 2.3,
            "52w_high": 200,
            "52w_low": 140
        },
        "technicals": {
            "rsi_14": 58,
            "trend": "BULLISH",
            "above_sma_20": True,
            "above_sma_50": True
        },
        "fundamentals": {
            "pe_ratio": 22,
            "profit_margin": 0.18,
            "revenue_growth": 0.12
        },
        "smart_money": {
            "signal": "ACCUMULATION",
            "smart_money_score": 65,
            "short_interest": {"short_percent_float": 8},
            "volume_analysis": {"volume_ratio_20d": 1.3}
        },
        "catalysts": [
            {"type": "EARNINGS", "date": "2026-01-28"}
        ]
    }
    
    result = engine.score_all_factors(test_data)
    print(engine.get_factor_summary(result))
    print("\n" + "="*50 + "\n")
    print("Individual Scores:")
    for factor, score in result["individual_scores"].items():
        print(f"  {factor}: {score}")
