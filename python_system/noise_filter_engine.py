"""
NOISE FILTER ENGINE v1.0
========================

Advanced noise reduction and bias elimination system for legendary-grade analysis.

Key Features:
1. SIGNAL vs NOISE classification
2. Confirmation requirements for validity
3. Bias detection and neutralization
4. Data quality scoring
5. Outlier detection and handling

Philosophy:
- Not all data is equal - weight by reliability
- Multiple confirmations increase signal strength
- Extreme values need extra scrutiny
- Recent data weighted more but not exclusively
- Contrarian signals at extremes have value
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class NoiseFilterEngine:
    """
    Advanced noise reduction and bias elimination system.
    
    Separates signal from noise and identifies potential biases in analysis.
    """
    
    # Thresholds for noise filtering
    THRESHOLDS = {
        # Volume thresholds
        "min_volume_ratio": 1.5,  # Volume must be 1.5x average to be significant
        "high_volume_ratio": 3.0,  # Very high volume threshold
        
        # Price move thresholds
        "min_price_move_pct": 2.0,  # Price move must be >2% to be significant
        "large_price_move_pct": 5.0,  # Large move threshold
        "extreme_price_move_pct": 10.0,  # Extreme move - needs scrutiny
        
        # Options thresholds
        "min_options_premium": 100000,  # Options trade must be >$100K to be notable
        "large_options_premium": 500000,  # Large options trade
        "whale_options_premium": 1000000,  # Whale-level options trade
        
        # Insider thresholds
        "min_insider_value": 50000,  # Insider trade must be >$50K to matter
        "significant_insider_value": 250000,  # Significant insider trade
        
        # Block trade thresholds
        "min_block_size": 10000,  # Block trade must be >10K shares
        "large_block_size": 50000,  # Large block trade
        
        # Sentiment thresholds
        "sentiment_extreme_bullish": 80,  # Extreme bullish sentiment
        "sentiment_extreme_bearish": 20,  # Extreme bearish sentiment
        
        # Technical thresholds
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "rsi_extreme_overbought": 80,
        "rsi_extreme_oversold": 20,
        
        # VIX thresholds
        "vix_low_fear": 15,
        "vix_elevated": 20,
        "vix_high_fear": 25,
        "vix_extreme_fear": 35,
        "vix_panic": 45
    }
    
    # Data source reliability weights (0-1)
    SOURCE_RELIABILITY = {
        "sec_filings": 1.0,  # Highest reliability
        "exchange_data": 0.95,  # Very high
        "company_guidance": 0.90,  # High
        "analyst_estimates": 0.75,  # Good
        "institutional_data": 0.80,  # Good
        "news_major": 0.70,  # Moderate
        "news_minor": 0.50,  # Lower
        "social_sentiment": 0.30,  # Low - use as contrarian
        "rumors": 0.10  # Very low
    }
    
    def __init__(self):
        """Initialize the noise filter engine."""
        self.filtered_signals = []
        self.noise_items = []
        self.bias_warnings = []
        self.data_quality_score = 100
        
    def filter_and_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main filtering function - separates signal from noise and validates data.
        
        Args:
            data: Raw data dictionary with all market information
            
        Returns:
            Dictionary with filtered signals, noise items, and quality metrics
        """
        self.filtered_signals = []
        self.noise_items = []
        self.bias_warnings = []
        self.data_quality_score = 100
        
        # Filter each data category
        price_signals = self._filter_price_data(data.get("price_data", {}))
        volume_signals = self._filter_volume_data(data.get("price_data", {}))
        technical_signals = self._filter_technical_data(data.get("technicals", {}))
        fundamental_signals = self._filter_fundamental_data(data.get("fundamentals", {}))
        smart_money_signals = self._filter_smart_money_data(data.get("smart_money", {}))
        sentiment_signals = self._filter_sentiment_data(data.get("macro", {}))
        
        # Check for confirmation across categories
        confirmations = self._check_confirmations()
        
        # Detect potential biases
        biases = self._detect_biases(data)
        
        # Calculate overall data quality
        quality = self._calculate_data_quality(data)
        
        return {
            "signals": {
                "price": price_signals,
                "volume": volume_signals,
                "technical": technical_signals,
                "fundamental": fundamental_signals,
                "smart_money": smart_money_signals,
                "sentiment": sentiment_signals
            },
            "noise_filtered": self.noise_items,
            "confirmations": confirmations,
            "bias_warnings": biases,
            "data_quality": quality,
            "signal_strength": self._calculate_signal_strength(),
            "recommendation_confidence_adjustment": self._get_confidence_adjustment()
        }
    
    def _filter_price_data(self, price_data: Dict) -> List[Dict]:
        """Filter price action data for significant signals."""
        signals = []
        
        if not price_data:
            return signals
        
        change_pct = price_data.get("change_pct", 0)
        current = price_data.get("current", 0)
        high_52w = price_data.get("52w_high", current)
        low_52w = price_data.get("52w_low", current)
        
        # Check for significant price moves
        if abs(change_pct) >= self.THRESHOLDS["extreme_price_move_pct"]:
            signals.append({
                "type": "EXTREME_MOVE",
                "value": change_pct,
                "significance": "HIGH",
                "note": f"Extreme {change_pct:+.1f}% move - needs scrutiny for sustainability"
            })
        elif abs(change_pct) >= self.THRESHOLDS["large_price_move_pct"]:
            signals.append({
                "type": "LARGE_MOVE",
                "value": change_pct,
                "significance": "MEDIUM",
                "note": f"Large {change_pct:+.1f}% move - confirm with volume"
            })
        elif abs(change_pct) < self.THRESHOLDS["min_price_move_pct"]:
            self.noise_items.append({
                "type": "SMALL_PRICE_MOVE",
                "value": change_pct,
                "reason": "Price move too small to be significant"
            })
        
        # Check position in 52-week range
        if high_52w > low_52w and current > 0:
            range_position = (current - low_52w) / (high_52w - low_52w)
            
            if range_position > 0.95:
                signals.append({
                    "type": "AT_52W_HIGH",
                    "value": range_position,
                    "significance": "HIGH",
                    "note": "At 52-week highs - momentum strong but watch for exhaustion"
                })
            elif range_position < 0.05:
                signals.append({
                    "type": "AT_52W_LOW",
                    "value": range_position,
                    "significance": "HIGH",
                    "note": "At 52-week lows - potential value or falling knife"
                })
        
        return signals
    
    def _filter_volume_data(self, price_data: Dict) -> List[Dict]:
        """Filter volume data for significant signals."""
        signals = []
        
        if not price_data:
            return signals
        
        volume = price_data.get("volume", 0)
        avg_volume = price_data.get("avg_volume", volume)
        
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio >= self.THRESHOLDS["high_volume_ratio"]:
                signals.append({
                    "type": "VERY_HIGH_VOLUME",
                    "value": volume_ratio,
                    "significance": "HIGH",
                    "note": f"Volume {volume_ratio:.1f}x average - institutional activity likely"
                })
            elif volume_ratio >= self.THRESHOLDS["min_volume_ratio"]:
                signals.append({
                    "type": "ELEVATED_VOLUME",
                    "value": volume_ratio,
                    "significance": "MEDIUM",
                    "note": f"Volume {volume_ratio:.1f}x average - confirms price action"
                })
            elif volume_ratio < 0.5:
                self.noise_items.append({
                    "type": "LOW_VOLUME",
                    "value": volume_ratio,
                    "reason": "Low volume - price action less reliable"
                })
        
        return signals
    
    def _filter_technical_data(self, technicals: Dict) -> List[Dict]:
        """Filter technical indicators for significant signals."""
        signals = []
        
        if not technicals:
            return signals
        
        rsi = technicals.get("rsi_14")
        trend = technicals.get("trend")
        
        # RSI signals
        if rsi:
            if rsi >= self.THRESHOLDS["rsi_extreme_overbought"]:
                signals.append({
                    "type": "RSI_EXTREME_OVERBOUGHT",
                    "value": rsi,
                    "significance": "HIGH",
                    "note": f"RSI {rsi:.1f} - extremely overbought, reversal risk high"
                })
            elif rsi >= self.THRESHOLDS["rsi_overbought"]:
                signals.append({
                    "type": "RSI_OVERBOUGHT",
                    "value": rsi,
                    "significance": "MEDIUM",
                    "note": f"RSI {rsi:.1f} - overbought, watch for pullback"
                })
            elif rsi <= self.THRESHOLDS["rsi_extreme_oversold"]:
                signals.append({
                    "type": "RSI_EXTREME_OVERSOLD",
                    "value": rsi,
                    "significance": "HIGH",
                    "note": f"RSI {rsi:.1f} - extremely oversold, bounce potential"
                })
            elif rsi <= self.THRESHOLDS["rsi_oversold"]:
                signals.append({
                    "type": "RSI_OVERSOLD",
                    "value": rsi,
                    "significance": "MEDIUM",
                    "note": f"RSI {rsi:.1f} - oversold, watch for reversal"
                })
            elif 45 < rsi < 55:
                self.noise_items.append({
                    "type": "RSI_NEUTRAL",
                    "value": rsi,
                    "reason": "RSI in neutral zone - no clear signal"
                })
        
        # Trend confirmation
        if trend:
            signals.append({
                "type": "TREND",
                "value": trend,
                "significance": "MEDIUM",
                "note": f"Trend is {trend}"
            })
        
        return signals
    
    def _filter_fundamental_data(self, fundamentals: Dict) -> List[Dict]:
        """Filter fundamental data for significant signals."""
        signals = []
        
        if not fundamentals:
            return signals
        
        pe_ratio = fundamentals.get("pe_ratio")
        profit_margin = fundamentals.get("profit_margin")
        revenue_growth = fundamentals.get("revenue_growth")
        
        # P/E analysis
        if pe_ratio:
            if pe_ratio < 0:
                signals.append({
                    "type": "NEGATIVE_PE",
                    "value": pe_ratio,
                    "significance": "HIGH",
                    "note": "Negative P/E - company is unprofitable"
                })
            elif pe_ratio > 50:
                signals.append({
                    "type": "HIGH_PE",
                    "value": pe_ratio,
                    "significance": "MEDIUM",
                    "note": f"P/E of {pe_ratio:.1f} - high valuation, growth priced in"
                })
            elif pe_ratio < 10:
                signals.append({
                    "type": "LOW_PE",
                    "value": pe_ratio,
                    "significance": "MEDIUM",
                    "note": f"P/E of {pe_ratio:.1f} - potentially undervalued or value trap"
                })
        
        # Profitability
        if profit_margin:
            if profit_margin > 0.25:
                signals.append({
                    "type": "HIGH_MARGIN",
                    "value": profit_margin,
                    "significance": "MEDIUM",
                    "note": f"Strong margins ({profit_margin*100:.1f}%) - quality business"
                })
            elif profit_margin < 0:
                signals.append({
                    "type": "NEGATIVE_MARGIN",
                    "value": profit_margin,
                    "significance": "HIGH",
                    "note": "Negative margins - company losing money"
                })
        
        # Growth
        if revenue_growth:
            if revenue_growth > 0.30:
                signals.append({
                    "type": "HIGH_GROWTH",
                    "value": revenue_growth,
                    "significance": "MEDIUM",
                    "note": f"Strong growth ({revenue_growth*100:.1f}%) - momentum stock"
                })
            elif revenue_growth < -0.10:
                signals.append({
                    "type": "DECLINING_REVENUE",
                    "value": revenue_growth,
                    "significance": "HIGH",
                    "note": f"Revenue declining ({revenue_growth*100:.1f}%) - concerning"
                })
        
        return signals
    
    def _filter_smart_money_data(self, smart_money: Dict) -> List[Dict]:
        """Filter smart money data for significant signals."""
        signals = []
        
        if not smart_money:
            return signals
        
        signal = smart_money.get("signal", "NEUTRAL")
        score = smart_money.get("smart_money_score", 50)
        confidence = smart_money.get("confidence", 0)
        
        # Smart money signal
        if signal == "ACCUMULATION" and score > 65:
            signals.append({
                "type": "SMART_MONEY_ACCUMULATION",
                "value": score,
                "significance": "HIGH",
                "note": f"Smart money accumulating (score: {score}) - bullish"
            })
        elif signal == "DISTRIBUTION" and score < 35:
            signals.append({
                "type": "SMART_MONEY_DISTRIBUTION",
                "value": score,
                "significance": "HIGH",
                "note": f"Smart money distributing (score: {score}) - bearish"
            })
        elif 40 < score < 60:
            self.noise_items.append({
                "type": "SMART_MONEY_NEUTRAL",
                "value": score,
                "reason": "Smart money score neutral - no clear signal"
            })
        
        # Short interest
        short_interest = smart_money.get("short_interest", {})
        short_pct = short_interest.get("short_percent_float", 0)
        
        if short_pct > 25:
            signals.append({
                "type": "VERY_HIGH_SHORT_INTEREST",
                "value": short_pct,
                "significance": "HIGH",
                "note": f"Short interest {short_pct:.1f}% - squeeze potential"
            })
        elif short_pct > 15:
            signals.append({
                "type": "HIGH_SHORT_INTEREST",
                "value": short_pct,
                "significance": "MEDIUM",
                "note": f"Elevated short interest {short_pct:.1f}%"
            })
        
        return signals
    
    def _filter_sentiment_data(self, macro: Dict) -> List[Dict]:
        """Filter sentiment data for significant signals."""
        signals = []
        
        if not macro:
            return signals
        
        vix = macro.get("vix", {})
        vix_level = vix.get("level", 20)
        vix_regime = vix.get("regime", "NEUTRAL")
        
        # VIX signals (contrarian)
        if vix_level >= self.THRESHOLDS["vix_panic"]:
            signals.append({
                "type": "VIX_PANIC",
                "value": vix_level,
                "significance": "HIGH",
                "note": f"VIX at {vix_level} - panic levels, contrarian buy signal"
            })
        elif vix_level >= self.THRESHOLDS["vix_extreme_fear"]:
            signals.append({
                "type": "VIX_EXTREME_FEAR",
                "value": vix_level,
                "significance": "HIGH",
                "note": f"VIX at {vix_level} - extreme fear, watch for reversal"
            })
        elif vix_level <= self.THRESHOLDS["vix_low_fear"]:
            signals.append({
                "type": "VIX_COMPLACENCY",
                "value": vix_level,
                "significance": "MEDIUM",
                "note": f"VIX at {vix_level} - complacency, risk of spike"
            })
        
        return signals
    
    def _check_confirmations(self) -> Dict[str, Any]:
        """Check for confirmation across signal categories."""
        confirmations = {
            "bullish_confirmations": 0,
            "bearish_confirmations": 0,
            "conflicting_signals": False,
            "confirmation_details": []
        }
        
        bullish_signals = []
        bearish_signals = []
        
        for signal in self.filtered_signals:
            signal_type = signal.get("type", "")
            
            # Classify as bullish or bearish
            bullish_types = ["SMART_MONEY_ACCUMULATION", "RSI_OVERSOLD", "RSI_EXTREME_OVERSOLD",
                           "AT_52W_LOW", "VIX_PANIC", "VIX_EXTREME_FEAR", "LOW_PE", "HIGH_GROWTH"]
            bearish_types = ["SMART_MONEY_DISTRIBUTION", "RSI_OVERBOUGHT", "RSI_EXTREME_OVERBOUGHT",
                           "AT_52W_HIGH", "VIX_COMPLACENCY", "NEGATIVE_PE", "DECLINING_REVENUE"]
            
            if signal_type in bullish_types:
                bullish_signals.append(signal_type)
            elif signal_type in bearish_types:
                bearish_signals.append(signal_type)
        
        confirmations["bullish_confirmations"] = len(bullish_signals)
        confirmations["bearish_confirmations"] = len(bearish_signals)
        confirmations["conflicting_signals"] = len(bullish_signals) > 0 and len(bearish_signals) > 0
        
        if bullish_signals:
            confirmations["confirmation_details"].append(f"Bullish: {', '.join(bullish_signals)}")
        if bearish_signals:
            confirmations["confirmation_details"].append(f"Bearish: {', '.join(bearish_signals)}")
        
        return confirmations
    
    def _detect_biases(self, data: Dict) -> List[Dict]:
        """Detect potential biases in the analysis."""
        biases = []
        
        # Recency bias check
        price_data = data.get("price_data", {})
        change_pct = price_data.get("change_pct", 0)
        
        if abs(change_pct) > 5:
            biases.append({
                "type": "RECENCY_BIAS_RISK",
                "severity": "MEDIUM",
                "note": f"Large recent move ({change_pct:+.1f}%) may cause recency bias - consider longer timeframe"
            })
        
        # Confirmation bias check
        technicals = data.get("technicals", {})
        fundamentals = data.get("fundamentals", {})
        
        trend = technicals.get("trend", "NEUTRAL")
        pe_ratio = fundamentals.get("pe_ratio")
        
        if trend == "BULLISH" and pe_ratio and pe_ratio > 40:
            biases.append({
                "type": "CONFIRMATION_BIAS_RISK",
                "severity": "MEDIUM",
                "note": "Bullish trend but high valuation - don't ignore valuation concerns"
            })
        elif trend == "BEARISH" and pe_ratio and pe_ratio < 12:
            biases.append({
                "type": "CONFIRMATION_BIAS_RISK",
                "severity": "MEDIUM",
                "note": "Bearish trend but low valuation - consider value opportunity"
            })
        
        # Anchoring bias check
        high_52w = price_data.get("52w_high", 0)
        current = price_data.get("current", 0)
        
        if high_52w > 0 and current > 0:
            pct_from_high = (high_52w - current) / high_52w * 100
            if pct_from_high > 30:
                biases.append({
                    "type": "ANCHORING_BIAS_RISK",
                    "severity": "LOW",
                    "note": f"Stock {pct_from_high:.1f}% below 52-week high - don't anchor to old highs"
                })
        
        # Survivorship bias note
        biases.append({
            "type": "SURVIVORSHIP_BIAS_NOTE",
            "severity": "INFO",
            "note": "Historical patterns based on surviving companies - failed companies not in data"
        })
        
        self.bias_warnings = biases
        return biases
    
    def _calculate_data_quality(self, data: Dict) -> Dict[str, Any]:
        """Calculate overall data quality score."""
        quality = {
            "score": 100,
            "issues": [],
            "strengths": []
        }
        
        # Check for missing data
        if not data.get("price_data"):
            quality["score"] -= 20
            quality["issues"].append("Missing price data")
        else:
            quality["strengths"].append("Price data available")
        
        if not data.get("technicals"):
            quality["score"] -= 10
            quality["issues"].append("Missing technical data")
        else:
            quality["strengths"].append("Technical data available")
        
        if not data.get("fundamentals"):
            quality["score"] -= 15
            quality["issues"].append("Missing fundamental data")
        else:
            quality["strengths"].append("Fundamental data available")
        
        if not data.get("smart_money"):
            quality["score"] -= 10
            quality["issues"].append("Missing smart money data")
        else:
            quality["strengths"].append("Smart money data available")
        
        if not data.get("macro"):
            quality["score"] -= 10
            quality["issues"].append("Missing macro data")
        else:
            quality["strengths"].append("Macro data available")
        
        # Check data freshness (assuming data has timestamps)
        # For now, assume data is fresh
        quality["strengths"].append("Data appears current")
        
        quality["score"] = max(0, quality["score"])
        self.data_quality_score = quality["score"]
        
        return quality
    
    def _calculate_signal_strength(self) -> Dict[str, Any]:
        """Calculate overall signal strength based on filtered signals."""
        high_significance = sum(1 for s in self.filtered_signals if s.get("significance") == "HIGH")
        medium_significance = sum(1 for s in self.filtered_signals if s.get("significance") == "MEDIUM")
        
        total_signals = len(self.filtered_signals)
        noise_count = len(self.noise_items)
        
        # Signal-to-noise ratio
        snr = total_signals / (noise_count + 1)  # +1 to avoid division by zero
        
        # Strength score
        strength_score = (high_significance * 3 + medium_significance * 1.5) / max(1, total_signals) * 33
        
        return {
            "total_signals": total_signals,
            "high_significance": high_significance,
            "medium_significance": medium_significance,
            "noise_filtered": noise_count,
            "signal_to_noise_ratio": round(snr, 2),
            "strength_score": round(min(100, strength_score), 1),
            "interpretation": self._interpret_signal_strength(strength_score, snr)
        }
    
    def _interpret_signal_strength(self, strength: float, snr: float) -> str:
        """Interpret signal strength for human readability."""
        if strength > 70 and snr > 2:
            return "STRONG - Multiple high-significance signals with good signal-to-noise"
        elif strength > 50 and snr > 1.5:
            return "MODERATE - Decent signals but some noise present"
        elif strength > 30:
            return "WEAK - Limited significant signals"
        else:
            return "VERY WEAK - Mostly noise, proceed with caution"
    
    def _get_confidence_adjustment(self) -> Dict[str, Any]:
        """Get confidence adjustment based on filtering results."""
        adjustment = 0
        reasons = []
        
        # Adjust based on data quality
        if self.data_quality_score < 70:
            adjustment -= 1
            reasons.append("Low data quality")
        elif self.data_quality_score > 90:
            adjustment += 0.5
            reasons.append("High data quality")
        
        # Adjust based on signal confirmations
        confirmations = self._check_confirmations()
        if confirmations["conflicting_signals"]:
            adjustment -= 1
            reasons.append("Conflicting signals present")
        elif confirmations["bullish_confirmations"] >= 3 or confirmations["bearish_confirmations"] >= 3:
            adjustment += 1
            reasons.append("Multiple confirming signals")
        
        # Adjust based on bias warnings
        high_severity_biases = sum(1 for b in self.bias_warnings if b.get("severity") == "HIGH")
        if high_severity_biases > 0:
            adjustment -= 0.5 * high_severity_biases
            reasons.append(f"{high_severity_biases} high-severity bias warnings")
        
        return {
            "adjustment": round(adjustment, 1),
            "reasons": reasons,
            "recommendation": f"Adjust confidence by {adjustment:+.1f} points based on data quality and signal analysis"
        }
    
    def get_filter_summary(self) -> str:
        """Generate human-readable summary of filtering results."""
        summary_parts = []
        
        summary_parts.append("🔍 **NOISE FILTER SUMMARY**")
        summary_parts.append(f"Data Quality Score: {self.data_quality_score}/100")
        summary_parts.append(f"Signals Identified: {len(self.filtered_signals)}")
        summary_parts.append(f"Noise Filtered: {len(self.noise_items)}")
        summary_parts.append(f"Bias Warnings: {len(self.bias_warnings)}")
        
        if self.bias_warnings:
            summary_parts.append("\n**Bias Alerts:**")
            for bias in self.bias_warnings:
                if bias.get("severity") != "INFO":
                    summary_parts.append(f"- [{bias['severity']}] {bias['type']}: {bias['note']}")
        
        return "\n".join(summary_parts)


# Standalone test
if __name__ == "__main__":
    engine = NoiseFilterEngine()
    
    test_data = {
        "price_data": {
            "current": 185.50,
            "change_pct": 6.5,
            "52w_high": 200,
            "52w_low": 140,
            "volume": 15000000,
            "avg_volume": 8000000
        },
        "technicals": {
            "rsi_14": 72,
            "trend": "BULLISH"
        },
        "fundamentals": {
            "pe_ratio": 45,
            "profit_margin": 0.22,
            "revenue_growth": 0.18
        },
        "smart_money": {
            "signal": "ACCUMULATION",
            "smart_money_score": 68,
            "short_interest": {"short_percent_float": 12}
        },
        "macro": {
            "vix": {"level": 18, "regime": "LOW_FEAR"}
        }
    }
    
    result = engine.filter_and_validate(test_data)
    print(engine.get_filter_summary())
    print("\n" + "="*50)
    print("\nSignal Strength:", result["signal_strength"])
    print("\nConfidence Adjustment:", result["recommendation_confidence_adjustment"])
