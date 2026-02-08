"""
TAAPI Cross-Validation Confidence Layer
========================================
Uses TAAPI.io (Pro) as an independent technical indicator source to cross-validate
locally-calculated indicators. When two independent data feeds agree on a signal,
confidence increases. When they disagree, confidence decreases and a warning is flagged.

Think of it like getting a second opinion from a different doctor — if both agree,
you can be much more confident in the diagnosis.

Integrates with:
- advanced_technicals.py (local RSI, MACD, Bollinger, ADX, Stochastic)
- candlestick_patterns.py (local Ichimoku)
- run_perfect_analysis.py (overall confidence scoring)
"""

import os
import sys
import math
from typing import Dict, Any, Optional, List, Tuple

try:
    from taapi_client import TaapiClient
except ImportError:
    TaapiClient = None


class TAAPICrossValidator:
    """
    Cross-validates locally-calculated technical indicators against TAAPI.io's
    independent calculations. Produces a confidence multiplier and discrepancy report.
    """
    
    AGREEMENT_THRESHOLD_RSI = 3.0       # RSI within 3 points = agreement
    AGREEMENT_THRESHOLD_MACD = 0.15     # MACD within 15% = agreement
    AGREEMENT_THRESHOLD_ADX = 3.0       # ADX within 3 points = agreement
    AGREEMENT_THRESHOLD_BB_PCT = 0.10   # Bollinger %B within 10% = agreement
    AGREEMENT_THRESHOLD_STOCH = 5.0     # Stochastic within 5 points = agreement
    
    # Confidence adjustments
    STRONG_AGREEMENT_BONUS = 0.08       # +8% when strongly agree
    AGREEMENT_BONUS = 0.05             # +5% when agree
    MINOR_DISAGREEMENT_PENALTY = -0.03  # -3% when slightly disagree
    MAJOR_DISAGREEMENT_PENALTY = -0.08  # -8% when significantly disagree
    
    def __init__(self, taapi_api_key: str = None):
        """Initialize with TAAPI API key."""
        self.api_key = taapi_api_key or os.environ.get('TAAPI_API_KEY', '')
        self.client = None
        if TaapiClient and self.api_key:
            self.client = TaapiClient(api_key=self.api_key)
    
    def cross_validate(self, symbol: str, local_technicals: Dict[str, Any],
                       interval: str = "1d") -> Dict[str, Any]:
        """
        Cross-validate local technical indicators against TAAPI.io.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            local_technicals: Dict containing locally-calculated indicators.
                Expected keys: rsi, macd, macd_signal, macd_histogram,
                               adx, bb_upper, bb_middle, bb_lower, bb_pct_b,
                               stoch_k, stoch_d, current_price
            interval: Timeframe for TAAPI (default "1d")
        
        Returns:
            Dict with:
                - confidence_multiplier: float (0.85 to 1.15)
                - agreements: list of agreeing indicators
                - disagreements: list of disagreeing indicators with details
                - taapi_values: raw TAAPI values for display
                - cross_validation_score: 0-100 score
                - summary: human-readable summary
        """
        result = {
            "available": False,
            "confidence_multiplier": 1.0,
            "agreements": [],
            "disagreements": [],
            "taapi_values": {},
            "cross_validation_score": 50,
            "indicator_comparisons": [],
            "summary": "TAAPI cross-validation not available"
        }
        
        if not self.client:
            result["summary"] = "TAAPI client not configured (missing API key)"
            return result
        
        # Fetch TAAPI indicators
        taapi_data = self._fetch_taapi_indicators(symbol, interval)
        
        if not taapi_data or taapi_data.get("error"):
            result["summary"] = f"TAAPI fetch failed: {taapi_data.get('error', 'unknown')}"
            return result
        
        result["available"] = True
        result["taapi_values"] = taapi_data
        
        # Compare each indicator
        comparisons = []
        total_adjustment = 0.0
        
        # 1. RSI Cross-Validation
        local_rsi = self._extract_local_rsi(local_technicals)
        taapi_rsi = taapi_data.get("rsi")
        if local_rsi is not None and taapi_rsi is not None:
            comp = self._compare_indicator(
                "RSI", local_rsi, taapi_rsi,
                self.AGREEMENT_THRESHOLD_RSI,
                weight=1.5  # RSI is heavily used, weight it more
            )
            comparisons.append(comp)
            total_adjustment += comp["adjustment"]
        
        # 2. MACD Cross-Validation
        local_macd = self._extract_local_macd(local_technicals)
        taapi_macd = taapi_data.get("macd_value")
        if local_macd is not None and taapi_macd is not None:
            # MACD comparison uses percentage difference since values vary widely
            comp = self._compare_macd(local_macd, taapi_macd,
                                       taapi_data.get("macd_signal"),
                                       taapi_data.get("macd_histogram"))
            comparisons.append(comp)
            total_adjustment += comp["adjustment"]
        
        # 3. ADX Cross-Validation
        local_adx = self._extract_local_adx(local_technicals)
        taapi_adx = taapi_data.get("adx")
        if local_adx is not None and taapi_adx is not None:
            comp = self._compare_indicator(
                "ADX", local_adx, taapi_adx,
                self.AGREEMENT_THRESHOLD_ADX,
                weight=1.0
            )
            comparisons.append(comp)
            total_adjustment += comp["adjustment"]
        
        # 4. Bollinger Band %B Cross-Validation
        local_bb = self._extract_local_bb(local_technicals)
        taapi_bb_upper = taapi_data.get("bb_upper")
        taapi_bb_lower = taapi_data.get("bb_lower")
        taapi_bb_middle = taapi_data.get("bb_middle")
        if local_bb is not None and taapi_bb_upper is not None:
            price = local_technicals.get("current_price", 0)
            if price > 0 and taapi_bb_upper and taapi_bb_lower:
                taapi_bb_pct = (price - taapi_bb_lower) / (taapi_bb_upper - taapi_bb_lower) if (taapi_bb_upper - taapi_bb_lower) > 0 else 0.5
                comp = self._compare_indicator(
                    "Bollinger %B", local_bb, taapi_bb_pct,
                    self.AGREEMENT_THRESHOLD_BB_PCT,
                    weight=0.8
                )
                comparisons.append(comp)
                total_adjustment += comp["adjustment"]
        
        # 5. Stochastic Cross-Validation
        local_stoch_k = self._extract_local_stoch(local_technicals)
        taapi_stoch_k = taapi_data.get("stoch_k")
        if local_stoch_k is not None and taapi_stoch_k is not None:
            comp = self._compare_indicator(
                "Stochastic %K", local_stoch_k, taapi_stoch_k,
                self.AGREEMENT_THRESHOLD_STOCH,
                weight=0.8
            )
            comparisons.append(comp)
            total_adjustment += comp["adjustment"]
        
        # 6. Supertrend direction cross-validation
        taapi_supertrend = taapi_data.get("supertrend_direction")
        local_trend = self._extract_local_trend(local_technicals)
        if taapi_supertrend is not None and local_trend is not None:
            trend_agrees = (taapi_supertrend > 0 and local_trend > 0) or (taapi_supertrend < 0 and local_trend < 0)
            comp = {
                "indicator": "Trend Direction",
                "local_value": "Bullish" if local_trend > 0 else "Bearish",
                "taapi_value": "Bullish" if taapi_supertrend > 0 else "Bearish",
                "agrees": trend_agrees,
                "difference": 0 if trend_agrees else 1,
                "adjustment": self.AGREEMENT_BONUS if trend_agrees else self.MAJOR_DISAGREEMENT_PENALTY,
                "note": "Both sources agree on trend direction" if trend_agrees else "CONFLICT: Sources disagree on trend direction"
            }
            comparisons.append(comp)
            total_adjustment += comp["adjustment"]
        
        # Calculate final results
        result["indicator_comparisons"] = comparisons
        
        agreements = [c for c in comparisons if c["agrees"]]
        disagreements = [c for c in comparisons if not c["agrees"]]
        result["agreements"] = [c["indicator"] for c in agreements]
        result["disagreements"] = [
            {"indicator": c["indicator"], "local": c["local_value"],
             "taapi": c["taapi_value"], "note": c.get("note", "")}
            for c in disagreements
        ]
        
        # Confidence multiplier: clamp between 0.85 and 1.15
        result["confidence_multiplier"] = max(0.85, min(1.15, 1.0 + total_adjustment))
        
        # Cross-validation score: 0-100
        if len(comparisons) > 0:
            agreement_pct = len(agreements) / len(comparisons)
            result["cross_validation_score"] = round(agreement_pct * 100)
        
        # Generate summary
        result["summary"] = self._generate_summary(comparisons, agreements, disagreements, result["confidence_multiplier"])
        
        return result
    
    def _fetch_taapi_indicators(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Fetch all needed indicators from TAAPI.io in a single bulk request if possible."""
        data = {}
        
        try:
            # TAAPI uses exchange:symbol format for stocks
            taapi_symbol = f"NASDAQ:{symbol}" if not ":" in symbol else symbol
            
            # Fetch RSI
            try:
                rsi_result = self.client.get_indicator("rsi", taapi_symbol, interval)
                if rsi_result and "value" in rsi_result:
                    data["rsi"] = round(float(rsi_result["value"]), 2)
            except Exception as e:
                print(f"  TAAPI RSI failed: {e}", file=sys.stderr, flush=True)
            
            # Fetch MACD
            try:
                macd_result = self.client.get_indicator("macd", taapi_symbol, interval)
                if macd_result:
                    data["macd_value"] = round(float(macd_result.get("valueMACD", 0)), 4)
                    data["macd_signal"] = round(float(macd_result.get("valueMACDSignal", 0)), 4)
                    data["macd_histogram"] = round(float(macd_result.get("valueMACDHist", 0)), 4)
            except Exception as e:
                print(f"  TAAPI MACD failed: {e}", file=sys.stderr, flush=True)
            
            # Fetch ADX
            try:
                adx_result = self.client.get_indicator("adx", taapi_symbol, interval)
                if adx_result and "value" in adx_result:
                    data["adx"] = round(float(adx_result["value"]), 2)
            except Exception as e:
                print(f"  TAAPI ADX failed: {e}", file=sys.stderr, flush=True)
            
            # Fetch Bollinger Bands
            try:
                bb_result = self.client.get_indicator("bbands", taapi_symbol, interval)
                if bb_result:
                    data["bb_upper"] = round(float(bb_result.get("valueUpperBand", 0)), 2)
                    data["bb_middle"] = round(float(bb_result.get("valueMiddleBand", 0)), 2)
                    data["bb_lower"] = round(float(bb_result.get("valueLowerBand", 0)), 2)
            except Exception as e:
                print(f"  TAAPI BBands failed: {e}", file=sys.stderr, flush=True)
            
            # Fetch Stochastic
            try:
                stoch_result = self.client.get_indicator("stoch", taapi_symbol, interval)
                if stoch_result:
                    data["stoch_k"] = round(float(stoch_result.get("valueK", 0)), 2)
                    data["stoch_d"] = round(float(stoch_result.get("valueD", 0)), 2)
            except Exception as e:
                print(f"  TAAPI Stochastic failed: {e}", file=sys.stderr, flush=True)
            
            # Fetch Supertrend
            try:
                st_result = self.client.get_indicator("supertrend", taapi_symbol, interval)
                if st_result:
                    data["supertrend_value"] = round(float(st_result.get("value", 0)), 2)
                    data["supertrend_direction"] = 1 if str(st_result.get("valueAdvice", "")).lower() == "long" else -1
            except Exception as e:
                print(f"  TAAPI Supertrend failed: {e}", file=sys.stderr, flush=True)
            
            return data
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compare_indicator(self, name: str, local_val: float, taapi_val: float,
                           threshold: float, weight: float = 1.0) -> Dict[str, Any]:
        """Compare a single indicator between local and TAAPI values."""
        diff = abs(local_val - taapi_val)
        agrees = diff <= threshold
        
        # Determine adjustment based on agreement level
        if agrees and diff <= threshold * 0.5:
            # Strong agreement
            adjustment = self.STRONG_AGREEMENT_BONUS * weight
            note = f"Strong agreement (diff: {diff:.2f})"
        elif agrees:
            # Normal agreement
            adjustment = self.AGREEMENT_BONUS * weight
            note = f"Agreement within threshold (diff: {diff:.2f})"
        elif diff <= threshold * 2:
            # Minor disagreement
            adjustment = self.MINOR_DISAGREEMENT_PENALTY * weight
            note = f"Minor disagreement (diff: {diff:.2f}, threshold: {threshold})"
        else:
            # Major disagreement
            adjustment = self.MAJOR_DISAGREEMENT_PENALTY * weight
            note = f"SIGNIFICANT DISAGREEMENT (diff: {diff:.2f}, threshold: {threshold})"
        
        return {
            "indicator": name,
            "local_value": round(local_val, 4),
            "taapi_value": round(taapi_val, 4),
            "difference": round(diff, 4),
            "threshold": threshold,
            "agrees": agrees,
            "adjustment": round(adjustment, 4),
            "note": note
        }
    
    def _compare_macd(self, local_macd: float, taapi_macd: float,
                      taapi_signal: Optional[float], taapi_hist: Optional[float]) -> Dict[str, Any]:
        """Compare MACD with special handling for sign agreement."""
        # For MACD, the most important thing is whether both agree on direction
        same_sign = (local_macd > 0 and taapi_macd > 0) or (local_macd < 0 and taapi_macd < 0) or (local_macd == 0 and taapi_macd == 0)
        
        # Also check percentage difference for magnitude
        if abs(taapi_macd) > 0.001:
            pct_diff = abs(local_macd - taapi_macd) / abs(taapi_macd)
        else:
            pct_diff = abs(local_macd - taapi_macd)
        
        agrees = same_sign and pct_diff < 0.30  # Within 30% and same direction
        
        if agrees and pct_diff < 0.15:
            adjustment = self.STRONG_AGREEMENT_BONUS
            note = f"Strong MACD agreement (same direction, {pct_diff:.1%} diff)"
        elif agrees:
            adjustment = self.AGREEMENT_BONUS
            note = f"MACD direction agrees ({pct_diff:.1%} magnitude diff)"
        elif same_sign:
            adjustment = self.MINOR_DISAGREEMENT_PENALTY
            note = f"MACD direction agrees but magnitude differs ({pct_diff:.1%})"
        else:
            adjustment = self.MAJOR_DISAGREEMENT_PENALTY
            note = f"MACD DIRECTION CONFLICT: Local={'bullish' if local_macd > 0 else 'bearish'}, TAAPI={'bullish' if taapi_macd > 0 else 'bearish'}"
        
        return {
            "indicator": "MACD",
            "local_value": round(local_macd, 4),
            "taapi_value": round(taapi_macd, 4),
            "difference": round(pct_diff, 4),
            "agrees": agrees,
            "adjustment": round(adjustment, 4),
            "note": note,
            "extra": {
                "taapi_signal": taapi_signal,
                "taapi_histogram": taapi_hist,
                "same_direction": same_sign
            }
        }
    
    def _extract_local_rsi(self, technicals: Dict) -> Optional[float]:
        """Extract RSI from various possible locations in the local technicals dict."""
        # Try direct field
        rsi = technicals.get("rsi")
        if rsi is not None:
            return float(rsi)
        
        # Try nested in momentum
        momentum = technicals.get("momentum", {})
        if isinstance(momentum, dict):
            rsi = momentum.get("rsi")
            if rsi is not None:
                return float(rsi)
        
        # Try nested in indicators
        indicators = technicals.get("indicators", {})
        if isinstance(indicators, dict):
            rsi = indicators.get("rsi")
            if rsi is not None:
                return float(rsi)
        
        return None
    
    def _extract_local_macd(self, technicals: Dict) -> Optional[float]:
        """Extract MACD value from local technicals."""
        macd = technicals.get("macd") or technicals.get("macd_line")
        if macd is not None:
            return float(macd)
        
        momentum = technicals.get("momentum", {})
        if isinstance(momentum, dict):
            macd = momentum.get("macd") or momentum.get("macd_line")
            if macd is not None:
                return float(macd)
        
        indicators = technicals.get("indicators", {})
        if isinstance(indicators, dict):
            macd = indicators.get("macd") or indicators.get("macd_line")
            if macd is not None:
                return float(macd)
        
        return None
    
    def _extract_local_adx(self, technicals: Dict) -> Optional[float]:
        """Extract ADX from local technicals."""
        adx = technicals.get("adx")
        if adx is not None:
            return float(adx)
        
        trend = technicals.get("trend", {})
        if isinstance(trend, dict):
            adx = trend.get("adx")
            if adx is not None:
                return float(adx)
        
        return None
    
    def _extract_local_bb(self, technicals: Dict) -> Optional[float]:
        """Extract Bollinger Band %B from local technicals."""
        bb_pct = technicals.get("bb_pct_b") or technicals.get("bollinger_pct_b")
        if bb_pct is not None:
            return float(bb_pct)
        
        volatility = technicals.get("volatility", {})
        if isinstance(volatility, dict):
            bb_pct = volatility.get("bb_pct_b") or volatility.get("bollinger_pct_b")
            if bb_pct is not None:
                return float(bb_pct)
        
        return None
    
    def _extract_local_stoch(self, technicals: Dict) -> Optional[float]:
        """Extract Stochastic %K from local technicals."""
        stoch = technicals.get("stoch_k") or technicals.get("stochastic_k")
        if stoch is not None:
            return float(stoch)
        
        momentum = technicals.get("momentum", {})
        if isinstance(momentum, dict):
            stoch = momentum.get("stoch_k") or momentum.get("stochastic_k")
            if stoch is not None:
                return float(stoch)
        
        return None
    
    def _extract_local_trend(self, technicals: Dict) -> Optional[int]:
        """Extract overall trend direction from local technicals."""
        # Try direct signal
        signal = technicals.get("signal") or technicals.get("overall_signal")
        if signal is not None:
            if isinstance(signal, str):
                signal_lower = signal.lower()
                if "buy" in signal_lower or "bullish" in signal_lower:
                    return 1
                elif "sell" in signal_lower or "bearish" in signal_lower:
                    return -1
                return 0
            return 1 if signal > 0 else (-1 if signal < 0 else 0)
        
        # Try from SMA crossover
        sma_20 = technicals.get("sma_20") or technicals.get("sma20")
        sma_50 = technicals.get("sma_50") or technicals.get("sma50")
        price = technicals.get("current_price")
        if price and sma_50:
            return 1 if price > sma_50 else -1
        
        return None
    
    def _generate_summary(self, comparisons: List, agreements: List,
                          disagreements: List, multiplier: float) -> str:
        """Generate a human-readable cross-validation summary."""
        total = len(comparisons)
        agreed = len(agreements)
        
        if total == 0:
            return "No indicators available for cross-validation"
        
        pct = agreed / total * 100
        
        # Build summary
        parts = []
        parts.append(f"Cross-validated {total} indicators against TAAPI.io: {agreed}/{total} agree ({pct:.0f}%)")
        
        if pct >= 80:
            parts.append("HIGH CONFIDENCE: Strong agreement between independent data sources.")
        elif pct >= 60:
            parts.append("MODERATE CONFIDENCE: Most indicators agree, some minor discrepancies.")
        elif pct >= 40:
            parts.append("MIXED SIGNALS: Significant disagreements between data sources — exercise caution.")
        else:
            parts.append("LOW CONFIDENCE: Major disagreements detected — data sources conflict significantly.")
        
        if multiplier > 1.05:
            parts.append(f"Confidence boost: +{(multiplier - 1) * 100:.1f}% (signals are reinforced)")
        elif multiplier < 0.95:
            parts.append(f"Confidence penalty: {(multiplier - 1) * 100:.1f}% (signals are weakened)")
        
        # Flag specific conflicts
        for d in disagreements:
            if not isinstance(d, dict):
                continue
            if "SIGNIFICANT" in str(d.get("note", "")) or "CONFLICT" in str(d.get("note", "")):
                parts.append(f"  ⚠ {d.get('indicator', 'Unknown')}: {d.get('note', '')}")
        
        return " | ".join(parts)
