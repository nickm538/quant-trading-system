"""
ADVANCED AI PATTERN RECOGNITION MODULE v2.0
=============================================

Institutional-grade historical pattern matching engine.
Uses multi-timeframe DTW, volume-weighted similarity, recency weighting,
and comprehensive historical regime database (1987-2025+).

Techniques:
- Dynamic Time Warping (DTW) for price shape similarity
- Volume-weighted pattern matching (not just price)
- Multi-timeframe analysis (15, 30, 60, 90 day windows)
- Recency-weighted scoring (recent patterns count more)
- Cross-market regime identification (SPY, VIX, sector rotation)
- Comprehensive historical regime database (14+ regimes)
- Forward outcome analysis with probability distributions

This is what separates institutional-grade systems from basic TA.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from fastdtw import fastdtw
import logging

logger = logging.getLogger(__name__)


class PatternRecognitionEngine:
    """
    Advanced pattern recognition using AI/ML techniques.
    Identifies similar historical patterns and predicts outcomes.
    Multi-timeframe, volume-aware, recency-weighted.
    """
    
    # Comprehensive historical market regimes with known outcomes
    HISTORICAL_REGIMES = {
        'black_monday_1987': {
            'period': '1987-10',
            'description': 'Black Monday - single-day 22% crash, program trading cascade',
            'outcome': 'Recovered fully within 2 years, +65% from bottom',
            'characteristics': {
                'pe_ratio': '18-22',
                'sentiment': 'panic',
                'volatility': 'extreme',
                'trend': 'vertical_drop',
                'vix_range': '>50'
            }
        },
        'dot_com_bubble_peak': {
            'period': '1999-12 to 2000-03',
            'description': 'Dot-com bubble peak - extreme valuations, euphoria, IPO mania',
            'outcome': 'Crashed -78% over 2.5 years (NASDAQ)',
            'characteristics': {
                'pe_ratio': '>80',
                'sentiment': 'extreme_greed',
                'volatility': 'rising',
                'trend': 'parabolic',
                'vix_range': '20-30'
            }
        },
        'dot_com_crash': {
            'period': '2000-03 to 2002-10',
            'description': 'Dot-com crash - capitulation, panic selling, accounting scandals',
            'outcome': 'Bottom formed, +400% recovery over 5 years',
            'characteristics': {
                'pe_ratio': '<15',
                'sentiment': 'extreme_fear',
                'volatility': 'extreme',
                'trend': 'downtrend',
                'vix_range': '>35'
            }
        },
        'housing_bubble_2006': {
            'period': '2005-06 to 2007-10',
            'description': 'Housing bubble peak - subprime lending, CDO mania, leverage excess',
            'outcome': 'S&P peaked Oct 2007, then -57% crash over 17 months',
            'characteristics': {
                'pe_ratio': '16-20',
                'sentiment': 'complacency',
                'volatility': 'low',
                'trend': 'steady_uptrend',
                'vix_range': '10-15'
            }
        },
        'financial_crisis_2008': {
            'period': '2008-09 to 2009-03',
            'description': 'Financial crisis - systemic risk, credit freeze, Lehman collapse',
            'outcome': 'Bottom formed Mar 2009, +350% recovery over 10 years',
            'characteristics': {
                'pe_ratio': '<12',
                'sentiment': 'panic',
                'volatility': 'extreme',
                'trend': 'freefall',
                'vix_range': '>60'
            }
        },
        'flash_crash_2010': {
            'period': '2010-05',
            'description': 'Flash crash - algorithmic trading cascade, 1000-point Dow drop in minutes',
            'outcome': 'Recovered within days, continued bull market',
            'characteristics': {
                'pe_ratio': '14-16',
                'sentiment': 'fear',
                'volatility': 'spike',
                'trend': 'v_recovery',
                'vix_range': '25-40'
            }
        },
        'eu_debt_crisis_2011': {
            'period': '2011-07 to 2011-10',
            'description': 'European debt crisis - Greek default fears, US downgrade',
            'outcome': '-19% correction, then strong recovery +30% in 12 months',
            'characteristics': {
                'pe_ratio': '12-15',
                'sentiment': 'fear',
                'volatility': 'high',
                'trend': 'correction',
                'vix_range': '30-45'
            }
        },
        'taper_tantrum_2013': {
            'period': '2013-05 to 2013-09',
            'description': 'Taper tantrum - Fed signals QE reduction, bond market selloff',
            'outcome': '-7% pullback, then resumed bull market +30% next year',
            'characteristics': {
                'pe_ratio': '15-18',
                'sentiment': 'anxiety',
                'volatility': 'moderate',
                'trend': 'pullback',
                'vix_range': '15-22'
            }
        },
        'bull_market_2017': {
            'period': '2016-11 to 2018-01',
            'description': 'Trump rally - tax cuts, deregulation optimism, low volatility',
            'outcome': '+40% gain, then -20% correction in Q4 2018',
            'characteristics': {
                'pe_ratio': '20-25',
                'sentiment': 'optimism',
                'volatility': 'low',
                'trend': 'steady_uptrend',
                'vix_range': '9-14'
            }
        },
        'volmageddon_2018': {
            'period': '2018-02',
            'description': 'Volmageddon - VIX spike, XIV collapse, algorithmic selling',
            'outcome': '-12% correction, recovered in 2 months, then -20% in Q4',
            'characteristics': {
                'pe_ratio': '22-26',
                'sentiment': 'fear',
                'volatility': 'spike',
                'trend': 'sharp_correction',
                'vix_range': '25-50'
            }
        },
        'fed_pivot_2018': {
            'period': '2018-10 to 2018-12',
            'description': 'Q4 2018 selloff - Fed tightening, trade war fears, growth scare',
            'outcome': '-20% bear market, then V-recovery after Fed pivot +30% in 2019',
            'characteristics': {
                'pe_ratio': '14-18',
                'sentiment': 'fear',
                'volatility': 'high',
                'trend': 'downtrend',
                'vix_range': '20-36'
            }
        },
        'covid_crash_2020': {
            'period': '2020-02 to 2020-03',
            'description': 'COVID crash - fastest bear market in history, global lockdowns',
            'outcome': 'V-shaped recovery, +100% in 18 months, fueled by stimulus',
            'characteristics': {
                'pe_ratio': '15-20',
                'sentiment': 'panic',
                'volatility': 'extreme',
                'trend': 'vertical_drop',
                'vix_range': '>65'
            }
        },
        'meme_stock_mania_2021': {
            'period': '2021-01 to 2021-06',
            'description': 'Meme stock mania - GME/AMC, retail frenzy, short squeeze cascade',
            'outcome': 'Speculative stocks crashed -60-90%, broad market continued up',
            'characteristics': {
                'pe_ratio': '>30',
                'sentiment': 'extreme_greed',
                'volatility': 'high',
                'trend': 'speculative_mania',
                'vix_range': '20-35'
            }
        },
        'inflation_scare_2022': {
            'period': '2022-01 to 2022-10',
            'description': 'Fed rate hikes - inflation fight, 40-year high CPI, recession fears',
            'outcome': '-25% drawdown, then recovery starting Oct 2022',
            'characteristics': {
                'pe_ratio': '15-18',
                'sentiment': 'fear',
                'volatility': 'high',
                'trend': 'downtrend',
                'vix_range': '20-35'
            }
        },
        'ai_rally_2023': {
            'period': '2023-01 to 2024-03',
            'description': 'AI rally - ChatGPT/NVIDIA boom, Magnificent 7 concentration',
            'outcome': 'S&P +45% from Oct 2022 lows, extreme concentration in tech',
            'characteristics': {
                'pe_ratio': '22-28',
                'sentiment': 'optimism',
                'volatility': 'moderate',
                'trend': 'strong_uptrend',
                'vix_range': '12-18'
            }
        },
        'tariff_volatility_2025': {
            'period': '2025-01 to present',
            'description': 'Tariff uncertainty - trade war escalation, policy whiplash, global supply chain disruption',
            'outcome': 'Ongoing - elevated volatility, sector rotation, defensive positioning',
            'characteristics': {
                'pe_ratio': '20-24',
                'sentiment': 'anxiety',
                'volatility': 'elevated',
                'trend': 'choppy',
                'vix_range': '18-30'
            }
        }
    }
    
    # Multiple analysis windows for multi-timeframe matching
    ANALYSIS_WINDOWS = [15, 30, 60, 90]
    
    def __init__(self):
        """Initialize pattern recognition engine"""
        logger.info("Pattern Recognition Engine v2.0 initialized")
    
    def analyze_patterns(self, price_data: pd.DataFrame, current_analysis: Dict) -> Dict:
        """
        MAIN METHOD: Analyze current pattern against ALL historical data.
        ALWAYS RUNS - not optional.
        
        Multi-timeframe analysis with volume weighting and recency bias.
        
        Args:
            price_data: DataFrame with OHLCV data
            current_analysis: Current stock analysis dict
            
        Returns:
            Dict with pattern matches, regime identification, and predictions
        """
        logger.info("=== PATTERN RECOGNITION v2.0 ACTIVE ===")
        
        # Multi-timeframe pattern matching
        all_patterns = []
        timeframe_results = {}
        
        for window in self.ANALYSIS_WINDOWS:
            patterns = self._find_similar_patterns(price_data, window=window)
            timeframe_results[f'{window}d'] = {
                'matches': len(patterns),
                'top_match_score': patterns[0]['similarity_score'] if patterns else None
            }
            all_patterns.extend(patterns)
        
        # Deduplicate overlapping patterns (keep best score for each period)
        all_patterns = self._deduplicate_patterns(all_patterns)
        
        # Apply recency weighting (recent patterns weighted 2x)
        all_patterns = self._apply_recency_weights(all_patterns, len(price_data))
        
        # Sort by weighted score
        all_patterns.sort(key=lambda x: x.get('weighted_score', x['similarity_score']))
        
        # Take top 15 matches
        top_patterns = all_patterns[:15]
        
        results = {
            'similar_patterns': top_patterns,
            'timeframe_analysis': timeframe_results,
            'regime_match': self._identify_regime(current_analysis),
            'historical_analogy': self._find_best_analogy(price_data, current_analysis),
            'pattern_prediction': None,
            'confidence': 0.0,
            'analysis_metadata': {
                'windows_analyzed': self.ANALYSIS_WINDOWS,
                'total_patterns_found': len(all_patterns),
                'data_points_analyzed': len(price_data),
                'volume_weighted': 'Volume' in price_data.columns,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Generate prediction based on similar patterns
        if results['similar_patterns']:
            results['pattern_prediction'] = self._predict_outcome(results['similar_patterns'])
            results['confidence'] = self._calculate_pattern_confidence(results['similar_patterns'])
        
        logger.info(f"Pattern recognition v2.0 complete: {len(results['similar_patterns'])} matches found across {len(self.ANALYSIS_WINDOWS)} timeframes")
        
        return results
    
    def _find_similar_patterns(self, price_data: pd.DataFrame, window: int = 30) -> List[Dict]:
        """
        Find similar price patterns using DTW with volume weighting.
        Multi-timeframe: compares last N days to ALL historical periods.
        """
        if len(price_data) < window * 2:
            logger.warning(f"Not enough data for {window}-day pattern matching: {len(price_data)} < {window * 2}")
            return []
        
        close = price_data['Close'].values
        has_volume = 'Volume' in price_data.columns
        volume = price_data['Volume'].values if has_volume else None
        
        # Normalize current pattern (last N days)
        current_pattern = close[-window:]
        current_std = current_pattern.std()
        if current_std < 1e-8:
            return []  # Flat price, no pattern to match
        current_normalized = (current_pattern - current_pattern.mean()) / current_std
        
        # Normalize current volume pattern if available
        current_vol_normalized = None
        if has_volume and volume is not None:
            current_vol = volume[-window:]
            vol_std = current_vol.std()
            if vol_std > 1e-8:
                current_vol_normalized = (current_vol - current_vol.mean()) / vol_std
        
        similar_patterns = []
        
        # Adaptive step size: smaller for shorter windows, larger for longer
        step = max(1, window // 10)
        
        # Slide window across historical data
        for i in range(len(close) - window * 2, 0, -step):
            if i < window:
                break
            
            # Get historical pattern
            hist_pattern = close[i:i+window]
            hist_std = hist_pattern.std()
            if hist_std < 1e-8:
                continue  # Skip flat periods
            hist_normalized = (hist_pattern - hist_pattern.mean()) / hist_std
            
            try:
                # Calculate DTW distance
                current_1d = current_normalized.reshape(-1)
                hist_1d = hist_normalized.reshape(-1)
                distance, _ = fastdtw(current_1d, hist_1d, dist=lambda x, y: np.abs(x - y))
                
                # Euclidean distance for shape comparison
                euclidean_dist = np.sqrt(np.sum((current_normalized - hist_normalized) ** 2))
                
                # Volume similarity bonus (if available)
                volume_bonus = 0.0
                if current_vol_normalized is not None and has_volume:
                    hist_vol = volume[i:i+window]
                    hist_vol_std = hist_vol.std()
                    if hist_vol_std > 1e-8:
                        hist_vol_normalized = (hist_vol - hist_vol.mean()) / hist_vol_std
                        # Correlation between volume patterns
                        vol_corr = np.corrcoef(current_vol_normalized, hist_vol_normalized)[0, 1]
                        if not np.isnan(vol_corr):
                            # Higher correlation = lower penalty (bonus up to -3.0)
                            volume_bonus = -3.0 * max(0, vol_corr)
                
                # Combined similarity score (lower is better)
                # Weight DTW more heavily as it handles time warping
                similarity_score = (distance * 0.6 + euclidean_dist * 0.4) + volume_bonus
                
                # Adaptive threshold based on window size
                threshold = 10.0 + (window / 15.0)  # Larger windows get more lenient threshold
                
                if similarity_score < threshold:
                    # Get outcome (what happened next) - multiple forward windows
                    outcomes = {}
                    for fwd_days in [5, 10, 20, 30, 60]:
                        future_end = min(i + window + fwd_days, len(close))
                        if future_end > i + window:
                            future_prices = close[i+window:future_end]
                            if len(future_prices) > 0:
                                outcome_return = (future_prices[-1] - hist_pattern[-1]) / hist_pattern[-1]
                                max_drawdown = (np.min(future_prices) - hist_pattern[-1]) / hist_pattern[-1]
                                max_gain = (np.max(future_prices) - hist_pattern[-1]) / hist_pattern[-1]
                                outcomes[f'{fwd_days}d'] = {
                                    'return': float(outcome_return),
                                    'max_drawdown': float(max_drawdown),
                                    'max_gain': float(max_gain),
                                    'days': len(future_prices)
                                }
                    
                    if outcomes:
                        # Primary outcome: 30-day forward (or longest available)
                        primary_key = '30d' if '30d' in outcomes else list(outcomes.keys())[-1]
                        primary_outcome = outcomes[primary_key]
                        
                        similar_patterns.append({
                            'start_index': i,
                            'window': window,
                            'similarity_score': float(similarity_score),
                            'dtw_distance': float(distance),
                            'euclidean_distance': float(euclidean_dist),
                            'volume_bonus': float(volume_bonus),
                            'outcome_return': float(primary_outcome['return']),
                            'outcome_days': primary_outcome['days'],
                            'multi_horizon_outcomes': outcomes,
                            'pattern_start_price': float(hist_pattern[0]),
                            'pattern_end_price': float(hist_pattern[-1]),
                            'pattern_return': float((hist_pattern[-1] - hist_pattern[0]) / hist_pattern[0]),
                            'recency_index': i  # For recency weighting later
                        })
            except Exception as e:
                logger.warning(f"DTW calculation failed at i={i}, window={window}: {str(e)}")
                continue
        
        # Sort by similarity (best matches first)
        similar_patterns.sort(key=lambda x: x['similarity_score'])
        
        # Return top 10 per timeframe
        return similar_patterns[:10]
    
    def _deduplicate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Remove overlapping patterns, keeping the best score for each period."""
        if not patterns:
            return []
        
        # Sort by score
        patterns.sort(key=lambda x: x['similarity_score'])
        
        deduplicated = []
        used_indices = set()
        
        for p in patterns:
            idx = p['start_index']
            window = p.get('window', 30)
            
            # Check if this overlaps with any already-selected pattern
            overlap = False
            for used_idx in used_indices:
                if abs(idx - used_idx) < window // 2:
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(p)
                used_indices.add(idx)
        
        return deduplicated
    
    def _apply_recency_weights(self, patterns: List[Dict], total_data_points: int) -> List[Dict]:
        """
        Apply recency weighting: more recent historical patterns count more.
        Rationale: market microstructure and participant behavior evolve over time.
        Recent patterns are more predictive of future behavior.
        """
        if not patterns or total_data_points == 0:
            return patterns
        
        for p in patterns:
            recency_idx = p.get('recency_index', 0)
            # Recency factor: 1.0 for most recent, decaying to 0.5 for oldest
            recency_factor = 0.5 + 0.5 * (recency_idx / total_data_points)
            
            # Weighted score: lower is better, so multiply by inverse of recency
            # More recent patterns get lower (better) weighted scores
            p['recency_factor'] = float(recency_factor)
            p['weighted_score'] = p['similarity_score'] / recency_factor
        
        return patterns
    
    def _identify_regime(self, analysis: Dict) -> Optional[Dict]:
        """
        Identify which historical market regime current conditions most resemble.
        Enhanced with VIX awareness and more nuanced matching.
        """
        fundamentals = analysis.get('fundamentals', {})
        technical = analysis.get('technical_analysis', {})
        sentiment = analysis.get('sentiment_score', 50)
        
        pe_ratio = fundamentals.get('pe_ratio') or fundamentals.get('forward_pe') or 20
        # Handle None/NaN pe_ratio
        try:
            pe_ratio = float(pe_ratio)
            if np.isnan(pe_ratio) or np.isinf(pe_ratio):
                pe_ratio = 20
        except (TypeError, ValueError):
            pe_ratio = 20
            
        volatility = technical.get('current_volatility', 0.2)
        trend_score = technical.get('trend_score', 50)
        vix = technical.get('vix', 20)
        adx = technical.get('adx', 25)
        
        # Score each regime based on multi-factor similarity
        regime_scores = {}
        
        for regime_name, regime_data in self.HISTORICAL_REGIMES.items():
            score = 0.0
            max_possible = 0.0
            characteristics = regime_data['characteristics']
            
            # PE ratio matching (25 points max)
            max_possible += 25
            pe_char = characteristics.get('pe_ratio', '')
            if '>80' in pe_char:
                if pe_ratio > 60:
                    score += 25
                elif pe_ratio > 40:
                    score += 15
            elif '>30' in pe_char:
                if pe_ratio > 30:
                    score += 25
                elif pe_ratio > 25:
                    score += 15
            elif '<12' in pe_char:
                if pe_ratio < 12:
                    score += 25
                elif pe_ratio < 15:
                    score += 15
            elif '<15' in pe_char:
                if pe_ratio < 15:
                    score += 20
                elif pe_ratio < 18:
                    score += 10
            elif '-' in pe_char:
                try:
                    parts = pe_char.split('-')
                    low = float(parts[0])
                    high = float(parts[1])
                    if low <= pe_ratio <= high:
                        score += 25
                    elif abs(pe_ratio - (low + high) / 2) < 5:
                        score += 12
                except (ValueError, IndexError):
                    pass
            
            # Sentiment matching (25 points max)
            max_possible += 25
            sentiment_char = characteristics.get('sentiment', '')
            if 'extreme_greed' in sentiment_char and sentiment > 80:
                score += 25
            elif 'extreme_greed' in sentiment_char and sentiment > 65:
                score += 12
            elif 'extreme_fear' in sentiment_char and sentiment < 20:
                score += 25
            elif 'panic' in sentiment_char and sentiment < 15:
                score += 25
            elif 'panic' in sentiment_char and sentiment < 25:
                score += 15
            elif 'fear' in sentiment_char and 15 <= sentiment < 35:
                score += 22
            elif 'fear' in sentiment_char and 35 <= sentiment < 45:
                score += 10
            elif 'anxiety' in sentiment_char and 30 <= sentiment < 50:
                score += 20
            elif 'complacency' in sentiment_char and 55 <= sentiment < 70:
                score += 20
            elif 'optimism' in sentiment_char and 60 <= sentiment < 80:
                score += 22
            
            # Volatility matching (25 points max)
            max_possible += 25
            vol_char = characteristics.get('volatility', '')
            if 'extreme' in vol_char and volatility > 0.5:
                score += 25
            elif 'extreme' in vol_char and volatility > 0.35:
                score += 15
            elif 'high' in vol_char and 0.25 <= volatility <= 0.5:
                score += 22
            elif 'elevated' in vol_char and 0.20 <= volatility <= 0.35:
                score += 22
            elif 'spike' in vol_char and volatility > 0.30:
                score += 20
            elif 'moderate' in vol_char and 0.15 <= volatility <= 0.25:
                score += 22
            elif 'rising' in vol_char and volatility > 0.20:
                score += 18
            elif 'low' in vol_char and volatility < 0.18:
                score += 22
            
            # Trend matching (25 points max)
            max_possible += 25
            trend_char = characteristics.get('trend', '')
            if 'parabolic' in trend_char and trend_score > 85:
                score += 25
            elif 'speculative_mania' in trend_char and trend_score > 80:
                score += 22
            elif 'strong_uptrend' in trend_char and 70 <= trend_score <= 90:
                score += 22
            elif 'steady_uptrend' in trend_char and 60 <= trend_score <= 80:
                score += 22
            elif 'choppy' in trend_char and 40 <= trend_score <= 60:
                score += 22
            elif 'pullback' in trend_char and 35 <= trend_score <= 50:
                score += 20
            elif 'correction' in trend_char and 30 <= trend_score <= 45:
                score += 20
            elif 'sharp_correction' in trend_char and 25 <= trend_score <= 40:
                score += 22
            elif 'downtrend' in trend_char and trend_score < 40:
                score += 22
            elif 'freefall' in trend_char and trend_score < 20:
                score += 25
            elif 'vertical_drop' in trend_char and trend_score < 15:
                score += 25
            elif 'v_recovery' in trend_char and trend_score > 55 and volatility > 0.25:
                score += 20
            
            # Normalize to percentage
            if max_possible > 0:
                regime_scores[regime_name] = (score / max_possible) * 100
            else:
                regime_scores[regime_name] = 0
        
        if not regime_scores:
            return None
        
        # Sort regimes by score
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        best_regime_name, best_score = sorted_regimes[0]
        
        # Only return if score is meaningful (>35 out of 100)
        if best_score < 35:
            return {
                'regime_name': 'unique_conditions',
                'description': 'Current conditions do not strongly match any major historical regime',
                'match_score': best_score,
                'closest_regime': best_regime_name,
                'closest_score': best_score,
                'top_3_regimes': [
                    {'name': name, 'score': score} 
                    for name, score in sorted_regimes[:3]
                ]
            }
        
        regime_info = self.HISTORICAL_REGIMES[best_regime_name].copy()
        regime_info['match_score'] = best_score
        regime_info['regime_name'] = best_regime_name
        regime_info['top_3_regimes'] = [
            {'name': name, 'score': score} 
            for name, score in sorted_regimes[:3]
        ]
        
        return regime_info
    
    def _find_best_analogy(self, price_data: pd.DataFrame, analysis: Dict) -> Optional[str]:
        """
        Generate human-readable historical analogy.
        "This setup is like [Historical Event] because [Reasons]"
        """
        regime = self._identify_regime(analysis)
        
        if not regime:
            return "Current market conditions don't strongly match any major historical regime. This could be a unique setup or transitional period."
        
        regime_name = regime.get('regime_name', 'unknown')
        
        if regime_name == 'unique_conditions':
            closest = regime.get('closest_regime', 'unknown')
            closest_score = regime.get('closest_score', 0)
            top_3 = regime.get('top_3_regimes', [])
            
            analogy = "**Current conditions are relatively unique** - no strong historical match.\n\n"
            analogy += f"**Closest Comparison**: {closest.replace('_', ' ').title()} ({closest_score:.0f}% match)\n\n"
            if top_3:
                analogy += "**Top 3 Historical Comparisons**:\n"
                for r in top_3:
                    regime_data = self.HISTORICAL_REGIMES.get(r['name'], {})
                    analogy += f"- {r['name'].replace('_', ' ').title()} ({r['score']:.0f}% match): {regime_data.get('outcome', 'N/A')}\n"
            return analogy
        
        period = regime.get('period', 'N/A')
        description = regime.get('description', 'N/A')
        outcome = regime.get('outcome', 'N/A')
        match_score = regime.get('match_score', 0)
        top_3 = regime.get('top_3_regimes', [])
        
        # Format analogy
        analogy = f"**This resembles {regime_name.replace('_', ' ').title()} ({period})**\n\n"
        analogy += f"**Historical Context**: {description}\n\n"
        analogy += f"**What Happened Next**: {outcome}\n\n"
        analogy += f"**Match Confidence**: {match_score:.0f}% similarity to historical conditions\n\n"
        
        # Add runner-up comparisons
        if len(top_3) > 1:
            analogy += "**Alternative Historical Comparisons**:\n"
            for r in top_3[1:]:
                alt_data = self.HISTORICAL_REGIMES.get(r['name'], {})
                analogy += f"- {r['name'].replace('_', ' ').title()} ({r['score']:.0f}% match): {alt_data.get('outcome', 'N/A')}\n"
            analogy += "\n"
        
        # Add key characteristics
        characteristics = regime.get('characteristics', {})
        if characteristics:
            analogy += "**Key Similarities**:\n"
            for key, value in characteristics.items():
                analogy += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return analogy
    
    def _predict_outcome(self, similar_patterns: List[Dict]) -> Dict:
        """
        Predict likely outcome based on similar historical patterns.
        Multi-horizon predictions with probability distributions.
        """
        if not similar_patterns:
            return {
                'expected_return': 0.0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'avg_days_to_outcome': 30,
                'horizons': {}
            }
        
        # Primary outcome statistics
        outcomes = [p['outcome_return'] for p in similar_patterns]
        
        expected_return = np.mean(outcomes)
        probability_up = sum(1 for r in outcomes if r > 0) / len(outcomes)
        probability_down = 1 - probability_up
        
        # Weight by similarity AND recency (better + more recent matches count more)
        weights = []
        for p in similar_patterns:
            sim_weight = 1 / (p['similarity_score'] + 1)
            recency_weight = p.get('recency_factor', 1.0)
            weights.append(sim_weight * recency_weight)
        
        weighted_return = np.average(outcomes, weights=weights)
        
        # Multi-horizon analysis
        horizons = {}
        for horizon in ['5d', '10d', '20d', '30d', '60d']:
            horizon_returns = []
            horizon_drawdowns = []
            horizon_gains = []
            
            for p in similar_patterns:
                multi = p.get('multi_horizon_outcomes', {})
                if horizon in multi:
                    horizon_returns.append(multi[horizon]['return'])
                    horizon_drawdowns.append(multi[horizon]['max_drawdown'])
                    horizon_gains.append(multi[horizon]['max_gain'])
            
            if horizon_returns:
                horizons[horizon] = {
                    'expected_return': float(np.mean(horizon_returns)),
                    'weighted_return': float(np.average(horizon_returns, weights=weights[:len(horizon_returns)])) if len(horizon_returns) == len(weights) else float(np.mean(horizon_returns)),
                    'probability_up': float(sum(1 for r in horizon_returns if r > 0) / len(horizon_returns)),
                    'median_return': float(np.median(horizon_returns)),
                    'best_case': float(np.max(horizon_returns)),
                    'worst_case': float(np.min(horizon_returns)),
                    'avg_max_drawdown': float(np.mean(horizon_drawdowns)) if horizon_drawdowns else None,
                    'avg_max_gain': float(np.mean(horizon_gains)) if horizon_gains else None,
                    'sample_size': len(horizon_returns)
                }
        
        return {
            'expected_return': float(weighted_return),
            'expected_return_unweighted': float(expected_return),
            'probability_up': float(probability_up),
            'probability_down': float(probability_down),
            'median_return': float(np.median(outcomes)),
            'return_std': float(np.std(outcomes)),
            'best_match_return': float(similar_patterns[0]['outcome_return']),
            'worst_match_return': float(similar_patterns[-1]['outcome_return']),
            'sample_size': len(similar_patterns),
            'horizons': horizons
        }
    
    def _calculate_pattern_confidence(self, similar_patterns: List[Dict]) -> float:
        """
        Calculate confidence in pattern prediction (0-100).
        Based on: number of matches, similarity scores, outcome consistency, recency
        """
        if not similar_patterns:
            return 0.0
        
        # Factor 1: Number of matches (more is better, up to 15)
        count_score = min(len(similar_patterns) / 15.0, 1.0) * 25
        
        # Factor 2: Average similarity (lower distance is better)
        avg_similarity = np.mean([p['similarity_score'] for p in similar_patterns])
        similarity_score = max(0, (15 - avg_similarity) / 15) * 30
        
        # Factor 3: Outcome consistency (all up or all down = high confidence)
        outcomes = [p['outcome_return'] for p in similar_patterns]
        up_count = sum(1 for r in outcomes if r > 0)
        down_count = len(outcomes) - up_count
        consistency = abs(up_count - down_count) / len(outcomes)
        consistency_score = consistency * 25
        
        # Factor 4: Recency of matches (more recent matches = higher confidence)
        recency_factors = [p.get('recency_factor', 0.5) for p in similar_patterns]
        avg_recency = np.mean(recency_factors)
        recency_score = avg_recency * 20
        
        total_confidence = count_score + similarity_score + consistency_score + recency_score
        
        return float(min(total_confidence, 100.0))
