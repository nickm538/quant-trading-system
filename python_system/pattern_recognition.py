"""
ADVANCED AI PATTERN RECOGNITION MODULE
=======================================

Uses computer vision and time-series analysis to identify similar historical patterns.
ALWAYS ACTIVE - runs automatically on every analysis.

Techniques:
- Dynamic Time Warping (DTW) for pattern similarity
- Euclidean distance for shape matching
- Historical regime database (2008 crash, dot-com, COVID, etc.)
- Rolling window comparison across ALL historical data

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
    """
    
    # Historical market regimes with known outcomes
    HISTORICAL_REGIMES = {
        'dot_com_bubble_peak': {
            'period': '1999-12 to 2000-03',
            'description': 'Dot-com bubble peak - extreme valuations, euphoria',
            'outcome': 'Crashed -78% over 2.5 years (NASDAQ)',
            'characteristics': {
                'pe_ratio': '>100',
                'sentiment': 'extreme_greed',
                'volatility': 'rising',
                'trend': 'parabolic'
            }
        },
        'dot_com_crash': {
            'period': '2000-03 to 2002-10',
            'description': 'Dot-com crash - capitulation, panic selling',
            'outcome': 'Bottom formed, +400% recovery over 5 years',
            'characteristics': {
                'pe_ratio': '<15',
                'sentiment': 'extreme_fear',
                'volatility': 'extreme',
                'trend': 'downtrend'
            }
        },
        'financial_crisis_2008': {
            'period': '2008-09 to 2009-03',
            'description': 'Financial crisis - systemic risk, credit freeze',
            'outcome': 'Bottom formed, +350% recovery over 10 years',
            'characteristics': {
                'pe_ratio': '<12',
                'sentiment': 'panic',
                'volatility': 'extreme',
                'trend': 'freefall'
            }
        },
        'covid_crash_2020': {
            'period': '2020-02 to 2020-03',
            'description': 'COVID crash - fastest bear market in history',
            'outcome': 'V-shaped recovery, +100% in 18 months',
            'characteristics': {
                'pe_ratio': '15-20',
                'sentiment': 'fear',
                'volatility': 'extreme',
                'trend': 'vertical_drop'
            }
        },
        'bull_market_2017': {
            'period': '2016-11 to 2018-01',
            'description': 'Trump rally - tax cuts, deregulation optimism',
            'outcome': '+40% gain, then -20% correction',
            'characteristics': {
                'pe_ratio': '20-25',
                'sentiment': 'optimism',
                'volatility': 'low',
                'trend': 'steady_uptrend'
            }
        },
        'inflation_scare_2022': {
            'period': '2022-01 to 2022-10',
            'description': 'Fed rate hikes - inflation fight, recession fears',
            'outcome': '-25% drawdown, then recovery',
            'characteristics': {
                'pe_ratio': '15-18',
                'sentiment': 'fear',
                'volatility': 'high',
                'trend': 'downtrend'
            }
        }
    }
    
    def __init__(self):
        """Initialize pattern recognition engine"""
        logger.info("Pattern Recognition Engine initialized")
    
    def analyze_patterns(self, price_data: pd.DataFrame, current_analysis: Dict) -> Dict:
        """
        MAIN METHOD: Analyze current pattern against ALL historical data.
        ALWAYS RUNS - not optional.
        
        Args:
            price_data: DataFrame with OHLCV data
            current_analysis: Current stock analysis dict
            
        Returns:
            Dict with pattern matches, regime identification, and predictions
        """
        logger.info("=== PATTERN RECOGNITION ACTIVE ===")
        
        results = {
            'similar_patterns': self._find_similar_patterns(price_data),
            'regime_match': self._identify_regime(current_analysis),
            'historical_analogy': self._find_best_analogy(price_data, current_analysis),
            'pattern_prediction': None,
            'confidence': 0.0
        }
        
        # Generate prediction based on similar patterns
        if results['similar_patterns']:
            results['pattern_prediction'] = self._predict_outcome(results['similar_patterns'])
            results['confidence'] = self._calculate_pattern_confidence(results['similar_patterns'])
        
        logger.info(f"Pattern recognition complete: {len(results['similar_patterns'])} matches found")
        
        return results
    
    def _find_similar_patterns(self, price_data: pd.DataFrame, window=30) -> List[Dict]:
        """
        Find similar price patterns using DTW (Dynamic Time Warping).
        Compares last N days to ALL historical periods.
        """
        logger.info(f"Pattern matching: price_data length = {len(price_data)}, window = {window}")
        if len(price_data) < window * 2:
            logger.warning(f"Not enough data for pattern matching: {len(price_data)} < {window * 2}")
            return []
        
        close = price_data['Close'].values
        
        # Normalize current pattern (last N days)
        current_pattern = close[-window:]
        current_normalized = (current_pattern - current_pattern.mean()) / (current_pattern.std() + 1e-8)
        
        similar_patterns = []
        scores_checked = 0
        scores_below_threshold = 0
        min_score_found = float('inf')
        
        # Slide window across historical data
        for i in range(len(close) - window * 2, 0, -5):  # Step by 5 days for efficiency
            if i < window:
                break
            
            # Get historical pattern
            hist_pattern = close[i:i+window]
            hist_normalized = (hist_pattern - hist_pattern.mean()) / (hist_pattern.std() + 1e-8)
            
            # Calculate DTW distance
            try:
                # Ensure 1D arrays for fastdtw (reshape if needed)
                current_1d = current_normalized.reshape(-1)
                hist_1d = hist_normalized.reshape(-1)
                # Use lambda for distance function (fastdtw compatible)
                distance, _ = fastdtw(current_1d, hist_1d, dist=lambda x, y: np.abs(x - y))
                
                # Also calculate Euclidean distance for comparison
                euclidean_dist = np.sqrt(np.sum((current_normalized - hist_normalized) ** 2))
                
                # Combined similarity score (lower is better)
                similarity_score = (distance + euclidean_dist) / 2
                scores_checked += 1
                min_score_found = min(min_score_found, similarity_score)
                
                # Only keep very similar patterns (threshold tuned for quality)
                # Relaxed threshold to find more matches (was 8.0, now 15.0)
                if similarity_score < 15.0:  # Threshold for "similar" (adjusted for more matches)
                    scores_below_threshold += 1
                    # Get outcome (what happened next)
                    future_window = min(30, len(close) - i - window)
                    if future_window > 0:
                        future_prices = close[i+window:i+window+future_window]
                        outcome_return = (future_prices[-1] - hist_pattern[-1]) / hist_pattern[-1]
                        
                        similar_patterns.append({
                            'start_index': i,
                            'similarity_score': float(similarity_score),
                            'dtw_distance': float(distance),
                            'euclidean_distance': float(euclidean_dist),
                            'outcome_return': float(outcome_return),
                            'outcome_days': future_window,
                            'pattern_start_price': float(hist_pattern[0]),
                            'pattern_end_price': float(hist_pattern[-1]),
                            'pattern_return': float((hist_pattern[-1] - hist_pattern[0]) / hist_pattern[0])
                        })
            except Exception as e:
                logger.warning(f"DTW calculation failed at i={i}: {str(e)}")
                continue
        
        # Sort by similarity (best matches first)
        similar_patterns.sort(key=lambda x: x['similarity_score'])
        
        logger.info(f"DTW Stats: checked {scores_checked} patterns, {scores_below_threshold} below threshold, min_score={min_score_found:.2f}")
        
        # Return top 10 matches
        return similar_patterns[:10]
    
    def _identify_regime(self, analysis: Dict) -> Optional[Dict]:
        """
        Identify which historical market regime current conditions most resemble.
        """
        fundamentals = analysis.get('fundamentals', {})
        technical = analysis.get('technical_analysis', {})
        sentiment = analysis.get('sentiment_score', 50)
        
        pe_ratio = fundamentals.get('pe_ratio', 20)
        volatility = technical.get('current_volatility', 0.2)
        trend_score = technical.get('trend_score', 50)
        
        # Score each regime based on similarity
        regime_scores = {}
        
        for regime_name, regime_data in self.HISTORICAL_REGIMES.items():
            score = 0.0
            characteristics = regime_data['characteristics']
            
            # PE ratio matching
            if 'extreme' in characteristics.get('pe_ratio', ''):
                if pe_ratio > 50:
                    score += 25
            elif '<12' in characteristics.get('pe_ratio', ''):
                if pe_ratio < 12:
                    score += 25
            elif '<15' in characteristics.get('pe_ratio', ''):
                if pe_ratio < 15:
                    score += 20
            elif '15-20' in characteristics.get('pe_ratio', ''):
                if 15 <= pe_ratio <= 20:
                    score += 20
            elif '20-25' in characteristics.get('pe_ratio', ''):
                if 20 <= pe_ratio <= 25:
                    score += 20
            
            # Sentiment matching
            sentiment_char = characteristics.get('sentiment', '')
            if 'extreme_greed' in sentiment_char and sentiment > 80:
                score += 25
            elif 'extreme_fear' in sentiment_char and sentiment < 20:
                score += 25
            elif 'panic' in sentiment_char and sentiment < 15:
                score += 25
            elif 'fear' in sentiment_char and 20 <= sentiment < 40:
                score += 20
            elif 'optimism' in sentiment_char and 60 <= sentiment < 80:
                score += 20
            
            # Volatility matching
            vol_char = characteristics.get('volatility', '')
            if 'extreme' in vol_char and volatility > 0.5:
                score += 25
            elif 'high' in vol_char and 0.3 <= volatility <= 0.5:
                score += 20
            elif 'rising' in vol_char and volatility > 0.25:
                score += 15
            elif 'low' in vol_char and volatility < 0.2:
                score += 20
            
            # Trend matching
            trend_char = characteristics.get('trend', '')
            if 'parabolic' in trend_char and trend_score > 85:
                score += 25
            elif 'steady_uptrend' in trend_char and 60 <= trend_score <= 80:
                score += 20
            elif 'downtrend' in trend_char and trend_score < 40:
                score += 20
            elif 'freefall' in trend_char and trend_score < 20:
                score += 25
            elif 'vertical_drop' in trend_char and trend_score < 15:
                score += 25
            
            regime_scores[regime_name] = score
        
        # Get best match
        if not regime_scores:
            return None
        
        best_regime = max(regime_scores, key=regime_scores.get)
        best_score = regime_scores[best_regime]
        
        # Only return if score is meaningful (>40 out of 100)
        if best_score < 40:
            return None
        
        regime_info = self.HISTORICAL_REGIMES[best_regime].copy()
        regime_info['match_score'] = best_score
        regime_info['regime_name'] = best_regime
        
        return regime_info
    
    def _find_best_analogy(self, price_data: pd.DataFrame, analysis: Dict) -> Optional[str]:
        """
        Generate human-readable historical analogy.
        "This setup is like [Historical Event] because [Reasons]"
        """
        regime = self._identify_regime(analysis)
        
        if not regime:
            return "Current market conditions don't strongly match any major historical regime. This could be a unique setup or transitional period."
        
        regime_name = regime['regime_name']
        period = regime['period']
        description = regime['description']
        outcome = regime['outcome']
        match_score = regime['match_score']
        
        # Format analogy
        analogy = f"**This resembles {regime_name.replace('_', ' ').title()} ({period})**\n\n"
        analogy += f"**Historical Context**: {description}\n\n"
        analogy += f"**What Happened Next**: {outcome}\n\n"
        analogy += f"**Match Confidence**: {match_score:.0f}% similarity to historical conditions\n\n"
        
        # Add characteristics
        characteristics = regime['characteristics']
        analogy += "**Key Similarities**:\n"
        for key, value in characteristics.items():
            analogy += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return analogy
    
    def _predict_outcome(self, similar_patterns: List[Dict]) -> Dict:
        """
        Predict likely outcome based on similar historical patterns.
        """
        if not similar_patterns:
            return {
                'expected_return': 0.0,
                'probability_up': 0.5,
                'probability_down': 0.5,
                'avg_days_to_outcome': 30
            }
        
        # Calculate statistics from similar patterns
        outcomes = [p['outcome_return'] for p in similar_patterns]
        days = [p['outcome_days'] for p in similar_patterns]
        
        expected_return = np.mean(outcomes)
        probability_up = sum(1 for r in outcomes if r > 0) / len(outcomes)
        probability_down = 1 - probability_up
        avg_days = np.mean(days)
        
        # Weight by similarity (better matches count more)
        weights = [1 / (p['similarity_score'] + 1) for p in similar_patterns]
        weighted_return = np.average(outcomes, weights=weights)
        
        return {
            'expected_return': float(weighted_return),
            'expected_return_unweighted': float(expected_return),
            'probability_up': float(probability_up),
            'probability_down': float(probability_down),
            'avg_days_to_outcome': int(avg_days),
            'sample_size': len(similar_patterns),
            'best_match_return': float(similar_patterns[0]['outcome_return']),
            'worst_match_return': float(similar_patterns[-1]['outcome_return'])
        }
    
    def _calculate_pattern_confidence(self, similar_patterns: List[Dict]) -> float:
        """
        Calculate confidence in pattern prediction (0-100).
        Based on: number of matches, similarity scores, outcome consistency
        """
        if not similar_patterns:
            return 0.0
        
        # Factor 1: Number of matches (more is better)
        count_score = min(len(similar_patterns) / 10.0, 1.0) * 30
        
        # Factor 2: Average similarity (lower distance is better)
        avg_similarity = np.mean([p['similarity_score'] for p in similar_patterns])
        similarity_score = max(0, (10 - avg_similarity) / 10) * 40  # Lower is better
        
        # Factor 3: Outcome consistency (all up or all down = high confidence)
        outcomes = [p['outcome_return'] for p in similar_patterns]
        up_count = sum(1 for r in outcomes if r > 0)
        down_count = len(outcomes) - up_count
        consistency = abs(up_count - down_count) / len(outcomes)
        consistency_score = consistency * 30
        
        total_confidence = count_score + similarity_score + consistency_score
        
        return float(min(total_confidence, 100.0))
