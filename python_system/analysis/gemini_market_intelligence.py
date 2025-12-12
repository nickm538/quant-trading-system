"""
Gemini-Powered Market Intelligence
Institutional-grade sentiment analysis, earnings interpretation, and market context
"""

import os
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class GeminiMarketIntelligence:
    """
    Uses Gemini Pro to provide institutional-grade market intelligence:
    - News sentiment analysis (more accurate than basic sentiment scores)
    - Earnings report interpretation
    - Market context and sector analysis
    """
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Initialize Gemini client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info("✓ Gemini Pro initialized for market intelligence")
        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            raise
    
    def analyze_news_sentiment(self, symbol: str, news_articles: List[Dict]) -> Dict:
        """
        Analyze news sentiment using Gemini Pro
        Returns: {
            'sentiment_score': float (-1 to 1),
            'confidence': float (0 to 1),
            'key_themes': List[str],
            'risk_factors': List[str],
            'catalysts': List[str]
        }
        """
        if not news_articles:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'key_themes': [],
                'risk_factors': [],
                'catalysts': []
            }
        
        try:
            # Prepare news summary for Gemini
            news_text = "\n\n".join([
                f"Headline: {article.get('headline', 'N/A')}\n"
                f"Summary: {article.get('summary', 'N/A')}\n"
                f"Source: {article.get('source', 'N/A')}\n"
                f"Date: {article.get('datetime', 'N/A')}"
                for article in news_articles[:10]  # Top 10 most recent
            ])
            
            prompt = f"""Analyze the following news articles for {symbol} and provide institutional-grade sentiment analysis.

NEWS ARTICLES:
{news_text}

Provide your analysis in the following JSON format:
{{
    "sentiment_score": <float between -1 (very bearish) and 1 (very bullish)>,
    "confidence": <float between 0 and 1 indicating confidence in the sentiment>,
    "key_themes": [<list of 3-5 main themes from the news>],
    "risk_factors": [<list of identified risks>],
    "catalysts": [<list of potential positive catalysts>],
    "summary": "<brief 2-sentence summary of overall sentiment>"
}}

Be objective and focus on actionable insights for trading decisions."""

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            # Parse JSON response
            result_text = response.text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            logger.info(f"✓ Gemini sentiment analysis for {symbol}: {result['sentiment_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini sentiment analysis failed for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'key_themes': [],
                'risk_factors': [],
                'catalysts': []
            }
    
    def analyze_earnings_context(self, symbol: str, earnings_data: List[Dict]) -> Dict:
        """
        Analyze earnings data using Gemini Pro
        Returns: {
            'earnings_quality': float (0 to 100),
            'growth_trajectory': str ('accelerating', 'stable', 'decelerating'),
            'beat_miss_pattern': str,
            'guidance_sentiment': float (-1 to 1)
        }
        """
        if not earnings_data:
            return {
                'earnings_quality': 50.0,
                'growth_trajectory': 'unknown',
                'beat_miss_pattern': 'insufficient_data',
                'guidance_sentiment': 0.0
            }
        
        try:
            earnings_text = "\n".join([
                f"Date: {e.get('date', 'N/A')}, "
                f"EPS Actual: {e.get('epsActual', 'N/A')}, "
                f"EPS Estimate: {e.get('epsEstimate', 'N/A')}, "
                f"Revenue Actual: {e.get('revenueActual', 'N/A')}, "
                f"Revenue Estimate: {e.get('revenueEstimate', 'N/A')}"
                for e in earnings_data[:8]  # Last 8 quarters
            ])
            
            prompt = f"""Analyze the earnings history for {symbol} and provide institutional-grade assessment.

EARNINGS DATA:
{earnings_text}

Provide your analysis in JSON format:
{{
    "earnings_quality": <float 0-100, quality of earnings beats/growth>,
    "growth_trajectory": "<accelerating|stable|decelerating>",
    "beat_miss_pattern": "<consistent_beats|mixed|consistent_misses>",
    "guidance_sentiment": <float -1 to 1, based on guidance trends>,
    "summary": "<brief assessment>"
}}"""

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            logger.info(f"✓ Gemini earnings analysis for {symbol}: quality={result['earnings_quality']:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Gemini earnings analysis failed for {symbol}: {e}")
            return {
                'earnings_quality': 50.0,
                'growth_trajectory': 'unknown',
                'beat_miss_pattern': 'error',
                'guidance_sentiment': 0.0
            }
    
    def get_market_context(self, symbol: str, sector: str = None) -> Dict:
        """
        Get broader market context for the stock
        Returns: {
            'sector_sentiment': float (-1 to 1),
            'market_regime': str ('bull', 'bear', 'neutral', 'volatile'),
            'relative_strength': str ('outperforming', 'inline', 'underperforming')
        }
        """
        try:
            prompt = f"""Provide current market context for {symbol}{f' in the {sector} sector' if sector else ''}.

Based on recent market conditions (December 2025), provide:
{{
    "sector_sentiment": <float -1 to 1>,
    "market_regime": "<bull|bear|neutral|volatile>",
    "relative_strength": "<outperforming|inline|underperforming>",
    "key_factors": [<list of 2-3 key market factors affecting this stock>]
}}

Be concise and focus on actionable context."""

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            logger.error(f"Gemini market context failed for {symbol}: {e}")
            return {
                'sector_sentiment': 0.0,
                'market_regime': 'unknown',
                'relative_strength': 'unknown',
                'key_factors': []
            }
