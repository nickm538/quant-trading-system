"""
Gemini-Powered Market Intelligence
Institutional-grade sentiment analysis, earnings interpretation, and market context
Uses Gemini 2.5 Pro for maximum reasoning depth and accuracy
"""

import os
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


def _build_context_preamble() -> str:
    """Build a rich date/time/geopolitical context preamble for every LLM call."""
    now = datetime.now()
    utc_now = datetime.utcnow()
    
    # Market hours context
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    is_weekday = now.weekday() < 5
    is_market_hours = is_weekday and market_open <= now <= market_close
    pre_market = is_weekday and now.replace(hour=4, minute=0) <= now < market_open
    after_hours = is_weekday and market_close < now <= now.replace(hour=20, minute=0)
    
    if is_market_hours:
        market_status = "MARKET IS OPEN (regular trading hours 9:30 AM - 4:00 PM ET)"
    elif pre_market:
        market_status = "PRE-MARKET SESSION (4:00 AM - 9:30 AM ET)"
    elif after_hours:
        market_status = "AFTER-HOURS SESSION (4:00 PM - 8:00 PM ET)"
    elif not is_weekday:
        market_status = "MARKET IS CLOSED (weekend)"
    else:
        market_status = "MARKET IS CLOSED (outside trading hours)"
    
    day_of_week = now.strftime('%A')
    
    # Options expiration context
    # Third Friday of the month = monthly options expiration
    import calendar
    cal = calendar.monthcalendar(now.year, now.month)
    third_friday = None
    friday_count = 0
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            friday_count += 1
            if friday_count == 3:
                third_friday = week[calendar.FRIDAY]
                break
    
    opex_context = ""
    if third_friday:
        opex_date = now.replace(day=third_friday)
        days_to_opex = (opex_date - now).days
        if days_to_opex == 0:
            opex_context = "TODAY IS MONTHLY OPTIONS EXPIRATION (OpEx) — expect elevated volatility, gamma exposure effects, and potential pin risk."
        elif 0 < days_to_opex <= 3:
            opex_context = f"Monthly options expiration (OpEx) is in {days_to_opex} day(s) — gamma exposure and dealer hedging may amplify moves."
        elif days_to_opex < 0 and days_to_opex >= -1:
            opex_context = "Monthly OpEx was yesterday — expect potential repositioning and new gamma exposure buildup."
    
    # Friday afternoon flag
    friday_context = ""
    if day_of_week == 'Friday' and now.hour >= 14:
        friday_context = "FRIDAY AFTERNOON — historically significant for end-of-week positioning, options decay acceleration (theta burn), and potential for low-volume moves."
    
    return f"""CURRENT CONTEXT (you MUST factor this into your analysis):
- Date: {now.strftime('%B %d, %Y')} ({day_of_week})
- Time: {now.strftime('%I:%M %p')} ET / {utc_now.strftime('%H:%M')} UTC
- {market_status}
{f'- {opex_context}' if opex_context else ''}
{f'- {friday_context}' if friday_context else ''}
- Current macro environment: Consider Fed policy stance, recent economic data releases, geopolitical tensions (trade policy, tariffs, conflicts), earnings season timing, and sector rotation trends.
- You are analyzing LIVE market data for a real trading system. Your analysis directly informs real trading decisions with real money. Be precise, specific, and honest about uncertainty.
"""


class GeminiMarketIntelligence:
    """
    Uses Gemini 2.5 Pro to provide institutional-grade market intelligence:
    - News sentiment analysis (more accurate than basic sentiment scores)
    - Earnings report interpretation
    - Market context and sector analysis
    """
    
    MODEL = "gemini-2.5-pro"  # Primary: Gemini 2.5 Pro for maximum reasoning
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Initialize Gemini client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"✓ Gemini 2.5 Pro initialized for market intelligence")
        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            raise
    
    def analyze_news_sentiment(self, symbol: str, news_articles: List[Dict]) -> Dict:
        """
        Analyze news sentiment using Gemini 2.5 Pro
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
            context = _build_context_preamble()
            
            # Prepare news summary for Gemini
            news_text = "\n\n".join([
                f"Headline: {article.get('headline', 'N/A')}\n"
                f"Summary: {article.get('summary', 'N/A')}\n"
                f"Source: {article.get('source', 'N/A')}\n"
                f"Date: {article.get('datetime', 'N/A')}"
                for article in news_articles[:10]  # Top 10 most recent
            ])
            
            prompt = f"""{context}

You are an institutional-grade financial analyst. Analyze the following news articles for {symbol} and provide rigorous sentiment analysis.

NEWS ARTICLES:
{news_text}

IMPORTANT INSTRUCTIONS:
- Weight recent articles more heavily than older ones
- Distinguish between noise (routine coverage) and signal (material events)
- Consider how each article might affect the stock in the next 1-5 trading days
- Factor in the current market environment and any relevant macro context
- Be skeptical of overly promotional or fear-mongering headlines

Provide your analysis in the following JSON format:
{{
    "sentiment_score": <float between -1 (very bearish) and 1 (very bullish)>,
    "confidence": <float between 0 and 1 indicating confidence in the sentiment>,
    "key_themes": [<list of 3-5 main themes from the news>],
    "risk_factors": [<list of identified risks with specifics>],
    "catalysts": [<list of potential positive catalysts with timeframes>],
    "summary": "<brief 2-sentence summary of overall sentiment and its likely market impact>"
}}

Be objective and focus on actionable insights for trading decisions."""

            response = self.client.models.generate_content(
                model=self.MODEL,
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
            logger.info(f"✓ Gemini 2.5 Pro sentiment analysis for {symbol}: {result['sentiment_score']:.2f}")
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
        Analyze earnings data using Gemini 2.5 Pro
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
            context = _build_context_preamble()
            
            earnings_text = "\n".join([
                f"Date: {e.get('date', 'N/A')}, "
                f"EPS Actual: {e.get('epsActual', 'N/A')}, "
                f"EPS Estimate: {e.get('epsEstimate', 'N/A')}, "
                f"Revenue Actual: {e.get('revenueActual', 'N/A')}, "
                f"Revenue Estimate: {e.get('revenueEstimate', 'N/A')}"
                for e in earnings_data[:8]  # Last 8 quarters
            ])
            
            prompt = f"""{context}

You are an institutional-grade earnings analyst. Analyze the earnings history for {symbol} and provide a rigorous assessment.

EARNINGS DATA (last 8 quarters, most recent first):
{earnings_text}

IMPORTANT INSTRUCTIONS:
- Look for acceleration or deceleration in EPS and revenue growth rates
- Identify the consistency of beats/misses and the magnitude of surprises
- Consider whether beats are narrowing (less upside surprise) or widening
- Factor in revenue quality (organic growth vs. acquisitions, one-time items)
- Assess the trajectory relative to the current macro environment

Provide your analysis in JSON format:
{{
    "earnings_quality": <float 0-100, quality of earnings beats/growth>,
    "growth_trajectory": "<accelerating|stable|decelerating>",
    "beat_miss_pattern": "<consistent_beats|mixed|consistent_misses>",
    "guidance_sentiment": <float -1 to 1, based on guidance trends>,
    "revenue_quality": "<high|medium|low>",
    "eps_trend": "<improving|flat|deteriorating>",
    "summary": "<brief assessment with specific numbers referenced>"
}}"""

            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=prompt
            )
            
            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            logger.info(f"✓ Gemini 2.5 Pro earnings analysis for {symbol}: quality={result['earnings_quality']:.1f}")
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
        Get broader market context for the stock using Gemini 2.5 Pro
        Returns: {
            'sector_sentiment': float (-1 to 1),
            'market_regime': str ('bull', 'bear', 'neutral', 'volatile'),
            'relative_strength': str ('outperforming', 'inline', 'underperforming')
        }
        """
        try:
            context = _build_context_preamble()
            
            prompt = f"""{context}

You are a senior macro strategist. Provide current market context for {symbol}{f' in the {sector} sector' if sector else ''}.

IMPORTANT INSTRUCTIONS:
- Reference specific, current macro factors (Fed rate decisions, CPI/PPI data, employment numbers)
- Consider geopolitical factors (trade tensions, tariffs, conflicts, sanctions)
- Assess sector rotation trends and where capital is flowing
- Factor in the VIX regime and credit spreads
- Consider the earnings cycle timing (are we in earnings season?)
- Be specific about timeframes and catalysts

Provide your analysis in JSON format:
{{
    "sector_sentiment": <float -1 to 1>,
    "market_regime": "<bull|bear|neutral|volatile>",
    "relative_strength": "<outperforming|inline|underperforming>",
    "key_factors": [<list of 3-5 specific, current market factors affecting this stock>],
    "geopolitical_risks": [<list of active geopolitical risks relevant to this stock/sector>],
    "upcoming_catalysts": [<list of scheduled events that could move this stock in next 2 weeks>],
    "macro_headwinds": [<specific headwinds>],
    "macro_tailwinds": [<specific tailwinds>]
}}

Be concise, specific, and reference real current events — not generic platitudes."""

            response = self.client.models.generate_content(
                model=self.MODEL,
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
