"""
EXA AI Client for Real-Time Web Search & Candlestick Chart Analysis
=====================================================================

Uses EXA AI's neural search to find and analyze:
1. Real-time candlestick charts for any stock/ETF
2. Recent technical analysis articles and expert opinions
3. Current market news and sentiment
4. Chart pattern breakdowns from financial sites

This is a critical component for ensuring the most precise, up-to-date
candlestick pattern recognition by cross-referencing algorithmic detection
with real-world expert chart analysis from the web.
"""

import requests
import logging
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# EXA AI API configuration
EXA_API_KEY = os.getenv(
    'EXA_API_KEY',
    '22f9b221-4031-477e-9175-750c5fa6985b'
)
EXA_BASE_URL = "https://api.exa.ai"
EXA_TIMEOUT = 30  # seconds


class ExaClient:
    """
    Client for EXA AI neural search API.
    Specializes in finding real-time financial chart analysis and market data.
    """
    
    def __init__(self, api_key: str = EXA_API_KEY):
        self.api_key = api_key
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        logger.info("EXA AI Client initialized")
    
    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """Make a request to the EXA API with error handling."""
        url = f"{EXA_BASE_URL}/{endpoint}"
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=EXA_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"EXA API timeout for {endpoint}")
            return {"error": "API timeout", "results": []}
        except requests.exceptions.HTTPError as e:
            logger.error(f"EXA API HTTP error: {e}")
            return {"error": str(e), "results": []}
        except Exception as e:
            logger.error(f"EXA API error: {e}")
            return {"error": str(e), "results": []}
    
    def search(self, query: str, num_results: int = 10, 
               category: str = None, search_type: str = "auto",
               include_domains: List[str] = None,
               start_published_date: str = None,
               end_published_date: str = None,
               include_text: List[str] = None) -> Dict:
        """
        Perform a neural search using EXA AI.
        
        Args:
            query: Search query string
            num_results: Number of results (max 100)
            category: Optional category filter (news, financial report, etc.)
            search_type: "auto", "neural", "fast", or "deep"
            include_domains: List of domains to restrict search to
            start_published_date: ISO 8601 date string
            end_published_date: ISO 8601 date string
            include_text: List of text strings that must appear in results
        """
        payload = {
            "query": query,
            "numResults": min(num_results, 100),
            "type": search_type,
            "contents": {
                "text": True,
                "summary": True,
                "highlights": {
                    "numSentences": 3
                }
            }
        }
        
        if category:
            payload["category"] = category
        if include_domains:
            payload["includeDomains"] = include_domains
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date
        if include_text:
            payload["includeText"] = include_text
        
        return self._make_request("search", payload)
    
    def get_contents(self, urls: List[str]) -> Dict:
        """Get full contents of specific URLs."""
        payload = {
            "ids": urls,
            "text": True,
            "summary": True,
            "highlights": {
                "numSentences": 5
            }
        }
        return self._make_request("contents", payload)
    
    # =========================================================================
    # CANDLESTICK CHART ANALYSIS METHODS
    # =========================================================================
    
    def find_candlestick_analysis(self, symbol: str, days_back: int = 30) -> Dict:
        """
        Find recent candlestick chart analysis for a given stock/ETF.
        Searches multiple financial domains for expert chart pattern analysis.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            days_back: How many days back to search (default 30)
            
        Returns:
            Dict with candlestick analysis from web sources
        """
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00.000Z")
        
        # Search for candlestick analysis from top financial sites
        chart_domains = [
            "tradingview.com",
            "stockcharts.com",
            "finviz.com",
            "barchart.com",
            "investing.com",
            "seekingalpha.com",
            "thestreet.com",
            "marketwatch.com",
            "yahoo.com",
            "benzinga.com",
            "schaeffersresearch.com",
            "stockanalysis.com",
            "chartmill.com"
        ]
        
        results = {
            "symbol": symbol,
            "search_date": datetime.now().isoformat(),
            "days_searched": days_back,
            "candlestick_analyses": [],
            "chart_pattern_mentions": [],
            "expert_opinions": [],
            "technical_summaries": [],
            "source_urls": [],
            "errors": []
        }
        
        # Query 1: Direct candlestick pattern search
        try:
            candlestick_search = self.search(
                query=f"{symbol} stock candlestick chart pattern analysis technical",
                num_results=10,
                search_type="auto",
                start_published_date=start_date,
                include_text=[symbol]
            )
            
            if candlestick_search.get("results"):
                for result in candlestick_search["results"]:
                    analysis = self._extract_chart_analysis(result, symbol)
                    if analysis:
                        results["candlestick_analyses"].append(analysis)
                        results["source_urls"].append(result.get("url", ""))
        except Exception as e:
            results["errors"].append(f"Candlestick search error: {str(e)}")
        
        # Query 2: Technical analysis and chart patterns
        try:
            tech_search = self.search(
                query=f"{symbol} technical analysis chart patterns support resistance moving average",
                num_results=10,
                search_type="auto",
                start_published_date=start_date,
                include_text=[symbol]
            )
            
            if tech_search.get("results"):
                for result in tech_search["results"]:
                    summary = self._extract_technical_summary(result, symbol)
                    if summary:
                        results["technical_summaries"].append(summary)
                        if result.get("url") not in results["source_urls"]:
                            results["source_urls"].append(result.get("url", ""))
        except Exception as e:
            results["errors"].append(f"Technical search error: {str(e)}")
        
        # Query 3: Recent news and price action
        try:
            news_search = self.search(
                query=f"{symbol} stock price action breakout breakdown trend",
                num_results=8,
                category="news",
                start_published_date=start_date,
                include_text=[symbol]
            )
            
            if news_search.get("results"):
                for result in news_search["results"]:
                    opinion = self._extract_expert_opinion(result, symbol)
                    if opinion:
                        results["expert_opinions"].append(opinion)
        except Exception as e:
            results["errors"].append(f"News search error: {str(e)}")
        
        # Query 4: Specific chart pattern identification
        try:
            pattern_search = self.search(
                query=f"{symbol} head and shoulders double top bottom cup handle flag pennant wedge triangle",
                num_results=8,
                search_type="auto",
                start_published_date=start_date,
                include_text=[symbol]
            )
            
            if pattern_search.get("results"):
                for result in pattern_search["results"]:
                    patterns = self._extract_pattern_mentions(result, symbol)
                    if patterns:
                        results["chart_pattern_mentions"].extend(patterns)
        except Exception as e:
            results["errors"].append(f"Pattern search error: {str(e)}")
        
        # Synthesize findings
        results["synthesis"] = self._synthesize_chart_findings(results)
        
        return results
    
    def find_market_sentiment(self, symbol: str) -> Dict:
        """
        Find current market sentiment and analyst opinions for a stock.
        """
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00.000Z")
        
        sentiment_result = {
            "symbol": symbol,
            "search_date": datetime.now().isoformat(),
            "analyst_opinions": [],
            "sentiment_signals": [],
            "key_headlines": []
        }
        
        try:
            sentiment_search = self.search(
                query=f"{symbol} stock analyst rating upgrade downgrade price target outlook",
                num_results=10,
                category="news",
                start_published_date=start_date,
                include_text=[symbol]
            )
            
            if sentiment_search.get("results"):
                for result in sentiment_search["results"]:
                    headline = result.get("title", "")
                    summary = result.get("summary", "")
                    url = result.get("url", "")
                    published = result.get("publishedDate", "")
                    
                    # Determine sentiment from content
                    text_content = (headline + " " + summary).lower()
                    sentiment = self._classify_sentiment(text_content)
                    
                    sentiment_result["analyst_opinions"].append({
                        "headline": headline,
                        "summary": summary[:500],
                        "url": url,
                        "published": published,
                        "sentiment": sentiment
                    })
                    
                    sentiment_result["key_headlines"].append(headline)
                    sentiment_result["sentiment_signals"].append(sentiment)
        except Exception as e:
            sentiment_result["error"] = str(e)
        
        # Calculate overall sentiment
        if sentiment_result["sentiment_signals"]:
            bullish = sum(1 for s in sentiment_result["sentiment_signals"] if s == "bullish")
            bearish = sum(1 for s in sentiment_result["sentiment_signals"] if s == "bearish")
            total = len(sentiment_result["sentiment_signals"])
            
            sentiment_result["overall_sentiment"] = {
                "bullish_pct": round(bullish / total * 100, 1),
                "bearish_pct": round(bearish / total * 100, 1),
                "neutral_pct": round((total - bullish - bearish) / total * 100, 1),
                "bias": "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
            }
        
        return sentiment_result
    
    def find_insider_and_dark_pool(self, symbol: str) -> Dict:
        """
        Search for insider trading activity, dark pool data, and unusual options activity.
        """
        start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%dT00:00:00.000Z")
        
        result = {
            "symbol": symbol,
            "insider_activity": [],
            "dark_pool_mentions": [],
            "unusual_options": [],
            "congress_trades": []
        }
        
        # Insider trading
        try:
            insider_search = self.search(
                query=f"{symbol} insider trading buy sell SEC filing",
                num_results=8,
                start_published_date=start_date,
                include_text=[symbol]
            )
            if insider_search.get("results"):
                for r in insider_search["results"]:
                    result["insider_activity"].append({
                        "title": r.get("title", ""),
                        "summary": r.get("summary", "")[:300],
                        "url": r.get("url", ""),
                        "date": r.get("publishedDate", "")
                    })
        except Exception as e:
            result["insider_error"] = str(e)
        
        # Dark pool / unusual options
        try:
            dark_search = self.search(
                query=f"{symbol} dark pool unusual options activity block trade flow",
                num_results=8,
                start_published_date=start_date,
                include_text=[symbol]
            )
            if dark_search.get("results"):
                for r in dark_search["results"]:
                    text = (r.get("title", "") + " " + r.get("summary", "")).lower()
                    if "dark pool" in text or "block" in text:
                        result["dark_pool_mentions"].append({
                            "title": r.get("title", ""),
                            "summary": r.get("summary", "")[:300],
                            "url": r.get("url", "")
                        })
                    if "unusual" in text or "options" in text or "sweep" in text:
                        result["unusual_options"].append({
                            "title": r.get("title", ""),
                            "summary": r.get("summary", "")[:300],
                            "url": r.get("url", "")
                        })
        except Exception as e:
            result["dark_pool_error"] = str(e)
        
        # Congress trades
        try:
            congress_search = self.search(
                query=f"{symbol} congress trade senator representative stock purchase",
                num_results=5,
                start_published_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00.000Z"),
                include_text=[symbol]
            )
            if congress_search.get("results"):
                for r in congress_search["results"]:
                    result["congress_trades"].append({
                        "title": r.get("title", ""),
                        "summary": r.get("summary", "")[:300],
                        "url": r.get("url", ""),
                        "date": r.get("publishedDate", "")
                    })
        except Exception as e:
            result["congress_error"] = str(e)
        
        return result
    
    # =========================================================================
    # EXTRACTION & ANALYSIS HELPERS
    # =========================================================================
    
    def _extract_chart_analysis(self, result: Dict, symbol: str) -> Optional[Dict]:
        """Extract candlestick chart analysis from a search result."""
        text = result.get("text", "")
        summary = result.get("summary", "")
        highlights = result.get("highlights", [])
        
        if not text and not summary:
            return None
        
        # Look for candlestick pattern mentions
        candlestick_patterns = self._find_candlestick_patterns_in_text(text)
        
        # Look for support/resistance levels
        levels = self._extract_price_levels(text, symbol)
        
        # Look for trend direction
        trend = self._extract_trend_direction(text)
        
        return {
            "source": result.get("url", ""),
            "title": result.get("title", ""),
            "published": result.get("publishedDate", ""),
            "summary": summary[:500] if summary else "",
            "key_highlights": highlights[:3] if highlights else [],
            "candlestick_patterns_found": candlestick_patterns,
            "price_levels": levels,
            "trend_direction": trend,
            "relevance_score": self._calculate_relevance(text, symbol)
        }
    
    def _extract_technical_summary(self, result: Dict, symbol: str) -> Optional[Dict]:
        """Extract technical analysis summary from a search result."""
        text = result.get("text", "")
        summary = result.get("summary", "")
        
        if not text and not summary:
            return None
        
        # Extract key technical indicators mentioned
        indicators = self._find_technical_indicators(text)
        
        return {
            "source": result.get("url", ""),
            "title": result.get("title", ""),
            "published": result.get("publishedDate", ""),
            "summary": summary[:500] if summary else "",
            "indicators_mentioned": indicators,
            "overall_bias": self._extract_trend_direction(text)
        }
    
    def _extract_expert_opinion(self, result: Dict, symbol: str) -> Optional[Dict]:
        """Extract expert opinion from a news result."""
        summary = result.get("summary", "")
        title = result.get("title", "")
        
        if not summary and not title:
            return None
        
        text_content = (title + " " + summary).lower()
        sentiment = self._classify_sentiment(text_content)
        
        return {
            "source": result.get("url", ""),
            "title": title,
            "published": result.get("publishedDate", ""),
            "opinion_summary": summary[:400] if summary else "",
            "sentiment": sentiment
        }
    
    def _extract_pattern_mentions(self, result: Dict, symbol: str) -> List[Dict]:
        """Extract specific chart pattern mentions from text."""
        text = result.get("text", "")
        patterns_found = []
        
        pattern_keywords = {
            "head and shoulders": "bearish_reversal",
            "inverse head and shoulders": "bullish_reversal",
            "double top": "bearish_reversal",
            "double bottom": "bullish_reversal",
            "triple top": "bearish_reversal",
            "triple bottom": "bullish_reversal",
            "cup and handle": "bullish_continuation",
            "ascending triangle": "bullish_continuation",
            "descending triangle": "bearish_continuation",
            "symmetrical triangle": "neutral_continuation",
            "bull flag": "bullish_continuation",
            "bear flag": "bearish_continuation",
            "pennant": "continuation",
            "rising wedge": "bearish_reversal",
            "falling wedge": "bullish_reversal",
            "channel up": "bullish_trend",
            "channel down": "bearish_trend",
            "breakout": "bullish_signal",
            "breakdown": "bearish_signal",
            "golden cross": "bullish_signal",
            "death cross": "bearish_signal",
            "doji": "indecision",
            "hammer": "bullish_reversal",
            "shooting star": "bearish_reversal",
            "engulfing": "reversal",
            "morning star": "bullish_reversal",
            "evening star": "bearish_reversal",
            "three white soldiers": "bullish_continuation",
            "three black crows": "bearish_continuation",
            "harami": "reversal",
            "spinning top": "indecision",
            "marubozu": "strong_trend",
            "tweezer": "reversal",
            "piercing line": "bullish_reversal",
            "dark cloud cover": "bearish_reversal",
            "abandoned baby": "reversal",
            "three inside up": "bullish_reversal",
            "three inside down": "bearish_reversal"
        }
        
        text_lower = text.lower()
        for pattern_name, pattern_type in pattern_keywords.items():
            if pattern_name in text_lower:
                # Find context around the pattern mention
                idx = text_lower.find(pattern_name)
                start = max(0, idx - 100)
                end = min(len(text), idx + len(pattern_name) + 200)
                context = text[start:end].strip()
                
                patterns_found.append({
                    "pattern": pattern_name,
                    "type": pattern_type,
                    "context": context[:300],
                    "source": result.get("url", ""),
                    "published": result.get("publishedDate", "")
                })
        
        return patterns_found
    
    def _find_candlestick_patterns_in_text(self, text: str) -> List[str]:
        """Find all candlestick pattern names mentioned in text."""
        patterns = []
        text_lower = text.lower()
        
        candlestick_names = [
            "doji", "hammer", "inverted hammer", "shooting star",
            "engulfing", "bullish engulfing", "bearish engulfing",
            "morning star", "evening star", "three white soldiers",
            "three black crows", "harami", "bullish harami", "bearish harami",
            "spinning top", "marubozu", "tweezer top", "tweezer bottom",
            "piercing line", "dark cloud cover", "abandoned baby",
            "three inside up", "three inside down", "kicker",
            "hanging man", "dragonfly doji", "gravestone doji",
            "long-legged doji", "rickshaw man", "belt hold",
            "counterattack", "gap up", "gap down",
            "rising three methods", "falling three methods",
            "on-neck", "in-neck", "thrusting",
            "tasuki gap", "side by side white lines",
            "advance block", "deliberation", "identical three crows",
            "concealing baby swallow", "stick sandwich",
            "homing pigeon", "ladder bottom", "matching low"
        ]
        
        for name in candlestick_names:
            if name in text_lower:
                patterns.append(name)
        
        return patterns
    
    def _find_technical_indicators(self, text: str) -> List[str]:
        """Find technical indicator mentions in text."""
        indicators = []
        text_lower = text.lower()
        
        indicator_names = [
            "rsi", "macd", "bollinger band", "moving average",
            "sma", "ema", "stochastic", "adx", "atr",
            "fibonacci", "ichimoku", "vwap", "obv",
            "volume", "momentum", "williams %r", "cci",
            "parabolic sar", "pivot point", "supertrend"
        ]
        
        for name in indicator_names:
            if name in text_lower:
                indicators.append(name)
        
        return indicators
    
    def _extract_price_levels(self, text: str, symbol: str) -> Dict:
        """Extract support and resistance price levels from text."""
        levels = {"support": [], "resistance": []}
        
        # Look for patterns like "support at $XXX" or "resistance near $XXX"
        support_pattern = r'support\s+(?:at|near|around|level)?\s*\$?([\d,]+\.?\d*)'
        resistance_pattern = r'resistance\s+(?:at|near|around|level)?\s*\$?([\d,]+\.?\d*)'
        
        for match in re.finditer(support_pattern, text.lower()):
            try:
                price = float(match.group(1).replace(',', ''))
                if price > 0:
                    levels["support"].append(price)
            except ValueError:
                pass
        
        for match in re.finditer(resistance_pattern, text.lower()):
            try:
                price = float(match.group(1).replace(',', ''))
                if price > 0:
                    levels["resistance"].append(price)
            except ValueError:
                pass
        
        # Deduplicate
        levels["support"] = sorted(list(set(levels["support"])))
        levels["resistance"] = sorted(list(set(levels["resistance"])))
        
        return levels
    
    def _extract_trend_direction(self, text: str) -> str:
        """Determine overall trend direction from text content."""
        text_lower = text.lower()
        
        bullish_words = [
            "bullish", "uptrend", "breakout", "rally", "surge",
            "higher", "gains", "positive", "buy", "accumulation",
            "support held", "bounce", "recovery", "outperform",
            "upgrade", "golden cross", "oversold bounce"
        ]
        bearish_words = [
            "bearish", "downtrend", "breakdown", "selloff", "decline",
            "lower", "losses", "negative", "sell", "distribution",
            "support broken", "crash", "correction", "underperform",
            "downgrade", "death cross", "overbought"
        ]
        
        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)
        
        if bullish_count > bearish_count + 2:
            return "strongly_bullish"
        elif bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count + 2:
            return "strongly_bearish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"
    
    def _classify_sentiment(self, text: str) -> str:
        """Classify sentiment of text as bullish/bearish/neutral."""
        bullish_terms = [
            "upgrade", "buy", "outperform", "bullish", "positive",
            "beat", "exceeded", "strong", "growth", "rally",
            "breakout", "surge", "upside", "overweight", "accumulate",
            "price target raised", "above consensus"
        ]
        bearish_terms = [
            "downgrade", "sell", "underperform", "bearish", "negative",
            "miss", "below", "weak", "decline", "selloff",
            "breakdown", "plunge", "downside", "underweight", "reduce",
            "price target lowered", "below consensus"
        ]
        
        text_lower = text.lower()
        bullish_score = sum(1 for t in bullish_terms if t in text_lower)
        bearish_score = sum(1 for t in bearish_terms if t in text_lower)
        
        if bullish_score > bearish_score:
            return "bullish"
        elif bearish_score > bullish_score:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_relevance(self, text: str, symbol: str) -> float:
        """Calculate how relevant a result is to the stock analysis."""
        score = 0.0
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Symbol mentioned frequently
        mentions = text_lower.count(symbol_lower)
        score += min(mentions * 5, 30)
        
        # Technical analysis terms
        tech_terms = ["chart", "pattern", "candlestick", "technical", "support", "resistance"]
        for term in tech_terms:
            if term in text_lower:
                score += 10
        
        # Recency bonus (handled by search date filter)
        score += 10
        
        return min(score, 100)
    
    def _synthesize_chart_findings(self, results: Dict) -> Dict:
        """
        Synthesize all chart findings into a coherent analysis.
        This is the key output - combining all EXA search results into
        actionable chart pattern intelligence.
        """
        synthesis = {
            "total_sources_analyzed": len(results.get("source_urls", [])),
            "candlestick_patterns_detected": [],
            "chart_patterns_detected": [],
            "consensus_trend": "neutral",
            "support_levels": [],
            "resistance_levels": [],
            "expert_consensus": "neutral",
            "confidence": 0,
            "key_findings": []
        }
        
        # Aggregate candlestick patterns
        all_candle_patterns = []
        for analysis in results.get("candlestick_analyses", []):
            all_candle_patterns.extend(analysis.get("candlestick_patterns_found", []))
        
        # Count pattern frequency
        pattern_counts = {}
        for p in all_candle_patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        synthesis["candlestick_patterns_detected"] = [
            {"pattern": k, "mentions": v}
            for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
        ]
        
        # Aggregate chart patterns
        chart_pattern_counts = {}
        for pm in results.get("chart_pattern_mentions", []):
            name = pm.get("pattern", "")
            ptype = pm.get("type", "")
            key = f"{name} ({ptype})"
            chart_pattern_counts[key] = chart_pattern_counts.get(key, 0) + 1
        
        synthesis["chart_patterns_detected"] = [
            {"pattern": k, "mentions": v}
            for k, v in sorted(chart_pattern_counts.items(), key=lambda x: -x[1])
        ]
        
        # Determine consensus trend
        trend_votes = []
        for analysis in results.get("candlestick_analyses", []):
            trend = analysis.get("trend_direction", "neutral")
            trend_votes.append(trend)
        for summary in results.get("technical_summaries", []):
            bias = summary.get("overall_bias", "neutral")
            trend_votes.append(bias)
        
        bullish_votes = sum(1 for t in trend_votes if "bullish" in t)
        bearish_votes = sum(1 for t in trend_votes if "bearish" in t)
        total_votes = len(trend_votes)
        
        if total_votes > 0:
            if bullish_votes > bearish_votes + 1:
                synthesis["consensus_trend"] = "bullish"
            elif bearish_votes > bullish_votes + 1:
                synthesis["consensus_trend"] = "bearish"
            else:
                synthesis["consensus_trend"] = "mixed"
        
        # Aggregate support/resistance levels
        all_support = []
        all_resistance = []
        for analysis in results.get("candlestick_analyses", []):
            levels = analysis.get("price_levels", {})
            all_support.extend(levels.get("support", []))
            all_resistance.extend(levels.get("resistance", []))
        
        synthesis["support_levels"] = sorted(list(set(all_support)))
        synthesis["resistance_levels"] = sorted(list(set(all_resistance)))
        
        # Expert consensus
        expert_sentiments = []
        for opinion in results.get("expert_opinions", []):
            expert_sentiments.append(opinion.get("sentiment", "neutral"))
        
        if expert_sentiments:
            bullish_experts = sum(1 for s in expert_sentiments if s == "bullish")
            bearish_experts = sum(1 for s in expert_sentiments if s == "bearish")
            total_experts = len(expert_sentiments)
            
            if bullish_experts > bearish_experts:
                synthesis["expert_consensus"] = "bullish"
            elif bearish_experts > bullish_experts:
                synthesis["expert_consensus"] = "bearish"
            else:
                synthesis["expert_consensus"] = "mixed"
        
        # Confidence based on data richness
        confidence = 0
        if synthesis["total_sources_analyzed"] >= 5:
            confidence += 30
        elif synthesis["total_sources_analyzed"] >= 2:
            confidence += 15
        
        if len(synthesis["candlestick_patterns_detected"]) > 0:
            confidence += 20
        if len(synthesis["chart_patterns_detected"]) > 0:
            confidence += 20
        if synthesis["consensus_trend"] != "neutral" and synthesis["consensus_trend"] != "mixed":
            confidence += 15
        if synthesis["expert_consensus"] != "neutral" and synthesis["expert_consensus"] != "mixed":
            confidence += 15
        
        synthesis["confidence"] = min(confidence, 100)
        
        # Key findings
        if synthesis["candlestick_patterns_detected"]:
            top_pattern = synthesis["candlestick_patterns_detected"][0]
            synthesis["key_findings"].append(
                f"Most frequently mentioned candlestick pattern: {top_pattern['pattern']} ({top_pattern['mentions']} sources)"
            )
        
        if synthesis["chart_patterns_detected"]:
            top_chart = synthesis["chart_patterns_detected"][0]
            synthesis["key_findings"].append(
                f"Most mentioned chart pattern: {top_chart['pattern']} ({top_chart['mentions']} sources)"
            )
        
        synthesis["key_findings"].append(
            f"Web consensus trend: {synthesis['consensus_trend']} (from {total_votes} sources)"
        )
        
        if synthesis["support_levels"]:
            synthesis["key_findings"].append(
                f"Key support levels from web analysis: {', '.join(f'${s:,.2f}' for s in synthesis['support_levels'][:3])}"
            )
        
        if synthesis["resistance_levels"]:
            synthesis["key_findings"].append(
                f"Key resistance levels from web analysis: {', '.join(f'${r:,.2f}' for r in synthesis['resistance_levels'][:3])}"
            )
        
        return synthesis
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS (combines all searches)
    # =========================================================================
    
    def get_comprehensive_stock_intelligence(self, symbol: str) -> Dict:
        """
        Run all EXA searches for a stock and return comprehensive intelligence.
        This is the main entry point for the analysis pipeline.
        """
        logger.info(f"Running comprehensive EXA intelligence for {symbol}")
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "source": "EXA AI Neural Search",
            "candlestick_analysis": {},
            "market_sentiment": {},
            "insider_dark_pool": {},
            "errors": []
        }
        
        # 1. Candlestick chart analysis
        try:
            result["candlestick_analysis"] = self.find_candlestick_analysis(symbol, days_back=30)
        except Exception as e:
            result["errors"].append(f"Candlestick analysis error: {str(e)}")
            result["candlestick_analysis"] = {"error": str(e)}
        
        # 2. Market sentiment
        try:
            result["market_sentiment"] = self.find_market_sentiment(symbol)
        except Exception as e:
            result["errors"].append(f"Sentiment error: {str(e)}")
            result["market_sentiment"] = {"error": str(e)}
        
        # 3. Insider/dark pool activity
        try:
            result["insider_dark_pool"] = self.find_insider_and_dark_pool(symbol)
        except Exception as e:
            result["errors"].append(f"Insider/dark pool error: {str(e)}")
            result["insider_dark_pool"] = {"error": str(e)}
        
        return result


# Convenience function for direct use
def get_exa_analysis(symbol: str) -> Dict:
    """Quick function to get full EXA analysis for a symbol."""
    client = ExaClient()
    return client.get_comprehensive_stock_intelligence(symbol)


if __name__ == "__main__":
    import sys
    import json
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Running EXA AI analysis for {symbol}...")
    
    client = ExaClient()
    result = client.get_comprehensive_stock_intelligence(symbol)
    print(json.dumps(result, indent=2, default=str))
