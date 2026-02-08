"""
Vision AI Chart Analyzer — Multi-Timeframe Edition
====================================================
Uses Gemini/Claude Vision AI to analyze stock charts at MULTIPLE timeframes:
  1. Daily chart from Finviz (swing/position trading view)
  2. 5-minute intraday chart from Polygon.io (day trading / precision view)
  3. Weekly chart from Finviz (macro trend context)

Each timeframe gets its own dedicated analysis, then a synthesis merges them
into a unified multi-timeframe signal with conflict detection.

NO PLACEHOLDERS — All analysis is real-time from actual chart images.
"""

import os
import sys
import json
import base64
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import time

logger = logging.getLogger(__name__)

# Try to import google-genai SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Try to import intraday chart renderer
try:
    from intraday_chart_renderer import IntradayChartRenderer
    HAS_INTRADAY = True
except ImportError:
    HAS_INTRADAY = False

# API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')


class VisionChartAnalyzer:
    """
    Expert-level chart analysis using Vision AI across multiple timeframes.
    Analyzes Finviz daily/weekly charts + Polygon.io intraday charts.
    """

    def __init__(self):
        self.gemini_key = GEMINI_API_KEY
        self.openrouter_key = OPENROUTER_API_KEY

        if not self.gemini_key and not self.openrouter_key:
            raise ValueError("Either GEMINI_API_KEY or OPENROUTER_API_KEY must be set")

        # Initialize Gemini client if SDK available
        if HAS_GENAI and self.gemini_key:
            self.client = genai.Client(api_key=self.gemini_key)
        else:
            self.client = None

        # Initialize intraday renderer
        self.intraday_renderer = None
        if HAS_INTRADAY:
            try:
                self.intraday_renderer = IntradayChartRenderer()
            except Exception as e:
                logger.warning(f"Intraday chart renderer not available: {e}")

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Multi-timeframe chart analysis using Vision AI.

        Analyzes:
        1. Finviz daily chart (primary — swing/position view)
        2. Polygon 5-minute intraday chart (precision — day trading view)
        3. Finviz weekly chart (context — macro trend)

        Returns unified analysis with per-timeframe breakdowns.
        """
        symbol = symbol.upper().strip()

        try:
            # ── Step 1: Fetch all chart images ──────────────────────────
            charts = {}

            # Daily chart (Finviz) — always the primary
            daily_data = self._fetch_finviz_chart(symbol, timeframe='d')
            if daily_data:
                charts['daily'] = {'data': daily_data, 'source': 'Finviz', 'label': 'Daily'}

            # Weekly chart (Finviz) — macro context
            weekly_data = self._fetch_finviz_chart(symbol, timeframe='w')
            if weekly_data:
                charts['weekly'] = {'data': weekly_data, 'source': 'Finviz', 'label': 'Weekly'}

            # 5-minute intraday chart (Polygon.io) — precision
            intraday_data = None
            intraday_summary = None
            if self.intraday_renderer:
                try:
                    intraday_data = self.intraday_renderer.render_chart(symbol, '5min')
                    intraday_summary = self.intraday_renderer.get_intraday_summary(symbol, '5min')
                    if intraday_data:
                        charts['intraday_5min'] = {
                            'data': intraday_data,
                            'source': 'Polygon.io',
                            'label': '5-Minute Intraday',
                        }
                except Exception as e:
                    logger.warning(f"Intraday chart failed for {symbol}: {e}")

            if not charts:
                return {
                    'success': False,
                    'error': f'Could not fetch any charts for {symbol}',
                    'symbol': symbol,
                }

            # ── Step 2: Analyze each chart with Vision AI ───────────────
            analyses = {}

            # Primary: Daily chart analysis (always runs)
            if 'daily' in charts:
                daily_analysis = self._analyze_single_chart(
                    symbol, charts['daily']['data'], 'daily',
                    f"This is a DAILY candlestick chart for {symbol} from Finviz. "
                    f"Each candle represents ONE TRADING DAY. "
                    f"Analyze the last 10-20 daily candles with extreme precision."
                )
                if daily_analysis:
                    analyses['daily'] = daily_analysis

            # Intraday: 5-minute chart analysis (if available)
            if 'intraday_5min' in charts:
                intraday_context = (
                    f"This is a 5-MINUTE intraday candlestick chart for {symbol} from Polygon.io. "
                    f"Each candle represents FIVE MINUTES of trading. "
                    f"The chart shows EMA 9 (orange), EMA 21 (teal), and VWAP (yellow dashed). "
                )
                if intraday_summary:
                    intraday_context += (
                        f"Current price: ${intraday_summary['current_price']}, "
                        f"VWAP: ${intraday_summary['vwap']}, "
                        f"EMA 9: ${intraday_summary['ema_9']}, EMA 21: ${intraday_summary['ema_21']}, "
                        f"Intraday range: {intraday_summary['range_pct']}%, "
                        f"Session change: {intraday_summary['change_pct']}%. "
                    )
                intraday_context += (
                    "Focus on the LAST 15-30 candles (most recent 1-2 hours). "
                    "Identify micro-patterns, intraday support/resistance, and momentum shifts."
                )
                intraday_analysis = self._analyze_single_chart(
                    symbol, charts['intraday_5min']['data'], 'intraday_5min', intraday_context
                )
                if intraday_analysis:
                    analyses['intraday_5min'] = intraday_analysis

            # Weekly: Macro trend context (if available, but don't block on it)
            if 'weekly' in charts and len(analyses) < 2:
                # Only analyze weekly if we don't already have 2 timeframes
                # (to save API calls and time)
                weekly_analysis = self._analyze_single_chart(
                    symbol, charts['weekly']['data'], 'weekly',
                    f"This is a WEEKLY candlestick chart for {symbol} from Finviz. "
                    f"Each candle represents ONE WEEK. "
                    f"Focus on the macro trend, major support/resistance, and long-term momentum."
                )
                if weekly_analysis:
                    analyses['weekly'] = weekly_analysis

            if not analyses:
                return {
                    'success': False,
                    'error': 'Vision AI could not analyze any charts',
                    'symbol': symbol,
                }

            # ── Step 3: Build unified response ──────────────────────────
            # Use daily as the primary analysis (backward compatible)
            primary = analyses.get('daily', analyses.get('intraday_5min', {}))

            result = {
                'success': True,
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
                'chart_source': 'Finviz + Polygon.io',
                'ai_model': primary.get('ai_model', 'Vision AI'),
                'timeframes_analyzed': list(analyses.keys()),
                # Primary fields (backward compatible with old single-chart format)
                'candlestick_patterns': primary.get('candlestick_patterns', []),
                'trend': primary.get('trend', {}),
                'support_levels': primary.get('support_levels', []),
                'resistance_levels': primary.get('resistance_levels', []),
                'volume_analysis': primary.get('volume_analysis', {}),
                'indicators': primary.get('indicators', {}),
                'recommendation': primary.get('recommendation', {}),
                'overall_bias': primary.get('overall_bias', 'NEUTRAL'),
                'key_observations': primary.get('key_observations', []),
                'pattern_strength': primary.get('pattern_strength', 'UNKNOWN'),
            }

            # Add per-timeframe breakdowns
            result['multi_timeframe'] = {}
            for tf_key, tf_analysis in analyses.items():
                result['multi_timeframe'][tf_key] = {
                    'source': charts.get(tf_key, {}).get('source', 'Unknown'),
                    'label': charts.get(tf_key, {}).get('label', tf_key),
                    'overall_bias': tf_analysis.get('overall_bias', 'NEUTRAL'),
                    'trend': tf_analysis.get('trend', {}),
                    'candlestick_patterns': tf_analysis.get('candlestick_patterns', []),
                    'recommendation': tf_analysis.get('recommendation', {}),
                    'support_levels': tf_analysis.get('support_levels', []),
                    'resistance_levels': tf_analysis.get('resistance_levels', []),
                    'key_observations': tf_analysis.get('key_observations', []),
                }

            # Add intraday summary data if available
            if intraday_summary:
                result['intraday_summary'] = intraday_summary

            # Multi-timeframe alignment check
            biases = {k: v.get('overall_bias', 'NEUTRAL').upper() for k, v in analyses.items()}
            bullish_count = sum(1 for b in biases.values() if 'BULL' in b)
            bearish_count = sum(1 for b in biases.values() if 'BEAR' in b)
            total = len(biases)

            if total > 1:
                if bullish_count == total:
                    result['timeframe_alignment'] = 'ALL_BULLISH'
                    result['timeframe_alignment_note'] = f'All {total} timeframes agree: BULLISH'
                elif bearish_count == total:
                    result['timeframe_alignment'] = 'ALL_BEARISH'
                    result['timeframe_alignment_note'] = f'All {total} timeframes agree: BEARISH'
                elif bullish_count > 0 and bearish_count > 0:
                    result['timeframe_alignment'] = 'CONFLICTING'
                    conflict_details = ', '.join(f'{k}={v}' for k, v in biases.items())
                    result['timeframe_alignment_note'] = (
                        f'Timeframe conflict detected: {conflict_details}. '
                        f'This means the short-term and long-term trends disagree — '
                        f'proceed with caution and use the higher timeframe as the dominant signal.'
                    )
                else:
                    result['timeframe_alignment'] = 'MIXED'
                    result['timeframe_alignment_note'] = f'Mixed signals: {biases}'

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
            }

    def _fetch_finviz_chart(self, symbol: str, timeframe: str = 'd') -> Optional[bytes]:
        """
        Fetch a technical analysis chart from Finviz.
        timeframe: 'd' (daily), 'w' (weekly), 'm' (monthly)
        Returns raw image bytes.
        """
        chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p={timeframe}&s=l"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finviz.com/',
        }

        try:
            response = requests.get(chart_url, headers=headers, timeout=15)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                logger.warning(f"Unexpected content type for {symbol} ({timeframe}): {content_type}")
                return None

            # Reject tiny images (paywall/error placeholders)
            if len(response.content) < 10000:
                logger.warning(f"Chart too small for {symbol} ({timeframe}): {len(response.content)} bytes — likely paywall")
                return None

            return response.content

        except Exception as e:
            logger.error(f"Error fetching Finviz chart ({timeframe}): {e}")
            return None

    def _analyze_single_chart(
        self, symbol: str, image_data: bytes, timeframe_key: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single chart image with Vision AI."""
        prompt = self._get_analysis_prompt(symbol, context)
        return self._analyze_with_retry(symbol, image_data, prompt, max_retries=2)

    def _analyze_with_retry(
        self, symbol: str, image_data: bytes, prompt: str, max_retries: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Analyze with model fallback chain."""
        last_error = None

        if self.openrouter_key:
            openrouter_models = [
                ('google/gemini-2.0-flash-001', 'Gemini 2.0 Flash'),
                ('anthropic/claude-sonnet-4', 'Claude Sonnet 4'),
                ('anthropic/claude-3.5-sonnet', 'Claude 3.5 Sonnet'),
            ]

            for model_id, model_name in openrouter_models:
                print(f"  Vision AI: Trying {model_name}...", file=sys.stderr)
                try:
                    result = self._analyze_with_openrouter(symbol, image_data, prompt, model=model_id)
                    if result and result.get('trend', {}).get('direction') != 'UNKNOWN':
                        result['ai_model'] = f'OpenRouter {model_name}'
                        return result
                except Exception as e:
                    last_error = str(e)
                    print(f"    {model_name} failed: {str(e)[:100]}", file=sys.stderr)
                    continue

        # Fallback: Gemini direct API
        if self.gemini_key and self.client:
            try:
                result = self._analyze_with_sdk(symbol, image_data, prompt)
                if result and result.get('trend', {}).get('direction') != 'UNKNOWN':
                    result['ai_model'] = 'Gemini 2.0 Flash (Direct)'
                    return result
            except Exception as e:
                last_error = str(e)

        logger.warning(f"All Vision AI models failed for {symbol}. Last error: {last_error}")
        return None

    def _analyze_with_sdk(self, symbol: str, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """Analyze using google-genai SDK."""
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        return self._parse_response(response.text)

    def _analyze_with_openrouter(
        self, symbol: str, image_data: bytes, prompt: str, model: str = 'google/gemini-2.0-flash-001'
    ) -> Dict[str, Any]:
        """Analyze using OpenRouter API with vision-capable model."""
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openrouter_key}',
            'HTTP-Referer': 'https://quant-trading-system.com',
            'X-Title': 'Quant Trading System',
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()

        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            text = result['choices'][0].get('message', {}).get('content', '')
            return self._parse_response(text)

        return {
            'candlestick_patterns': [],
            'trend': {'direction': 'UNKNOWN', 'strength': 'UNKNOWN', 'momentum': 'UNKNOWN'},
            'overall_bias': 'NEUTRAL',
            'error': 'No response from OpenRouter',
        }

    def _get_analysis_prompt(self, symbol: str, context: str = '') -> str:
        """Get the expert trading prompt for chart analysis."""
        return f"""You are a world-class technical analyst and candlestick pattern expert.
Analyze this stock chart for {symbol} with extreme precision and detail.

CHART CONTEXT: {context}

CRITICAL INSTRUCTIONS:
- Read the EXACT prices from the Y-axis. Do NOT guess or round.
- Count the EXACT number of candles. Each candle's color (green=up, red=down) matters.
- Read the EXACT moving average values from the legend/labels on the chart.
- If you see volume bars at the bottom, analyze the volume pattern precisely.
- Identify support/resistance by finding price levels where candles repeatedly reverse.

REQUIRED ANALYSIS (provide ALL of these):

1. CANDLESTICK PATTERNS DETECTED:
   - List ALL candlestick patterns visible in the last 10-20 candles
   - For each pattern: name, signal (bullish/bearish/neutral), reliability (high/moderate/low)
   - Include: Doji, Hammer, Engulfing, Morning/Evening Star, Three White Soldiers, etc.

2. CURRENT TREND:
   - Primary trend direction (UPTREND/DOWNTREND/SIDEWAYS)
   - Trend strength (STRONG/MODERATE/WEAK)
   - Recent momentum (ACCELERATING/DECELERATING/STABLE)

3. SUPPORT & RESISTANCE LEVELS:
   - Identify key support levels (at least 2) — read exact prices from chart
   - Identify key resistance levels (at least 2) — read exact prices from chart
   - Note which levels are being tested

4. VOLUME ANALYSIS:
   - Volume trend (INCREASING/DECREASING/STABLE)
   - Any volume divergences
   - Smart money signals

5. TECHNICAL INDICATORS (if visible):
   - Moving average positions and crossovers — read exact MA values
   - RSI/MACD signals if shown
   - VWAP position relative to price (if shown)

6. TRADING RECOMMENDATION:
   - Signal: BUY/SELL/HOLD
   - Confidence: 0-100%
   - Entry zone (price range)
   - Stop loss level
   - Target price(s)
   - Risk/Reward ratio

7. OVERALL BIAS:
   - BULLISH/BEARISH/NEUTRAL
   - Key reasons (3-5 bullet points)

Respond in this exact JSON format:
{{
    "candlestick_patterns": [
        {{"name": "pattern_name", "signal": "bullish/bearish/neutral", "reliability": "high/moderate/low", "description": "brief description"}}
    ],
    "trend": {{
        "direction": "UPTREND/DOWNTREND/SIDEWAYS",
        "strength": "STRONG/MODERATE/WEAK",
        "momentum": "ACCELERATING/DECELERATING/STABLE"
    }},
    "support_levels": [price1, price2],
    "resistance_levels": [price1, price2],
    "volume_analysis": {{
        "trend": "INCREASING/DECREASING/STABLE",
        "divergence": "none/bullish/bearish",
        "smart_money_signal": "accumulation/distribution/neutral"
    }},
    "indicators": {{
        "ma_signal": "bullish/bearish/neutral",
        "rsi_signal": "overbought/oversold/neutral",
        "macd_signal": "bullish/bearish/neutral",
        "vwap_signal": "above/below/at"
    }},
    "recommendation": {{
        "signal": "BUY/SELL/HOLD",
        "confidence": 75,
        "entry_zone": [low_price, high_price],
        "stop_loss": price,
        "targets": [target1, target2],
        "risk_reward": "1:2"
    }},
    "overall_bias": "BULLISH/BEARISH/NEUTRAL",
    "key_observations": ["observation 1", "observation 2", "observation 3"],
    "pattern_strength": "STRONG/MODERATE/WEAK"
}}"""

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        return {
            'candlestick_patterns': [],
            'trend': {'direction': 'UNKNOWN', 'strength': 'UNKNOWN', 'momentum': 'UNKNOWN'},
            'overall_bias': 'NEUTRAL',
            'error': 'Could not parse AI response',
        }


def main():
    """Test the Vision Chart Analyzer."""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python vision_chart_analyzer.py <symbol>"}))
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        analyzer = VisionChartAnalyzer()
        result = analyzer.analyze(symbol)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
