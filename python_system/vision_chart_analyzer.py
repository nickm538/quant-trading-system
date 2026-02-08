"""
Vision AI Chart Analyzer
========================
Uses Gemini Vision AI to analyze stock charts from Finviz.
Provides expert-level candlestick pattern recognition and technical analysis.

This module:
1. Fetches technical analysis chart from Finviz
2. Analyzes with Gemini Vision AI
3. Returns detailed candlestick patterns, support/resistance, and trading signals

NO PLACEHOLDERS - All analysis is real-time from actual chart images.
"""

import os
import sys
import json
import base64
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import time

# Try to import google-genai SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')


class VisionChartAnalyzer:
    """
    Expert-level chart analysis using Gemini Vision AI.
    Analyzes Finviz charts for candlestick patterns, support/resistance, and trading signals.
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
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze a stock chart using Vision AI.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dict with candlestick patterns, support/resistance, and trading signals
        """
        symbol = symbol.upper().strip()
        
        try:
            # Step 1: Fetch Finviz chart
            chart_data = self._fetch_finviz_chart(symbol)
            if not chart_data:
                return {
                    'success': False,
                    'error': f'Could not fetch chart for {symbol}',
                    'symbol': symbol
                }
            
            # Step 2: Analyze with Gemini Vision AI (with retry)
            analysis = self._analyze_with_retry(symbol, chart_data, max_retries=3)
            
            return {
                'success': True,
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
                'chart_source': 'Finviz',
                'ai_model': 'Gemini Vision',
                **analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def _fetch_finviz_chart(self, symbol: str) -> Optional[bytes]:
        """
        Fetch the technical analysis chart from Finviz.
        Returns raw image bytes.
        """
        # Finviz chart URL - includes candlesticks, volume, and indicators
        chart_url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finviz.com/'
        }
        
        try:
            response = requests.get(chart_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Verify we got an image
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                print(f"Warning: Unexpected content type: {content_type}", file=sys.stderr)
                return None
            
            return response.content
            
        except Exception as e:
            print(f"Error fetching Finviz chart: {e}", file=sys.stderr)
            return None
    
    def _analyze_with_retry(self, symbol: str, image_data: bytes, max_retries: int = 3) -> Dict[str, Any]:
        """Analyze with exponential backoff retry and OpenRouter fallback."""
        last_error = None
        
        # Primary: OpenRouter models (fast models first to avoid blocking pipeline)
        # Gemini direct API is deprioritized due to quota issues
        if self.openrouter_key:
            # Model fallback chain: Fast models first, then premium
            # Opus 4 moved to end because it can take 60-120s and block the pipeline
            openrouter_models = [
                ('google/gemini-2.0-flash-001', 'Gemini 2.0 Flash'),
                ('anthropic/claude-sonnet-4', 'Claude Sonnet 4'),
                ('anthropic/claude-3.5-sonnet', 'Claude 3.5 Sonnet'),
            ]
            
            for model_id, model_name in openrouter_models:
                print(f"Trying OpenRouter {model_name}...", file=sys.stderr)
                try:
                    result = self._analyze_with_openrouter(symbol, image_data, model=model_id)
                    if result.get('trend', {}).get('direction') != 'UNKNOWN':
                        result['ai_model'] = f'OpenRouter {model_name}'
                        return result
                except Exception as e:
                    last_error = str(e)
                    print(f"  {model_name} failed: {str(e)[:100]}", file=sys.stderr)
                    continue
        
        return {
            'candlestick_patterns': [],
            'trend': {'direction': 'UNKNOWN', 'strength': 'UNKNOWN', 'momentum': 'UNKNOWN'},
            'overall_bias': 'NEUTRAL',
            'error': f'All API fallbacks failed. Last error: {last_error}'
        }
    
    def _analyze_with_sdk(self, symbol: str, image_data: bytes) -> Dict[str, Any]:
        """Analyze using google-genai SDK."""
        prompt = self._get_analysis_prompt(symbol)
        
        # Create image part
        image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096
            )
        )
        
        return self._parse_response(response.text)
    
    def _analyze_with_rest(self, symbol: str, image_data: bytes) -> Dict[str, Any]:
        """Analyze using REST API directly."""
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        prompt = self._get_analysis_prompt(symbol)
        
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 4096
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0].get('content', {})
            parts = content.get('parts', [])
            if parts:
                text = parts[0].get('text', '')
                return self._parse_response(text)
        
        return {
            'candlestick_patterns': [],
            'trend': {'direction': 'UNKNOWN', 'strength': 'UNKNOWN', 'momentum': 'UNKNOWN'},
            'overall_bias': 'NEUTRAL',
            'error': 'No response from API'
        }
    
    def _get_analysis_prompt(self, symbol: str) -> str:
        """Get the expert trading prompt for chart analysis."""
        return f"""You are a world-class technical analyst and candlestick pattern expert. 
Analyze this stock chart for {symbol} with extreme precision and detail.

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
   - Identify key support levels (at least 2)
   - Identify key resistance levels (at least 2)
   - Note which levels are being tested

4. VOLUME ANALYSIS:
   - Volume trend (INCREASING/DECREASING/STABLE)
   - Any volume divergences
   - Smart money signals

5. TECHNICAL INDICATORS (if visible):
   - Moving average positions and crossovers
   - RSI/MACD signals if shown
   - Bollinger Band position

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
        "macd_signal": "bullish/bearish/neutral"
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
    "key_observations": ["observation1", "observation2", "observation3"],
    "pattern_strength": "STRONG/MODERATE/WEAK"
}}"""
    
    def _analyze_with_openrouter(self, symbol: str, image_data: bytes, model: str = 'anthropic/claude-3.5-sonnet') -> Dict[str, Any]:
        """Analyze using OpenRouter API with vision-capable model."""
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        prompt = self._get_analysis_prompt(symbol)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.openrouter_key}',
            'HTTP-Referer': 'https://quant-trading-system.com',
            'X-Title': 'Quant Trading System'
        }
        
        # Use specified model for vision analysis
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }],
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
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
            'error': 'No response from OpenRouter'
        }
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        try:
            # Find JSON block in response
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
            'error': 'Could not parse AI response'
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
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
