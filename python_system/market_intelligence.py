"""
MARKET INTELLIGENCE MODULE
===========================
Real-time market context awareness including:
- Market open/closed detection (EST, holidays)
- Live VIX reading
- Market regime detection (bull/bear/volatile/ranging)
- Historical pattern matching
- Sentiment/catalysts/geopolitical factors
- News and social momentum

NO PLACEHOLDERS. ALL LIVE DATA.
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

# API Keys
FINNHUB_API_KEY = os.environ.get('KEY', os.environ.get('FINNHUB_API_KEY', 'd55b3ohr01qljfdeghm0d55b3ohr01qljfdeghmg'))
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY', '')


class MarketIntelligence:
    """
    Comprehensive market intelligence with live data.
    """
    
    # US Market Holidays 2024-2026 (NYSE/NASDAQ closed)
    MARKET_HOLIDAYS = {
        # 2024
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
        '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        # 2025
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
        # 2026
        '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
        '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    }
    
    # Early close days (1:00 PM EST)
    EARLY_CLOSE_DAYS = {
        '2024-07-03', '2024-11-29', '2024-12-24',
        '2025-07-03', '2025-11-28', '2025-12-24',
        '2026-07-03', '2026-11-27', '2026-12-24',
    }
    
    def __init__(self):
        self.est_tz = pytz.timezone('America/New_York')
    
    def get_current_est_time(self) -> datetime:
        """Get current time in EST/EDT"""
        return datetime.now(self.est_tz)
    
    def is_market_open(self) -> Dict:
        """
        Determine if US stock market is currently open.
        Returns detailed status including next open/close time.
        """
        now = self.get_current_est_time()
        date_str = now.strftime('%Y-%m-%d')
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        
        result = {
            'current_time_est': now.strftime('%Y-%m-%d %H:%M:%S EST'),
            'day_of_week': now.strftime('%A'),
            'date': date_str,
            'is_open': False,
            'status': 'CLOSED',
            'reason': '',
            'next_open': None,
            'next_close': None,
            'session': None
        }
        
        # Check if weekend
        if day_of_week >= 5:  # Saturday or Sunday
            result['reason'] = 'Weekend'
            result['next_open'] = self._get_next_market_open(now)
            return result
        
        # Check if holiday
        if date_str in self.MARKET_HOLIDAYS:
            result['reason'] = 'Market Holiday'
            result['next_open'] = self._get_next_market_open(now)
            return result
        
        # Regular trading hours: 9:30 AM - 4:00 PM EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check for early close
        if date_str in self.EARLY_CLOSE_DAYS:
            market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
        
        # Pre-market: 4:00 AM - 9:30 AM
        pre_market_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
        
        # After-hours: 4:00 PM - 8:00 PM
        after_hours_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        if now < pre_market_open:
            result['status'] = 'CLOSED'
            result['reason'] = 'Before pre-market'
            result['session'] = 'overnight'
            result['next_open'] = pre_market_open.strftime('%H:%M EST')
        elif now < market_open:
            result['status'] = 'PRE-MARKET'
            result['reason'] = 'Pre-market session active'
            result['session'] = 'pre-market'
            result['is_open'] = True
            result['next_close'] = market_open.strftime('%H:%M EST') + ' (regular open)'
        elif now < market_close:
            result['status'] = 'OPEN'
            result['reason'] = 'Regular trading hours'
            result['session'] = 'regular'
            result['is_open'] = True
            result['next_close'] = market_close.strftime('%H:%M EST')
        elif now < after_hours_close:
            result['status'] = 'AFTER-HOURS'
            result['reason'] = 'After-hours session active'
            result['session'] = 'after-hours'
            result['is_open'] = True
            result['next_close'] = after_hours_close.strftime('%H:%M EST')
        else:
            result['status'] = 'CLOSED'
            result['reason'] = 'After trading hours'
            result['session'] = 'closed'
            result['next_open'] = self._get_next_market_open(now)
        
        return result
    
    def _get_next_market_open(self, from_time: datetime) -> str:
        """Calculate next market open time"""
        next_day = from_time + timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Skip weekends and holidays
        while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in self.MARKET_HOLIDAYS:
            next_day += timedelta(days=1)
        
        return next_day.strftime('%A %m/%d at 9:30 AM EST')
    
    def get_live_vix(self) -> Dict:
        """
        Fetch real-time VIX (CBOE Volatility Index) from Finnhub.
        """
        try:
            # VIX is traded as ^VIX on some APIs, VIX on others
            url = f"https://finnhub.io/api/v1/quote?symbol=VIX&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('c') and data['c'] > 0:
                current = data['c']
                prev_close = data.get('pc', current)
                change = current - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                
                # VIX interpretation
                if current < 12:
                    level = 'EXTREMELY_LOW'
                    interpretation = 'Extreme complacency - potential reversal risk'
                elif current < 15:
                    level = 'LOW'
                    interpretation = 'Low fear - bullish environment'
                elif current < 20:
                    level = 'NORMAL'
                    interpretation = 'Normal volatility - balanced market'
                elif current < 25:
                    level = 'ELEVATED'
                    interpretation = 'Elevated fear - caution advised'
                elif current < 30:
                    level = 'HIGH'
                    interpretation = 'High volatility - defensive positioning'
                else:
                    level = 'EXTREME'
                    interpretation = 'Extreme fear - potential capitulation or crisis'
                
                return {
                    'success': True,
                    'vix': round(current, 2),
                    'previous_close': round(prev_close, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_pct, 2),
                    'level': level,
                    'interpretation': interpretation,
                    'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
                }
            
            # Fallback to Yahoo Finance scraping via yfinance
            try:
                import yfinance as yf
                vix = yf.Ticker('^VIX')
                hist = vix.history(period='2d')
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    
                    if current < 12:
                        level = 'EXTREMELY_LOW'
                        interpretation = 'Extreme complacency - potential reversal risk'
                    elif current < 15:
                        level = 'LOW'
                        interpretation = 'Low fear - bullish environment'
                    elif current < 20:
                        level = 'NORMAL'
                        interpretation = 'Normal volatility - balanced market'
                    elif current < 25:
                        level = 'ELEVATED'
                        interpretation = 'Elevated fear - caution advised'
                    elif current < 30:
                        level = 'HIGH'
                        interpretation = 'High volatility - defensive positioning'
                    else:
                        level = 'EXTREME'
                        interpretation = 'Extreme fear - potential capitulation or crisis'
                    
                    return {
                        'success': True,
                        'vix': round(float(current), 2),
                        'previous_close': round(float(prev_close), 2),
                        'change': round(float(change), 2),
                        'change_percent': round(float(change_pct), 2),
                        'level': level,
                        'interpretation': interpretation,
                        'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST'),
                        'source': 'yahoo_finance'
                    }
            except Exception as yf_error:
                print(f"YFinance VIX fallback failed: {yf_error}", file=sys.stderr)
            
            return {
                'success': False,
                'error': 'VIX data not available',
                'vix': None
            }
            
        except Exception as e:
            print(f"VIX fetch error: {e}", file=sys.stderr)
            return {
                'success': False,
                'error': str(e),
                'vix': None
            }
    
    def detect_market_regime(self) -> Dict:
        """
        Detect current market regime using SPY price action and technical indicators.
        Regimes: BULL, BEAR, VOLATILE, RANGING
        """
        try:
            import yfinance as yf
            import numpy as np
            
            # Get SPY data for regime detection
            spy = yf.Ticker('SPY')
            hist = spy.history(period='6mo')
            
            if hist.empty or len(hist) < 50:
                return {'success': False, 'error': 'Insufficient data for regime detection'}
            
            closes = hist['Close'].values
            current_price = closes[-1]
            
            # Calculate key indicators
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:])
            sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
            
            # Trend direction
            trend_20 = (current_price - sma_20) / sma_20 * 100
            trend_50 = (current_price - sma_50) / sma_50 * 100
            
            # Volatility (20-day standard deviation of returns)
            returns = np.diff(closes[-21:]) / closes[-21:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
            
            # Recent momentum (20-day return)
            momentum_20d = (closes[-1] / closes[-20] - 1) * 100
            
            # Regime detection logic
            regime = 'RANGING'
            confidence = 0
            characteristics = []
            
            if current_price > sma_50 and current_price > sma_200 and trend_20 > 0:
                regime = 'BULL'
                confidence = min(90, 50 + abs(trend_20) * 5 + abs(trend_50) * 3)
                characteristics = [
                    f'Price above 50-day SMA ({sma_50:.2f})',
                    f'Price above 200-day SMA ({sma_200:.2f})',
                    f'20-day momentum: +{momentum_20d:.1f}%'
                ]
            elif current_price < sma_50 and current_price < sma_200 and trend_20 < 0:
                regime = 'BEAR'
                confidence = min(90, 50 + abs(trend_20) * 5 + abs(trend_50) * 3)
                characteristics = [
                    f'Price below 50-day SMA ({sma_50:.2f})',
                    f'Price below 200-day SMA ({sma_200:.2f})',
                    f'20-day momentum: {momentum_20d:.1f}%'
                ]
            elif volatility > 25:
                regime = 'VOLATILE'
                confidence = min(90, 50 + (volatility - 20) * 2)
                characteristics = [
                    f'Annualized volatility: {volatility:.1f}%',
                    'High intraday swings expected',
                    'Options premiums elevated'
                ]
            else:
                regime = 'RANGING'
                confidence = 60
                characteristics = [
                    'No clear directional trend',
                    f'Trading between {min(closes[-20:]):.2f} - {max(closes[-20:]):.2f}',
                    'Mean reversion strategies favored'
                ]
            
            return {
                'success': True,
                'regime': regime,
                'confidence': round(confidence, 1),
                'spy_price': round(current_price, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'sma_200': round(sma_200, 2),
                'volatility_annualized': round(volatility, 1),
                'momentum_20d': round(momentum_20d, 2),
                'characteristics': characteristics,
                'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
            }
            
        except Exception as e:
            print(f"Regime detection error: {e}", file=sys.stderr)
            return {'success': False, 'error': str(e)}
    
    def get_market_sentiment(self) -> Dict:
        """
        Get market sentiment from multiple sources using Perplexity for real-time analysis.
        """
        try:
            if not PERPLEXITY_API_KEY:
                # Fallback to Finnhub market news
                return self._get_finnhub_sentiment()
            
            # Use Perplexity Sonar for real-time market sentiment
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst. Provide a brief, factual summary of current market conditions. Be concise and data-driven."
                    },
                    {
                        "role": "user",
                        "content": """Analyze current US stock market conditions as of today. Include:
1. Overall market sentiment (bullish/bearish/neutral) with confidence
2. Top 3 market-moving catalysts right now
3. Key geopolitical factors affecting markets
4. Any significant economic data releases today
5. Social/retail sentiment trends

Format as JSON with keys: sentiment, confidence, catalysts (array), geopolitical (array), economic_data (array), social_sentiment"""
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                
                # Try to parse as JSON
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        sentiment_data = json.loads(json_match.group())
                        sentiment_data['success'] = True
                        sentiment_data['source'] = 'perplexity_sonar'
                        sentiment_data['timestamp'] = datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
                        return sentiment_data
                except json.JSONDecodeError:
                    pass
                
                # Return raw content if JSON parsing fails
                return {
                    'success': True,
                    'sentiment': 'See analysis',
                    'analysis': content,
                    'source': 'perplexity_sonar',
                    'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
                }
            
            return self._get_finnhub_sentiment()
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}", file=sys.stderr)
            return self._get_finnhub_sentiment()
    
    def _get_finnhub_sentiment(self) -> Dict:
        """Fallback sentiment from Finnhub market news"""
        try:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=10)
            news = response.json()
            
            if news and len(news) > 0:
                # Get top 5 headlines
                headlines = [n.get('headline', '')[:100] for n in news[:5]]
                
                return {
                    'success': True,
                    'sentiment': 'See headlines',
                    'catalysts': headlines,
                    'source': 'finnhub_news',
                    'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
                }
            
            return {
                'success': False,
                'error': 'No news data available'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def match_historical_patterns(self) -> Dict:
        """
        Match current market conditions to historical patterns.
        """
        try:
            import yfinance as yf
            import numpy as np
            
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1y')
            
            if hist.empty or len(hist) < 50:
                return {'success': False, 'error': 'Insufficient data'}
            
            closes = hist['Close'].values
            
            # Current 20-day pattern
            current_pattern = closes[-20:]
            current_normalized = (current_pattern - np.mean(current_pattern)) / np.std(current_pattern)
            
            # Find similar patterns in history
            matches = []
            for i in range(50, len(closes) - 40):
                historical_pattern = closes[i:i+20]
                historical_normalized = (historical_pattern - np.mean(historical_pattern)) / np.std(historical_pattern)
                
                # Correlation as similarity measure
                correlation = np.corrcoef(current_normalized, historical_normalized)[0, 1]
                
                if correlation > 0.8:  # Strong correlation threshold
                    # What happened after this pattern?
                    future_return = (closes[i+40] / closes[i+20] - 1) * 100 if i + 40 < len(closes) else None
                    
                    if future_return is not None:
                        matches.append({
                            'date': hist.index[i+20].strftime('%Y-%m-%d'),
                            'correlation': round(correlation, 3),
                            'subsequent_20d_return': round(future_return, 2)
                        })
            
            if matches:
                # Sort by correlation
                matches.sort(key=lambda x: x['correlation'], reverse=True)
                top_matches = matches[:5]
                
                # Average subsequent return
                avg_return = np.mean([m['subsequent_20d_return'] for m in top_matches])
                bullish_count = sum(1 for m in top_matches if m['subsequent_20d_return'] > 0)
                
                return {
                    'success': True,
                    'patterns_found': len(matches),
                    'top_matches': top_matches,
                    'average_subsequent_return': round(avg_return, 2),
                    'bullish_probability': round(bullish_count / len(top_matches) * 100, 1),
                    'interpretation': f"Based on {len(matches)} similar patterns, {bullish_count}/{len(top_matches)} were followed by gains",
                    'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
                }
            
            return {
                'success': True,
                'patterns_found': 0,
                'interpretation': 'No strongly matching historical patterns found',
                'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST')
            }
            
        except Exception as e:
            print(f"Pattern matching error: {e}", file=sys.stderr)
            return {'success': False, 'error': str(e)}
    
    def get_full_market_intelligence(self) -> Dict:
        """
        Comprehensive market intelligence combining all sources.
        """
        result = {
            'timestamp': datetime.now(self.est_tz).strftime('%Y-%m-%d %H:%M:%S EST'),
            'market_status': self.is_market_open(),
            'vix': self.get_live_vix(),
            'regime': self.detect_market_regime(),
            'sentiment': self.get_market_sentiment(),
            'patterns': self.match_historical_patterns()
        }
        
        # Overall assessment
        vix_level = result['vix'].get('level', 'UNKNOWN') if result['vix'].get('success') else 'UNKNOWN'
        regime = result['regime'].get('regime', 'UNKNOWN') if result['regime'].get('success') else 'UNKNOWN'
        
        # Trading conditions assessment
        if regime == 'BULL' and vix_level in ['LOW', 'NORMAL']:
            conditions = 'FAVORABLE'
            recommendation = 'Conditions favor long positions and momentum strategies'
        elif regime == 'BEAR' or vix_level in ['HIGH', 'EXTREME']:
            conditions = 'CAUTIOUS'
            recommendation = 'Elevated risk - consider defensive positioning or reduced size'
        elif regime == 'VOLATILE':
            conditions = 'VOLATILE'
            recommendation = 'High volatility - options strategies may be advantageous'
        else:
            conditions = 'NEUTRAL'
            recommendation = 'Mixed signals - selective stock picking recommended'
        
        result['overall'] = {
            'conditions': conditions,
            'recommendation': recommendation
        }
        
        return result


def run_market_intelligence():
    """Entry point for running market intelligence from command line"""
    mi = MarketIntelligence()
    result = mi.get_full_market_intelligence()
    print(json.dumps(result, default=str))


if __name__ == '__main__':
    run_market_intelligence()
