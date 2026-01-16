"""
DARK POOL SCANNER
==================
Free dark pool and off-exchange trading analysis.

Data Sources:
1. FINRA Short Volume Data (daily)
2. Stockgrid.io Dark Pool API (end-of-day)
3. Yahoo Finance for price/volume context

Features:
- Dark pool volume tracking
- Short volume ratio analysis
- Net dark pool position
- Buy/Sell classification using tick rule
- Unusual dark pool activity detection

For integration with Streamlit dashboard.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import io
import yfinance as yf


class DarkPoolScanner:
    """
    Dark Pool Scanner - Free Implementation
    
    Combines multiple free data sources to provide dark pool insights
    similar to paid services like Unusual Whales.
    """
    
    def __init__(self):
        """Initialize Dark Pool Scanner."""
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.stockgrid_url = "https://www.stockgrid.io/get_dark_pool_data"
        self.finra_base_url = "https://cdn.finra.org/equity/regsho/daily"
    
    def get_dark_pool_analysis(self, ticker: str) -> Dict:
        """
        Get comprehensive dark pool analysis for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with dark pool metrics and analysis
        """
        try:
            # Check cache
            cache_key = f"{ticker}_darkpool"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            # Fetch data from multiple sources
            stockgrid_data = self._fetch_stockgrid_data(ticker)
            finra_data = self._fetch_finra_short_volume(ticker)
            price_data = self._fetch_price_context(ticker)
            
            # Combine and analyze
            result = self._analyze_dark_pool(ticker, stockgrid_data, finra_data, price_data)
            
            # Cache result
            self.cache[cache_key] = (time.time(), result)
            
            return result
            
        except Exception as e:
            return self._empty_result(ticker, str(e))
    
    def _fetch_stockgrid_data(self, ticker: str) -> Dict:
        """
        Fetch dark pool data from Stockgrid.io.
        
        Returns net dark pool position and recent activity.
        """
        try:
            # Stockgrid provides top dark pool positions
            url = f"{self.stockgrid_url}?top=Dark%20Pools%20Position%20$&minmax=desc"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Data is nested under 'data' key
                data = response_data.get('data', [])
                
                # Find the ticker in the data
                for item in data:
                    if item.get('Ticker', '').upper() == ticker.upper():
                        return {
                            'found': True,
                        'net_position': item.get('Dark Pools Position', 0),
                        'net_position_dollar': item.get('Dark Pools Position $', 0),
                        'dp_volume': item.get('Dark Pools Position', 0),
                        'dp_volume_dollar': item.get('Dark Pools Position $', 0),
                        'short_volume': item.get('Short Volume', 0),
                        'short_volume_pct': item.get('Short Volume %', 0),
                            'date': item.get('Date', ''),
                            'source': 'stockgrid.io'
                        }
                
                return {'found': False, 'source': 'stockgrid.io'}
            
            return {'found': False, 'error': f'HTTP {response.status_code}'}
            
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def _fetch_finra_short_volume(self, ticker: str) -> Dict:
        """
        Fetch short volume data from FINRA.
        
        FINRA provides daily short sale volume for all securities.
        """
        try:
            # Try to get recent FINRA short volume data
            # FINRA files are named by date: CNMSshvol20251220.txt
            
            today = datetime.now()
            
            # Try last 5 business days
            for days_back in range(7):
                check_date = today - timedelta(days=days_back)
                
                # Skip weekends
                if check_date.weekday() >= 5:
                    continue
                
                date_str = check_date.strftime('%Y%m%d')
                url = f"{self.finra_base_url}/CNMSshvol{date_str}.txt"
                
                try:
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        # Parse the pipe-delimited file
                        df = pd.read_csv(io.StringIO(response.text), delimiter='|')
                        
                        # Find the ticker
                        ticker_data = df[df['Symbol'].str.upper() == ticker.upper()]
                        
                        if not ticker_data.empty:
                            row = ticker_data.iloc[0]
                            
                            short_volume = row.get('ShortVolume', 0)
                            short_exempt = row.get('ShortExemptVolume', 0)
                            total_volume = row.get('TotalVolume', 0)
                            
                            # Calculate short ratio
                            short_ratio = (short_volume / total_volume * 100) if total_volume > 0 else 0
                            
                            return {
                                'found': True,
                                'date': check_date.strftime('%Y-%m-%d'),
                                'short_volume': int(short_volume),
                                'short_exempt_volume': int(short_exempt),
                                'total_volume': int(total_volume),
                                'short_ratio': round(short_ratio, 2),
                                'market': row.get('Market', 'Unknown'),
                                'source': 'FINRA'
                            }
                
                except Exception:
                    continue
            
            return {'found': False, 'source': 'FINRA'}
            
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def _fetch_price_context(self, ticker: str) -> Dict:
        """
        Fetch price and volume context from Yahoo Finance.
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get recent history
            hist = stock.history(period='5d')
            
            if hist.empty:
                return {'found': False}
            
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            
            # Calculate metrics
            price_change = latest['Close'] - prev['Close']
            price_change_pct = (price_change / prev['Close'] * 100) if prev['Close'] > 0 else 0
            
            # Volume analysis
            avg_volume = hist['Volume'].mean()
            volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
            
            return {
                'found': True,
                'current_price': round(latest['Close'], 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'volume': int(latest['Volume']),
                'avg_volume': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'high': round(latest['High'], 2),
                'low': round(latest['Low'], 2)
            }
            
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def _analyze_dark_pool(self, ticker: str, stockgrid: Dict, finra: Dict, price: Dict) -> Dict:
        """
        Combine all data sources and perform analysis.
        """
        # Initialize result
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'error': None,
            
            # Dark Pool Metrics
            'has_dark_pool_data': False,
            'net_dp_position': 0,
            'net_dp_position_dollar': 0,
            'dp_volume': 0,
            'dp_sentiment': 'UNKNOWN',
            'dp_sentiment_color': '#9E9E9E',
            
            # Short Volume Metrics
            'has_short_data': False,
            'short_volume': 0,
            'short_ratio': 0,
            'short_sentiment': 'UNKNOWN',
            
            # Price Context
            'current_price': 0,
            'price_change_pct': 0,
            'volume_ratio': 1.0,
            
            # Combined Analysis
            'overall_sentiment': 'NEUTRAL',
            'overall_score': 50,
            'signals': []
        }
        
        signals = []
        score_adjustments = []
        
        # Process Stockgrid data
        if stockgrid.get('found'):
            result['has_dark_pool_data'] = True
            result['net_dp_position'] = stockgrid.get('net_position', 0)
            result['net_dp_position_dollar'] = stockgrid.get('net_position_dollar', 0)
            result['dp_volume'] = stockgrid.get('dp_volume', 0)
            result['dp_date'] = stockgrid.get('date', '')
            
            # Analyze dark pool sentiment
            net_pos = result['net_dp_position']
            if net_pos > 1000000:
                result['dp_sentiment'] = 'VERY_BULLISH'
                result['dp_sentiment_color'] = '#00c851'
                signals.append('🟢 Large NET BUYING in dark pools')
                score_adjustments.append(25)
            elif net_pos > 100000:
                result['dp_sentiment'] = 'BULLISH'
                result['dp_sentiment_color'] = '#4CAF50'
                signals.append('🟢 Net buying in dark pools')
                score_adjustments.append(15)
            elif net_pos < -1000000:
                result['dp_sentiment'] = 'VERY_BEARISH'
                result['dp_sentiment_color'] = '#F44336'
                signals.append('🔴 Large NET SELLING in dark pools')
                score_adjustments.append(-25)
            elif net_pos < -100000:
                result['dp_sentiment'] = 'BEARISH'
                result['dp_sentiment_color'] = '#FF5722'
                signals.append('🔴 Net selling in dark pools')
                score_adjustments.append(-15)
            else:
                result['dp_sentiment'] = 'NEUTRAL'
                result['dp_sentiment_color'] = '#9E9E9E'
        
        # Process FINRA short data
        if finra.get('found'):
            result['has_short_data'] = True
            result['short_volume'] = finra.get('short_volume', 0)
            result['short_ratio'] = finra.get('short_ratio', 0)
            result['short_total_volume'] = finra.get('total_volume', 0)
            result['short_date'] = finra.get('date', '')
            
            # Analyze short sentiment
            short_ratio = result['short_ratio']
            if short_ratio > 60:
                result['short_sentiment'] = 'VERY_BEARISH'
                signals.append(f'🔴 High short ratio: {short_ratio}%')
                score_adjustments.append(-20)
            elif short_ratio > 50:
                result['short_sentiment'] = 'BEARISH'
                signals.append(f'🟠 Elevated short ratio: {short_ratio}%')
                score_adjustments.append(-10)
            elif short_ratio < 30:
                result['short_sentiment'] = 'BULLISH'
                signals.append(f'🟢 Low short ratio: {short_ratio}%')
                score_adjustments.append(10)
            else:
                result['short_sentiment'] = 'NEUTRAL'
        
        # Process price context
        if price.get('found'):
            result['current_price'] = price.get('current_price', 0)
            result['price_change_pct'] = price.get('price_change_pct', 0)
            result['volume_ratio'] = price.get('volume_ratio', 1.0)
            result['avg_volume'] = price.get('avg_volume', 0)
            result['today_volume'] = price.get('volume', 0)
            
            # Volume analysis
            if result['volume_ratio'] > 2.0:
                signals.append(f'📊 Unusual volume: {result["volume_ratio"]:.1f}x average')
            
            # Price direction
            if result['price_change_pct'] > 3:
                signals.append(f'📈 Strong price move: +{result["price_change_pct"]:.1f}%')
                score_adjustments.append(5)
            elif result['price_change_pct'] < -3:
                signals.append(f'📉 Strong price drop: {result["price_change_pct"]:.1f}%')
                score_adjustments.append(-5)
        
        # Calculate overall score (0-100, 50 = neutral)
        base_score = 50
        total_adjustment = sum(score_adjustments)
        result['overall_score'] = max(0, min(100, base_score + total_adjustment))
        
        # Determine overall sentiment (symmetric thresholds around 50)
        # BULLISH: 65+ | SLIGHTLY_BULLISH: 55-64 | NEUTRAL: 45-54 | SLIGHTLY_BEARISH: 35-44 | BEARISH: <35
        if result['overall_score'] >= 65:
            result['overall_sentiment'] = 'BULLISH'
            result['overall_color'] = '#00c851'
        elif result['overall_score'] >= 55:
            result['overall_sentiment'] = 'SLIGHTLY_BULLISH'
            result['overall_color'] = '#4CAF50'
        elif result['overall_score'] <= 35:
            result['overall_sentiment'] = 'BEARISH'
            result['overall_color'] = '#F44336'
        elif result['overall_score'] <= 45:
            result['overall_sentiment'] = 'SLIGHTLY_BEARISH'
            result['overall_color'] = '#FF5722'
        else:
            result['overall_sentiment'] = 'NEUTRAL'
            result['overall_color'] = '#9E9E9E'
        
        result['signals'] = signals if signals else ['No significant dark pool signals detected']
        
        return result
    
    def get_top_dark_pool_activity(self, limit: int = 20) -> List[Dict]:
        """
        Get top stocks by dark pool activity.
        
        Returns list of stocks with highest dark pool volume.
        """
        try:
            url = f"{self.stockgrid_url}?top=Dark%20Pools%20Position%20$&minmax=desc"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                response_data = response.json()
                data = response_data.get('data', [])
                
                results = []
                for item in data[:limit]:
                    net_pos = item.get('Dark Pools Position', 0)
                    
                    # Determine sentiment
                    if net_pos > 500000:
                        sentiment = 'BULLISH'
                        color = '#00c851'
                    elif net_pos < -500000:
                        sentiment = 'BEARISH'
                        color = '#F44336'
                    else:
                        sentiment = 'NEUTRAL'
                        color = '#9E9E9E'
                    
                    results.append({
                        'ticker': item.get('Ticker', ''),
                        'net_position': net_pos,
                        'net_position_dollar': item.get('Dark Pools Position $', 0),
                        'dp_volume': item.get('Dark Pools Position', 0),
                        'short_volume': item.get('Short Volume', 0),
                        'short_volume_pct': item.get('Short Volume %', 0),
                        'sentiment': sentiment,
                        'sentiment_color': color,
                        'date': item.get('Date', '')
                    })
                
                return results
            
            return []
            
        except Exception as e:
            return []
    
    def _empty_result(self, ticker: str, error: str) -> Dict:
        """Return empty result structure with error."""
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error,
            
            'has_dark_pool_data': False,
            'net_dp_position': 0,
            'net_dp_position_dollar': 0,
            'dp_volume': 0,
            'dp_sentiment': 'UNKNOWN',
            'dp_sentiment_color': '#9E9E9E',
            
            'has_short_data': False,
            'short_volume': 0,
            'short_ratio': 0,
            'short_sentiment': 'UNKNOWN',
            
            'current_price': 0,
            'price_change_pct': 0,
            'volume_ratio': 1.0,
            
            'overall_sentiment': 'UNKNOWN',
            'overall_score': 50,
            'signals': ['Error fetching dark pool data']
        }


class BuySellClassifier:
    """
    Buy/Sell Volume Classifier
    
    Uses tick rule and bid/ask analysis to classify trades.
    Accuracy: ~77-81%
    """
    
    @staticmethod
    def classify_by_tick_rule(prices: List[float]) -> List[str]:
        """
        Classify trades using the tick rule.
        
        - If price > previous price → BUY
        - If price < previous price → SELL
        - If price = previous price → Use previous classification
        
        Args:
            prices: List of trade prices
            
        Returns:
            List of classifications ('BUY', 'SELL', 'UNKNOWN')
        """
        if not prices:
            return []
        
        classifications = ['UNKNOWN']  # First trade is unknown
        last_direction = 'UNKNOWN'
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                classifications.append('BUY')
                last_direction = 'BUY'
            elif prices[i] < prices[i-1]:
                classifications.append('SELL')
                last_direction = 'SELL'
            else:
                # Zero tick - use last known direction
                classifications.append(last_direction)
        
        return classifications
    
    @staticmethod
    def classify_by_quote(trade_price: float, bid: float, ask: float) -> str:
        """
        Classify a single trade using the quote rule.
        
        - Trade at ASK → BUY (buyer aggressive)
        - Trade at BID → SELL (seller aggressive)
        - Trade between → UNKNOWN
        
        Args:
            trade_price: Execution price
            bid: Bid price at time of trade
            ask: Ask price at time of trade
            
        Returns:
            Classification ('BUY', 'SELL', 'UNKNOWN')
        """
        if trade_price >= ask:
            return 'BUY'
        elif trade_price <= bid:
            return 'SELL'
        else:
            # Trade inside spread - use midpoint rule
            midpoint = (bid + ask) / 2
            if trade_price > midpoint:
                return 'BUY'
            elif trade_price < midpoint:
                return 'SELL'
            else:
                return 'UNKNOWN'
    
    @staticmethod
    def estimate_buy_sell_ratio(volume: int, price_change_pct: float, 
                                 short_ratio: float = 50) -> Dict:
        """
        Estimate buy/sell ratio from available metrics.
        
        This is an approximation when true order flow is not available.
        
        Args:
            volume: Total volume
            price_change_pct: Price change percentage
            short_ratio: Short volume ratio (from FINRA)
            
        Returns:
            Dict with estimated buy/sell breakdown
        """
        # Base estimate: 50/50
        buy_pct = 50
        sell_pct = 50
        
        # Adjust based on price movement
        # Rising prices suggest more buying pressure
        if price_change_pct > 0:
            buy_adjustment = min(price_change_pct * 2, 20)  # Cap at 20%
            buy_pct += buy_adjustment
            sell_pct -= buy_adjustment
        elif price_change_pct < 0:
            sell_adjustment = min(abs(price_change_pct) * 2, 20)
            sell_pct += sell_adjustment
            buy_pct -= sell_adjustment
        
        # Adjust based on short ratio
        # Higher short ratio suggests more selling
        if short_ratio > 50:
            short_adjustment = min((short_ratio - 50) * 0.5, 15)
            sell_pct += short_adjustment
            buy_pct -= short_adjustment
        elif short_ratio < 50:
            long_adjustment = min((50 - short_ratio) * 0.5, 15)
            buy_pct += long_adjustment
            sell_pct -= long_adjustment
        
        # Ensure percentages are valid
        buy_pct = max(10, min(90, buy_pct))
        sell_pct = 100 - buy_pct
        
        # Calculate volumes
        buy_volume = int(volume * buy_pct / 100)
        sell_volume = int(volume * sell_pct / 100)
        
        return {
            'buy_pct': round(buy_pct, 1),
            'sell_pct': round(sell_pct, 1),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_volume': buy_volume - sell_volume,
            'confidence': 'ESTIMATED',
            'method': 'price_and_short_ratio'
        }


# Test function
if __name__ == "__main__":
    print("Testing Dark Pool Scanner...")
    print("=" * 60)
    
    scanner = DarkPoolScanner()
    
    # Test single ticker analysis
    print("\n📊 Testing AAPL Dark Pool Analysis:")
    result = scanner.get_dark_pool_analysis("AAPL")
    
    print(f"\nTicker: {result['ticker']}")
    print(f"Status: {result['status']}")
    
    if result['has_dark_pool_data']:
        print(f"\n🏊 Dark Pool Data (Stockgrid):")
        print(f"   Net Position: {result['net_dp_position']:,} shares")
        print(f"   Net Position $: ${result['net_dp_position_dollar']:,.0f}")
        print(f"   DP Sentiment: {result['dp_sentiment']}")
    
    if result['has_short_data']:
        print(f"\n📉 Short Volume Data (FINRA):")
        print(f"   Short Volume: {result['short_volume']:,}")
        print(f"   Short Ratio: {result['short_ratio']}%")
        print(f"   Short Sentiment: {result['short_sentiment']}")
    
    print(f"\n🎯 Overall Analysis:")
    print(f"   Score: {result['overall_score']}/100")
    print(f"   Sentiment: {result['overall_sentiment']}")
    print(f"\n📋 Signals:")
    for signal in result['signals']:
        print(f"   {signal}")
    
    # Test top dark pool activity
    print("\n" + "=" * 60)
    print("\n🔝 Top 10 Dark Pool Activity:")
    top_stocks = scanner.get_top_dark_pool_activity(10)
    
    for i, stock in enumerate(top_stocks, 1):
        sentiment_emoji = '🟢' if stock['sentiment'] == 'BULLISH' else '🔴' if stock['sentiment'] == 'BEARISH' else '⚪'
        print(f"   {i}. {stock['ticker']}: {stock['net_position']:,} shares ({sentiment_emoji} {stock['sentiment']})")
    
    # Test buy/sell classifier
    print("\n" + "=" * 60)
    print("\n🔄 Buy/Sell Estimation:")
    classifier = BuySellClassifier()
    
    estimate = classifier.estimate_buy_sell_ratio(
        volume=result.get('today_volume', 50000000),
        price_change_pct=result.get('price_change_pct', 0),
        short_ratio=result.get('short_ratio', 50)
    )
    
    print(f"   Buy Volume: {estimate['buy_volume']:,} ({estimate['buy_pct']}%)")
    print(f"   Sell Volume: {estimate['sell_volume']:,} ({estimate['sell_pct']}%)")
    print(f"   Net: {estimate['net_volume']:,}")
    print(f"   Confidence: {estimate['confidence']}")
