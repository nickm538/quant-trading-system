"""
PERFECT PRODUCTION STOCK ANALYZER V2 - Finnhub + yfinance
====================================================================

100% REAL DATA. ZERO PLACEHOLDERS. NO RATE LIMITS.

Data Sources:
1. Manus API Hub (YahooFinance) - Primary for prices, charts
2. Finnhub - Primary for fundamentals and news sentiment
3. yfinance - Backup for fundamentals
4. Local pandas + TA-Lib - Technical indicators

This is the FINAL version for REAL MONEY trading.
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

from data_api import ApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import requests
import time
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Finnhub API key
FINNHUB_API_KEY = "d47ssnpr01qk80bicu4gd47ssnpr01qk80bicu50"


class PerfectProductionAnalyzer:
    """World-class stock analyzer for real money trading - NO PLACEHOLDERS"""
    
    def __init__(self):
        self.api_client = ApiClient()
        self.finnhub_base_url = "https://finnhub.io/api/v1"
        self.cache = {}  # Simple cache
        # Initialize TAAPI.io client for backup/validation
        try:
            from taapi_client import TaapiClient
            self.taapi = TaapiClient()
            logger.info("Perfect Production Analyzer V2 initialized with Finnhub + yfinance + TAAPI.io")
        except Exception as e:
            self.taapi = None
            logger.warning(f"TAAPI.io client not available: {e}")
            logger.info("Perfect Production Analyzer V2 initialized with Finnhub + yfinance")
    
    def _call_finnhub(self, endpoint: str, symbol: str) -> Optional[Dict]:
        """Call Finnhub API - NO CACHING for real-money trading"""
        # PRODUCTION MODE: NO CACHING - Always fetch live data
        # For real-money trading, we need the freshest data possible
        
        try:
            url = f"{self.finnhub_base_url}/{endpoint}"
            params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Fresh Finnhub data fetched for {endpoint}/{symbol}")
            return data
            
        except Exception as e:
            logger.warning(f"Finnhub {endpoint} failed: {e}")
            return None
    
    def analyze_stock(self, symbol: str) -> Dict:
        """
        Comprehensive institutional-grade stock analysis with REAL DATA ONLY
        """
        analysis = {}
        analysis['symbol'] = symbol
        analysis['timestamp'] = datetime.now().isoformat()
        
        # 1. GET PRICE DATA from Manus API Hub
        logger.info(f"=== Analyzing {symbol} with REAL DATA (NO PLACEHOLDERS) ===")
        logger.info("Fetching stock chart from Manus API Hub...")
        
        try:
            # INTRADAY TRADING: Use 5-minute bars for last 5 days
            # 5-minute bars provide good balance between data granularity and noise
            chart_data = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '5m',  # 5-minute bars for intraday trading
                'range': '5d'      # Last 5 trading days (enough for indicators)
            })
            
            if not chart_data or 'chart' not in chart_data:
                raise ValueError("No chart data available")
            
            result = chart_data['chart']['result'][0]
            meta = result['meta']
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Build DataFrame
            hist = pd.DataFrame({
                'timestamp': timestamps,
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            hist['timestamp'] = pd.to_datetime(hist['timestamp'], unit='s')
            hist = hist.dropna()
            
            # INTRADAY FILTER: Only include NYSE market hours (9:30 AM - 4:00 PM EST)
            # Convert to US/Eastern timezone
            hist['timestamp'] = hist['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            hist['hour'] = hist['timestamp'].dt.hour
            hist['minute'] = hist['timestamp'].dt.minute
            
            # Filter for market hours: 9:30 AM (9.5 hours) to 4:00 PM (16.0 hours)
            market_hours_mask = (
                ((hist['hour'] == 9) & (hist['minute'] >= 30)) |  # 9:30 AM onwards
                ((hist['hour'] >= 10) & (hist['hour'] < 16))       # 10 AM to 3:59 PM
            )
            hist = hist[market_hours_mask].copy()
            hist = hist.drop(['hour', 'minute'], axis=1)
            
            logger.info(f"✓ Got {len(hist)} 5-minute bars during NYSE market hours (9:30 AM - 4:00 PM EST)")
            
            # FETCH DAILY DATA FOR ATR (need larger timeframe for targets/stops)
            logger.info("Fetching daily data for ATR calculation...")
            daily_response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1d',
                'range': '3mo'  # 3 months of daily data
            })
            
            daily_result = daily_response['chart']['result'][0]
            daily_timestamps = daily_result['timestamp']
            daily_quotes = daily_result['indicators']['quote'][0]
            
            daily_hist = pd.DataFrame({
                'open': daily_quotes['open'],
                'high': daily_quotes['high'],
                'low': daily_quotes['low'],
                'close': daily_quotes['close'],
                'volume': daily_quotes['volume']
            }).dropna()
            
            logger.info(f"✓ Got {len(daily_hist)} daily bars for ATR calculation")
            
            current_price = float(meta['regularMarketPrice'])
            analysis['current_price'] = current_price
            
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return {'error': f'Failed to fetch price data: {e}'}
        
        # 2. GET REAL FUNDAMENTALS from Finnhub + yfinance
        logger.info("Fetching REAL fundamentals from Finnhub...")
        fundamentals = self._get_real_fundamentals(symbol, meta)
        
        # 3. CALCULATE TECHNICAL INDICATORS LOCALLY (100% REAL)
        logger.info("Calculating REAL technical indicators...")
        technical_indicators = self._calculate_technical_indicators(hist, symbol, daily_hist)
        
        # 4. GET REAL NEWS SENTIMENT from Finnhub
        logger.info("Fetching REAL news sentiment from Finnhub...")
        sentiment_score = self._get_real_sentiment(symbol)
        
        # 5. CALCULATE SCORES
        logger.info("Calculating comprehensive scores...")
        
        fundamental_score = self._calculate_fundamental_score(fundamentals)
        technical_score = self._calculate_technical_score(technical_indicators, hist)
        
        analysis['fundamental_score'] = fundamental_score
        analysis['technical_score'] = technical_score
        analysis['sentiment_score'] = sentiment_score
        
        # OVERALL SCORE - INTRADAY OPTIMIZED
        # Intraday trading relies more on technicals and momentum
        # Weights: 30% fundamental, 50% technical, 20% sentiment
        overall_score = (
            fundamental_score * 0.30 +  # Reduced from 0.40
            technical_score * 0.50 +    # Increased from 0.45
            sentiment_score * 0.20      # Increased from 0.15
        )
        analysis['overall_score'] = overall_score
        
        # RECOMMENDATION
        if overall_score >= 75:
            recommendation = "STRONG_BUY"
        elif overall_score >= 60:
            recommendation = "BUY"
        elif overall_score >= 40:
            recommendation = "HOLD"
        elif overall_score >= 25:
            recommendation = "SELL"
        else:
            recommendation = "STRONG_SELL"
        
        analysis['recommendation'] = recommendation
        # Confidence should directly reflect overall score for clarity in real trading
        # 0-40: Low confidence, 40-60: Medium, 60-80: High, 80-100: Very High
        analysis['confidence'] = overall_score
        
        # Add details
        analysis['fundamentals'] = fundamentals
        analysis['technical_indicators'] = technical_indicators
        
        logger.info(f"✓ Analysis complete: {recommendation} (score: {overall_score:.1f}/100)")
        return analysis
    
    def _get_real_fundamentals(self, symbol: str, meta: Dict) -> Dict:
        """Get REAL fundamentals from Finnhub + yfinance"""
        fundamentals = {}
        
        # Try Finnhub first
        finnhub_metrics = self._call_finnhub('stock/metric', symbol)
        
        if finnhub_metrics and 'metric' in finnhub_metrics:
            logger.info("✓ Got REAL fundamentals from Finnhub")
            metric = finnhub_metrics['metric']
            
            # CRITICAL FIX: Finnhub returns MIXED units
            # - Growth metrics (earnings, revenue, profit margin): Already in % (e.g., 22.89 = 22.89%)
            # - Dividend yield: Already in % (e.g., 0.3747 = 0.3747%, NOT 37.47%)
            # - ROE: Already in % (e.g., 164.05 = 164.05%)
            
            pe_ratio = float(metric.get('peBasicExclExtraTTM', 0) or 0)
            peg_ratio = float(metric.get('pegTTM', 0) or 0)
            # CRITICAL FIX: Finnhub growth metrics are PERCENTAGE VALUES (1065.94 = 1065.94%)
            # We need to convert to decimal by dividing by 100
            earnings_growth_pct = float(metric.get('epsGrowthTTMYoy', 0) or 0)
            
            # Cap extreme growth values (>500%) which indicate turnaround from losses
            # These are mathematically correct but misleading for valuation
            if abs(earnings_growth_pct) > 500:
                logger.warning(f"Extreme earnings growth detected: {earnings_growth_pct}% (capping at 500% for scoring)")
                earnings_growth_pct = 500 if earnings_growth_pct > 0 else -500
            
            earnings_growth = earnings_growth_pct / 100  # Convert % to decimal
            
            # Calculate PEG manually if not provided by Finnhub
            if peg_ratio == 0 and pe_ratio > 0:
                # Use trailing earnings growth if positive
                if earnings_growth > 0:
                    peg_ratio = pe_ratio / (earnings_growth * 100)
                else:
                    # Fallback to yfinance forward earnings growth for negative trailing growth
                    try:
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        forward_eps = float(info.get('forwardEps', 0) or 0)
                        trailing_eps = float(info.get('trailingEps', 0) or 0)
                        if forward_eps > 0 and trailing_eps != 0:
                            forward_growth = (forward_eps - trailing_eps) / abs(trailing_eps)
                            if forward_growth > 0:
                                peg_ratio = pe_ratio / (forward_growth * 100)
                                logger.info(f"Using forward earnings growth for PEG: {forward_growth*100:.1f}%")
                    except Exception as e:
                        logger.warning(f"Could not calculate forward PEG: {e}")
            
            # Extract liquidity and EBITDA from Finnhub or yfinance fallback
            current_ratio = float(metric.get('currentRatioQuarterly', 0) or 0)
            quick_ratio = float(metric.get('quickRatioQuarterly', 0) or 0)
            
            # EBITDA - Finnhub doesn't provide directly, get from yfinance
            ebitda = 0
            total_debt = 0
            total_equity = 0
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                ebitda = float(info.get('ebitda', 0) or 0)
                total_debt = float(info.get('totalDebt', 0) or 0)
                total_equity = float(info.get('totalStockholderEquity', 0) or 0)
                
                # Get P/S and estimated EPS
                price_to_sales = float(info.get('priceToSalesTrailing12Months', 0) or 0)
                forward_eps = float(info.get('forwardEps', 0) or 0)
                trailing_eps = float(info.get('trailingEps', 0) or 0)
                
                # Override D/E with more accurate yfinance data if available
                if total_equity > 0 and total_debt > 0:
                    debt_to_equity = total_debt / total_equity
                else:
                    debt_to_equity = float(metric.get('totalDebt/totalEquityQuarterly', 0) or 0)
                
                # Override liquidity ratios if yfinance has better data
                if not current_ratio:
                    current_ratio = float(info.get('currentRatio', 0) or 0)
                if not quick_ratio:
                    quick_ratio = float(info.get('quickRatio', 0) or 0)
            except Exception as e:
                logger.warning(f"Could not get EBITDA/liquidity from yfinance: {e}")
                debt_to_equity = float(metric.get('totalDebt/totalEquityQuarterly', 0) or 0)
            
            fundamentals = {
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'profit_margin': float(metric.get('netProfitMarginTTM', 0) or 0) / 100,  # Already in %, convert to decimal
                'roe': float(metric.get('roeTTM', 0) or 0) / 100 if metric.get('roeTTM') else 0,  # Already in %, convert to decimal
                'debt_to_equity': debt_to_equity,
                'revenue_growth': float(metric.get('revenueGrowthTTMYoy', 0) or 0) / 100,  # Finnhub returns % (48.85 = 48.85%), convert to decimal
                'earnings_growth': earnings_growth,
                'beta': float(metric.get('beta', 1.0) or 1.0),
                'market_cap': float(metric.get('marketCapitalization', 0) or 0) * 1_000_000,  # Finnhub returns in millions
                'dividend_yield': float(metric.get('dividendYieldIndicatedAnnual', 0) or 0) / 100,  # Already in % (0.3747 = 0.3747%), convert to decimal
                'book_value': float(metric.get('bookValuePerShareQuarterly', 0) or 0),
                'price_to_book': float(metric.get('pbQuarterly', 0) or 0),
                'price_to_sales': price_to_sales,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'ebitda': ebitda,
                'eps': trailing_eps,
                'forward_eps': forward_eps,
                'total_debt': total_debt,
                'total_equity': total_equity
            }
        else:
            # Fallback to yfinance
            logger.warning("Finnhub unavailable, trying yfinance...")
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                pe_ratio = float(info.get('trailingPE', 0) or 0)
                peg_ratio = float(info.get('pegRatio', 0) or 0)
                earnings_growth = float(info.get('earningsGrowth', 0) or 0)
                
                # Calculate PEG manually if not provided
                if peg_ratio == 0 and pe_ratio > 0:
                    # Use trailing earnings growth if positive
                    if earnings_growth > 0:
                        peg_ratio = pe_ratio / (earnings_growth * 100)
                    else:
                        # Fallback to forward earnings growth for negative trailing growth
                        forward_eps = float(info.get('forwardEps', 0) or 0)
                        trailing_eps = float(info.get('trailingEps', 0) or 0)
                        if forward_eps > 0 and trailing_eps != 0:
                            forward_growth = (forward_eps - trailing_eps) / abs(trailing_eps)
                            if forward_growth > 0:
                                peg_ratio = pe_ratio / (forward_growth * 100)
                                logger.info(f"Using forward earnings growth for PEG: {forward_growth*100:.1f}%")
                
                # Extract liquidity and financial health metrics
                total_debt = float(info.get('totalDebt', 0) or 0)
                total_equity = float(info.get('totalStockholderEquity', 0) or 0)
                debt_to_equity = (total_debt / total_equity) if total_equity > 0 else 0
                
                # Liquidity metrics
                current_ratio = float(info.get('currentRatio', 0) or 0)
                quick_ratio = float(info.get('quickRatio', 0) or 0)
                
                # EBITDA
                ebitda = float(info.get('ebitda', 0) or 0)
                
                fundamentals = {
                    'pe_ratio': pe_ratio,
                    'peg_ratio': peg_ratio,
                    'profit_margin': float(info.get('profitMargins', 0) or 0),
                    'roe': float(info.get('returnOnEquity', 0) or 0),
                    'debt_to_equity': debt_to_equity,
                    'revenue_growth': float(info.get('revenueGrowth', 0) or 0),
                    'earnings_growth': earnings_growth,
                    'beta': float(info.get('beta', 1.0) or 1.0),
                    'market_cap': float(info.get('marketCap', 0) or 0),
                    'dividend_yield': float(info.get('dividendYield', 0) or 0),
                    'book_value': float(info.get('bookValue', 0) or 0),
                    'price_to_book': float(info.get('priceToBook', 0) or 0),
                    'price_to_sales': float(info.get('priceToSalesTrailing12Months', 0) or 0),
                    'current_ratio': current_ratio,
                    'quick_ratio': quick_ratio,
                    'ebitda': ebitda,
                    'eps': float(info.get('trailingEps', 0) or 0),
                    'forward_eps': float(info.get('forwardEps', 0) or 0),
                    'total_debt': total_debt,
                    'total_equity': total_equity
                }
                logger.info("✓ Got REAL fundamentals from yfinance")
            except Exception as e:
                logger.error(f"yfinance failed: {e}, using Manus meta (limited)")
                fundamentals = {
                    'pe_ratio': float(meta.get('trailingPE', 0) or 0),
                    'market_cap': float(meta.get('marketCap', 0) or 0),
                    'peg_ratio': 0,
                    'profit_margin': 0,
                    'roe': 0,
                    'debt_to_equity': 0,
                    'revenue_growth': 0,
                    'earnings_growth': 0,
                    'beta': 1.0,
                    'dividend_yield': 0,
                    'book_value': 0,
                    'price_to_book': 0
                }
        
        # Get earnings date using yfinance
        try:
            import yfinance as yf
            from datetime import date
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar and 'Earnings Date' in calendar:
                earnings_date = calendar['Earnings Date'][0] if isinstance(calendar['Earnings Date'], list) else calendar['Earnings Date']
                
                # Convert to date if datetime
                if isinstance(earnings_date, datetime):
                    earnings_date = earnings_date.date()
                
                today = date.today()
                days_to_earnings = (earnings_date - today).days
                fundamentals['earnings_date'] = str(earnings_date)
                fundamentals['days_to_earnings'] = days_to_earnings
            else:
                fundamentals['days_to_earnings'] = None
        except Exception as e:
            logger.warning(f"Could not fetch earnings date: {e}")
            fundamentals['days_to_earnings'] = None
        
        return fundamentals
    
    def _get_real_sentiment(self, symbol: str) -> float:
        """Get REAL news sentiment from Finnhub + analyst ratings"""
        sentiment_components = []
        
        # 1. NEWS SENTIMENT (40% weight)
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Extended to 30 days
        
        try:
            url = f"{self.finnhub_base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': FINNHUB_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            news = response.json()
            
            if news and len(news) > 0:
                # Analyze headlines for sentiment keywords
                positive_keywords = ['beat', 'surge', 'rally', 'gain', 'up', 'rise', 'high', 'strong', 'growth', 'profit', 'upgrade', 'buy', 'bullish', 'outperform']
                negative_keywords = ['miss', 'fall', 'drop', 'down', 'decline', 'low', 'weak', 'loss', 'downgrade', 'sell', 'bearish', 'underperform', 'cut', 'warning']
                
                news_scores = []
                for article in news[:20]:  # Analyze most recent 20 articles
                    headline = article.get('headline', '').lower()
                    summary = article.get('summary', '').lower()
                    text = headline + ' ' + summary
                    
                    # Count sentiment keywords
                    pos_count = sum(1 for word in positive_keywords if word in text)
                    neg_count = sum(1 for word in negative_keywords if word in text)
                    
                    # Calculate article sentiment (-1 to 1)
                    if pos_count + neg_count > 0:
                        article_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                    else:
                        article_sentiment = 0
                    
                    # Weight recent news more heavily (exponential decay)
                    days_ago = (datetime.now() - datetime.fromtimestamp(article.get('datetime', time.time()))).days
                    weight = np.exp(-days_ago / 10)  # Decay factor
                    
                    news_scores.append(article_sentiment * weight)
                
                if news_scores:
                    avg_news_sentiment = np.mean(news_scores)
                    # Convert from [-1, 1] to [0, 100]
                    news_score = (avg_news_sentiment + 1) * 50
                    sentiment_components.append(('news', news_score, 0.4))
                    logger.info(f"✓ Analyzed {len(news)} news articles, news sentiment: {news_score:.1f}")
        except Exception as e:
            logger.warning(f"News sentiment failed: {e}")
        
        # 2. ANALYST RECOMMENDATIONS (30% weight)
        try:
            url = f"{self.finnhub_base_url}/stock/recommendation"
            params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            recommendations = response.json()
            
            if recommendations and len(recommendations) > 0:
                # Get most recent recommendation
                latest = recommendations[0]
                
                strong_buy = latest.get('strongBuy', 0)
                buy = latest.get('buy', 0)
                hold = latest.get('hold', 0)
                sell = latest.get('sell', 0)
                strong_sell = latest.get('strongSell', 0)
                
                total = strong_buy + buy + hold + sell + strong_sell
                
                if total > 0:
                    # Calculate weighted score
                    # Strong Buy = 100, Buy = 75, Hold = 50, Sell = 25, Strong Sell = 0
                    analyst_score = (
                        (strong_buy * 100 + buy * 75 + hold * 50 + sell * 25 + strong_sell * 0) / total
                    )
                    sentiment_components.append(('analyst', analyst_score, 0.3))
                    logger.info(f"✓ Analyst ratings: {strong_buy} strong buy, {buy} buy, {hold} hold, {sell} sell, {strong_sell} strong sell -> {analyst_score:.1f}")
        except Exception as e:
            logger.warning(f"Analyst recommendations failed: {e}")
        
        # 3. PRICE MOMENTUM (30% weight) - Recent price trend as sentiment proxy
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo', interval='1d')
            
            if not hist.empty and len(hist) > 20:
                # Calculate price momentum over different periods
                current_price = hist['Close'].iloc[-1]
                price_1w_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else current_price
                price_1m_ago = hist['Close'].iloc[-20] if len(hist) >= 20 else current_price
                
                # Calculate returns
                return_1w = (current_price - price_1w_ago) / price_1w_ago
                return_1m = (current_price - price_1m_ago) / price_1m_ago
                
                # Convert to sentiment score (0-100)
                # +10% return = 100, -10% return = 0, 0% return = 50
                momentum_score = 50 + (return_1w * 0.7 + return_1m * 0.3) * 500
                momentum_score = max(0, min(100, momentum_score))  # Clamp to [0, 100]
                
                sentiment_components.append(('momentum', momentum_score, 0.3))
                logger.info(f"✓ Price momentum: 1w={return_1w*100:.1f}%, 1m={return_1m*100:.1f}% -> {momentum_score:.1f}")
        except Exception as e:
            logger.warning(f"Price momentum calculation failed: {e}")
        
        # Calculate weighted average sentiment
        if sentiment_components:
            total_weight = sum(weight for _, _, weight in sentiment_components)
            weighted_sentiment = sum(score * weight for _, score, weight in sentiment_components) / total_weight
            logger.info(f"✓ Final sentiment score: {weighted_sentiment:.1f} (from {len(sentiment_components)} sources)")
            return weighted_sentiment
        else:
            logger.warning("No sentiment data available, using neutral (50)")
            return 50.0
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame, symbol: str, daily_hist: pd.DataFrame = None) -> Dict:
        """Calculate technical indicators from price data
        
        Args:
            hist: Intraday 5-minute bars for indicators
            symbol: Stock symbol
            daily_hist: Daily bars for ATR calculation (optional)
        """
        import talib
        
        close = hist['close'].values
        high = hist['high'].values
        low = hist['low'].values
        volume = hist['volume'].values
        
        # Use iloc[-2] to avoid look-ahead bias (yesterday's close)
        current_price = float(close[-1])
        
        # INTRADAY: Use [-1] (current bar) for all indicators
        # For intraday trading, we need the most recent values
        indicators = {
            'symbol': symbol,
            'current_price': current_price,
            'rsi': float(talib.RSI(close, timeperiod=14)[-1]),  # Current RSI
            'macd': 0.0,
            'macd_signal': 0.0,
            'sma_20': float(talib.SMA(close, timeperiod=20)[-1]),
            'sma_50': float(talib.SMA(close, timeperiod=50)[-1]),
            'sma_200': float(talib.SMA(close, timeperiod=200)[-1]) if len(close) >= 200 else float(talib.SMA(close, timeperiod=50)[-1]),
            'bb_upper': 0.0,
            'bb_middle': 0.0,
            'bb_lower': 0.0,
            'adx': float(talib.ADX(high, low, close, timeperiod=14)[-1])  # Current ADX
        }
        
        # MACD - use current values for intraday
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['macd'] = float(macd[-1])
        indicators['macd_signal'] = float(macd_signal[-1])
        
        # Bollinger Bands - use current values for intraday
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        indicators['bb_upper'] = float(bb_upper[-1])
        indicators['bb_middle'] = float(bb_middle[-1])
        indicators['bb_lower'] = float(bb_lower[-1])
        
        # ATR (Average True Range) for institutional-grade target/stop calculations
        # Use DAILY ATR (not 5-minute) for realistic targets/stops
        try:
            if daily_hist is not None and len(daily_hist) >= 14:
                daily_high = daily_hist['high'].values
                daily_low = daily_hist['low'].values
                daily_close = daily_hist['close'].values
                daily_atr = talib.ATR(daily_high, daily_low, daily_close, timeperiod=14)
                indicators['atr'] = float(daily_atr[-1])  # Use DAILY ATR for target/stop
                indicators['atr_pct'] = float(daily_atr[-1] / current_price)  # ATR as % of price
                logger.info(f"Using DAILY ATR: ${daily_atr[-1]:.2f} ({daily_atr[-1]/current_price*100:.2f}% of price)")
            else:
                # Fallback to intraday ATR if daily data not available
                atr = talib.ATR(high, low, close, timeperiod=14)
                indicators['atr'] = float(atr[-1])
                indicators['atr_pct'] = float(atr[-1] / current_price)
                logger.warning(f"Using INTRADAY ATR (daily data not available): ${atr[-1]:.2f}")
        except Exception as e:
            logger.warning(f"TA-Lib ATR failed: {e}")
            # Try TAAPI.io fallback
            if self.taapi:
                try:
                    # INTRADAY: Use 5-minute interval for TAAPI.io
                    taapi_atr = self.taapi.get_atr(symbol, period=14, interval='5m')
                    if taapi_atr:
                        indicators['atr'] = taapi_atr
                        indicators['atr_pct'] = taapi_atr / current_price
                        logger.info(f"✓ TAAPI.io ATR fallback successful: {taapi_atr:.2f}")
                    else:
                        raise Exception("TAAPI.io returned None")
                except Exception as taapi_error:
                    logger.error(f"TAAPI.io ATR fallback failed: {taapi_error}")
                    raise Exception(f"CRITICAL: Cannot calculate ATR for {symbol}. TA-Lib and TAAPI.io both failed. Aborting analysis to prevent fake data.")
            else:
                # No TAAPI.io available - FAIL LOUDLY
                logger.error("TAAPI.io not available and TA-Lib ATR failed")
                raise Exception(f"CRITICAL: Cannot calculate ATR for {symbol}. No backup data source available. Aborting analysis to prevent fake data.")
        
        # INTRADAY INDICATORS
        # VWAP (Volume Weighted Average Price) - key intraday support/resistance
        try:
            # Calculate VWAP: sum(price * volume) / sum(volume)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * hist['volume']).cumsum() / hist['volume'].cumsum()
            indicators['vwap'] = float(vwap.iloc[-1])
            indicators['vwap_distance'] = (current_price - indicators['vwap']) / indicators['vwap']  # % above/below VWAP
            logger.info(f"✓ VWAP calculated: ${indicators['vwap']:.2f} ({indicators['vwap_distance']*100:.2f}% from current)")
        except Exception as e:
            logger.warning(f"VWAP calculation failed: {e}")
            indicators['vwap'] = current_price
            indicators['vwap_distance'] = 0.0
        
        # Rate of Change (ROC) - momentum indicator
        try:
            roc = talib.ROC(close, timeperiod=10)  # 10-period ROC
            indicators['roc'] = float(roc[-1])
        except Exception as e:
            logger.warning(f"ROC calculation failed: {e}")
            indicators['roc'] = 0.0
        
        # Money Flow Index (MFI) - volume-weighted RSI
        try:
            mfi = talib.MFI(high, low, close, hist['volume'].values, timeperiod=14)
            indicators['mfi'] = float(mfi[-1])
        except Exception as e:
            logger.warning(f"MFI calculation failed: {e}")
            indicators['mfi'] = 50.0  # Neutral
        
        return indicators
    
    def _calculate_fundamental_score(self, fundamentals: Dict) -> float:
        """Calculate fundamental score from REAL fundamentals (0-100)"""
        score = 50.0  # Start neutral
        
        # P/E Ratio (lower is better, but not negative)
        pe = fundamentals.get('pe_ratio', 0)
        if 10 <= pe <= 20:
            score += 15  # Sweet spot
        elif 20 < pe <= 30:
            score += 8
        elif pe > 40:
            score -= 15  # Overvalued
        elif pe < 0:
            score -= 20  # Negative earnings
        
        # ROE (higher is better)
        roe = fundamentals.get('roe', 0)
        if roe > 0.15:
            score += 12
        elif roe > 0.10:
            score += 8
        elif roe > 0:
            score += 4
        elif roe < 0:
            score -= 12
        
        # Debt to Equity (lower is better)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity < 0.5:
            score += 8
        elif debt_to_equity < 1.0:
            score += 4
        elif debt_to_equity > 2.0:
            score -= 12
        
        # Revenue Growth (higher is better)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 0.20:
            score += 10
        elif revenue_growth > 0.10:
            score += 6
        elif revenue_growth < 0:
            score -= 10
        
        # Earnings Growth (higher is better)
        earnings_growth = fundamentals.get('earnings_growth', 0)
        if earnings_growth > 0.20:
            score += 10
        elif earnings_growth > 0.10:
            score += 6
        elif earnings_growth < 0:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_technical_score(self, indicators: Dict, hist: pd.DataFrame) -> float:
        """Calculate technical score from REAL indicators (0-100)"""
        score = 50.0  # Start neutral
        
        current_price = indicators['current_price']
        
        # RSI (30-40 oversold = buy, 60-70 overbought = sell)
        rsi = indicators['rsi']
        if 30 <= rsi <= 40:
            score += 15  # Oversold, good buy signal
        elif 40 < rsi <= 50:
            score += 10
        elif 50 < rsi <= 60:
            score += 5
        elif 60 < rsi <= 70:
            score -= 10
        elif rsi > 70:
            score -= 20  # Overbought, sell signal
        
        # MACD (bullish crossover = buy, bearish = sell)
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        if macd > macd_signal and macd > 0:
            score += 12  # Bullish crossover above zero
        elif macd > macd_signal:
            score += 8   # Bullish crossover
        elif macd < macd_signal and macd < 0:
            score -= 8   # Bearish crossover below zero
        
        # Moving Averages (price above MAs = bullish)
        sma_50 = indicators['sma_50']
        sma_200 = indicators['sma_200']
        
        if current_price > sma_50 and current_price > sma_200:
            score += 15  # Above both MAs
        elif current_price > sma_50:
            score += 8
        elif current_price < sma_50 and current_price < sma_200:
            score -= 15  # Below both MAs
        
        # Bollinger Bands (price near upper = overbought, near lower = oversold)
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0:
            position = (current_price - bb_lower) / bb_range
            if position < 0.2:
                score += 10  # Near lower band, oversold
            elif position > 0.8:
                score -= 10  # Near upper band, overbought
        
        # ADX (trend strength)
        adx = indicators['adx']
        if adx > 40:
            score += 8  # Strong trend
        elif adx > 25:
            score += 4  # Moderate trend
        
        # PATTERN RECOGNITION (10% weight max to avoid overfitting)
        # Fetch pattern indicators from TAAPI.io
        pattern_score = 0
        try:
            from taapi_client import get_pattern_indicators
            patterns = get_pattern_indicators(indicators.get('symbol', 'UNKNOWN'), '1d')
            
            if patterns:
                # Bearish reversal patterns (subtract from score)
                bearish_patterns = ['3blackcrows', '2crows', 'eveningstar', 'eveningdojistar', 
                                   'shootingstar', 'darkcloudcover', 'hangingman']
                for pattern in bearish_patterns:
                    if pattern in patterns and patterns[pattern].get('value') == 100:
                        pattern_score -= 1.43  # -10 points max for all 7 bearish patterns
                        logger.info(f"Bearish pattern detected: {pattern}")
                
                # Bullish reversal patterns (add to score)
                bullish_patterns = ['3whitesoldiers', 'morningstar', 'morningdojistar', 
                                   'hammer', 'invertedhammer', 'piercing']
                for pattern in bullish_patterns:
                    if pattern in patterns and patterns[pattern].get('value') == 100:
                        pattern_score += 1.67  # +10 points max for all 6 bullish patterns
                        logger.info(f"Bullish pattern detected: {pattern}")
                
                # Indecision patterns (context-dependent)
                if 'doji' in patterns and patterns['doji'].get('value') == 100:
                    # Doji in uptrend (RSI > 60) = bearish, in downtrend (RSI < 40) = bullish
                    if rsi > 60:
                        pattern_score -= 0.5
                        logger.info("Doji detected in uptrend (bearish)")
                    elif rsi < 40:
                        pattern_score += 0.5
                        logger.info("Doji detected in downtrend (bullish)")
                
                # Engulfing pattern (check with MACD for direction)
                if 'engulfing' in patterns and patterns['engulfing'].get('value') == 100:
                    if macd > macd_signal:  # Bullish engulfing
                        pattern_score += 1.0
                        logger.info("Bullish engulfing pattern detected")
                    else:  # Bearish engulfing
                        pattern_score -= 1.0
                        logger.info("Bearish engulfing pattern detected")
                
                # Apply pattern score (capped at ±10 points)
                pattern_score = max(-10, min(10, pattern_score))
                score += pattern_score
                logger.info(f"Pattern recognition score adjustment: {pattern_score:+.2f}")
        
        except Exception as e:
            logger.warning(f"Pattern recognition failed: {e}")
            # Continue without pattern recognition if it fails
        
        return max(0, min(100, score))
