"""
Dark Pool & ARIMA Integration Module
=====================================
Provides access to institutional-grade dark pool data from FINRA,
ARIMA forecasting, and factor analysis.

Data Sources:
- Dark Pools: FINRA RegSHO short volume data (direct from FINRA CDN)
- ARIMA: Time series forecasting using statsmodels
- Factor Analysis: Calculated from price/volume data

Author: Quant Trading System
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import warnings
from io import StringIO
warnings.filterwarnings('ignore')

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA features disabled.")


class StockGridIntegration:
    """
    Integration for dark pool data (FINRA), ARIMA forecasting,
    and factor analysis.
    """
    
    # FINRA RegSHO data URLs
    FINRA_BASE_URL = "https://cdn.finra.org/equity/regsho/daily"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self._cache = {}
        self._cache_time = {}
        self.cache_duration = 1800  # 30 minutes
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_time:
            return False
        return (datetime.now() - self._cache_time[key]).seconds < self.cache_duration
    
    def get_dark_pool_data(self, ticker: str) -> Dict:
        """
        Fetch dark pool data for a specific ticker from FINRA RegSHO.
        
        Returns:
            Dict with dark pool metrics including:
            - short_volume: Today's short volume
            - total_volume: Today's total volume
            - short_ratio: Short volume / Total volume
            - net_short_volume: Calculated net short (short - (total - short))
            - position: 20-day cumulative net short position
            - position_dollar: Dollar value of position
            - sentiment: Bullish/Bearish interpretation
        """
        cache_key = f"darkpool_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Fetch historical FINRA data (last 20 trading days)
            history = self._fetch_finra_history(ticker.upper(), days=25)
            
            if not history:
                return self._empty_dark_pool_result(ticker, "No FINRA data available")
            
            # Get current stock price for dollar calculations
            current_price = 0
            try:
                import os
                poly_key = os.environ.get('POLYGON_API_KEY', '')
                if poly_key:
                    r = requests.get(
                        f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={poly_key}',
                        timeout=5
                    )
                    if r.status_code == 200:
                        results = r.json().get('results', [])
                        if results:
                            current_price = results[0].get('c', 0)
                if not current_price:
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice', 0)
            except:
                current_price = 0
            
            # Calculate metrics
            latest = history[0] if history else {}
            
            # Calculate net short volume (short volume - non-short volume)
            # Net short = short_volume - (total_volume - short_volume) = 2*short_volume - total_volume
            net_short_volume = 0
            if latest.get('short_volume') and latest.get('total_volume'):
                net_short_volume = (2 * latest['short_volume']) - latest['total_volume']
            
            # Calculate 20-day position
            position = 0
            for day in history[:20]:
                if day.get('short_volume') and day.get('total_volume'):
                    daily_net = (2 * day['short_volume']) - day['total_volume']
                    position += daily_net
            
            # Calculate dollar values
            net_short_dollar = net_short_volume * current_price if current_price else 0
            position_dollar = position * current_price if current_price else 0
            
            result = {
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'source': 'FINRA RegSHO (Direct)',
                'status': 'success',
                'error': None,
                'date': latest.get('date'),
                'short_volume': latest.get('short_volume'),
                'short_exempt_volume': latest.get('short_exempt_volume'),
                'total_volume': latest.get('total_volume'),
                'short_ratio': latest.get('short_ratio'),
                'net_short_volume': net_short_volume,
                'net_short_volume_dollar': net_short_dollar,
                'position': position,
                'position_dollar': position_dollar,
                'current_price': current_price,
                'short_volume_history': history[:15],
            }
            
            # Calculate sentiment interpretation
            result['sentiment'] = self._interpret_dark_pool_sentiment(result)
            
            # Add explanation
            result['explanation'] = self._get_dark_pool_explanation(result)
            
            # Cache the result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return self._empty_dark_pool_result(ticker, str(e))
    
    def _fetch_finra_history(self, ticker: str, days: int = 25) -> List[Dict]:
        """
        Fetch historical FINRA RegSHO short volume data.
        
        FINRA publishes daily files at:
        https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
        """
        history = []
        current_date = datetime.now()
        
        # Try to fetch data for the last N trading days
        attempts = 0
        max_attempts = days + 10  # Account for weekends/holidays
        
        while len(history) < days and attempts < max_attempts:
            date_str = current_date.strftime('%Y%m%d')
            
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                try:
                    url = f"{self.FINRA_BASE_URL}/CNMSshvol{date_str}.txt"
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        # Parse the pipe-delimited file
                        data = self._parse_finra_file(response.text, ticker)
                        if data:
                            data['date'] = current_date.strftime('%Y-%m-%d')
                            history.append(data)
                except Exception as e:
                    pass  # Skip failed days
            
            current_date -= timedelta(days=1)
            attempts += 1
        
        return history
    
    def _parse_finra_file(self, content: str, ticker: str) -> Optional[Dict]:
        """Parse FINRA RegSHO file and extract data for specific ticker."""
        try:
            # File format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            for line in content.strip().split('\n'):
                if line.startswith('Date|'):
                    continue  # Skip header
                
                parts = line.split('|')
                if len(parts) >= 5 and parts[1].upper() == ticker.upper():
                    short_vol = int(parts[2])
                    short_exempt = int(parts[3])
                    total_vol = int(parts[4])
                    
                    return {
                        'short_volume': short_vol,
                        'short_exempt_volume': short_exempt,
                        'total_volume': total_vol,
                        'short_ratio': round(short_vol / total_vol, 4) if total_vol > 0 else 0,
                        'market': parts[5] if len(parts) > 5 else 'N/A'
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _interpret_dark_pool_sentiment(self, data: Dict) -> Dict:
        """
        Interpret dark pool data for trading sentiment.
        
        Key insight from SqueezeMetrics research:
        - NEGATIVE net short volume = More buying than shorting = BULLISH
        - POSITIVE net short volume = More shorting than buying = BEARISH
        
        This is counterintuitive but backed by research showing that
        dark pool "short volume" mostly represents market makers
        selling shares to meet buyer demand.
        """
        sentiment = {
            'sentiment': 'NEUTRAL',
            'score': 50,
            'signal': 'No clear signal',
            'daily_trend': 'NEUTRAL'
        }
        
        position_dollar = data.get('position_dollar', 0) or 0
        net_short = data.get('net_short_volume', 0) or 0
        short_ratio = data.get('short_ratio', 0.5) or 0.5
        
        # Score based on 20-day position (in dollars)
        # Negative position = bullish (more buying than shorting)
        if position_dollar < -10_000_000_000:  # < -$10B
            sentiment['sentiment'] = 'VERY_BULLISH'
            sentiment['score'] = 90
            sentiment['signal'] = 'Extreme institutional buying detected'
        elif position_dollar < -1_000_000_000:  # < -$1B
            sentiment['sentiment'] = 'VERY_BULLISH'
            sentiment['score'] = 80
            sentiment['signal'] = 'Strong institutional buying detected'
        elif position_dollar < -100_000_000:  # < -$100M
            sentiment['sentiment'] = 'BULLISH'
            sentiment['score'] = 70
            sentiment['signal'] = 'Moderate institutional buying detected'
        elif position_dollar < 0:
            sentiment['sentiment'] = 'SLIGHTLY_BULLISH'
            sentiment['score'] = 60
            sentiment['signal'] = 'Light institutional buying detected'
        elif position_dollar > 10_000_000_000:  # > $10B
            sentiment['sentiment'] = 'VERY_BEARISH'
            sentiment['score'] = 10
            sentiment['signal'] = 'Extreme institutional selling/shorting detected'
        elif position_dollar > 1_000_000_000:  # > $1B
            sentiment['sentiment'] = 'VERY_BEARISH'
            sentiment['score'] = 20
            sentiment['signal'] = 'Strong institutional selling/shorting detected'
        elif position_dollar > 100_000_000:  # > $100M
            sentiment['sentiment'] = 'BEARISH'
            sentiment['score'] = 30
            sentiment['signal'] = 'Moderate institutional selling/shorting detected'
        elif position_dollar > 0:
            sentiment['sentiment'] = 'SLIGHTLY_BEARISH'
            sentiment['score'] = 40
            sentiment['signal'] = 'Light institutional selling detected'
        
        # Daily trend based on today's net short
        if net_short < 0:
            sentiment['daily_trend'] = 'BULLISH'
        elif net_short > 0:
            sentiment['daily_trend'] = 'BEARISH'
        
        # Adjust for short ratio
        if short_ratio > 0.6:
            sentiment['signal'] += ' (High short ratio - potential squeeze setup)'
        elif short_ratio < 0.3:
            sentiment['signal'] += ' (Low short ratio - limited short interest)'
        
        return sentiment
    
    def _get_dark_pool_explanation(self, data: Dict) -> str:
        """Generate explanation of dark pool data."""
        position = data.get('position', 0) or 0
        position_dollar = data.get('position_dollar', 0) or 0
        short_ratio = data.get('short_ratio', 0) or 0
        
        explanation = []
        
        if position < 0:
            explanation.append(
                f"20-day net position is {abs(position):,.0f} shares negative "
                f"(${abs(position_dollar)/1e9:.2f}B), indicating net buying pressure."
            )
        else:
            explanation.append(
                f"20-day net position is {position:,.0f} shares positive "
                f"(${position_dollar/1e9:.2f}B), indicating net selling pressure."
            )
        
        explanation.append(
            f"Short volume ratio is {short_ratio:.1%}, "
            f"{'above' if short_ratio > 0.45 else 'below'} the typical 45% average."
        )
        
        return ' '.join(explanation)
    
    def _empty_dark_pool_result(self, ticker: str, error: str) -> Dict:
        """Return empty result structure with error."""
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'source': 'FINRA RegSHO',
            'status': 'error',
            'error': error,
            'net_short_volume': None,
            'net_short_volume_dollar': None,
            'position': None,
            'position_dollar': None,
            'short_volume_history': [],
            'sentiment': {
                'sentiment': 'UNKNOWN',
                'score': 50,
                'signal': f'Data unavailable: {error}',
                'daily_trend': 'UNKNOWN'
            }
        }
    
    def calculate_arima_forecast(self, ticker: str, forecast_days: int = 5) -> Dict:
        """
        Calculate ARIMA forecast for a stock.
        
        Args:
            ticker: Stock symbol
            forecast_days: Number of days to forecast (default 5)
            
        Returns:
            Dict with ARIMA forecast results
        """
        if not ARIMA_AVAILABLE:
            return {
                'status': 'error',
                'error': 'statsmodels not installed',
                'ticker': ticker
            }
        
        cache_key = f"arima_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Fetch historical price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty or len(hist) < 60:
                return {
                    'status': 'error',
                    'error': 'Insufficient historical data',
                    'ticker': ticker
                }
            
            # Use closing prices
            prices = hist['Close'].dropna()
            current_price = float(prices.iloc[-1])
            
            # Test for stationarity
            adf_result = adfuller(prices, autolag='AIC')
            is_stationary = adf_result[1] < 0.05
            
            # Fit ARIMA model
            # Using (1,1,1) as a robust default - can be optimized
            model = ARIMA(prices, order=(1, 1, 1))
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.get_forecast(steps=forecast_days)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
            
            # Build forecast results
            forecast_dates = pd.date_range(
                start=prices.index[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='B'  # Business days
            )
            
            forecast_data = []
            for i in range(forecast_days):
                forecast_data.append({
                    'date': forecast_dates[i].strftime('%Y-%m-%d'),
                    'predicted': float(forecast_mean.iloc[i]),
                    'lower_95': float(forecast_ci.iloc[i, 0]),
                    'upper_95': float(forecast_ci.iloc[i, 1])
                })
            
            # Calculate expected change
            final_forecast = forecast_data[-1]['predicted']
            expected_change = final_forecast - current_price
            expected_change_pct = (expected_change / current_price) * 100
            
            # Determine signal
            if expected_change_pct > 3:
                signal = 'STRONG_BUY'
                confidence = 'HIGH'
            elif expected_change_pct > 1:
                signal = 'BUY'
                confidence = 'MEDIUM'
            elif expected_change_pct < -3:
                signal = 'STRONG_SELL'
                confidence = 'HIGH'
            elif expected_change_pct < -1:
                signal = 'SELL'
                confidence = 'MEDIUM'
            else:
                signal = 'HOLD'
                confidence = 'LOW'
            
            result = {
                'status': 'success',
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'model': 'ARIMA(1,1,1)',
                'current_price': current_price,
                'forecast': forecast_data,
                'model_stats': {
                    'aic': float(fitted.aic),
                    'bic': float(fitted.bic),
                    'observations': len(prices),
                    'is_stationary': is_stationary
                },
                'interpretation': {
                    'expected_change': expected_change,
                    'expected_change_pct': expected_change_pct,
                    'signal': signal,
                    'confidence': confidence,
                    'summary': f"ARIMA predicts {expected_change_pct:+.1f}% change over forecast period"
                }
            }
            
            # Cache result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'ticker': ticker
            }
    
    def get_factor_analysis(self, ticker: str) -> Dict:
        """
        Calculate factor analysis for a stock.
        
        Factors calculated:
        - Momentum (price change over various periods)
        - Volatility (standard deviation of returns)
        - Volume trend
        - Relative strength
        """
        cache_key = f"factors_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            if hist.empty or len(hist) < 60:
                return {
                    'status': 'error',
                    'error': 'Insufficient data',
                    'ticker': ticker
                }
            
            prices = hist['Close']
            volumes = hist['Volume']
            returns = prices.pct_change().dropna()
            
            # Calculate factors
            factors = {}
            
            # Momentum factors
            factors['momentum_1m'] = float((prices.iloc[-1] / prices.iloc[-21] - 1) * 100) if len(prices) > 21 else 0
            factors['momentum_3m'] = float((prices.iloc[-1] / prices.iloc[-63] - 1) * 100) if len(prices) > 63 else 0
            factors['momentum_6m'] = float((prices.iloc[-1] / prices.iloc[-126] - 1) * 100) if len(prices) > 126 else 0
            
            # Volatility
            factors['volatility_20d'] = float(returns.tail(20).std() * np.sqrt(252) * 100)
            factors['volatility_60d'] = float(returns.tail(60).std() * np.sqrt(252) * 100)
            
            # Volume trend
            avg_vol_20 = volumes.tail(20).mean()
            avg_vol_60 = volumes.tail(60).mean()
            factors['volume_trend'] = float((avg_vol_20 / avg_vol_60 - 1) * 100) if avg_vol_60 > 0 else 0
            
            # Price relative to moving averages
            sma_20 = prices.tail(20).mean()
            sma_50 = prices.tail(50).mean()
            sma_200 = prices.tail(200).mean() if len(prices) >= 200 else prices.mean()
            
            current_price = float(prices.iloc[-1])
            factors['price_vs_sma20'] = float((current_price / sma_20 - 1) * 100)
            factors['price_vs_sma50'] = float((current_price / sma_50 - 1) * 100)
            factors['price_vs_sma200'] = float((current_price / sma_200 - 1) * 100)
            
            # Calculate composite score
            score = 50  # Start neutral
            
            # Momentum contribution (max ±20)
            if factors['momentum_1m'] > 5:
                score += 10
            elif factors['momentum_1m'] < -5:
                score -= 10
            
            if factors['momentum_3m'] > 10:
                score += 10
            elif factors['momentum_3m'] < -10:
                score -= 10
            
            # Trend contribution (max ±20)
            if factors['price_vs_sma20'] > 0 and factors['price_vs_sma50'] > 0:
                score += 10
            elif factors['price_vs_sma20'] < 0 and factors['price_vs_sma50'] < 0:
                score -= 10
            
            if factors['price_vs_sma200'] > 0:
                score += 10
            else:
                score -= 10
            
            # Volume contribution (max ±10)
            if factors['volume_trend'] > 20:
                score += 5
            elif factors['volume_trend'] < -20:
                score -= 5
            
            # Determine signal
            if score >= 70:
                signal = 'STRONG_BUY'
            elif score >= 60:
                signal = 'BUY'
            elif score <= 30:
                signal = 'STRONG_SELL'
            elif score <= 40:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            result = {
                'status': 'success',
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'factors': factors,
                'composite_score': score,
                'signal': signal,
                'interpretation': self._interpret_factors(factors, score)
            }
            
            # Cache result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'ticker': ticker
            }
    
    def _interpret_factors(self, factors: Dict, score: int) -> str:
        """Generate interpretation of factor analysis."""
        parts = []
        
        # Momentum interpretation
        mom_1m = factors.get('momentum_1m', 0)
        mom_3m = factors.get('momentum_3m', 0)
        
        if mom_1m > 10 and mom_3m > 15:
            parts.append("Strong positive momentum across timeframes")
        elif mom_1m < -10 and mom_3m < -15:
            parts.append("Strong negative momentum across timeframes")
        elif mom_1m > 0 and mom_3m > 0:
            parts.append("Positive momentum trend")
        elif mom_1m < 0 and mom_3m < 0:
            parts.append("Negative momentum trend")
        else:
            parts.append("Mixed momentum signals")
        
        # Trend interpretation
        vs_sma20 = factors.get('price_vs_sma20', 0)
        vs_sma200 = factors.get('price_vs_sma200', 0)
        
        if vs_sma20 > 0 and vs_sma200 > 0:
            parts.append("Trading above key moving averages (bullish)")
        elif vs_sma20 < 0 and vs_sma200 < 0:
            parts.append("Trading below key moving averages (bearish)")
        else:
            parts.append("Mixed trend signals")
        
        # Volume interpretation
        vol_trend = factors.get('volume_trend', 0)
        if vol_trend > 30:
            parts.append("Significantly elevated volume (high interest)")
        elif vol_trend < -30:
            parts.append("Declining volume (waning interest)")
        
        return '. '.join(parts) + '.'
    
    def get_full_stockgrid_analysis(self, ticker: str) -> Dict:
        """
        Get complete analysis including dark pools, ARIMA, and factors.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dict with all analysis components
        """
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'dark_pools': self.get_dark_pool_data(ticker),
            'arima': self.calculate_arima_forecast(ticker),
            'factor_analysis': self.get_factor_analysis(ticker)
        }


# Test function
if __name__ == '__main__':
    sg = StockGridIntegration()
    
    print("Testing AAPL analysis...")
    result = sg.get_full_stockgrid_analysis('AAPL')
    
    print("\n=== DARK POOLS ===")
    dp = result['dark_pools']
    print(f"Status: {dp.get('status')}")
    print(f"Short Volume: {dp.get('short_volume'):,}" if dp.get('short_volume') else "Short Volume: N/A")
    print(f"Total Volume: {dp.get('total_volume'):,}" if dp.get('total_volume') else "Total Volume: N/A")
    print(f"Short Ratio: {dp.get('short_ratio'):.1%}" if dp.get('short_ratio') else "Short Ratio: N/A")
    print(f"Net Short Volume: {dp.get('net_short_volume'):,}" if dp.get('net_short_volume') else "Net Short: N/A")
    print(f"20-Day Position: {dp.get('position'):,}" if dp.get('position') else "Position: N/A")
    print(f"Position $: ${dp.get('position_dollar', 0)/1e9:.2f}B")
    print(f"Sentiment: {dp.get('sentiment', {}).get('sentiment')}")
    print(f"Signal: {dp.get('sentiment', {}).get('signal')}")
    
    print("\n=== ARIMA ===")
    arima = result['arima']
    print(f"Status: {arima.get('status')}")
    if arima.get('status') == 'success':
        print(f"Current Price: ${arima.get('current_price'):.2f}")
        print(f"5-Day Forecast: ${arima['forecast'][-1]['predicted']:.2f}")
        print(f"Signal: {arima.get('interpretation', {}).get('signal')}")
    
    print("\n=== FACTORS ===")
    fa = result['factor_analysis']
    print(f"Status: {fa.get('status')}")
    if fa.get('status') == 'success':
        print(f"Composite Score: {fa.get('composite_score')}/100")
        print(f"Signal: {fa.get('signal')}")
