"""
StockGrid.io Integration Module
================================
Provides access to institutional-grade dark pool data, ARIMA forecasting, 
and factor analysis from StockGrid.io.

Data Sources:
- Dark Pools: FINRA TRF short volume data
- ARIMA: Time series forecasting using statsmodels
- Factor Analysis: Signal-based trading factors

Author: Quant Trading System
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import re
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA features disabled.")


class StockGridIntegration:
    """
    Integration with StockGrid.io for dark pool data, ARIMA forecasting,
    and factor analysis.
    """
    
    BASE_URL = "https://www.stockgrid.io"
    
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
        Fetch dark pool data for a specific ticker from StockGrid.
        
        Returns:
            Dict with dark pool metrics including:
            - net_short_volume: Today's net short volume
            - net_short_volume_dollar: Dollar value of net short volume
            - position: 20-day cumulative net short position
            - position_dollar: Dollar value of position
            - short_volume_history: Historical short volume data
            - sentiment: Bullish/Bearish interpretation
        """
        cache_key = f"darkpool_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Try Firecrawl MCP first (handles JavaScript rendering)
            text = self._fetch_with_firecrawl(f"{self.BASE_URL}/darkpools/{ticker.upper()}")
            
            if not text:
                # Fallback to requests
                url = f"{self.BASE_URL}/darkpools/{ticker.upper()}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    return self._empty_dark_pool_result(ticker, f"HTTP {response.status_code}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
            
            # Extract dark pool metrics using regex
            result = self._parse_dark_pool_data(ticker, text)
            
            # Cache the result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return self._empty_dark_pool_result(ticker, str(e))
    
    def _fetch_with_firecrawl(self, url: str) -> Optional[str]:
        """Fetch page content using Firecrawl MCP for JavaScript rendering."""
        import subprocess
        import json
        
        try:
            cmd = [
                'manus-mcp-cli', 'tool', 'call', 'firecrawl_scrape',
                '--server', 'firecrawl',
                '--input', json.dumps({
                    'url': url,
                    'formats': ['markdown'],
                    'waitFor': 5000
                })
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # Parse the JSON output
                output = result.stdout
                # Find the JSON part
                json_start = output.find('{')
                if json_start >= 0:
                    json_str = output[json_start:]
                    data = json.loads(json_str)
                    return data.get('markdown', '')
            
            return None
            
        except Exception as e:
            print(f"Firecrawl error: {e}")
            return None
    
    def _parse_dark_pool_data(self, ticker: str, text: str) -> Dict:
        """Parse dark pool data from page text."""
        result = {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'source': 'StockGrid.io (FINRA TRF)',
            'status': 'success',
            'error': None
        }
        
        # Extract Net Short Volume
        net_short_match = re.search(r'Net Short Volume\s*([+-]?[\d,]+)', text)
        if net_short_match:
            result['net_short_volume'] = int(net_short_match.group(1).replace(',', ''))
        else:
            result['net_short_volume'] = None
        
        # Extract Net Short Volume $
        net_short_dollar_match = re.search(r'Net Short Volume \$\s*([+-]?[\d,]+)', text)
        if net_short_dollar_match:
            result['net_short_volume_dollar'] = int(net_short_dollar_match.group(1).replace(',', ''))
        else:
            result['net_short_volume_dollar'] = None
        
        # Extract Position (20-day)
        position_match = re.search(r'Position\s*([+-]?[\d,]+)', text)
        if position_match:
            result['position'] = int(position_match.group(1).replace(',', ''))
        else:
            result['position'] = None
        
        # Extract Position $
        position_dollar_match = re.search(r'Position \$\s*([+-]?[\d,]+)', text)
        if position_dollar_match:
            result['position_dollar'] = int(position_dollar_match.group(1).replace(',', ''))
        else:
            result['position_dollar'] = None
        
        # Parse short volume history table
        result['short_volume_history'] = self._parse_short_volume_table(ticker, text)
        
        # Calculate sentiment interpretation
        result['sentiment'] = self._interpret_dark_pool_sentiment(result)
        
        # Add explanation
        result['explanation'] = self._get_dark_pool_explanation(result)
        
        return result
    
    def _parse_short_volume_table(self, ticker: str, text: str) -> List[Dict]:
        """Parse the short volume history table."""
        history = []
        
        # Look for table rows with the ticker
        pattern = rf'{ticker.upper()}\s+(\d{{4}}-\d{{2}}-\d{{2}})\s+([\d,]+)\s+([\d.]+)\s+([\d,]+)\s+([\d,]+)'
        matches = re.findall(pattern, text)
        
        for match in matches[:15]:  # Last 15 days
            try:
                history.append({
                    'date': match[0],
                    'short_volume': int(match[1].replace(',', '')),
                    'short_volume_pct': float(match[2]),
                    'short_exempt_volume': int(match[3].replace(',', '')),
                    'total_volume': int(match[4].replace(',', ''))
                })
            except (ValueError, IndexError):
                continue
        
        return history
    
    def _interpret_dark_pool_sentiment(self, data: Dict) -> Dict:
        """
        Interpret dark pool data for trading sentiment.
        
        Key insight from SqueezeMetrics research:
        - NEGATIVE net short volume = More buying than shorting = BULLISH
        - POSITIVE net short volume = More shorting than buying = BEARISH
        
        This is counterintuitive but backed by research showing that
        short volume mostly represents market makers selling to buyers.
        """
        position = data.get('position', 0) or 0
        position_dollar = data.get('position_dollar', 0) or 0
        net_short = data.get('net_short_volume', 0) or 0
        
        # Determine sentiment based on position
        if position_dollar < -10_000_000_000:  # < -$10B
            sentiment = 'VERY_BULLISH'
            signal = 'Strong institutional buying detected'
            score = 85
        elif position_dollar < -1_000_000_000:  # < -$1B
            sentiment = 'BULLISH'
            signal = 'Institutional buying detected'
            score = 70
        elif position_dollar < -100_000_000:  # < -$100M
            sentiment = 'MODERATELY_BULLISH'
            signal = 'Moderate institutional buying'
            score = 60
        elif position_dollar > 10_000_000_000:  # > $10B
            sentiment = 'VERY_BEARISH'
            signal = 'Strong institutional selling detected'
            score = 15
        elif position_dollar > 1_000_000_000:  # > $1B
            sentiment = 'BEARISH'
            signal = 'Institutional selling detected'
            score = 30
        elif position_dollar > 100_000_000:  # > $100M
            sentiment = 'MODERATELY_BEARISH'
            signal = 'Moderate institutional selling'
            score = 40
        else:
            sentiment = 'NEUTRAL'
            signal = 'No significant institutional bias'
            score = 50
        
        # Daily trend
        if net_short < -1_000_000:
            daily_trend = 'BUYING'
        elif net_short > 1_000_000:
            daily_trend = 'SELLING'
        else:
            daily_trend = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'signal': signal,
            'score': score,
            'daily_trend': daily_trend,
            'interpretation': f"20-day position: ${position_dollar:,.0f}" if position_dollar else "N/A"
        }
    
    def _get_dark_pool_explanation(self, data: Dict) -> str:
        """Generate explanation text for dark pool data."""
        return """
**Dark Pool Data Interpretation:**

Dark pools are off-exchange trading venues where institutional investors execute large orders.
FINRA reports short volume data from these venues daily.

**Key Insight (SqueezeMetrics Research):**
- **Negative Position** = More buying than shorting = **BULLISH** for the stock
- **Positive Position** = More shorting than buying = **BEARISH** for the stock

This is counterintuitive because "short volume" mostly represents market makers 
selling shares short to meet buyer demand. High short volume often indicates 
strong buying pressure from institutions.

**How to Use:**
- Position $ < -$1B: Strong institutional accumulation (bullish)
- Position $ > $1B: Strong institutional distribution (bearish)
- Daily Net Short < 0: Today's flow is bullish
- Daily Net Short > 0: Today's flow is bearish
"""
    
    def _empty_dark_pool_result(self, ticker: str, error: str) -> Dict:
        """Return empty result structure."""
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'source': 'StockGrid.io (FINRA TRF)',
            'status': 'error',
            'error': error,
            'net_short_volume': None,
            'net_short_volume_dollar': None,
            'position': None,
            'position_dollar': None,
            'short_volume_history': [],
            'sentiment': {
                'sentiment': 'UNKNOWN',
                'signal': 'Data unavailable',
                'score': 50,
                'daily_trend': 'UNKNOWN'
            },
            'explanation': 'Dark pool data unavailable.'
        }
    
    def get_arima_forecast(self, ticker: str, periods: int = 5, 
                          p: int = 1, d: int = 1, q: int = 1) -> Dict:
        """
        Generate ARIMA forecast for a stock.
        
        Args:
            ticker: Stock symbol
            periods: Number of periods to forecast
            p: Autoregression order
            d: Differencing order
            q: Moving average order
            
        Returns:
            Dict with forecast data, confidence intervals, and model statistics
        """
        if not ARIMA_AVAILABLE:
            return self._empty_arima_result(ticker, "statsmodels not installed")
        
        cache_key = f"arima_{ticker}_{p}_{d}_{q}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Get historical price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty or len(hist) < 100:
                return self._empty_arima_result(ticker, "Insufficient historical data")
            
            # Use closing prices
            prices = hist['Close'].dropna()
            
            # Fit ARIMA model
            model = ARIMA(prices, order=(p, d, q))
            fitted = model.fit()
            
            # Generate forecast
            forecast = fitted.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            # Calculate model diagnostics
            aic = fitted.aic
            bic = fitted.bic
            
            # Stationarity test on original series
            adf_result = adfuller(prices)
            is_stationary = adf_result[1] < 0.05
            
            # Calculate forecast dates
            last_date = prices.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='B'  # Business days
            )
            
            # Build result
            result = {
                'ticker': ticker.upper(),
                'timestamp': datetime.now().isoformat(),
                'model': f'ARIMA({p},{d},{q})',
                'status': 'success',
                'error': None,
                
                # Current price info
                'current_price': float(prices.iloc[-1]),
                'last_date': str(prices.index[-1].date()),
                
                # Forecast
                'forecast': [
                    {
                        'date': str(forecast_dates[i].date()),
                        'predicted': float(forecast_mean.iloc[i]),
                        'lower_95': float(conf_int.iloc[i, 0]),
                        'upper_95': float(conf_int.iloc[i, 1])
                    }
                    for i in range(periods)
                ],
                
                # Model statistics
                'model_stats': {
                    'aic': float(aic),
                    'bic': float(bic),
                    'is_stationary': is_stationary,
                    'adf_pvalue': float(adf_result[1]),
                    'observations': len(prices)
                },
                
                # Historical data for charting
                'historical': [
                    {
                        'date': str(prices.index[i].date()),
                        'price': float(prices.iloc[i])
                    }
                    for i in range(-30, 0)  # Last 30 days
                ],
                
                # Interpretation
                'interpretation': self._interpret_arima_forecast(
                    prices.iloc[-1], forecast_mean, conf_int
                ),
                
                # Explanation
                'explanation': self._get_arima_explanation()
            }
            
            # Cache result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return self._empty_arima_result(ticker, str(e))
    
    def _interpret_arima_forecast(self, current: float, forecast: pd.Series, 
                                  conf_int: pd.DataFrame) -> Dict:
        """Interpret ARIMA forecast for trading signals."""
        final_forecast = float(forecast.iloc[-1])
        change_pct = ((final_forecast - current) / current) * 100
        
        # Confidence interval width as % of price
        ci_width = (conf_int.iloc[-1, 1] - conf_int.iloc[-1, 0]) / current * 100
        
        # Determine signal
        if change_pct > 5 and ci_width < 20:
            signal = 'STRONG_BUY'
            confidence = 'HIGH'
        elif change_pct > 2:
            signal = 'BUY'
            confidence = 'MEDIUM' if ci_width < 30 else 'LOW'
        elif change_pct < -5 and ci_width < 20:
            signal = 'STRONG_SELL'
            confidence = 'HIGH'
        elif change_pct < -2:
            signal = 'SELL'
            confidence = 'MEDIUM' if ci_width < 30 else 'LOW'
        else:
            signal = 'HOLD'
            confidence = 'MEDIUM'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'expected_change_pct': round(change_pct, 2),
            'forecast_price': round(final_forecast, 2),
            'ci_width_pct': round(ci_width, 2),
            'summary': f"ARIMA predicts {change_pct:+.1f}% change over forecast period"
        }
    
    def _get_arima_explanation(self) -> str:
        """Generate explanation text for ARIMA model."""
        return """
**ARIMA Model Explanation:**

ARIMA (AutoRegressive Integrated Moving Average) is a statistical model for 
time series forecasting that combines three components:

- **AR (p)**: Autoregression - uses past values to predict future values
- **I (d)**: Integration - differencing to make the series stationary
- **MA (q)**: Moving Average - uses past forecast errors

**How to Interpret:**
- **Forecast Line**: Predicted price trajectory
- **Confidence Interval**: 95% probability range for actual price
- **Narrow CI**: Higher confidence in prediction
- **Wide CI**: Lower confidence, more uncertainty

**Limitations:**
- ARIMA assumes patterns continue (may miss sudden changes)
- Works best for short-term forecasts (1-5 days)
- Should be combined with fundamental and technical analysis

**Best Practices:**
- Use as one input among many, not sole decision maker
- More reliable for liquid, established stocks
- Re-run daily as new data becomes available
"""
    
    def _empty_arima_result(self, ticker: str, error: str) -> Dict:
        """Return empty ARIMA result."""
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'model': 'ARIMA',
            'status': 'error',
            'error': error,
            'current_price': None,
            'forecast': [],
            'model_stats': {},
            'historical': [],
            'interpretation': {
                'signal': 'UNKNOWN',
                'confidence': 'NONE',
                'summary': f'ARIMA forecast unavailable: {error}'
            },
            'explanation': 'ARIMA forecast unavailable.'
        }
    
    def get_factor_analysis(self, ticker: str) -> Dict:
        """
        Get factor analysis signals from StockGrid.
        
        Factors include:
        - New Uptrend
        - Filtered Momentum
        - Momentum
        - New High
        - Relative Strength (S&P500)
        - ROC Curve
        - Sustained Momentum
        - Dark Pools
        """
        cache_key = f"factor_{ticker}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            url = f"{self.BASE_URL}/factor-analysis/{ticker.upper()}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return self._empty_factor_result(ticker, f"HTTP {response.status_code}")
            
            text = response.text
            result = self._parse_factor_analysis(ticker, text)
            
            # Cache result
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return self._empty_factor_result(ticker, str(e))
    
    def _parse_factor_analysis(self, ticker: str, text: str) -> Dict:
        """Parse factor analysis data from page."""
        soup = BeautifulSoup(text, 'html.parser')
        page_text = soup.get_text()
        
        result = {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'source': 'StockGrid.io Factor Analysis',
            'status': 'success',
            'error': None,
            'factors': {},
            'explanation': self._get_factor_explanation()
        }
        
        # Extract last signal information
        signal_match = re.search(
            r'last signal.*?(\d{4}-\d{2}-\d{2}).*?expected return.*?([\d.]+)%.*?probability.*?([\d.]+)%',
            page_text, re.IGNORECASE | re.DOTALL
        )
        
        if signal_match:
            result['last_signal'] = {
                'date': signal_match.group(1),
                'expected_return': float(signal_match.group(2)),
                'profit_probability': float(signal_match.group(3))
            }
        else:
            result['last_signal'] = None
        
        # Extract factor statistics from tables
        # Look for statistics patterns
        stats_pattern = r'profitable%\s+([\d.]+)'
        stats_match = re.search(stats_pattern, page_text)
        
        if stats_match:
            result['profit_probability'] = float(stats_match.group(1))
        
        # Determine overall signal
        result['overall_signal'] = self._determine_factor_signal(result)
        
        return result
    
    def _determine_factor_signal(self, data: Dict) -> Dict:
        """Determine overall trading signal from factor analysis."""
        last_signal = data.get('last_signal')
        
        if not last_signal:
            return {
                'signal': 'NEUTRAL',
                'confidence': 'LOW',
                'summary': 'No recent factor signals detected'
            }
        
        expected_return = last_signal.get('expected_return', 0)
        prob = last_signal.get('profit_probability', 50)
        
        if expected_return > 10 and prob > 70:
            signal = 'STRONG_BUY'
            confidence = 'HIGH'
        elif expected_return > 5 and prob > 60:
            signal = 'BUY'
            confidence = 'MEDIUM'
        elif expected_return < -5 and prob < 40:
            signal = 'SELL'
            confidence = 'MEDIUM'
        else:
            signal = 'HOLD'
            confidence = 'LOW'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'expected_return': expected_return,
            'profit_probability': prob,
            'summary': f"Factor signal: {expected_return:+.1f}% expected return, {prob:.0f}% win rate"
        }
    
    def _get_factor_explanation(self) -> str:
        """Generate explanation for factor analysis."""
        return """
**Factor Analysis Explanation:**

Factor analysis generates trade signals based on historical return distributions 
of specific market factors. Each factor has been backtested to determine its 
predictive power.

**Available Factors:**
- **New Uptrend**: Detects start of new bullish trends
- **Momentum**: Measures price momentum strength
- **Relative Strength**: Compares performance vs S&P 500
- **Dark Pools**: Uses institutional flow data
- **New High**: Identifies breakout to new highs

**How to Use:**
- **Expected Return**: Historical average return after signal
- **Profit Probability**: % of times signal was profitable
- **Profit-Loss Ratio**: Avg win / Avg loss

**Best Signals:**
- Expected Return > 10%
- Profit Probability > 70%
- Profit-Loss Ratio > 1.5

**Note**: Past performance doesn't guarantee future results.
Factor signals work best when combined with other analysis.
"""
    
    def _empty_factor_result(self, ticker: str, error: str) -> Dict:
        """Return empty factor analysis result."""
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'source': 'StockGrid.io Factor Analysis',
            'status': 'error',
            'error': error,
            'factors': {},
            'last_signal': None,
            'overall_signal': {
                'signal': 'UNKNOWN',
                'confidence': 'NONE',
                'summary': f'Factor analysis unavailable: {error}'
            },
            'explanation': 'Factor analysis unavailable.'
        }
    
    def get_full_stockgrid_analysis(self, ticker: str) -> Dict:
        """
        Get comprehensive StockGrid analysis including all data sources.
        
        Returns combined dark pool, ARIMA, and factor analysis data.
        """
        return {
            'ticker': ticker.upper(),
            'timestamp': datetime.now().isoformat(),
            'dark_pools': self.get_dark_pool_data(ticker),
            'arima': self.get_arima_forecast(ticker),
            'factor_analysis': self.get_factor_analysis(ticker),
            'combined_signal': self._calculate_combined_signal(ticker)
        }
    
    def _calculate_combined_signal(self, ticker: str) -> Dict:
        """Calculate combined signal from all StockGrid sources."""
        dark_pool = self.get_dark_pool_data(ticker)
        arima = self.get_arima_forecast(ticker)
        factor = self.get_factor_analysis(ticker)
        
        scores = []
        signals = []
        
        # Dark pool score
        dp_score = dark_pool.get('sentiment', {}).get('score', 50)
        scores.append(dp_score)
        
        # ARIMA score
        arima_signal = arima.get('interpretation', {}).get('signal', 'HOLD')
        arima_score = {
            'STRONG_BUY': 90, 'BUY': 70, 'HOLD': 50, 
            'SELL': 30, 'STRONG_SELL': 10, 'UNKNOWN': 50
        }.get(arima_signal, 50)
        scores.append(arima_score)
        
        # Factor score
        factor_signal = factor.get('overall_signal', {}).get('signal', 'NEUTRAL')
        factor_score = {
            'STRONG_BUY': 90, 'BUY': 70, 'HOLD': 50,
            'SELL': 30, 'NEUTRAL': 50, 'UNKNOWN': 50
        }.get(factor_signal, 50)
        scores.append(factor_score)
        
        # Calculate weighted average
        avg_score = sum(scores) / len(scores)
        
        # Determine combined signal
        if avg_score >= 75:
            combined = 'STRONG_BUY'
        elif avg_score >= 60:
            combined = 'BUY'
        elif avg_score <= 25:
            combined = 'STRONG_SELL'
        elif avg_score <= 40:
            combined = 'SELL'
        else:
            combined = 'HOLD'
        
        return {
            'signal': combined,
            'score': round(avg_score, 1),
            'components': {
                'dark_pool_score': dp_score,
                'arima_score': arima_score,
                'factor_score': factor_score
            },
            'summary': f"Combined StockGrid signal: {combined} (score: {avg_score:.0f}/100)"
        }


# Convenience functions for direct use
def get_dark_pool_data(ticker: str) -> Dict:
    """Get dark pool data for a ticker."""
    return StockGridIntegration().get_dark_pool_data(ticker)

def get_arima_forecast(ticker: str, periods: int = 5) -> Dict:
    """Get ARIMA forecast for a ticker."""
    return StockGridIntegration().get_arima_forecast(ticker, periods)

def get_factor_analysis(ticker: str) -> Dict:
    """Get factor analysis for a ticker."""
    return StockGridIntegration().get_factor_analysis(ticker)

def get_full_stockgrid_analysis(ticker: str) -> Dict:
    """Get full StockGrid analysis for a ticker."""
    return StockGridIntegration().get_full_stockgrid_analysis(ticker)


if __name__ == "__main__":
    # Test the integration
    import json
    
    print("Testing StockGrid Integration...")
    print("=" * 60)
    
    # Test dark pool data
    print("\n1. Dark Pool Data for AAPL:")
    dp_data = get_dark_pool_data("AAPL")
    print(f"   Net Short Volume: {dp_data.get('net_short_volume'):,}" if dp_data.get('net_short_volume') else "   Net Short Volume: N/A")
    print(f"   Position $: ${dp_data.get('position_dollar'):,.0f}" if dp_data.get('position_dollar') else "   Position $: N/A")
    print(f"   Sentiment: {dp_data.get('sentiment', {}).get('sentiment', 'N/A')}")
    
    # Test ARIMA forecast
    print("\n2. ARIMA Forecast for AAPL:")
    arima_data = get_arima_forecast("AAPL")
    if arima_data.get('status') == 'success':
        print(f"   Current Price: ${arima_data.get('current_price'):.2f}")
        print(f"   Model: {arima_data.get('model')}")
        print(f"   Signal: {arima_data.get('interpretation', {}).get('signal', 'N/A')}")
        if arima_data.get('forecast'):
            print(f"   5-Day Forecast: ${arima_data['forecast'][-1]['predicted']:.2f}")
    else:
        print(f"   Error: {arima_data.get('error')}")
    
    # Test factor analysis
    print("\n3. Factor Analysis for AAPL:")
    factor_data = get_factor_analysis("AAPL")
    print(f"   Signal: {factor_data.get('overall_signal', {}).get('signal', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("StockGrid Integration Test Complete!")
