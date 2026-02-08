"""
POLYGON DATA PROVIDER - Single Source of Truth for All Price Data
================================================================
Replaces all yfinance calls across the pipeline with Polygon.io as primary,
yfinance as fallback. Fetches once, caches, and distributes to all modules.

Key design:
- Singleton pattern: one instance per analysis run, shared across all modules
- Caches all fetched data to eliminate redundant API calls
- Provides yfinance-compatible output formats so existing modules work with minimal changes
- Falls back to yfinance if Polygon fails (graceful degradation)
- Also integrates FinancialDatasets.ai for fundamental data
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── Polygon.io ───
try:
    from polygon import RESTClient as PolygonRESTClient
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False

# ─── yfinance (fallback only) ───
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ─── FinancialDatasets.ai ───
try:
    from financial_datasets_client import FinancialDatasetsClient
    HAS_FD = True
except ImportError:
    HAS_FD = False

# ─── TAAPI.io ───
try:
    from taapi_client import TaapiClient
    HAS_TAAPI = True
except ImportError:
    HAS_TAAPI = False


class PolygonDataProvider:
    """
    Centralized data provider that fetches once and distributes everywhere.
    
    Usage:
        provider = PolygonDataProvider.get_instance(symbol)
        # or
        provider = PolygonDataProvider(symbol)
        
        # Get data in various formats
        df = provider.get_daily_ohlcv()          # pandas DataFrame (yfinance-compatible)
        df = provider.get_intraday_ohlcv()       # 5-min bars
        info = provider.get_stock_info()          # dict like yf.Ticker.info
        fundamentals = provider.get_fundamentals() # from FinancialDatasets.ai
    """
    
    _instances: Dict[str, 'PolygonDataProvider'] = {}
    
    @classmethod
    def get_instance(cls, symbol: str, polygon_key: str = None) -> 'PolygonDataProvider':
        """Get or create a cached instance for this symbol."""
        symbol = symbol.upper()
        if symbol not in cls._instances:
            cls._instances[symbol] = cls(symbol, polygon_key)
        return cls._instances[symbol]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances (call between analysis runs)."""
        cls._instances.clear()
    
    def __init__(self, symbol: str, polygon_key: str = None):
        self.symbol = symbol.upper()
        self.polygon_key = polygon_key or os.environ.get('POLYGON_API_KEY', '')
        self._polygon_client = None
        self._fd_client = None
        self._taapi_client = None
        
        # Data caches
        self._daily_df: Optional[pd.DataFrame] = None
        self._intraday_df: Optional[pd.DataFrame] = None
        self._weekly_df: Optional[pd.DataFrame] = None
        self._stock_info: Optional[Dict] = None
        self._fundamentals: Optional[Dict] = None
        self._taapi_indicators: Optional[Dict] = None
        self._ticker_details: Optional[Dict] = None
        self._last_trade: Optional[Dict] = None
        
        # Data source tracking
        self.data_sources_used: Dict[str, str] = {}
        self._fetch_times: Dict[str, float] = {}
        
        # Initialize clients
        if HAS_POLYGON and self.polygon_key:
            try:
                self._polygon_client = PolygonRESTClient(api_key=self.polygon_key)
                print(f"  ✓ PolygonDataProvider: Polygon.io client initialized", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"  ✗ PolygonDataProvider: Polygon init failed: {e}", file=sys.stderr, flush=True)
        
        if HAS_FD:
            try:
                self._fd_client = FinancialDatasetsClient()
            except Exception:
                pass
        
        if HAS_TAAPI:
            try:
                self._taapi_client = TaapiClient()
            except Exception:
                pass
    
    # ═══════════════════════════════════════════════════════════════
    # PRICE DATA (OHLCV)
    # ═══════════════════════════════════════════════════════════════
    
    def get_daily_ohlcv(self, days: int = 400, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV data as a pandas DataFrame.
        Returns yfinance-compatible format: columns = [Open, High, Low, Close, Volume]
        Index = DatetimeIndex
        """
        if self._daily_df is not None and not force_refresh:
            # Return cached, possibly trimmed
            if len(self._daily_df) >= days:
                return self._daily_df.tail(days).copy()
            return self._daily_df.copy()
        
        t0 = time.time()
        
        # Try Polygon first
        df = self._fetch_daily_polygon(days)
        if df is not None and len(df) > 50:
            self._daily_df = df
            self.data_sources_used['daily_ohlcv'] = 'polygon'
            self._fetch_times['daily_ohlcv'] = time.time() - t0
            print(f"  ✓ Daily OHLCV: {len(df)} bars from Polygon ({self._fetch_times['daily_ohlcv']:.1f}s)", file=sys.stderr, flush=True)
            return df.copy()
        
        # Fallback to yfinance
        df = self._fetch_daily_yfinance(days)
        if df is not None and len(df) > 50:
            self._daily_df = df
            self.data_sources_used['daily_ohlcv'] = 'yfinance'
            self._fetch_times['daily_ohlcv'] = time.time() - t0
            print(f"  ✓ Daily OHLCV: {len(df)} bars from yfinance ({self._fetch_times['daily_ohlcv']:.1f}s)", file=sys.stderr, flush=True)
            return df.copy()
        
        print(f"  ✗ Daily OHLCV: No data available for {self.symbol}", file=sys.stderr, flush=True)
        return None
    
    def get_intraday_ohlcv(self, interval_minutes: int = 5, days: int = 5, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get intraday OHLCV data.
        Returns yfinance-compatible format.
        """
        if self._intraday_df is not None and not force_refresh:
            return self._intraday_df.copy()
        
        t0 = time.time()
        
        # Try Polygon
        df = self._fetch_intraday_polygon(interval_minutes, days)
        if df is not None and len(df) > 20:
            self._intraday_df = df
            self.data_sources_used['intraday_ohlcv'] = 'polygon'
            self._fetch_times['intraday_ohlcv'] = time.time() - t0
            return df.copy()
        
        # Fallback to yfinance
        df = self._fetch_intraday_yfinance(interval_minutes, days)
        if df is not None and len(df) > 20:
            self._intraday_df = df
            self.data_sources_used['intraday_ohlcv'] = 'yfinance'
            self._fetch_times['intraday_ohlcv'] = time.time() - t0
            return df.copy()
        
        return None
    
    def get_weekly_ohlcv(self, weeks: int = 104, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Get weekly OHLCV data (for longer-term analysis)."""
        if self._weekly_df is not None and not force_refresh:
            return self._weekly_df.copy()
        
        t0 = time.time()
        
        # Try Polygon
        if self._polygon_client:
            try:
                end = datetime.now().strftime('%Y-%m-%d')
                start = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
                aggs = list(self._polygon_client.list_aggs(
                    self.symbol, 1, 'week', start, end, limit=5000
                ))
                if aggs and len(aggs) > 10:
                    df = self._aggs_to_dataframe(aggs)
                    self._weekly_df = df
                    self.data_sources_used['weekly_ohlcv'] = 'polygon'
                    return df.copy()
            except Exception as e:
                print(f"  Polygon weekly failed: {e}", file=sys.stderr, flush=True)
        
        # Fallback: resample daily data
        daily = self.get_daily_ohlcv(days=weeks * 7)
        if daily is not None and len(daily) > 20:
            weekly = daily.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            self._weekly_df = weekly
            self.data_sources_used['weekly_ohlcv'] = 'resampled_daily'
            return weekly.copy()
        
        return None
    
    def get_current_price(self) -> Optional[float]:
        """Get the most recent closing price."""
        # Try last trade from Polygon
        if self._polygon_client:
            try:
                trade = self._polygon_client.get_last_trade(self.symbol)
                if trade and hasattr(trade, 'price') and trade.price:
                    return float(trade.price)
            except Exception:
                pass
        
        # Fallback to latest bar
        df = self.get_daily_ohlcv(days=5)
        if df is not None and len(df) > 0:
            return float(df['Close'].iloc[-1])
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # STOCK INFO (replaces yf.Ticker.info)
    # ═══════════════════════════════════════════════════════════════
    
    def get_stock_info(self, force_refresh: bool = False) -> Dict:
        """
        Get stock info dict compatible with yf.Ticker.info.
        Combines Polygon ticker details + FinancialDatasets fundamentals.
        """
        if self._stock_info is not None and not force_refresh:
            return self._stock_info.copy()
        
        info = {}
        
        # 1. Polygon ticker details
        if self._polygon_client:
            try:
                details = self._polygon_client.get_ticker_details(self.symbol)
                if details:
                    info.update({
                        'symbol': self.symbol,
                        'shortName': getattr(details, 'name', self.symbol),
                        'longName': getattr(details, 'name', self.symbol),
                        'sector': getattr(details, 'sic_description', ''),
                        'industry': getattr(details, 'sic_description', ''),
                        'marketCap': getattr(details, 'market_cap', None),
                        'sharesOutstanding': getattr(details, 'share_class_shares_outstanding', None) or getattr(details, 'weighted_shares_outstanding', None),
                        'description': getattr(details, 'description', ''),
                        'exchange': getattr(details, 'primary_exchange', ''),
                        'currency': getattr(details, 'currency_name', 'USD'),
                        'homepage_url': getattr(details, 'homepage_url', ''),
                        'locale': getattr(details, 'locale', 'us'),
                        'type': getattr(details, 'type', ''),
                    })
                    self._ticker_details = info.copy()
                    self.data_sources_used['ticker_details'] = 'polygon'
            except Exception as e:
                print(f"  Polygon ticker details failed: {e}", file=sys.stderr, flush=True)
        
        # 2. Add price data
        price = self.get_current_price()
        if price:
            info['currentPrice'] = price
            info['regularMarketPrice'] = price
        
        daily = self.get_daily_ohlcv(days=5)
        if daily is not None and len(daily) >= 2:
            info['previousClose'] = float(daily['Close'].iloc[-2])
            info['open'] = float(daily['Open'].iloc[-1])
            info['dayHigh'] = float(daily['High'].iloc[-1])
            info['dayLow'] = float(daily['Low'].iloc[-1])
            info['volume'] = int(daily['Volume'].iloc[-1])
            info['averageVolume'] = int(daily['Volume'].tail(20).mean()) if len(daily) >= 20 else int(daily['Volume'].mean())
        
        # 3. Add 52-week high/low
        daily_year = self.get_daily_ohlcv(days=260)
        if daily_year is not None and len(daily_year) > 50:
            info['fiftyTwoWeekHigh'] = float(daily_year['High'].max())
            info['fiftyTwoWeekLow'] = float(daily_year['Low'].min())
            info['fiftyDayAverage'] = float(daily_year['Close'].tail(50).mean())
            info['twoHundredDayAverage'] = float(daily_year['Close'].tail(200).mean()) if len(daily_year) >= 200 else None
        
        # 4. Add fundamental data from FinancialDatasets.ai
        fundamentals = self.get_fundamentals()
        if fundamentals:
            metrics = fundamentals.get('financial_metrics', {})
            if isinstance(metrics, dict):
                fm_list = metrics.get('financial_metrics', [])
            elif isinstance(metrics, list):
                fm_list = metrics
            else:
                fm_list = []
            
            if fm_list and isinstance(fm_list, list) and len(fm_list) > 0:
                latest = fm_list[0] if isinstance(fm_list[0], dict) else {}
                info.update({
                    'trailingPE': latest.get('pe_ratio'),
                    'forwardPE': latest.get('forward_pe_ratio'),
                    'priceToBook': latest.get('price_to_book'),
                    'returnOnEquity': latest.get('roe'),
                    'debtToEquity': latest.get('debt_to_equity'),
                    'currentRatio': latest.get('current_ratio'),
                    'revenueGrowth': latest.get('revenue_growth'),
                    'earningsGrowth': latest.get('earnings_growth'),
                    'profitMargins': latest.get('net_margin'),
                    'operatingMargins': latest.get('operating_margin'),
                    'grossMargins': latest.get('gross_margin'),
                    'dividendYield': latest.get('dividend_yield'),
                    'payoutRatio': latest.get('payout_ratio'),
                    'beta': latest.get('beta'),
                })
            
            # Company facts
            facts = fundamentals.get('company_facts', {})
            if isinstance(facts, dict):
                info.update({
                    'sector': facts.get('sector', info.get('sector', '')),
                    'industry': facts.get('industry', info.get('industry', '')),
                    'fullTimeEmployees': facts.get('employees'),
                    'country': facts.get('country'),
                    'city': facts.get('city'),
                    'state': facts.get('state'),
                })
        
        # 5. Fallback to yfinance for missing critical fields
        critical_missing = [k for k in ['trailingPE', 'marketCap', 'sector'] if not info.get(k)]
        if critical_missing and HAS_YF:
            try:
                ticker = yf.Ticker(self.symbol)
                yf_info = ticker.info
                for key in critical_missing:
                    if key in yf_info and yf_info[key]:
                        info[key] = yf_info[key]
                # Also grab analyst targets if available
                for key in ['targetMeanPrice', 'targetHighPrice', 'targetLowPrice',
                            'recommendationKey', 'numberOfAnalystOpinions', 'earningsDate']:
                    if key in yf_info and yf_info[key]:
                        info[key] = yf_info[key]
                self.data_sources_used['stock_info_yf_supplement'] = 'yfinance'
            except Exception:
                pass
        
        self._stock_info = info
        return info.copy()
    
    # ═══════════════════════════════════════════════════════════════
    # FUNDAMENTAL DATA (FinancialDatasets.ai primary)
    # ═══════════════════════════════════════════════════════════════
    
    def get_fundamentals(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get comprehensive fundamental data from FinancialDatasets.ai.
        Returns dict with: financial_metrics, price_snapshot, company_facts
        """
        if self._fundamentals is not None and not force_refresh:
            return self._fundamentals.copy()
        
        if not self._fd_client:
            return None
        
        t0 = time.time()
        result = {}
        
        try:
            # Financial metrics (P/E, ROE, debt ratios, etc.)
            metrics = self._fd_client.get_financial_metrics(self.symbol, limit=10)
            if metrics:
                result['financial_metrics'] = metrics
            
            # Price snapshot
            snapshot = self._fd_client.get_stock_price_snapshot(self.symbol)
            if snapshot:
                result['price_snapshot'] = snapshot
            
            # Company facts
            facts = self._fd_client.get_company_facts(self.symbol)
            if facts:
                result['company_facts'] = facts
            
            if result:
                self._fundamentals = result
                self.data_sources_used['fundamentals'] = 'financialdatasets'
                self._fetch_times['fundamentals'] = time.time() - t0
                print(f"  ✓ Fundamentals: {len(result)} sections from FinancialDatasets ({self._fetch_times['fundamentals']:.1f}s)", file=sys.stderr, flush=True)
                return result.copy()
        except Exception as e:
            print(f"  FinancialDatasets failed: {e}", file=sys.stderr, flush=True)
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # TAAPI CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    def get_taapi_indicators(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get technical indicators from TAAPI.io for cross-validation.
        Returns dict with: rsi, macd, supertrend, adx, bbands, stoch, ema, sma
        """
        if self._taapi_indicators is not None and not force_refresh:
            return self._taapi_indicators.copy()
        
        if not self._taapi_client:
            return None
        
        t0 = time.time()
        indicators = {}
        
        indicator_list = ['rsi', 'macd', 'supertrend', 'adx', 'bbands', 'stoch', 'ema', 'sma']
        
        for ind in indicator_list:
            try:
                result = self._taapi_client.get_indicator(ind, self.symbol, '1d')
                if result:
                    indicators[ind] = result
            except Exception:
                pass
        
        if indicators:
            self._taapi_indicators = indicators
            self.data_sources_used['taapi'] = 'taapi.io'
            self._fetch_times['taapi'] = time.time() - t0
            print(f"  ✓ TAAPI: {len(indicators)} indicators ({self._fetch_times['taapi']:.1f}s)", file=sys.stderr, flush=True)
            return indicators.copy()
        
        return None
    
    def cross_validate_technicals(self, local_indicators: Dict) -> Dict:
        """
        Cross-validate locally-calculated technical indicators against TAAPI.io.
        
        Returns a confidence adjustment dict:
        {
            'confidence_modifier': float (-10 to +10),
            'agreements': [...],
            'divergences': [...],
            'taapi_data': {...}
        }
        """
        taapi = self.get_taapi_indicators()
        if not taapi:
            return {'confidence_modifier': 0, 'agreements': [], 'divergences': [], 'taapi_data': None}
        
        agreements = []
        divergences = []
        confidence_modifier = 0.0
        
        # RSI comparison
        local_rsi = local_indicators.get('rsi')
        taapi_rsi = taapi.get('rsi', {}).get('value')
        if local_rsi is not None and taapi_rsi is not None:
            rsi_diff = abs(float(local_rsi) - float(taapi_rsi))
            if rsi_diff <= 3:
                agreements.append(f"RSI agrees: local={local_rsi:.1f}, TAAPI={taapi_rsi:.1f} (diff={rsi_diff:.1f})")
                confidence_modifier += 3.0
            elif rsi_diff <= 7:
                agreements.append(f"RSI close: local={local_rsi:.1f}, TAAPI={taapi_rsi:.1f} (diff={rsi_diff:.1f})")
                confidence_modifier += 1.0
            else:
                divergences.append(f"RSI DIVERGES: local={local_rsi:.1f}, TAAPI={taapi_rsi:.1f} (diff={rsi_diff:.1f})")
                confidence_modifier -= 5.0
        
        # MACD comparison (direction agreement)
        local_macd = local_indicators.get('macd')
        taapi_macd = taapi.get('macd', {})
        if local_macd is not None and taapi_macd:
            taapi_macd_val = taapi_macd.get('valueMACD')
            taapi_signal = taapi_macd.get('valueMACDSignal')
            if taapi_macd_val is not None and taapi_signal is not None:
                taapi_bullish = float(taapi_macd_val) > float(taapi_signal)
                local_bullish = float(local_macd) > 0  # Simplified: positive MACD = bullish
                if taapi_bullish == local_bullish:
                    agreements.append(f"MACD direction agrees: both {'bullish' if taapi_bullish else 'bearish'}")
                    confidence_modifier += 2.0
                else:
                    divergences.append(f"MACD direction DIVERGES: local={'bullish' if local_bullish else 'bearish'}, TAAPI={'bullish' if taapi_bullish else 'bearish'}")
                    confidence_modifier -= 3.0
        
        # ADX comparison (trend strength)
        local_adx = local_indicators.get('adx')
        taapi_adx = taapi.get('adx', {}).get('value')
        if local_adx is not None and taapi_adx is not None:
            adx_diff = abs(float(local_adx) - float(taapi_adx))
            if adx_diff <= 5:
                agreements.append(f"ADX agrees: local={local_adx:.1f}, TAAPI={taapi_adx:.1f}")
                confidence_modifier += 2.0
            elif adx_diff > 10:
                divergences.append(f"ADX DIVERGES: local={local_adx:.1f}, TAAPI={taapi_adx:.1f}")
                confidence_modifier -= 2.0
        
        # Bollinger Bands comparison
        taapi_bb = taapi.get('bbands', {})
        local_bb_upper = local_indicators.get('bb_upper')
        local_bb_lower = local_indicators.get('bb_lower')
        if taapi_bb and local_bb_upper is not None and local_bb_lower is not None:
            taapi_upper = taapi_bb.get('valueUpperBand')
            taapi_lower = taapi_bb.get('valueLowerBand')
            if taapi_upper and taapi_lower:
                upper_diff_pct = abs(float(local_bb_upper) - float(taapi_upper)) / float(taapi_upper) * 100
                if upper_diff_pct <= 1.0:
                    agreements.append(f"Bollinger Bands agree (upper diff: {upper_diff_pct:.2f}%)")
                    confidence_modifier += 1.5
                elif upper_diff_pct > 3.0:
                    divergences.append(f"Bollinger Bands diverge (upper diff: {upper_diff_pct:.2f}%)")
                    confidence_modifier -= 1.5
        
        # Supertrend direction
        taapi_st = taapi.get('supertrend', {})
        if taapi_st:
            st_val = taapi_st.get('value')
            st_dir = taapi_st.get('valueAdvice')  # 'long' or 'short'
            local_signal = local_indicators.get('signal', '').upper()
            if st_dir and local_signal:
                taapi_bullish = st_dir.lower() == 'long'
                local_bullish = local_signal in ('BUY', 'STRONG_BUY')
                if taapi_bullish == local_bullish:
                    agreements.append(f"Supertrend agrees: both {'bullish' if taapi_bullish else 'bearish'}")
                    confidence_modifier += 2.0
                else:
                    divergences.append(f"Supertrend DIVERGES: local={local_signal}, TAAPI={st_dir}")
                    confidence_modifier -= 3.0
        
        # Cap the modifier
        confidence_modifier = max(-10.0, min(10.0, confidence_modifier))
        
        return {
            'confidence_modifier': round(confidence_modifier, 1),
            'agreements': agreements,
            'divergences': divergences,
            'agreement_count': len(agreements),
            'divergence_count': len(divergences),
            'cross_validation_score': round(len(agreements) / max(len(agreements) + len(divergences), 1) * 100, 1),
            'taapi_data': taapi
        }
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY / DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════
    
    def get_data_summary(self) -> Dict:
        """Get a summary of all data sources used and fetch times."""
        return {
            'symbol': self.symbol,
            'data_sources': self.data_sources_used.copy(),
            'fetch_times': {k: round(v, 2) for k, v in self._fetch_times.items()},
            'cache_status': {
                'daily_ohlcv': self._daily_df is not None,
                'intraday_ohlcv': self._intraday_df is not None,
                'weekly_ohlcv': self._weekly_df is not None,
                'stock_info': self._stock_info is not None,
                'fundamentals': self._fundamentals is not None,
                'taapi_indicators': self._taapi_indicators is not None,
            },
            'daily_bars': len(self._daily_df) if self._daily_df is not None else 0,
            'intraday_bars': len(self._intraday_df) if self._intraday_df is not None else 0,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: Polygon fetchers
    # ═══════════════════════════════════════════════════════════════
    
    def _fetch_daily_polygon(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV from Polygon.io."""
        if not self._polygon_client:
            return None
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            # Request extra days to account for weekends/holidays
            start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime('%Y-%m-%d')
            aggs = list(self._polygon_client.list_aggs(
                self.symbol, 1, 'day', start, end, limit=5000
            ))
            if aggs and len(aggs) > 0:
                return self._aggs_to_dataframe(aggs)
        except Exception as e:
            print(f"  Polygon daily fetch failed: {e}", file=sys.stderr, flush=True)
        return None
    
    def _fetch_intraday_polygon(self, interval_minutes: int, days: int) -> Optional[pd.DataFrame]:
        """Fetch intraday OHLCV from Polygon.io."""
        if not self._polygon_client:
            return None
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            aggs = list(self._polygon_client.list_aggs(
                self.symbol, interval_minutes, 'minute', start, end, limit=50000
            ))
            if aggs and len(aggs) > 0:
                return self._aggs_to_dataframe(aggs)
        except Exception as e:
            print(f"  Polygon intraday fetch failed: {e}", file=sys.stderr, flush=True)
        return None
    
    def _aggs_to_dataframe(self, aggs: list) -> pd.DataFrame:
        """Convert Polygon aggregates to yfinance-compatible DataFrame."""
        data = []
        for a in aggs:
            ts = getattr(a, 'timestamp', None)
            if ts:
                # Polygon timestamps are in milliseconds
                dt = pd.Timestamp(ts, unit='ms', tz='UTC').tz_convert('US/Eastern').tz_localize(None)
            else:
                continue
            data.append({
                'Date': dt,
                'Open': float(getattr(a, 'open', 0)),
                'High': float(getattr(a, 'high', 0)),
                'Low': float(getattr(a, 'low', 0)),
                'Close': float(getattr(a, 'close', 0)),
                'Volume': int(getattr(a, 'volume', 0)),
            })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='last')]
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # PRIVATE: yfinance fallbacks
    # ═══════════════════════════════════════════════════════════════
    
    def _fetch_daily_yfinance(self, days: int) -> Optional[pd.DataFrame]:
        """Fallback: fetch daily OHLCV from yfinance."""
        if not HAS_YF:
            return None
        try:
            period = '2y' if days > 365 else '1y' if days > 180 else '6mo' if days > 90 else '3mo'
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=period)
            if hist is not None and len(hist) > 0:
                # Normalize column names
                hist = hist.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                })
                # Keep only OHLCV columns
                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in hist.columns]
                return hist[cols]
        except Exception as e:
            print(f"  yfinance daily fallback failed: {e}", file=sys.stderr, flush=True)
        return None
    
    def _fetch_intraday_yfinance(self, interval_minutes: int, days: int) -> Optional[pd.DataFrame]:
        """Fallback: fetch intraday OHLCV from yfinance."""
        if not HAS_YF:
            return None
        try:
            interval_map = {1: '1m', 5: '5m', 15: '15m', 30: '30m', 60: '1h'}
            interval = interval_map.get(interval_minutes, '5m')
            period = f'{days}d'
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=period, interval=interval)
            if hist is not None and len(hist) > 0:
                hist = hist.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'
                })
                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in hist.columns]
                return hist[cols]
        except Exception as e:
            print(f"  yfinance intraday fallback failed: {e}", file=sys.stderr, flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (for modules that just need a DataFrame)
# ═══════════════════════════════════════════════════════════════════

def get_stock_data(symbol: str, days: int = 400) -> Optional[pd.DataFrame]:
    """Quick helper: get daily OHLCV for a symbol."""
    provider = PolygonDataProvider.get_instance(symbol)
    return provider.get_daily_ohlcv(days=days)

def get_stock_info(symbol: str) -> Dict:
    """Quick helper: get stock info dict for a symbol."""
    provider = PolygonDataProvider.get_instance(symbol)
    return provider.get_stock_info()

def get_current_price(symbol: str) -> Optional[float]:
    """Quick helper: get current price for a symbol."""
    provider = PolygonDataProvider.get_instance(symbol)
    return provider.get_current_price()
