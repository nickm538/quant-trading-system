"""
Intraday Chart Renderer
=======================
Generates professional candlestick chart images from Polygon.io intraday data.
Supports 5-minute, 15-minute, and hourly timeframes.

These chart images are fed to Vision AI for multi-timeframe analysis,
giving the system 5-minute-level precision that Finviz (paywalled) cannot provide.

Uses mplfinance for institutional-quality chart rendering with:
- OHLC candlesticks with proper coloring
- Volume bars
- 9-period and 21-period EMAs (fast/slow for intraday)
- VWAP overlay
- Price labels and grid
"""

import os
import sys
import io
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Polygon.io
try:
    from polygon import RESTClient as PolygonClient
    HAS_POLYGON = True
except ImportError:
    HAS_POLYGON = False

# mplfinance for candlestick charts
try:
    import mplfinance as mpf
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    HAS_MPF = True
except ImportError:
    HAS_MPF = False

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')


class IntradayChartRenderer:
    """
    Renders professional candlestick charts from Polygon.io intraday data.
    Returns chart images as PNG bytes suitable for Vision AI analysis.
    """

    # Timeframe configurations
    TIMEFRAMES = {
        '5min': {'multiplier': 5, 'timespan': 'minute', 'bars': 78, 'label': '5-Minute', 'days_back': 1},
        '15min': {'multiplier': 15, 'timespan': 'minute', 'bars': 52, 'label': '15-Minute', 'days_back': 2},
        '1hour': {'multiplier': 1, 'timespan': 'hour', 'bars': 40, 'label': '1-Hour', 'days_back': 5},
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is required for intraday data")
        if not HAS_POLYGON:
            raise ImportError("polygon-api-client is required: pip install polygon-api-client")
        if not HAS_MPF:
            raise ImportError("mplfinance is required: pip install mplfinance")

        self.client = PolygonClient(api_key=self.api_key)

    def fetch_intraday_data(self, symbol: str, timeframe: str = '5min') -> Optional[pd.DataFrame]:
        """
        Fetch intraday OHLCV data from Polygon.io.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            timeframe: One of '5min', '15min', '1hour'

        Returns:
            DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
            Returns None if data is unavailable.
        """
        config = self.TIMEFRAMES.get(timeframe)
        if not config:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use: {list(self.TIMEFRAMES.keys())}")

        symbol = symbol.upper().strip()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days_back'])

        # For weekends/holidays, extend lookback
        if end_date.weekday() >= 5:  # Saturday or Sunday
            start_date = end_date - timedelta(days=config['days_back'] + 3)

        try:
            aggs = list(self.client.list_aggs(
                ticker=symbol,
                multiplier=config['multiplier'],
                timespan=config['timespan'],
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                limit=50000,
                sort='asc',
            ))

            if not aggs or len(aggs) < 10:
                logger.warning(f"Insufficient intraday data for {symbol} ({timeframe}): {len(aggs) if aggs else 0} bars")
                return None

            # Convert to DataFrame
            data = []
            for a in aggs:
                ts = datetime.fromtimestamp(a.timestamp / 1000)
                data.append({
                    'Date': ts,
                    'Open': float(a.open),
                    'High': float(a.high),
                    'Low': float(a.low),
                    'Close': float(a.close),
                    'Volume': int(a.volume),
                })

            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            df.index = pd.DatetimeIndex(df.index)

            # Take the last N bars for the chart
            max_bars = config['bars']
            if len(df) > max_bars:
                df = df.tail(max_bars)

            return df

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price)."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cum_tp_vol = (typical_price * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum()
        vwap = cum_tp_vol / cum_vol
        return vwap

    def render_chart(
        self,
        symbol: str,
        timeframe: str = '5min',
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[bytes]:
        """
        Render a professional candlestick chart as PNG bytes.

        Args:
            symbol: Stock ticker
            timeframe: '5min', '15min', or '1hour'
            df: Pre-fetched DataFrame (optional, will fetch if not provided)

        Returns:
            PNG image bytes, or None if rendering fails
        """
        if df is None:
            df = self.fetch_intraday_data(symbol, timeframe)

        if df is None or len(df) < 10:
            return None

        config = self.TIMEFRAMES.get(timeframe, self.TIMEFRAMES['5min'])
        label = config['label']

        try:
            # Calculate overlays
            ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
            ema_21 = df['Close'].ewm(span=21, adjust=False).mean()
            vwap = self._calculate_vwap(df)

            # Build addplot list for overlays
            ap = [
                mpf.make_addplot(ema_9, color='#FF6B35', width=1.2, label='EMA 9'),
                mpf.make_addplot(ema_21, color='#4ECDC4', width=1.2, label='EMA 21'),
                mpf.make_addplot(vwap, color='#FFD700', width=1.5, linestyle='--', label='VWAP'),
            ]

            # Chart style - dark professional theme
            mc = mpf.make_marketcolors(
                up='#26A69A',       # Green for up candles
                down='#EF5350',     # Red for down candles
                edge='inherit',
                wick='inherit',
                volume={'up': '#26A69A', 'down': '#EF5350'},
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                figcolor='#1E1E2E',
                facecolor='#1E1E2E',
                edgecolor='#333',
                gridcolor='#333',
                gridstyle='--',
                gridaxis='both',
                y_on_right=True,
                rc={
                    'font.size': 9,
                    'axes.labelcolor': '#CCC',
                    'xtick.color': '#CCC',
                    'ytick.color': '#CCC',
                },
            )

            # Get current price info for title
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0
            change_sign = '+' if change >= 0 else ''

            title = (
                f"{symbol} {label} Chart | "
                f"${current_price:.2f} ({change_sign}{change:.2f}, {change_sign}{change_pct:.2f}%) | "
                f"EMA9/21 + VWAP"
            )

            # Render to bytes
            buf = io.BytesIO()
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=style,
                title=title,
                volume=True,
                addplot=ap,
                figsize=(14, 8),
                tight_layout=True,
                returnfig=True,
                datetime_format='%H:%M' if timeframe in ['5min', '15min'] else '%m/%d %H:%M',
                xrotation=30,
            )

            # Add legend manually
            ax_main = axes[0]
            ax_main.legend(
                ['EMA 9', 'EMA 21', 'VWAP'],
                loc='upper left',
                fontsize=8,
                facecolor='#1E1E2E',
                edgecolor='#555',
                labelcolor='#CCC',
            )

            # Add watermark
            ax_main.text(
                0.99, 0.01,
                f'Polygon.io | {datetime.now().strftime("%Y-%m-%d %H:%M")} EST',
                transform=ax_main.transAxes,
                fontsize=7,
                color='#666',
                ha='right',
                va='bottom',
            )

            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                        facecolor='#1E1E2E', edgecolor='none')
            plt.close(fig)

            buf.seek(0)
            return buf.read()

        except Exception as e:
            logger.error(f"Error rendering chart for {symbol} ({timeframe}): {e}")
            return None

    def render_multi_timeframe(self, symbol: str) -> Dict[str, Optional[bytes]]:
        """
        Render charts for all available timeframes.

        Returns:
            Dict of timeframe -> PNG bytes (or None if unavailable)
        """
        results = {}
        for tf in ['5min', '15min', '1hour']:
            try:
                results[tf] = self.render_chart(symbol, tf)
            except Exception as e:
                logger.error(f"Failed to render {tf} chart for {symbol}: {e}")
                results[tf] = None
        return results

    def get_intraday_summary(self, symbol: str, timeframe: str = '5min') -> Optional[Dict[str, Any]]:
        """
        Get a numerical summary of the intraday data (for non-visual analysis).
        This supplements the chart image with precise numbers.
        """
        df = self.fetch_intraday_data(symbol, timeframe)
        if df is None or len(df) < 10:
            return None

        current = df['Close'].iloc[-1]
        open_price = df['Open'].iloc[0]
        high = df['High'].max()
        low = df['Low'].min()
        avg_volume = df['Volume'].mean()
        total_volume = df['Volume'].sum()

        ema_9 = df['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
        ema_21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
        vwap = self._calculate_vwap(df).iloc[-1]

        # Determine intraday trend
        first_third = df['Close'].iloc[:len(df)//3].mean()
        last_third = df['Close'].iloc[-len(df)//3:].mean()
        if last_third > first_third * 1.002:
            intraday_trend = 'UPTREND'
        elif last_third < first_third * 0.998:
            intraday_trend = 'DOWNTREND'
        else:
            intraday_trend = 'SIDEWAYS'

        # EMA cross signal
        if ema_9 > ema_21:
            ema_signal = 'BULLISH'
        elif ema_9 < ema_21:
            ema_signal = 'BEARISH'
        else:
            ema_signal = 'NEUTRAL'

        # Price vs VWAP
        if current > vwap * 1.001:
            vwap_signal = 'ABOVE_VWAP'
        elif current < vwap * 0.999:
            vwap_signal = 'BELOW_VWAP'
        else:
            vwap_signal = 'AT_VWAP'

        # Volume trend (last 10 bars vs first 10 bars)
        if len(df) >= 20:
            early_vol = df['Volume'].iloc[:10].mean()
            late_vol = df['Volume'].iloc[-10:].mean()
            if early_vol > 0:
                vol_ratio = late_vol / early_vol
                if vol_ratio > 1.3:
                    volume_trend = 'INCREASING'
                elif vol_ratio < 0.7:
                    volume_trend = 'DECREASING'
                else:
                    volume_trend = 'STABLE'
            else:
                volume_trend = 'N/A'
        else:
            volume_trend = 'N/A'

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'bars': len(df),
            'current_price': round(current, 2),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'range_pct': round((high - low) / low * 100, 2) if low > 0 else 0,
            'change_pct': round((current - open_price) / open_price * 100, 2) if open_price > 0 else 0,
            'ema_9': round(ema_9, 2),
            'ema_21': round(ema_21, 2),
            'vwap': round(vwap, 2),
            'avg_volume': int(avg_volume),
            'total_volume': int(total_volume),
            'intraday_trend': intraday_trend,
            'ema_signal': ema_signal,
            'vwap_signal': vwap_signal,
            'volume_trend': volume_trend,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
        }


# CLI test
if __name__ == '__main__':
    import json

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'AAPL'
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '5min'

    renderer = IntradayChartRenderer()

    # Get summary
    summary = renderer.get_intraday_summary(symbol, timeframe)
    if summary:
        print(json.dumps(summary, indent=2))
    else:
        print(f"No intraday data available for {symbol} ({timeframe})")

    # Render chart
    chart_bytes = renderer.render_chart(symbol, timeframe)
    if chart_bytes:
        out_path = f'/tmp/{symbol}_{timeframe}_chart.png'
        with open(out_path, 'wb') as f:
            f.write(chart_bytes)
        print(f"\nChart saved to {out_path} ({len(chart_bytes)} bytes)")
    else:
        print(f"\nCould not render chart for {symbol} ({timeframe})")
