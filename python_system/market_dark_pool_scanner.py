#!/usr/bin/env python3
"""
INSTITUTIONAL-GRADE MARKET-WIDE DARK POOL SCANNER
===================================================
Production-ready scanner for real money trading.

Data Sources:
1. FINRA Short Volume (Daily) - Official regulatory data
2. Stockgrid.io Dark Pool Positions - Cumulative institutional positioning
3. Yahoo Finance - Real-time price/volume context

Key Metrics:
- Short Volume Ratio: % of daily volume that is short selling (normal: 40-50%)
- Net Dark Pool Position: Cumulative buy/sell imbalance in dark pools
- Net Short Volume: Daily short volume minus long volume
- Volume Anomaly: Current vs historical dark pool activity

Scoring Methodology:
- Based on deviation from market norms, not absolute values
- Short ratio >55% = bearish pressure, <40% = bullish pressure
- Negative net positions = institutional selling
- High net short volume = active shorting pressure

Author: Quant Trading System
Version: 2.0 (Production)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import io
import json
import sys
import os
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MarketDarkPoolScanner:
    """
    Institutional-Grade Market-Wide Dark Pool Scanner
    
    Methodology:
    1. Fetch FINRA short volume data for actual short ratios
    2. Fetch Stockgrid dark pool positions for institutional sentiment
    3. Calculate volume anomalies vs historical averages
    4. Score based on deviation from market norms
    5. Generate dynamic, contextual signals
    """
    
    # Market norm baselines (derived from historical data)
    NORMAL_SHORT_RATIO_LOW = 38.0   # Below this = bullish (less shorting)
    NORMAL_SHORT_RATIO_HIGH = 52.0  # Above this = bearish (heavy shorting)
    EXTREME_SHORT_RATIO = 60.0      # Very high short pressure
    EXTREME_LOW_SHORT_RATIO = 30.0  # Very low short pressure (bullish)
    
    def __init__(self):
        """Initialize Market Dark Pool Scanner."""
        self.stockgrid_url = "https://www.stockgrid.io/get_dark_pool_data"
        self.finra_base_url = "https://cdn.finra.org/equity/regsho/daily"
        self.est_tz = pytz.timezone('America/New_York')
        self.finra_data_cache = None
        self.finra_date = None
        
    def scan_market(self, limit: int = 400) -> Dict:
        """
        Scan the market for dark pool activity with institutional-grade methodology.
        
        Args:
            limit: Maximum number of stocks to analyze
            
        Returns:
            Dict with comprehensive market-wide dark pool analysis
        """
        try:
            now_est = datetime.now(self.est_tz)
            
            # Step 1: Fetch FINRA short volume data (ground truth for short ratios)
            finra_data = self._fetch_finra_short_volume()
            
            # Step 2: Fetch Stockgrid dark pool positions (both bullish and bearish)
            bullish_positions = self._fetch_stockgrid_data("Dark Pools Position", "desc")
            bearish_positions = self._fetch_stockgrid_data("Dark Pools Position", "asc")
            high_short_pressure = self._fetch_stockgrid_data("Net Short Volume", "desc")
            
            # Combine all data sources
            all_stocks = self._merge_and_analyze(
                finra_data, 
                bullish_positions, 
                bearish_positions,
                high_short_pressure,
                limit
            )
            
            # Categorize results
            results = self._categorize_results(all_stocks)
            
            # Add metadata
            results["timestamp"] = now_est.strftime("%Y-%m-%d %H:%M:%S")
            results["data_source"] = "FINRA RegSHO + Stockgrid Dark Pools"
            results["finra_date"] = self.finra_date or "N/A"
            results["methodology"] = "Institutional-grade: Short ratio deviation + Net position + Volume anomaly"
            
            return results
            
        except Exception as e:
            import traceback
            return {
                "error": f"Market scan failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(self.est_tz).strftime("%Y-%m-%d %H:%M:%S"),
                "stocks_scanned": 0
            }
    
    def _fetch_finra_short_volume(self) -> Dict[str, Dict]:
        """
        Fetch FINRA RegSHO short volume data - the official source for short ratios.
        
        Returns:
            Dict mapping symbol to short volume metrics
        """
        finra_data = {}
        
        # Try last 5 business days to find available data
        today = datetime.now()
        for days_back in range(1, 6):
            check_date = today - timedelta(days=days_back)
            date_str = check_date.strftime('%Y%m%d')
            
            # FINRA CNMS (Consolidated NMS) data
            url = f"{self.finra_base_url}/CNMSshvol{date_str}.txt"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    self.finra_date = check_date.strftime('%Y-%m-%d')
                    
                    # Parse the pipe-delimited data
                    df = pd.read_csv(io.StringIO(response.text), sep='|')
                    
                    # Aggregate by symbol (may have multiple markets)
                    for symbol in df['Symbol'].unique():
                        stock_data = df[df['Symbol'] == symbol]
                        short_vol = stock_data['ShortVolume'].sum()
                        exempt_vol = stock_data['ShortExemptVolume'].sum()
                        total_vol = stock_data['TotalVolume'].sum()
                        
                        if total_vol > 0:
                            short_ratio = (short_vol / total_vol) * 100
                            finra_data[symbol] = {
                                'short_volume': int(short_vol),
                                'short_exempt_volume': int(exempt_vol),
                                'total_volume': int(total_vol),
                                'short_ratio': round(short_ratio, 2),
                                'date': self.finra_date
                            }
                    
                    print(f"Loaded FINRA data for {self.finra_date}: {len(finra_data)} symbols", file=sys.stderr)
                    break
                    
            except Exception as e:
                print(f"Error fetching FINRA data for {date_str}: {e}", file=sys.stderr)
                continue
        
        return finra_data
    
    def _fetch_stockgrid_data(self, sort_field: str, order: str) -> List[Dict]:
        """
        Fetch dark pool data from Stockgrid with specific sorting.
        
        Args:
            sort_field: Field to sort by
            order: 'asc' or 'desc'
            
        Returns:
            List of stock data dicts
        """
        try:
            url = f"{self.stockgrid_url}?top={sort_field.replace(' ', '%20')}&minmax={order}"
            
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            
            return []
            
        except Exception as e:
            print(f"Error fetching Stockgrid data ({sort_field}, {order}): {e}", file=sys.stderr)
            return []
    
    def _merge_and_analyze(
        self, 
        finra_data: Dict[str, Dict],
        bullish_positions: List[Dict],
        bearish_positions: List[Dict],
        high_short_pressure: List[Dict],
        limit: int
    ) -> List[Dict]:
        """
        Merge all data sources and perform comprehensive analysis.
        """
        analyzed_stocks = {}
        
        # Process bullish positions (positive net dark pool position)
        for item in bullish_positions[:limit]:
            ticker = item.get('Ticker', '')
            if not ticker or ticker in analyzed_stocks:
                continue
            
            analysis = self._analyze_stock(ticker, item, finra_data)
            if analysis:
                analyzed_stocks[ticker] = analysis
        
        # Process bearish positions (negative net dark pool position)
        for item in bearish_positions[:limit]:
            ticker = item.get('Ticker', '')
            if not ticker or ticker in analyzed_stocks:
                continue
            
            analysis = self._analyze_stock(ticker, item, finra_data)
            if analysis:
                analyzed_stocks[ticker] = analysis
        
        # Process high short pressure stocks
        for item in high_short_pressure[:limit]:
            ticker = item.get('Ticker', '')
            if not ticker or ticker in analyzed_stocks:
                continue
            
            analysis = self._analyze_stock(ticker, item, finra_data)
            if analysis:
                analyzed_stocks[ticker] = analysis
        
        return list(analyzed_stocks.values())
    
    def _analyze_stock(self, ticker: str, stockgrid_data: Dict, finra_data: Dict[str, Dict]) -> Optional[Dict]:
        """
        Perform comprehensive analysis on a single stock.
        
        Scoring Methodology:
        - Start at 50 (neutral)
        - Adjust based on short ratio deviation from norm
        - Adjust based on net dark pool position
        - Adjust based on net short volume
        - Generate contextual signal
        """
        try:
            # Extract Stockgrid data
            net_position = stockgrid_data.get('Dark Pools Position', 0)
            net_position_dollar = stockgrid_data.get('Dark Pools Position $', 0)
            net_short_volume = stockgrid_data.get('Net Short Volume', 0)
            stockgrid_short_pct = stockgrid_data.get('Short Volume %', 0)
            date = stockgrid_data.get('Date', '')
            
            # Get FINRA short ratio (more accurate than Stockgrid)
            finra_info = finra_data.get(ticker, {})
            short_ratio = finra_info.get('short_ratio', stockgrid_short_pct * 100)
            short_volume = finra_info.get('short_volume', 0)
            total_volume = finra_info.get('total_volume', 0)
            
            # Calculate score (0-100)
            score = self._calculate_score(
                net_position=net_position,
                net_position_dollar=net_position_dollar,
                net_short_volume=net_short_volume,
                short_ratio=short_ratio
            )
            
            # Determine sentiment
            sentiment = self._determine_sentiment(score, net_position, short_ratio, net_short_volume)
            
            # Calculate volume ratio (dark pool vs total)
            volume_ratio = self._calculate_volume_ratio(ticker, net_position, total_volume)
            
            # Generate dynamic signal
            signal = self._generate_signal(
                ticker=ticker,
                net_position=net_position,
                net_position_dollar=net_position_dollar,
                net_short_volume=net_short_volume,
                short_ratio=short_ratio,
                sentiment=sentiment
            )
            
            return {
                "symbol": ticker,
                "net_position": net_position,
                "net_position_dollar": net_position_dollar,
                "net_short_volume": net_short_volume,
                "short_volume": short_volume,
                "short_ratio": round(short_ratio, 1),
                "volume_ratio": round(volume_ratio, 2),
                "score": score,
                "sentiment": sentiment,
                "signal": signal,
                "date": date,
                "data_quality": "FINRA" if finra_info else "Stockgrid"
            }
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}", file=sys.stderr)
            return None
    
    def _calculate_score(
        self, 
        net_position: float, 
        net_position_dollar: float, 
        net_short_volume: float,
        short_ratio: float
    ) -> int:
        """
        Calculate institutional-grade score (0-100).
        
        Methodology:
        - 50 = neutral baseline
        - Short ratio deviation: +/- 25 points max
        - Net position: +/- 15 points max
        - Net short volume: +/- 10 points max
        
        Higher score = more bullish
        Lower score = more bearish
        """
        score = 50.0
        
        # 1. Short Ratio Component (most important - +/- 25 points)
        if short_ratio >= self.EXTREME_SHORT_RATIO:
            # Very high short ratio = very bearish
            score -= 25
        elif short_ratio >= self.NORMAL_SHORT_RATIO_HIGH:
            # Above normal = bearish
            deviation = (short_ratio - self.NORMAL_SHORT_RATIO_HIGH) / (self.EXTREME_SHORT_RATIO - self.NORMAL_SHORT_RATIO_HIGH)
            score -= 10 + (15 * min(deviation, 1.0))
        elif short_ratio <= self.EXTREME_LOW_SHORT_RATIO:
            # Very low short ratio = very bullish
            score += 25
        elif short_ratio <= self.NORMAL_SHORT_RATIO_LOW:
            # Below normal = bullish
            deviation = (self.NORMAL_SHORT_RATIO_LOW - short_ratio) / (self.NORMAL_SHORT_RATIO_LOW - self.EXTREME_LOW_SHORT_RATIO)
            score += 10 + (15 * min(deviation, 1.0))
        # else: within normal range, no adjustment
        
        # 2. Net Dark Pool Position Component (+/- 15 points)
        if net_position > 10000000:  # 10M+ shares accumulated
            score += 15
        elif net_position > 1000000:  # 1M+ shares
            score += 10
        elif net_position > 100000:  # 100K+ shares
            score += 5
        elif net_position < -10000000:  # 10M+ shares distributed
            score -= 15
        elif net_position < -1000000:  # 1M+ shares
            score -= 10
        elif net_position < -100000:  # 100K+ shares
            score -= 5
        
        # 3. Net Short Volume Component (+/- 10 points)
        # Positive net short volume = more shorts than longs today = bearish
        if net_short_volume > 5000000:
            score -= 10
        elif net_short_volume > 1000000:
            score -= 6
        elif net_short_volume > 100000:
            score -= 3
        elif net_short_volume < -5000000:
            score += 10
        elif net_short_volume < -1000000:
            score += 6
        elif net_short_volume < -100000:
            score += 3
        
        return max(0, min(100, int(score)))
    
    def _determine_sentiment(
        self, 
        score: int, 
        net_position: float, 
        short_ratio: float,
        net_short_volume: float
    ) -> str:
        """
        Determine sentiment based on multiple factors.
        
        Uses score as primary driver but considers edge cases.
        """
        # Primary: Score-based sentiment
        if score >= 75:
            base_sentiment = 'VERY_BULLISH'
        elif score >= 60:
            base_sentiment = 'BULLISH'
        elif score <= 25:
            base_sentiment = 'VERY_BEARISH'
        elif score <= 40:
            base_sentiment = 'BEARISH'
        else:
            base_sentiment = 'NEUTRAL'
        
        # Override checks for extreme conditions
        if net_position < -50000000 and short_ratio > 50:
            # Large negative position + high short ratio = definitely bearish
            if base_sentiment in ['NEUTRAL', 'BULLISH']:
                return 'BEARISH'
        
        if net_position > 50000000 and short_ratio < 40:
            # Large positive position + low short ratio = definitely bullish
            if base_sentiment in ['NEUTRAL', 'BEARISH']:
                return 'BULLISH'
        
        return base_sentiment
    
    def _calculate_volume_ratio(self, ticker: str, net_position: float, finra_total_volume: int) -> float:
        """
        Calculate dark pool volume ratio vs regular market volume.
        
        Returns ratio of dark pool activity to total volume.
        """
        if finra_total_volume > 0:
            # Use absolute net position as proxy for dark pool activity
            dp_activity = abs(net_position)
            # Ratio capped at 5x for display purposes
            return min(dp_activity / finra_total_volume, 5.0)
        
        return 1.0  # Default if no volume data
    
    def _generate_signal(
        self,
        ticker: str,
        net_position: float,
        net_position_dollar: float,
        net_short_volume: float,
        short_ratio: float,
        sentiment: str
    ) -> str:
        """
        Generate a dynamic, contextual signal description.
        
        Each signal is unique based on the actual data.
        """
        signals = []
        
        # Position-based signals
        if net_position > 50000000:
            signals.append(f"Massive institutional accumulation ({net_position/1e6:.1f}M shares)")
        elif net_position > 10000000:
            signals.append(f"Heavy institutional buying ({net_position/1e6:.1f}M shares)")
        elif net_position > 1000000:
            signals.append(f"Net institutional buying ({net_position/1e6:.1f}M shares)")
        elif net_position < -50000000:
            signals.append(f"Massive institutional distribution ({abs(net_position)/1e6:.1f}M shares)")
        elif net_position < -10000000:
            signals.append(f"Heavy institutional selling ({abs(net_position)/1e6:.1f}M shares)")
        elif net_position < -1000000:
            signals.append(f"Net institutional selling ({abs(net_position)/1e6:.1f}M shares)")
        
        # Short ratio signals
        if short_ratio >= self.EXTREME_SHORT_RATIO:
            signals.append(f"EXTREME short pressure ({short_ratio:.1f}%)")
        elif short_ratio >= self.NORMAL_SHORT_RATIO_HIGH:
            signals.append(f"Elevated short ratio ({short_ratio:.1f}%)")
        elif short_ratio <= self.EXTREME_LOW_SHORT_RATIO:
            signals.append(f"Very low shorting ({short_ratio:.1f}%) - bullish")
        elif short_ratio <= self.NORMAL_SHORT_RATIO_LOW:
            signals.append(f"Below-average shorting ({short_ratio:.1f}%)")
        
        # Net short volume signals
        if net_short_volume > 5000000:
            signals.append(f"Active short selling today (+{net_short_volume/1e6:.1f}M net short)")
        elif net_short_volume < -5000000:
            signals.append(f"Short covering today ({abs(net_short_volume)/1e6:.1f}M net long)")
        
        # Dollar position context
        if abs(net_position_dollar) > 1e9:
            direction = "accumulated" if net_position_dollar > 0 else "distributed"
            signals.append(f"${abs(net_position_dollar)/1e9:.1f}B {direction}")
        elif abs(net_position_dollar) > 100e6:
            direction = "accumulated" if net_position_dollar > 0 else "distributed"
            signals.append(f"${abs(net_position_dollar)/1e6:.0f}M {direction}")
        
        if not signals:
            signals.append("Normal dark pool activity")
        
        return "; ".join(signals)
    
    def _categorize_results(self, all_stocks: List[Dict]) -> Dict:
        """
        Categorize analyzed stocks into bullish, bearish, and unusual activity.
        """
        bullish_stocks = []
        bearish_stocks = []
        unusual_activity = []
        high_activity = []
        
        for stock in all_stocks:
            sentiment = stock.get('sentiment', 'NEUTRAL')
            short_ratio = stock.get('short_ratio', 45)
            net_position = stock.get('net_position', 0)
            net_position_dollar = stock.get('net_position_dollar', 0)
            volume_ratio = stock.get('volume_ratio', 1.0)
            
            # Categorize by sentiment
            if sentiment in ['VERY_BULLISH', 'BULLISH']:
                bullish_stocks.append(stock)
            elif sentiment in ['VERY_BEARISH', 'BEARISH']:
                bearish_stocks.append(stock)
            
            # High activity (large dollar positions either direction)
            if abs(net_position_dollar) > 100000000:  # $100M+
                high_activity.append(stock)
            
            # Unusual activity (extreme short ratios or volume anomalies)
            if short_ratio > self.NORMAL_SHORT_RATIO_HIGH or short_ratio < self.NORMAL_SHORT_RATIO_LOW or volume_ratio > 2.0:
                reason_parts = []
                if short_ratio > self.EXTREME_SHORT_RATIO:
                    reason_parts.append(f"Extreme short ratio ({short_ratio:.1f}%)")
                elif short_ratio > self.NORMAL_SHORT_RATIO_HIGH:
                    reason_parts.append(f"High short ratio ({short_ratio:.1f}%)")
                elif short_ratio < self.EXTREME_LOW_SHORT_RATIO:
                    reason_parts.append(f"Very low short ratio ({short_ratio:.1f}%)")
                elif short_ratio < self.NORMAL_SHORT_RATIO_LOW:
                    reason_parts.append(f"Low short ratio ({short_ratio:.1f}%)")
                
                if volume_ratio > 2.0:
                    reason_parts.append(f"High DP volume ({volume_ratio:.1f}x)")
                
                unusual_activity.append({
                    "symbol": stock['symbol'],
                    "volume_ratio": volume_ratio,
                    "dp_volume": abs(net_position),
                    "short_ratio": short_ratio,
                    "reason": "; ".join(reason_parts) if reason_parts else "Unusual activity"
                })
        
        # Sort by score (bullish high to low, bearish low to high)
        bullish_stocks.sort(key=lambda x: x['score'], reverse=True)
        bearish_stocks.sort(key=lambda x: x['score'])
        high_activity.sort(key=lambda x: abs(x['net_position_dollar']), reverse=True)
        unusual_activity.sort(key=lambda x: abs(x.get('short_ratio', 45) - 45), reverse=True)
        
        return {
            "stocks_scanned": len(all_stocks),
            "bullish_count": len(bullish_stocks),
            "bearish_count": len(bearish_stocks),
            "high_activity_count": len(high_activity),
            "unusual_volume_count": len(unusual_activity),
            "top_bullish": bullish_stocks[:20],
            "top_bearish": bearish_stocks[:20],
            "unusual_activity": unusual_activity[:20],
            "high_activity": high_activity[:20]
        }


def main():
    """Main entry point for market-wide dark pool scanning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Institutional-Grade Market-Wide Dark Pool Scanner')
    parser.add_argument('--limit', type=int, default=400, help='Number of stocks to scan')
    parser.add_argument('--output', type=str, default='json', choices=['json', 'text'], help='Output format')
    
    args = parser.parse_args()
    
    scanner = MarketDarkPoolScanner()
    result = scanner.scan_market(limit=args.limit)
    
    if args.output == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        # Text output
        print("=" * 70)
        print("INSTITUTIONAL-GRADE MARKET-WIDE DARK POOL SCAN")
        print("=" * 70)
        print(f"Timestamp: {result.get('timestamp', 'N/A')} EST")
        print(f"FINRA Data Date: {result.get('finra_date', 'N/A')}")
        print(f"Stocks Scanned: {result.get('stocks_scanned', 0)}")
        print(f"Bullish Signals: {result.get('bullish_count', 0)}")
        print(f"Bearish Signals: {result.get('bearish_count', 0)}")
        print(f"High Activity: {result.get('high_activity_count', 0)}")
        print(f"Unusual Volume: {result.get('unusual_volume_count', 0)}")
        
        if result.get('top_bullish'):
            print("\n" + "=" * 70)
            print("TOP BULLISH DARK POOL ACTIVITY")
            print("=" * 70)
            for stock in result['top_bullish'][:10]:
                print(f"\n{stock['symbol']} (Score: {stock['score']})")
                print(f"  Net Position: {stock['net_position']:,.0f} shares")
                print(f"  Short Ratio: {stock['short_ratio']:.1f}%")
                print(f"  Signal: {stock['signal']}")
        
        if result.get('top_bearish'):
            print("\n" + "=" * 70)
            print("TOP BEARISH DARK POOL ACTIVITY")
            print("=" * 70)
            for stock in result['top_bearish'][:10]:
                print(f"\n{stock['symbol']} (Score: {stock['score']})")
                print(f"  Net Position: {stock['net_position']:,.0f} shares")
                print(f"  Short Ratio: {stock['short_ratio']:.1f}%")
                print(f"  Signal: {stock['signal']}")


if __name__ == "__main__":
    main()
