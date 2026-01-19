#!/usr/bin/env python3
"""
MARKET-WIDE DARK POOL SCANNER
==============================
Scans multiple stocks for dark pool activity and institutional movements.

Data Sources:
1. Stockgrid.io Dark Pool API (end-of-day)
2. FINRA Short Volume Data (daily)
3. Yahoo Finance for price/volume context

Features:
- Scan 400+ stocks for dark pool activity
- Identify bullish/bearish institutional positioning
- Detect unusual dark pool volume
- Rank stocks by institutional activity

For integration with the quant trading system frontend.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import io
import json
import sys
import os
import pytz

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MarketDarkPoolScanner:
    """
    Market-Wide Dark Pool Scanner
    
    Scans the entire market for dark pool activity and institutional movements.
    """
    
    def __init__(self):
        """Initialize Market Dark Pool Scanner."""
        self.stockgrid_url = "https://www.stockgrid.io/get_dark_pool_data"
        self.finra_base_url = "https://cdn.finra.org/equity/regsho/daily"
        self.est_tz = pytz.timezone('America/New_York')
        
        # Top 400+ stocks to scan (S&P 500 + popular stocks)
        self.scan_universe = [
            # Mega-cap tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
            "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "AMAT",
            "MU", "LRCX", "ADI", "KLAC", "SNPS", "CDNS", "MRVL", "NXPI", "MCHP", "ON",
            
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
            "PNC", "TFC", "COF", "BK", "STT", "FITB", "HBAN", "KEY", "RF", "CFG",
            "MTB", "NTRS", "ZION", "CMA", "FRC", "SIVB", "ALLY", "SYF", "DFS", "AIG",
            
            # Healthcare
            "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "ISRG", "VRTX", "REGN", "MDT", "SYK", "BSX", "EW", "ZBH",
            "DXCM", "IDXX", "IQV", "MTD", "A", "BIO", "TECH", "HOLX", "ALGN", "PODD",
            
            # Consumer
            "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "COST", "WMT",
            "PG", "KO", "PEP", "PM", "MO", "CL", "EL", "KMB", "GIS", "K",
            "MDLZ", "HSY", "KHC", "SJM", "CAG", "CPB", "HRL", "MKC", "CLX", "CHD",
            
            # Industrial
            "CAT", "DE", "HON", "UNP", "UPS", "RTX", "BA", "LMT", "GD", "NOC",
            "GE", "MMM", "EMR", "ITW", "ETN", "ROK", "PH", "CMI", "PCAR", "FAST",
            "IR", "DOV", "SWK", "AME", "ROP", "OTIS", "CARR", "TT", "XYL", "IEX",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
            "DVN", "HES", "FANG", "HAL", "BKR", "KMI", "WMB", "OKE", "TRGP", "LNG",
            
            # Communication
            "GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR", "PARA",
            "WBD", "FOX", "FOXA", "OMC", "IPG", "TTWO", "EA", "ATVI", "MTCH", "ZG",
            
            # Real Estate
            "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "DLR", "WELL", "AVB",
            "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "INVH", "SUI", "ELS", "CPT",
            
            # Materials
            "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
            "PPG", "ALB", "EMN", "CE", "LYB", "DOW", "CTVA", "FMC", "MOS", "CF",
            
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL", "ED", "EXC", "WEC",
            "ES", "AWK", "AEE", "DTE", "CMS", "CNP", "NI", "EVRG", "ATO", "NRG",
            
            # Popular retail/meme stocks
            "GME", "AMC", "BB", "BBBY", "PLTR", "SOFI", "LCID", "RIVN", "NIO", "XPEV",
            "HOOD", "COIN", "RBLX", "SNAP", "PINS", "TWTR", "DKNG", "PENN", "WYNN", "LVS",
            
            # ETFs (for market context)
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "XLF", "XLE", "XLK",
            "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC", "GLD", "SLV",
            
            # Additional popular stocks
            "SHOP", "SQ", "PYPL", "V", "MA", "ADP", "PAYX", "INTU", "ADSK", "ANSS",
            "PANW", "CRWD", "ZS", "FTNT", "NET", "DDOG", "SNOW", "MDB", "TEAM", "OKTA",
            "UBER", "LYFT", "ABNB", "DASH", "BKNG", "EXPE", "MAR", "HLT", "H", "IHG",
            "LUV", "DAL", "UAL", "AAL", "JBLU", "ALK", "SAVE", "HA", "SKYW", "MESA",
            "F", "GM", "STLA", "TM", "HMC", "RACE", "TSLA", "RIVN", "LCID", "FSR",
        ]
        
        # Remove duplicates
        self.scan_universe = list(dict.fromkeys(self.scan_universe))
    
    def scan_market(self, limit: int = 400) -> Dict:
        """
        Scan the market for dark pool activity.
        
        Args:
            limit: Maximum number of stocks to scan
            
        Returns:
            Dict with market-wide dark pool analysis
        """
        try:
            # Get current EST time
            now_est = datetime.now(self.est_tz)
            
            # Fetch Stockgrid data (contains all dark pool positions)
            stockgrid_data = self._fetch_all_stockgrid_data()
            
            if not stockgrid_data:
                return {
                    "error": "Failed to fetch dark pool data from Stockgrid",
                    "timestamp": now_est.strftime("%Y-%m-%d %H:%M:%S"),
                    "stocks_scanned": 0
                }
            
            # Process and analyze data
            results = self._analyze_market_data(stockgrid_data, limit)
            
            # Add metadata
            results["timestamp"] = now_est.strftime("%Y-%m-%d %H:%M:%S")
            results["data_source"] = "Stockgrid.io + FINRA"
            
            return results
            
        except Exception as e:
            return {
                "error": f"Market scan failed: {str(e)}",
                "timestamp": datetime.now(self.est_tz).strftime("%Y-%m-%d %H:%M:%S"),
                "stocks_scanned": 0
            }
    
    def _fetch_all_stockgrid_data(self) -> List[Dict]:
        """
        Fetch all dark pool data from Stockgrid.
        """
        try:
            # Fetch top dark pool positions (sorted by dollar value)
            url = f"{self.stockgrid_url}?top=Dark%20Pools%20Position%20$&minmax=desc"
            
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get('data', [])
            
            return []
            
        except Exception as e:
            print(f"Error fetching Stockgrid data: {e}", file=sys.stderr)
            return []
    
    def _analyze_market_data(self, data: List[Dict], limit: int) -> Dict:
        """
        Analyze market-wide dark pool data.
        """
        # Initialize counters
        bullish_stocks = []
        bearish_stocks = []
        unusual_activity = []
        high_activity = []
        
        bullish_count = 0
        bearish_count = 0
        high_activity_count = 0
        unusual_volume_count = 0
        
        stocks_processed = 0
        
        for item in data[:limit]:
            ticker = item.get('Ticker', '')
            if not ticker:
                continue
                
            stocks_processed += 1
            
            net_position = item.get('Dark Pools Position', 0)
            net_position_dollar = item.get('Dark Pools Position $', 0)
            short_volume = item.get('Short Volume', 0)
            short_volume_pct = item.get('Short Volume %', 0)
            date = item.get('Date', '')
            
            # Calculate score based on position and short ratio
            score = self._calculate_score(net_position, net_position_dollar, short_volume_pct)
            
            # Determine sentiment
            sentiment = self._determine_sentiment(net_position, net_position_dollar, short_volume_pct)
            
            # Create stock entry
            stock_entry = {
                "symbol": ticker,
                "net_position": net_position,
                "net_position_dollar": net_position_dollar,
                "short_volume": short_volume,
                "short_ratio": short_volume_pct,
                "score": score,
                "sentiment": sentiment,
                "signal": self._generate_signal(net_position, net_position_dollar, short_volume_pct),
                "date": date
            }
            
            # Categorize
            if sentiment in ['VERY_BULLISH', 'BULLISH']:
                bullish_count += 1
                if len(bullish_stocks) < 20:
                    bullish_stocks.append(stock_entry)
            elif sentiment in ['VERY_BEARISH', 'BEARISH']:
                bearish_count += 1
                if len(bearish_stocks) < 20:
                    bearish_stocks.append(stock_entry)
            
            # Check for high activity (large dollar positions)
            if abs(net_position_dollar) > 50000000:  # $50M+
                high_activity_count += 1
                if len(high_activity) < 20:
                    high_activity.append(stock_entry)
            
            # Check for unusual volume ratio
            # If short volume is very high or very low, it's unusual
            if short_volume_pct > 60 or short_volume_pct < 25:
                unusual_volume_count += 1
                if len(unusual_activity) < 20:
                    unusual_activity.append({
                        "symbol": ticker,
                        "volume_ratio": 1.0,  # Placeholder - would need actual volume data
                        "dp_volume": abs(net_position),
                        "reason": f"Short ratio: {short_volume_pct:.1f}% ({'High' if short_volume_pct > 60 else 'Low'})"
                    })
        
        # Sort by score
        bullish_stocks.sort(key=lambda x: x['score'], reverse=True)
        bearish_stocks.sort(key=lambda x: x['score'])
        high_activity.sort(key=lambda x: abs(x['net_position_dollar']), reverse=True)
        
        return {
            "stocks_scanned": stocks_processed,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "high_activity_count": high_activity_count,
            "unusual_volume_count": unusual_volume_count,
            "top_bullish": bullish_stocks[:15],
            "top_bearish": bearish_stocks[:15],
            "unusual_activity": unusual_activity[:15],
            "high_activity": high_activity[:15]
        }
    
    def _calculate_score(self, net_position: float, net_position_dollar: float, short_ratio: float) -> int:
        """
        Calculate a 0-100 score based on dark pool metrics.
        """
        score = 50  # Start neutral
        
        # Adjust based on net position
        if net_position > 5000000:
            score += 25
        elif net_position > 1000000:
            score += 15
        elif net_position > 100000:
            score += 8
        elif net_position < -5000000:
            score -= 25
        elif net_position < -1000000:
            score -= 15
        elif net_position < -100000:
            score -= 8
        
        # Adjust based on dollar value
        if net_position_dollar > 100000000:  # $100M+
            score += 10 if net_position > 0 else -10
        elif net_position_dollar > 50000000:  # $50M+
            score += 5 if net_position > 0 else -5
        
        # Adjust based on short ratio
        if short_ratio > 60:
            score -= 15
        elif short_ratio > 50:
            score -= 8
        elif short_ratio < 30:
            score += 10
        elif short_ratio < 40:
            score += 5
        
        return max(0, min(100, score))
    
    def _determine_sentiment(self, net_position: float, net_position_dollar: float, short_ratio: float) -> str:
        """
        Determine overall sentiment based on metrics.
        """
        score = self._calculate_score(net_position, net_position_dollar, short_ratio)
        
        if score >= 75:
            return 'VERY_BULLISH'
        elif score >= 60:
            return 'BULLISH'
        elif score <= 25:
            return 'VERY_BEARISH'
        elif score <= 40:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _generate_signal(self, net_position: float, net_position_dollar: float, short_ratio: float) -> str:
        """
        Generate a human-readable signal description.
        """
        signals = []
        
        if net_position > 5000000:
            signals.append("Heavy institutional buying")
        elif net_position > 1000000:
            signals.append("Net institutional buying")
        elif net_position < -5000000:
            signals.append("Heavy institutional selling")
        elif net_position < -1000000:
            signals.append("Net institutional selling")
        
        if short_ratio > 60:
            signals.append(f"High short ratio ({short_ratio:.1f}%)")
        elif short_ratio < 30:
            signals.append(f"Low short ratio ({short_ratio:.1f}%)")
        
        if abs(net_position_dollar) > 100000000:
            signals.append("Large dollar position")
        
        return "; ".join(signals) if signals else "Normal activity"


def main():
    """Main entry point for market-wide dark pool scanning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market-Wide Dark Pool Scanner')
    parser.add_argument('--limit', type=int, default=400, help='Number of stocks to scan')
    parser.add_argument('--output', type=str, default='json', choices=['json', 'text'], help='Output format')
    
    args = parser.parse_args()
    
    scanner = MarketDarkPoolScanner()
    result = scanner.scan_market(limit=args.limit)
    
    if args.output == 'json':
        # Clean output for JSON - remove emojis
        print(json.dumps(result, indent=2, default=str))
    else:
        # Text output
        print("=" * 60)
        print("MARKET-WIDE DARK POOL SCAN")
        print("=" * 60)
        print(f"Timestamp: {result.get('timestamp', 'N/A')} EST")
        print(f"Stocks Scanned: {result.get('stocks_scanned', 0)}")
        print(f"Bullish Signals: {result.get('bullish_count', 0)}")
        print(f"Bearish Signals: {result.get('bearish_count', 0)}")
        print(f"High Activity: {result.get('high_activity_count', 0)}")
        print(f"Unusual Volume: {result.get('unusual_volume_count', 0)}")
        
        if result.get('top_bullish'):
            print("\nTOP BULLISH:")
            for stock in result['top_bullish'][:10]:
                print(f"  {stock['symbol']}: Score {stock['score']}, Net {stock['net_position']:,}")
        
        if result.get('top_bearish'):
            print("\nTOP BEARISH:")
            for stock in result['top_bearish'][:10]:
                print(f"  {stock['symbol']}: Score {stock['score']}, Net {stock['net_position']:,}")


if __name__ == "__main__":
    main()
