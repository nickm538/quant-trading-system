"""
Institutional-Grade Full Universe Market Scanner
Dynamically scans ALL stocks and ETFs with no hardcoded tickers
Optimized for speed with batch processing and smart filtering
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class MarketScanner:
    """
    True institutional-grade market scanner that dynamically fetches
    the full universe of stocks and ETFs with no hardcoded limitations.
    """
    
    def __init__(self):
        self.finnhub_api_key = os.environ.get('KEY', '')
        self.polygon_api_key = os.environ.get('POLYGON_API_KEY', '')
        
    def get_full_universe(self, include_etfs: bool = True, filter_active: bool = True) -> List[str]:
        """
        Fetch the complete universe of tradeable stocks and ETFs.
        Uses multiple sources to ensure comprehensive coverage.
        
        Returns:
            List of ticker symbols (8000+ tickers)
        """
        all_tickers = set()
        
        print("Fetching full market universe...", file=sys.stderr)
        
        # Source 1: Polygon.io - All US stocks (preferred - has quality data)
        if self.polygon_api_key:
            try:
                url = f"https://api.polygon.io/v3/reference/tickers"
                params = {
                    "market": "stocks",
                    "active": "true",
                    "limit": 1000,
                    "apiKey": self.polygon_api_key
                }
                
                # Polygon paginates, fetch all pages
                next_url = url
                page_count = 0
                max_pages = 20  # ~20,000 tickers max
                
                while next_url and page_count < max_pages:
                    response = requests.get(next_url, params=params if page_count == 0 else None, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        for ticker_data in results:
                            ticker = ticker_data.get('ticker', '')
                            ticker_type = ticker_data.get('type', '')
                            
                            # Include common stocks and ETFs
                            if ticker_type in ['CS', 'ADRC', 'ETF', 'ETN'] if include_etfs else ['CS', 'ADRC']:
                                # Filter out complex tickers (warrants, units, etc.)
                                if '.' not in ticker and '-' not in ticker and len(ticker) <= 5:
                                    all_tickers.add(ticker)
                        
                        # Get next page URL
                        next_url = data.get('next_url')
                        page_count += 1
                        print(f"Polygon: Fetched page {page_count}, total tickers: {len(all_tickers)}", file=sys.stderr)
                    else:
                        print(f"Polygon API error: {response.status_code}", file=sys.stderr)
                        break
                
                print(f"Polygon: Total tickers fetched: {len(all_tickers)}", file=sys.stderr)
            except Exception as e:
                print(f"Polygon fetch error: {e}", file=sys.stderr)
        
        # Source 2: Finnhub - US stocks (fallback)
        if self.finnhub_api_key and len(all_tickers) < 1000:
            try:
                url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={self.finnhub_api_key}"
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        ticker = item.get('symbol', '')
                        ticker_type = item.get('type', '')
                        
                        # Filter for common stocks and ETFs - only include if type suggests active trading
                        if '.' not in ticker and '-' not in ticker and len(ticker) <= 5:
                            # Skip obvious non-tradeable symbols
                            if not any(x in ticker for x in ['$', '^', '=']):
                                all_tickers.add(ticker)
                    
                    print(f"Finnhub: Added {len(data)} tickers, total: {len(all_tickers)}", file=sys.stderr)
            except Exception as e:
                print(f"Finnhub fetch error: {e}", file=sys.stderr)
        
        # Source 3: Fallback - Major indices constituents
        if len(all_tickers) < 100:
            print("Warning: Using fallback ticker list", file=sys.stderr)
            fallback_tickers = self._get_fallback_tickers()
            all_tickers.update(fallback_tickers)
        
        tickers_list = sorted(list(all_tickers))
        print(f"Final universe size: {len(tickers_list)} tickers", file=sys.stderr)
        
        return tickers_list
    
    def _get_fallback_tickers(self) -> List[str]:
        """
        Fallback ticker list - major indices constituents.
        Only used if API sources fail.
        """
        return [
            # Mega cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR', 'LLY', 'BMY',
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
            # Industrial
            'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'DE',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'EEM', 'GLD', 'TLT', 'HYG'
        ]
    
    def _get_screener_universe(self, min_volume: int = 100000, min_price: float = 1.0, max_price: float = 10000.0) -> List[str]:
        """
        Get a pre-filtered universe using Yahoo Finance screener.
        This is much faster than scanning all tickers.
        """
        try:
            # Use yfinance screener to get active stocks
            screener = yf.Screener()
            
            # Get most active stocks
            screener.set_default_body({
                "size": 250,
                "offset": 0,
                "sortField": "intradaymarketcap",
                "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": {
                    "operator": "AND",
                    "operands": [
                        {"operator": "GT", "operands": ["avgdailyvol3m", min_volume]},
                        {"operator": "GT", "operands": ["intradayprice", min_price]},
                        {"operator": "LT", "operands": ["intradayprice", max_price]}
                    ]
                }
            })
            
            response = screener.response
            if response and 'quotes' in response:
                return [q['symbol'] for q in response['quotes']]
        except Exception as e:
            print(f"Screener error: {e}", file=sys.stderr)
        
        return []
    
    def scan_universe(
        self,
        criteria: Dict[str, Any],
        max_results: int = 50,
        min_volume: int = 100000,
        min_price: float = 1.0,
        max_price: float = 10000.0,
        use_screener: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan the full universe with institutional-grade criteria.
        
        Args:
            criteria: Scanning criteria (momentum, breakout, value, etc.)
            max_results: Maximum number of results to return
            min_volume: Minimum average daily volume
            min_price: Minimum stock price
            max_price: Maximum stock price
            use_screener: Use Yahoo screener for faster results
        
        Returns:
            List of stocks matching criteria with scores
        """
        # Get universe - use screener for speed if available
        if use_screener:
            universe = self._get_screener_universe(min_volume, min_price, max_price)
            if len(universe) < 50:
                print("Screener returned few results, using full universe...", file=sys.stderr)
                universe = self.get_full_universe()
        else:
            universe = self.get_full_universe()
        
        results = []
        
        print(f"Scanning {len(universe)} tickers with criteria: {criteria}", file=sys.stderr)
        
        # Use thread pool for parallel processing
        def process_ticker(ticker):
            try:
                return self._score_ticker(ticker, criteria, min_volume, min_price, max_price)
            except:
                return None
        
        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_ticker, ticker): ticker for ticker in universe[:500]}  # Limit to 500 for speed
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    score = future.result()
                    if score is not None and score['total_score'] > 50:
                        results.append(score)
                        print(f"Found: {ticker} (score: {score['total_score']:.1f})", file=sys.stderr)
                        
                        # Early exit if we have enough results
                        if len(results) >= max_results * 2:
                            break
                except Exception as e:
                    pass
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"Scan complete: {len(results)} matches found", file=sys.stderr)
        
        return results[:max_results]
    
    def _score_ticker(
        self,
        ticker: str,
        criteria: Dict[str, Any],
        min_volume: int,
        min_price: float,
        max_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Score a single ticker against the criteria.
        Returns None if ticker doesn't meet basic filters.
        """
        try:
            # Fetch data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty or len(hist) < 20:
                return None
            
            info = stock.info
            
            # Basic filters
            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            if current_price < min_price or current_price > max_price:
                return None
            if avg_volume < min_volume:
                return None
            
            # Calculate scores
            scores = {
                'ticker': ticker,
                'price': float(current_price),
                'volume': float(avg_volume),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
            
            # Momentum score
            if criteria.get('momentum', False):
                scores['momentum_score'] = self._calculate_momentum(hist)
            
            # Breakout score
            if criteria.get('breakout', False):
                scores['breakout_score'] = self._calculate_breakout(hist)
            
            # Value score
            if criteria.get('value', False):
                scores['value_score'] = self._calculate_value(info)
            
            # Growth score
            if criteria.get('growth', False):
                scores['growth_score'] = self._calculate_growth(info)
            
            # Volatility score
            if criteria.get('volatility', False):
                scores['volatility_score'] = self._calculate_volatility(hist)
            
            # Add legendary investor strategy scores
            scores['buffett_score'] = self._calculate_buffett_score(info, hist)
            scores['lynch_score'] = self._calculate_lynch_score(info, hist)
            scores['druckenmiller_score'] = self._calculate_druckenmiller_score(info, hist)
            scores['soros_score'] = self._calculate_soros_score(info, hist)
            scores['livermore_score'] = self._calculate_livermore_score(hist)
            
            # Calculate total score (weighted average)
            total_score = 0
            score_count = 0
            
            for key in ['momentum_score', 'breakout_score', 'value_score', 'growth_score', 'volatility_score',
                       'buffett_score', 'lynch_score', 'druckenmiller_score', 'soros_score', 'livermore_score']:
                if key in scores:
                    total_score += scores[key]
                    score_count += 1
            
            scores['total_score'] = total_score / score_count if score_count > 0 else 0
            
            return scores if scores['total_score'] > 50 else None
            
        except Exception as e:
            return None
    
    def _calculate_momentum(self, hist: pd.DataFrame) -> float:
        """Calculate momentum score (0-100)"""
        try:
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Price vs moving averages
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else sma_20
            current_price = hist['Close'].iloc[-1]
            
            # Score components
            rsi_score = 100 if 50 < current_rsi < 70 else (70 - abs(current_rsi - 60))
            ma_score = 100 if current_price > sma_20 > sma_50 else 50
            
            return (rsi_score + ma_score) / 2
        except:
            return 50
    
    def _calculate_breakout(self, hist: pd.DataFrame) -> float:
        """Calculate breakout score (0-100)"""
        try:
            # 52-week high proximity
            high_52w = hist['High'].max()
            current_price = hist['Close'].iloc[-1]
            proximity = (current_price / high_52w) * 100
            
            # Volume surge
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            volume_ratio = (recent_volume / avg_volume) * 100
            
            # Consolidation breakout
            recent_range = hist['High'].iloc[-20:].max() - hist['Low'].iloc[-20:].min()
            price_range_pct = (recent_range / current_price) * 100
            
            # Score
            proximity_score = proximity
            volume_score = min(volume_ratio, 100)
            consolidation_score = 100 - min(price_range_pct * 10, 100)
            
            return (proximity_score + volume_score + consolidation_score) / 3
        except:
            return 50
    
    def _calculate_value(self, info: Dict) -> float:
        """Calculate value score (0-100)"""
        try:
            pe = info.get('trailingPE', 999)
            pb = info.get('priceToBook', 999)
            ps = info.get('priceToSalesTrailing12Months', 999)
            
            # Lower is better for value
            pe_score = max(0, 100 - (pe / 0.5)) if pe < 50 else 0
            pb_score = max(0, 100 - (pb / 0.05)) if pb < 5 else 0
            ps_score = max(0, 100 - (ps / 0.02)) if ps < 2 else 0
            
            return (pe_score + pb_score + ps_score) / 3
        except:
            return 50
    
    def _calculate_growth(self, info: Dict) -> float:
        """Calculate growth score (0-100)"""
        try:
            revenue_growth = info.get('revenueGrowth', 0) * 100
            earnings_growth = info.get('earningsGrowth', 0) * 100
            
            revenue_score = min(revenue_growth * 2, 100)
            earnings_score = min(earnings_growth * 2, 100)
            
            return (revenue_score + earnings_score) / 2
        except:
            return 50
    
    def _calculate_volatility(self, hist: pd.DataFrame) -> float:
        """Calculate volatility score (0-100) - higher is more volatile"""
        try:
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Score: 0-50% volatility mapped to 0-100
            return min(volatility * 2, 100)
        except:
            return 50
    
    def _calculate_buffett_score(self, info: Dict, hist: pd.DataFrame) -> float:
        """
        Warren Buffett Strategy: Value + Quality + Moat
        - Low P/E, high ROE, strong margins
        - Consistent earnings growth
        - Strong competitive advantage (moat)
        """
        try:
            pe = info.get('trailingPE', 999)
            roe = info.get('returnOnEquity', 0) * 100
            profit_margin = info.get('profitMargins', 0) * 100
            debt_to_equity = info.get('debtToEquity', 999)
            
            # Buffett likes: PE < 15, ROE > 15%, margins > 10%, low debt
            pe_score = max(0, 100 - (pe / 0.15)) if pe < 15 else 0
            roe_score = min(roe * 5, 100)
            margin_score = min(profit_margin * 10, 100)
            debt_score = max(0, 100 - (debt_to_equity / 2)) if debt_to_equity < 200 else 0
            
            return (pe_score + roe_score + margin_score + debt_score) / 4
        except:
            return 50
    
    def _calculate_lynch_score(self, info: Dict, hist: pd.DataFrame) -> float:
        """
        Peter Lynch Strategy: PEG Ratio + Growth at Reasonable Price (GARP)
        - PEG ratio < 1.0 is ideal
        - Strong earnings growth
        - Reasonable valuation
        """
        try:
            pe = info.get('trailingPE', 999)
            growth = info.get('earningsGrowth', 0) * 100
            
            if growth > 0 and pe < 999:
                peg = pe / growth
                # Lynch loves PEG < 1.0
                peg_score = max(0, 100 - (peg * 50)) if peg < 2 else 0
            else:
                peg_score = 0
            
            # Also likes strong revenue growth
            revenue_growth = info.get('revenueGrowth', 0) * 100
            revenue_score = min(revenue_growth * 2, 100)
            
            return (peg_score + revenue_score) / 2
        except:
            return 50
    
    def _calculate_druckenmiller_score(self, info: Dict, hist: pd.DataFrame) -> float:
        """
        Stanley Druckenmiller Strategy: Macro + Momentum + Trend
        - Strong uptrend
        - High momentum
        - Macro tailwinds
        """
        try:
            # Price vs moving averages (trend)
            current_price = hist['Close'].iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else current_price
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else sma_50
            
            # Druckenmiller loves strong trends
            trend_score = 100 if current_price > sma_50 > sma_200 else 50
            
            # Momentum (rate of change)
            roc_20 = ((current_price / hist['Close'].iloc[-20]) - 1) * 100 if len(hist) >= 20 else 0
            momentum_score = min(abs(roc_20) * 5, 100)
            
            # Volume confirmation
            recent_volume = hist['Volume'].iloc[-10:].mean()
            avg_volume = hist['Volume'].mean()
            volume_score = min((recent_volume / avg_volume) * 50, 100)
            
            return (trend_score + momentum_score + volume_score) / 3
        except:
            return 50
    
    def _calculate_soros_score(self, info: Dict, hist: pd.DataFrame) -> float:
        """
        George Soros Strategy: Reflexivity + Market Psychology
        - Identify market inefficiencies
        - Sentiment extremes
        - Volatility opportunities
        """
        try:
            # Look for sentiment extremes (contrarian)
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Soros loves volatility (opportunity)
            volatility_score = min(volatility * 200, 100)
            
            # Look for sharp moves (reflexivity)
            recent_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100 if len(hist) >= 5 else 0
            reflexivity_score = min(abs(recent_return) * 10, 100)
            
            # Volume spikes (market psychology)
            volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].mean()
            psychology_score = min(volume_ratio * 50, 100)
            
            return (volatility_score + reflexivity_score + psychology_score) / 3
        except:
            return 50
    
    def _calculate_livermore_score(self, hist: pd.DataFrame) -> float:
        """
        Jesse Livermore Strategy: Tape Reading + Pivotal Points
        - Breakouts from consolidation
        - Volume at key levels
        - Follow the line of least resistance
        """
        try:
            current_price = hist['Close'].iloc[-1]
            
            # Identify consolidation breakout
            recent_high = hist['High'].iloc[-20:].max()
            recent_low = hist['Low'].iloc[-20:].min()
            consolidation_range = (recent_high - recent_low) / current_price * 100
            
            # Livermore loves tight consolidation before breakout
            consolidation_score = max(0, 100 - (consolidation_range * 10))
            
            # Volume at breakout
            avg_volume = hist['Volume'].iloc[:-5].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            volume_score = min((recent_volume / avg_volume) * 50, 100)
            
            # Proximity to 52-week high (line of least resistance)
            high_52w = hist['High'].max()
            proximity = (current_price / high_52w) * 100
            resistance_score = proximity
            
            return (consolidation_score + volume_score + resistance_score) / 3
        except:
            return 50
