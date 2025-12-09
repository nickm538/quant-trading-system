"""
INSTITUTIONAL-GRADE MARKET SCANNER
Multi-tier filtering system to find best opportunities across US markets
Maximum computational power - pushing beyond limitations
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from data.enhanced_data_ingestion import EnhancedDataIngestion
from models.technical_indicators import TechnicalIndicators
from main_trading_system import InstitutionalTradingSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Multi-tier market scanner for finding best opportunities
    Tier 1: Quick filter (~3000 stocks) -> Top 200
    Tier 2: Medium analysis (200 stocks) -> Top 50
    Tier 3: Deep analysis (50 stocks) -> Top 10-20
    """
    
    def __init__(self):
        self.data_ingestion = EnhancedDataIngestion()
        self.technical_indicators = TechnicalIndicators()
        self.trading_system = InstitutionalTradingSystem()
        
        # Major US market indices
        self.SP500_SYMBOLS = self._get_sp500_symbols()
        self.NASDAQ100_SYMBOLS = self._get_nasdaq100_symbols()
        self.DOW30_SYMBOLS = self._get_dow30_symbols()
        self.RUSSELL2000_SYMBOLS = self._get_russell2000_symbols()
        
        logger.info(f"Market Scanner initialized")
        logger.info(f"  S&P 500: {len(self.SP500_SYMBOLS)} symbols")
        logger.info(f"  NASDAQ 100: {len(self.NASDAQ100_SYMBOLS)} symbols")
        logger.info(f"  Dow 30: {len(self.DOW30_SYMBOLS)} symbols")
        logger.info(f"  Russell 2000: {len(self.RUSSELL2000_SYMBOLS)} symbols")
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols - top 100 most liquid for speed"""
        # Top 100 most liquid S&P 500 stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
            'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD', 'LLY', 'ABBV',
            'MRK', 'PEP', 'KO', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            'DHR', 'VZ', 'ADBE', 'CRM', 'NKE', 'NFLX', 'CMCSA', 'TXN', 'DIS', 'PM',
            'INTC', 'UPS', 'NEE', 'BMY', 'ORCL', 'HON', 'QCOM', 'AMD', 'UNP', 'RTX',
            'LOW', 'SPGI', 'INTU', 'CAT', 'BA', 'AMGN', 'GE', 'SBUX', 'IBM', 'AMAT',
            'DE', 'AXP', 'BLK', 'MDT', 'GILD', 'TJX', 'MMC', 'BKNG', 'CVS', 'AMT',
            'SCHW', 'SYK', 'PLD', 'C', 'ADP', 'MDLZ', 'TMUS', 'CI', 'ZTS', 'ISRG',
            'MO', 'CB', 'REGN', 'SO', 'DUK', 'PNC', 'BSX', 'EOG', 'VRTX', 'USB',
            'CL', 'NOC', 'MMM', 'GD', 'ITW', 'SLB', 'APD', 'HUM', 'CME', 'TGT'
        ]
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols - top tech stocks"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
            'ASML', 'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'CMCSA', 'INTC', 'TXN',
            'QCOM', 'INTU', 'AMGN', 'HON', 'AMAT', 'SBUX', 'BKNG', 'ISRG', 'GILD', 'ADI',
            'VRTX', 'REGN', 'PANW', 'MU', 'LRCX', 'ADP', 'MDLZ', 'PYPL', 'SNPS', 'KLAC',
            'CDNS', 'MELI', 'CRWD', 'MAR', 'ABNB', 'ORLY', 'CTAS', 'NXPI', 'MRVL', 'FTNT'
        ]
    
    def _get_dow30_symbols(self) -> List[str]:
        """Get Dow Jones 30 symbols"""
        return [
            'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'CAT', 'MCD', 'AMGN', 'V', 'BA',
            'TRV', 'AXP', 'JPM', 'IBM', 'HON', 'JNJ', 'PG', 'CVX', 'MRK', 'WMT',
            'DIS', 'CRM', 'NKE', 'CSCO', 'KO', 'DOW', 'VZ', 'INTC', 'MMM', 'WBA'
        ]
    
    def _get_russell2000_symbols(self) -> List[str]:
        """Get Russell 2000 symbols - top 50 small caps for speed"""
        # Top 50 most liquid Russell 2000 stocks
        return [
            'SIRI', 'PLUG', 'LCID', 'RIVN', 'SOFI', 'PLTR', 'COIN', 'HOOD', 'RBLX', 'U',
            'DKNG', 'OPEN', 'AFRM', 'UPST', 'CVNA', 'LAZR', 'BLNK', 'CHPT', 'QS', 'GOEV',
            'NKLA', 'RIDE', 'FSR', 'WKHS', 'HYLN', 'SPCE', 'ASTS', 'RKLB', 'MNTS', 'IONQ',
            'RGTI', 'QUBT', 'SMCI', 'MARA', 'RIOT', 'CLSK', 'BITF', 'HUT', 'CIFR', 'CORZ',
            'IREN', 'WULF', 'BTBT', 'CAN', 'ARBK', 'SDIG', 'APLD', 'BTDR', 'BKKT', 'GREE'
        ]
    
    def tier1_quick_filter(self, symbols: List[str], max_workers: int = 20) -> List[Dict]:
        """
        Tier 1: Quick filter with basic metrics
        Parallel processing for maximum speed
        """
        logger.info("=" * 80)
        logger.info(f"TIER 1: QUICK FILTER - Scanning {len(symbols)} symbols")
        logger.info("=" * 80)
        
        results = []
        start_time = time.time()
        
        def analyze_symbol(symbol: str) -> Dict:
            try:
                # Get basic price data using yfinance directly (fast)
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                
                # Get 3 months of daily data
                df = ticker.history(period='3mo', interval='1d')
                
                if df.empty:
                    return None
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    'Close': 'close',
                    'Volume': 'volume',
                    'High': 'high',
                    'Low': 'low'
                })
                df = df[['close', 'volume', 'high', 'low']].dropna()
                
                # Get market cap from info
                try:
                    info = ticker.info
                    market_cap = info.get('marketCap', 0)
                except:
                    market_cap = 0
                
                if len(df) < 20:
                    return None
                
                # Quick metrics
                current_price = df['close'].iloc[-1]
                avg_volume = df['volume'].mean()
                
                # Liquidity filter
                if avg_volume < 500000:  # Min 500k shares/day
                    return None
                
                # Calculate quick technical score
                returns = df['close'].pct_change()
                volatility = returns.std() * np.sqrt(252)
                
                # Momentum (20-day)
                momentum_20 = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) if len(df) >= 20 else 0
                
                # Simple RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                
                # Quick score (0-100)
                momentum_score = min(max(momentum_20 * 200 + 50, 0), 100)
                rsi_score = 100 - abs(current_rsi - 50)
                vol_score = max(100 - volatility * 200, 0)
                
                quick_score = (momentum_score + rsi_score + vol_score) / 3
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'avg_volume': avg_volume,
                    'volatility': volatility,
                    'momentum_20d': momentum_20,
                    'rsi': current_rsi,
                    'quick_score': quick_score,
                    'market_cap': market_cap
                }
                
            except Exception as e:
                logger.warning(f"  Error analyzing {symbol}: {str(e)}")
                return None
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"  Progress: {completed}/{len(symbols)} symbols analyzed")
                
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by quick score
        results.sort(key=lambda x: x['quick_score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Tier 1 Complete: {len(results)} candidates in {elapsed:.1f}s")
        logger.info(f"  Top 5: {', '.join([r['symbol'] for r in results[:5]])}")
        
        return results[:200]  # Top 200
    
    def tier2_medium_analysis(self, candidates: List[Dict], max_workers: int = 10) -> List[Dict]:
        """
        Tier 2: Medium analysis with full technical indicators
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"TIER 2: MEDIUM ANALYSIS - Analyzing {len(candidates)} candidates")
        logger.info("=" * 80)
        
        results = []
        start_time = time.time()
        
        def deep_analyze(candidate: Dict) -> Dict:
            try:
                symbol = candidate['symbol']
                
                # Get full price data
                try:
                    complete_data = self.data_ingestion.get_complete_stock_data(symbol)
                    price_data = complete_data['price_data']
                except Exception as e:
                    logger.warning(f"  Failed to get data for {symbol}: {str(e)}")
                    return None
                
                if price_data.empty or len(price_data) < 100:
                    return None
                
                # Calculate ALL 50+ technical indicators
                df_with_indicators = self.technical_indicators.calculate_all_indicators(price_data)
                latest = df_with_indicators.iloc[-1]
                
                # Technical scores
                rsi = latest['rsi_14']
                momentum_score = 100 - abs(rsi - 50)
                
                if latest['close'] > latest['sma_50'] > latest['sma_200']:
                    trend_score = 80 + (latest['adx'] / 100 * 20)
                elif latest['close'] < latest['sma_50'] < latest['sma_200']:
                    trend_score = 20 - (latest['adx'] / 100 * 20)
                else:
                    trend_score = 50
                
                volatility_score = 100 - (latest['hist_vol_20'] * 100)
                technical_score = (momentum_score + trend_score + volatility_score) / 3
                
                # News sentiment
                news_articles = complete_data.get('news', [])
                sentiment_score = 0
                if news_articles:
                    sentiments = [article.get('sentiment', 0) for article in news_articles if 'sentiment' in article]
                    sentiment_score = np.mean(sentiments) if sentiments else 0
                
                candidate.update({
                    'technical_score': technical_score,
                    'momentum_score': momentum_score,
                    'trend_score': trend_score,
                    'volatility_score': volatility_score,
                    'rsi': rsi,
                    'macd': latest['macd'],
                    'adx': latest['adx'],
                    'sentiment': sentiment_score,
                    'num_news': len(news_articles)
                })
                
                return candidate
                
            except Exception as e:
                logger.warning(f"  Error in tier 2 for {candidate['symbol']}: {str(e)}")
                return None
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(deep_analyze, cand): cand for cand in candidates}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 20 == 0:
                    logger.info(f"  Progress: {completed}/{len(candidates)} analyzed")
                
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by technical score
        results.sort(key=lambda x: x['technical_score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Tier 2 Complete: {len(results)} candidates in {elapsed:.1f}s")
        logger.info(f"  Top 5: {', '.join([r['symbol'] for r in results[:5]])}")
        
        return results[:50]  # Top 50
    
    def tier3_deep_analysis(self, candidates: List[Dict], top_n: int = 20) -> List[Dict]:
        """
        Tier 3: Full institutional-grade analysis
        Complete Monte Carlo, GARCH, options, everything
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"TIER 3: DEEP INSTITUTIONAL ANALYSIS - {len(candidates)} candidates")
        logger.info("=" * 80)
        
        results = []
        start_time = time.time()
        
        for i, candidate in enumerate(candidates):
            try:
                symbol = candidate['symbol']
                logger.info(f"\n[{i+1}/{len(candidates)}] Deep analysis: {symbol}")
                
                # Full comprehensive analysis
                # Reduced Monte Carlo sims to 5000 for stability in parallel processing
                analysis = self.trading_system.analyze_stock_comprehensive(
                    symbol=symbol,
                    monte_carlo_sims=5000,
                    forecast_days=30,
                    bankroll=1000.0
                )
                
                if not analysis or 'recommendation' not in analysis:
                    continue
                
                rec = analysis['recommendation']
                
                # Combined score - WORLD-CLASS RANKING
                confidence = rec.get('confidence', 0)
                expected_return = analysis['stochastic_analysis']['monte_carlo'].get('expected_return', 0)
                var_95 = rec.get('var_95', 0.05)  # Default 5% risk
                risk_reward = rec.get('risk_reward_ratio', 1.0)
                
                # Risk-adjusted return (Sharpe-like)
                risk_adjusted_return = expected_return / max(abs(var_95), 0.01)
                
                # Kelly Criterion approximation
                win_prob = confidence / 100  # Convert percentage to decimal probability
                kelly_fraction = win_prob - ((1 - win_prob) / max(risk_reward, 0.1))
                kelly_score = max(kelly_fraction, 0) * 100
                
                # Optimal score = Risk-adjusted return × R/R × Confidence × Kelly
                # This finds trades with: HIGH return, LOW risk, HIGH confidence, OPTIMAL sizing
                opportunity_score = (
                    risk_adjusted_return *  # Return per unit of risk
                    risk_reward *           # Reward/Risk ratio
                    confidence *            # Model confidence
                    (1 + kelly_score/100) * # Kelly optimal sizing bonus
                    100
                )
                
                results.append({
                    'symbol': symbol,
                    'signal': rec['signal_type'],
                    'confidence': confidence,
                    'current_price': rec['current_price'],
                    'target_price': rec['target_price'],
                    'expected_return': expected_return * 100,
                    'var_95': rec['var_95'] * 100,
                    'cvar_95': rec['cvar_95'] * 100,
                    'position_size': rec['position_size'] * 100,
                    'shares': rec.get('shares', 0),
                    'dollar_risk': rec.get('dollar_risk', 0),
                    'dollar_reward': rec.get('dollar_reward', 0),
                    'risk_reward_ratio': rec.get('risk_reward_ratio', 0),
                    'technical_score': rec['technical_score'],
                    'sentiment': rec.get('news_sentiment', 0),
                    'opportunity_score': opportunity_score,
                    'full_analysis': analysis
                })
                
                logger.info(f"  ✓ {symbol}: Score={opportunity_score:.1f}, Signal={rec['signal_type']}, Return={expected_return*100:.2f}%")
                
            except Exception as e:
                logger.error(f"  ✗ Error analyzing {candidate['symbol']}: {str(e)}")
                continue
        
        # Sort by opportunity score
        results.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Tier 3 Complete: {len(results)} analyzed in {elapsed:.1f}s")
        
        return results[:top_n]
    
    def scan_market(self, top_n: int = 20) -> Dict:
        """
        Complete market scan across all tiers
        Returns top opportunities
        """
        logger.info("\n" + "=" * 100)
        logger.info("COMPREHENSIVE MARKET SCAN - MAXIMUM COMPUTATIONAL POWER")
        logger.info("=" * 100)
        
        scan_start = time.time()
        
        # Combine all symbols (remove duplicates)
        all_symbols = list(set(
            self.SP500_SYMBOLS + 
            self.NASDAQ100_SYMBOLS + 
            self.DOW30_SYMBOLS + 
            self.RUSSELL2000_SYMBOLS
        ))
        
        logger.info(f"\nTotal universe: {len(all_symbols)} unique symbols")
        
        # Tier 1: Quick filter
        tier1_results = self.tier1_quick_filter(all_symbols, max_workers=30)
        
        # Tier 2: Medium analysis
        tier2_results = self.tier2_medium_analysis(tier1_results, max_workers=15)
        
        # Tier 3: Deep analysis
        tier3_results = self.tier3_deep_analysis(tier2_results, top_n=top_n)
        
        total_elapsed = time.time() - scan_start
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("MARKET SCAN COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
        logger.info(f"Symbols scanned: {len(all_symbols)}")
        logger.info(f"Tier 1 passed: {len(tier1_results)}")
        logger.info(f"Tier 2 passed: {len(tier2_results)}")
        logger.info(f"Final opportunities: {len(tier3_results)}")
        
        if tier3_results:
            logger.info(f"\n🏆 TOP OPPORTUNITIES:")
            for i, opp in enumerate(tier3_results[:10], 1):
                logger.info(f"  {i}. {opp['symbol']:6s} - Score: {opp['opportunity_score']:6.1f} | "
                          f"Return: {opp['expected_return']:6.2f}% | "
                          f"Signal: {opp['signal']:4s} | "
                          f"Confidence: {opp['confidence']:.1f}%")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'scan_time_minutes': total_elapsed / 60,
            'symbols_scanned': len(all_symbols),
            'opportunities': tier3_results,
            'summary': {
                'tier1_passed': len(tier1_results),
                'tier2_passed': len(tier2_results),
                'final_count': len(tier3_results)
            }
        }


if __name__ == "__main__":
    scanner = MarketScanner()
    results = scanner.scan_market(top_n=20)
    
    # Save results (disabled for production)
    # output_file = f"/home/ubuntu/quant_trading_system/logs/market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # with open(output_file, 'w') as f:
    #     # Remove full_analysis for file size
    #     simplified = results.copy()
    #     simplified['opportunities'] = [
    #         {k: v for k, v in opp.items() if k != 'full_analysis'}
    #         for opp in results['opportunities']
    #     ]
    #     json.dump(simplified, f, indent=2, default=str)
    # 
    # print(f"\n✓ Results saved to: {output_file}")
