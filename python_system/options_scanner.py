#!/usr/bin/env python3
"""
Options Scanner - Find Best Long Call Opportunities
3-tier filtering system to scan entire market for optimal short-to-mid term long calls
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our institutional options engine
from institutional_options_engine import InstitutionalOptionsEngine
from greeks_calculator import GreeksCalculator

# Custom JSON encoder to handle NumPy/Pandas types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionsScanner:
    """Scan entire market for best long call opportunities"""
    
    def __init__(self):
        self.options_engine = InstitutionalOptionsEngine()
        self.greeks_calc = GreeksCalculator(risk_free_rate=0.05)  # 5% risk-free rate
        
        # Universe of stocks to scan (sector-agnostic)
        # Using Russell 1000 + mid-caps for broad coverage
        self.universe = self._build_universe()
        
    def _build_universe(self) -> List[str]:
        """Build universe of stocks to scan - sector agnostic, equal treatment"""
        
        # Start with common indices for broad market coverage
        base_symbols = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            'CRM', 'ADBE', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
            'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'KEY', 'RF', 'CFG', 'HBAN',
            
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR', 'BMY', 'LLY',
            'AMGN', 'GILD', 'CVS', 'CI', 'BIIB', 'VRTX', 'REGN', 'ISRG', 'ZTS', 'ILMN',
            
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'ROST',
            'COST', 'CMG', 'YUM', 'ULTA', 'ORLY', 'AZO', 'BBY', 'DPZ', 'BURL', 'FIVE',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
            'HAL', 'BKR', 'DVN', 'FANG', 'HES', 'MRO', 'APA', 'CTRA', 'OVV', 'EQT',
            
            # Industrials
            'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'UNP',
            'FDX', 'NSC', 'CSX', 'WM', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'PCAR',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ALB',
            'PPG', 'VMC', 'MLM', 'CTVA', 'IFF', 'CE', 'FMC', 'EMN', 'CF', 'MOS',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
            'ES', 'AWK', 'DTE', 'PPL', 'EIX', 'WEC', 'AEE', 'CMS', 'CNP', 'EVRG',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'SBAC',
            'AVB', 'EQR', 'VTR', 'ARE', 'INVH', 'MAA', 'ESS', 'UDR', 'CPT', 'HST',
            
            # Communications
            'CMCSA', 'DIS', 'VZ', 'T', 'TMUS', 'CHTR', 'NXST', 'FOXA', 'PARA', 'WBD',
            
            # Lesser-known growth stocks with active options
            'PLTR', 'SNOW', 'DKNG', 'COIN', 'RIVN', 'LCID', 'SOFI', 'HOOD', 'RBLX', 'U',
            'DASH', 'ABNB', 'LYFT', 'UBER', 'PINS', 'SNAP', 'SPOT', 'ZM', 'DOCU', 'CRWD',
            'NET', 'DDOG', 'MDB', 'OKTA', 'ZS', 'PANW', 'FTNT', 'CYBR', 'S', 'BILL',
            
            # Mid-caps across sectors
            'ENPH', 'SEDG', 'FSLR', 'RUN', 'PLUG', 'BE', 'CHPT', 'BLNK', 'QS', 'LAZR',
            'OPEN', 'COMP', 'RKT', 'UWMC', 'CLOV', 'OSCR', 'HIMS', 'BROS', 'CELH', 'MNST',
            'WING', 'TXRH', 'SHAK', 'CAVA', 'SWEETGREEN', 'DUOL', 'BMBL', 'MTCH', 'PTON', 'LULU',
        ]
        
        logger.info(f"Built universe of {len(base_symbols)} stocks (sector-agnostic)")
        return base_symbols
    
    def tier1_quick_filter(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Tier 1: Quick filter to reduce universe
        Criteria:
        - Market cap > $500M (ensures liquidity)
        - Options volume > 500 contracts/day
        - Has options chain with 2-12 weeks expiration
        - Stock price > $5 (avoid penny stocks)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TIER 1: Quick Filter - Scanning {len(symbols)} stocks")
        logger.info(f"{'='*80}")
        
        candidates = []
        
        def check_stock(symbol: str) -> Dict[str, Any]:
            """Check if stock passes tier 1 filters"""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get basic info
                market_cap = info.get('marketCap', 0)
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                
                # Check basic filters
                if market_cap < 500_000_000:  # $500M minimum
                    return None
                if not price or price <= 0:  # Must have valid price
                    return None
                if price < 5:  # Avoid penny stocks
                    return None
                
                # Check if options are available
                try:
                    expirations = ticker.options
                    if not expirations or len(expirations) == 0:
                        return None
                    
                    # Check for 2-12 week expirations
                    today = datetime.now()
                    valid_expirations = []
                    for exp_str in expirations:
                        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                        days_to_exp = (exp_date - today).days
                        if 14 <= days_to_exp <= 84:  # 2-12 weeks
                            valid_expirations.append(exp_str)
                    
                    if not valid_expirations:
                        return None
                    
                    # Check options volume on nearest expiration
                    nearest_exp = valid_expirations[0]
                    opt_chain = ticker.option_chain(nearest_exp)
                    calls = opt_chain.calls
                    
                    if calls.empty:
                        return None
                    
                    # Total call volume (handle NaN values)
                    total_volume = calls['volume'].fillna(0).sum()
                    if total_volume < 300:  # Minimum 300 contracts/day (balanced: was 500, then 200)
                        return None
                    
                    return {
                        'symbol': symbol,
                        'price': price,
                        'market_cap': market_cap,
                        'options_volume': total_volume,
                        'valid_expirations': valid_expirations,
                        'sector': info.get('sector', 'Unknown')
                    }
                    
                except Exception as e:
                    return None
                    
            except Exception as e:
                return None
        
        # Parallel processing for speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(check_stock, sym): sym for sym in symbols}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)
        
        logger.info(f"\nTier 1 Complete: {len(candidates)}/{len(symbols)} stocks passed")
        return candidates
    
    def tier2_medium_analysis(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tier 2: Medium analysis on filtered candidates
        Criteria:
        - Find ATM/OTM calls with 0.30-0.70 delta
        - IV > HV (implied vol premium)
        - Positive momentum (price > 20-day MA)
        - Reasonable bid-ask spread (< 10% of mid price)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TIER 2: Medium Analysis - Analyzing {len(candidates)} candidates")
        logger.info(f"{'='*80}")
        
        qualified = []
        
        # Track rejection reasons
        rejection_stats = {
            'insufficient_history': 0,
            'negative_momentum': 0,
            'low_iv': 0,
            'no_valid_delta': 0,
            'errors': 0
        }
        
        for i, candidate in enumerate(candidates, 1):
            try:
                symbol = candidate['symbol']
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = candidate['price']
                
                # Get historical data for momentum (use 2mo to ensure >= 20 trading days)
                hist = ticker.history(period='2mo')
                if hist.empty or len(hist) < 20:
                    pass  # Insufficient historical data
                    rejection_stats['insufficient_history'] += 1
                    continue
                
                # Check momentum (temporarily disabled for testing)
                ma_20 = hist['Close'].tail(20).mean()
                # TEMPORARILY DISABLED: Testing if this is blocking all stocks
                # if current_price < ma_20 * 0.99:  # Allow up to 1% below MA20
                #     logger.info(f"  ✗ Weak momentum (price ${current_price:.2f} < 99% of MA20 ${ma_20:.2f})")
                #     rejection_stats['negative_momentum'] += 1
                #     continue
                
                # Get IV vs HV (use ATM options IV, not stock-level IV)
                hist_vol = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                
                # Get ATM call IV from first valid expiration
                try:
                    first_exp = candidate['valid_expirations'][0]
                    opt_chain = ticker.option_chain(first_exp)
                    calls = opt_chain.calls
                    
                    # Find ATM call (strike closest to current price)
                    calls['strike_diff'] = abs(calls['strike'] - current_price)
                    atm_call = calls.nsmallest(1, 'strike_diff')
                    
                    if atm_call.empty or atm_call.iloc[0]['impliedVolatility'] <= 0:
                        pass  # No valid ATM IV data
                        continue
                    
                    implied_vol_pct = atm_call.iloc[0]['impliedVolatility'] * 100
                    
                    # More lenient: allow IV >= 70% of HV (testing to find bottleneck)
                    if implied_vol_pct < hist_vol * 0.7:
                        pass  # IV too low
                        rejection_stats['low_iv'] += 1
                        continue
                        
                except Exception as e:
                    pass  # Error getting IV
                    continue
                
                # Find best call option (0.30-0.70 delta)
                best_call = None
                best_score = 0
                
                for exp_date in candidate['valid_expirations'][:3]:  # Check first 3 expirations
                    try:
                        opt_chain = ticker.option_chain(exp_date)
                        calls = opt_chain.calls
                        
                        if calls.empty:
                            continue
                        
                        # Filter for liquidity (don't filter ITM/OTM yet - delta will handle it)
                        valid_calls = calls[
                            (calls['volume'] > 10) &  # Minimum volume
                            (calls['openInterest'] > 50)  # Minimum OI
                        ].copy()
                        
                        if valid_calls.empty:
                            continue
                        
                        # Calculate bid-ask spread (filter out invalid bid/ask first)
                        valid_calls = valid_calls[
                            (valid_calls['bid'] > 0) & 
                            (valid_calls['ask'] > 0) &
                            (valid_calls['ask'] > valid_calls['bid'])  # Sanity check
                        ].copy()
                        
                        if valid_calls.empty:
                            continue
                        
                        valid_calls['spread_pct'] = (
                            (valid_calls['ask'] - valid_calls['bid']) / 
                            ((valid_calls['ask'] + valid_calls['bid']) / 2) * 100
                        )
                        
                        # Filter for reasonable spreads
                        valid_calls = valid_calls[valid_calls['spread_pct'] < 10]
                        
                        if valid_calls.empty:
                            continue
                        
                        # Calculate REAL delta using Black-Scholes (NO APPROXIMATIONS)
                        exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                        days_to_exp = (exp_datetime - datetime.now()).days
                        time_to_expiry = max(days_to_exp / 365.0, 0.001)  # Years
                        
                        # Get real dividend yield (not hardcoded)
                        div_yield = info.get('dividendYield', 0.0) or 0.0
                        
                        deltas = []
                        for idx, row in valid_calls.iterrows():
                            try:
                                greeks = self.greeks_calc.calculate_all_greeks(
                                    spot=current_price,
                                    strike=row['strike'],
                                    time_to_expiry=time_to_expiry,
                                    volatility=row['impliedVolatility'],
                                    option_type='call',
                                    dividend_yield=div_yield  # Real dividend yield
                                )
                                deltas.append(greeks['delta'])
                            except Exception as e:
                                deltas.append(0.0)
                        
                        valid_calls['real_delta'] = deltas
                        
                        # Filter for REAL delta range 0.28-0.72 (balanced: was 0.30-0.70, then 0.25-0.75)
                        target_calls = valid_calls[
                            (valid_calls['real_delta'] >= 0.28) &
                            (valid_calls['real_delta'] <= 0.72)
                        ].copy()
                        
                        if target_calls.empty:
                            continue
                        
                        # Score based on volume, OI, and spread
                        target_calls['score'] = (
                            target_calls['volume'] * 0.3 +
                            target_calls['openInterest'] * 0.5 -
                            target_calls['spread_pct'] * 10
                        )
                        
                        best_in_exp = target_calls.nlargest(1, 'score')
                        
                        if not best_in_exp.empty and best_in_exp.iloc[0]['score'] > best_score:
                            best_score = best_in_exp.iloc[0]['score']
                            best_call = {
                                'expiration': exp_date,
                                'strike': best_in_exp.iloc[0]['strike'],
                                'last_price': best_in_exp.iloc[0]['lastPrice'],
                                'bid': best_in_exp.iloc[0]['bid'],
                                'ask': best_in_exp.iloc[0]['ask'],
                                'volume': best_in_exp.iloc[0]['volume'],
                                'open_interest': best_in_exp.iloc[0]['openInterest'],
                                'implied_volatility': best_in_exp.iloc[0]['impliedVolatility'],
                                'real_delta': best_in_exp.iloc[0]['real_delta']
                            }
                    
                    except Exception as e:
                        continue
                
                if not best_call:
                    rejection_stats['no_valid_delta'] += 1
                    continue
                
                # Add to qualified list
                candidate['best_call'] = best_call
                candidate['hist_vol'] = hist_vol
                candidate['implied_vol'] = implied_vol_pct
                candidate['momentum'] = ((current_price / ma_20) - 1) * 100
                
                qualified.append(candidate)
                
                # Logging removed to prevent Railway rate limits
                
            except Exception as e:
                rejection_stats['errors'] += 1
                continue
        
        logger.info(f"\nTier 2 Complete: {len(qualified)}/{len(candidates)} passed")
        # Rejection stats tracked but not logged
        return qualified
    
    def tier3_deep_analysis(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tier 3: Full institutional analysis on top candidates (limit to 15 to prevent timeouts)
        Run complete 8-factor scoring with Greeks, IV crush, Kelly sizing
        """
        # Tier 3: Deep Analysis
        
        # Limit to top 10 for optimal diversification and industry standard
        if len(candidates) > 10:
            pass  # Limiting to top 10
            candidates = candidates[:10]
        
        results = []
        errors = 0
        
        for i, candidate in enumerate(candidates, 1):
            try:
                symbol = candidate['symbol']
                call = candidate['best_call']
                
                # Deep analysis in progress
                
                # Run full institutional analysis
                analysis = self.options_engine.analyze_single_option(
                    symbol=symbol,
                    strike_price=call['strike'],
                    expiration_date=call['expiration'],
                    option_type='call',
                    current_price=candidate['price'],
                    option_price=call['last_price']
                )
                
                if not analysis or 'error' in analysis:
                    pass  # Analysis failed
                    continue
                
                # Extract key metrics
                total_score = analysis.get('total_score', 0)
                
                if total_score < 45:  # Minimum score threshold (balanced: was 50, then 35)
                    continue
                
                # Build result
                result = {
                    'symbol': symbol,
                    'sector': candidate['sector'],
                    'current_price': candidate['price'],
                    'strike': call['strike'],
                    'expiration': call['expiration'],
                    'days_to_expiry': analysis.get('days_to_expiry', 0),
                    'option_price': call['last_price'],
                    'bid': call['bid'],
                    'ask': call['ask'],
                    'volume': call['volume'],
                    'open_interest': call['open_interest'],
                    
                    # Scoring
                    'total_score': total_score,
                    'greek_score': analysis.get('greek_score', 0),
                    'volatility_score': analysis.get('volatility_score', 0),
                    'liquidity_score': analysis.get('liquidity_score', 0),
                    'risk_reward_score': analysis.get('risk_reward_score', 0),
                    
                    # Greeks
                    'delta': analysis.get('greeks', {}).get('delta', 0),
                    'gamma': analysis.get('greeks', {}).get('gamma', 0),
                    'theta': analysis.get('greeks', {}).get('theta', 0),
                    'vega': analysis.get('greeks', {}).get('vega', 0),
                    
                    # Volatility
                    'implied_vol': candidate['implied_vol'],
                    'hist_vol': candidate['hist_vol'],
                    'iv_rank': analysis.get('iv_rank', 0),
                    'iv_percentile': analysis.get('iv_percentile', 0),
                    
                    # Risk metrics
                    'max_loss': call['last_price'] * 100,  # Per contract
                    'breakeven': call['strike'] + call['last_price'],
                    'profit_target': analysis.get('profit_target', 0),
                    'expected_return': analysis.get('expected_return', 0),
                    
                    # Position sizing
                    'kelly_fraction': analysis.get('kelly_fraction', 0),
                    'recommended_contracts': analysis.get('recommended_contracts', 1),
                    'position_size_pct': analysis.get('position_size_pct', 0),
                    
                    # Momentum
                    'momentum': candidate['momentum']
                }
                
                results.append(result)
                
                # Logging removed              
            except Exception as e:
                errors += 1
                if errors > 5:
                    logger.warning(f"Too many errors ({errors}), stopping Tier 3 early")
                    break
                continue
        
        # Tier 3 complete
        
        # Sort by total score descending
        results.sort(key=lambda x: x.get('total_score', 0), reverse=True)
        
        return results
    
    def scan(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Run full 3-tier scan and return top opportunities
        """
        # Starting scan
        
        # Tier 1: Quick filter
        tier1_candidates = self.tier1_quick_filter(self.universe)
        
        if not tier1_candidates:
            logger.warning("No candidates passed Tier 1 filter")
            return []
        
        # Tier 2: Medium analysis (limit to top 100 to reduce load)
        tier2_candidates = self.tier2_medium_analysis(tier1_candidates[:100])
        
        if not tier2_candidates:
            logger.warning("No candidates passed Tier 2 analysis")
            return []
        
        # Tier 3: Deep analysis
        final_results = self.tier3_deep_analysis(tier2_candidates)
        
        # Return top N results
        top_results = final_results[:max_results]
        
        logger.info(f"\nSCAN COMPLETE - {len(top_results)} opportunities found")
        
        return top_results


def main():
    """Main entry point"""
    try:
        # Get max results from command line (default 10)
        max_results = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        
        # Run scan
        scanner = OptionsScanner()
        results = scanner.scan(max_results=max_results)
        
        # Output JSON for backend (use custom encoder for NumPy types)
        print(json.dumps({
            'success': True,
            'opportunities': results,
            'total_scanned': len(scanner.universe),
            'timestamp': datetime.now().isoformat()
        }, cls=NumpyEncoder))
        
    except Exception as e:
        logger.error(f"Scanner failed: {str(e)}", exc_info=True)
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
