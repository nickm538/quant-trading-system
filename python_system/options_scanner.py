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
                    
                    # Total call volume
                    total_volume = calls['volume'].sum()
                    if total_volume < 500:  # Minimum 500 contracts/day
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
            
            for i, future in enumerate(as_completed(futures), 1):
                if i % 20 == 0:
                    logger.info(f"  Progress: {i}/{len(symbols)} stocks checked...")
                
                result = future.result()
                if result:
                    candidates.append(result)
                    logger.info(f"  ✓ {result['symbol']}: ${result['price']:.2f}, "
                              f"MCap ${result['market_cap']/1e9:.1f}B, "
                              f"Vol {result['options_volume']:.0f}, "
                              f"{result['sector']}")
        
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
        
        for i, candidate in enumerate(candidates, 1):
            try:
                symbol = candidate['symbol']
                logger.info(f"\n[{i}/{len(candidates)}] Analyzing {symbol}...")
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = candidate['price']
                
                # Get historical data for momentum
                hist = ticker.history(period='1mo')
                if hist.empty or len(hist) < 20:
                    logger.info(f"  ✗ Insufficient historical data")
                    continue
                
                # Check momentum
                ma_20 = hist['Close'].tail(20).mean()
                if current_price < ma_20:
                    logger.info(f"  ✗ Negative momentum (price ${current_price:.2f} < MA20 ${ma_20:.2f})")
                    continue
                
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
                        logger.info(f"  ✗ No valid ATM IV data")
                        continue
                    
                    implied_vol_pct = atm_call.iloc[0]['impliedVolatility'] * 100
                    
                    if implied_vol_pct <= hist_vol:
                        logger.info(f"  ✗ IV ({implied_vol_pct:.1f}%) <= HV ({hist_vol:.1f}%)")
                        continue
                        
                except Exception as e:
                    logger.info(f"  ✗ Error getting IV: {str(e)}")
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
                        
                        # Calculate bid-ask spread
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
                        
                        deltas = []
                        for idx, row in valid_calls.iterrows():
                            try:
                                greeks = self.greeks_calc.calculate_all_greeks(
                                    spot=current_price,
                                    strike=row['strike'],
                                    time_to_expiry=time_to_expiry,
                                    volatility=row['impliedVolatility'],
                                    option_type='call',
                                    dividend_yield=0.0
                                )
                                deltas.append(greeks['delta'])
                            except:
                                deltas.append(0.0)
                        
                        valid_calls['real_delta'] = deltas
                        
                        # Filter for REAL delta range 0.30-0.70 (medium risk/reward)
                        target_calls = valid_calls[
                            (valid_calls['real_delta'] >= 0.30) &
                            (valid_calls['real_delta'] <= 0.70)
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
                    logger.info(f"  ✗ No suitable calls found (need 0.30-0.70 delta range)")
                    continue
                
                # Add to qualified list
                candidate['best_call'] = best_call
                candidate['hist_vol'] = hist_vol
                candidate['implied_vol'] = implied_vol_pct
                candidate['momentum'] = ((current_price / ma_20) - 1) * 100
                
                qualified.append(candidate)
                
                logger.info(f"  ✓ Found call: ${best_call['strike']:.2f} exp {best_call['expiration']}")
                logger.info(f"    Premium: ${best_call['last_price']:.2f}, "
                          f"Delta: {best_call['real_delta']:.3f}, "
                          f"Vol: {best_call['volume']:.0f}, "
                          f"OI: {best_call['open_interest']:.0f}")
                logger.info(f"    IV: {implied_vol_pct:.1f}% vs HV: {hist_vol:.1f}%, "
                          f"Momentum: +{candidate['momentum']:.1f}%")
                
            except Exception as e:
                logger.error(f"  ✗ Error analyzing {candidate['symbol']}: {str(e)}")
                continue
        
        logger.info(f"\nTier 2 Complete: {len(qualified)}/{len(candidates)} passed")
        return qualified
    
    def tier3_deep_analysis(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tier 3: Full institutional analysis on top candidates
        Run complete 8-factor scoring with Greeks, IV crush, Kelly sizing
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TIER 3: Deep Analysis - Full institutional scoring")
        logger.info(f"{'='*80}")
        
        results = []
        
        for i, candidate in enumerate(candidates, 1):
            try:
                symbol = candidate['symbol']
                call = candidate['best_call']
                
                logger.info(f"\n[{i}/{len(candidates)}] Deep analysis: {symbol}")
                logger.info(f"  Analyzing ${call['strike']:.2f} call exp {call['expiration']}")
                
                # Run full institutional analysis
                analysis = self.options_engine.analyze_options(
                    symbol=symbol,
                    strike_price=call['strike'],
                    expiration_date=call['expiration'],
                    option_type='call',
                    current_price=candidate['price'],
                    option_price=call['last_price']
                )
                
                if not analysis or 'error' in analysis:
                    logger.info(f"  ✗ Analysis failed")
                    continue
                
                # Extract key metrics
                total_score = analysis.get('total_score', 0)
                
                if total_score < 50:  # Minimum quality threshold
                    logger.info(f"  ✗ Score too low: {total_score:.1f}/100")
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
                
                logger.info(f"  ✓ Score: {total_score:.1f}/100")
                logger.info(f"    Delta: {result['delta']:.3f}, "
                          f"Theta: {result['theta']:.3f}, "
                          f"Vega: {result['vega']:.3f}")
                logger.info(f"    Max Loss: ${result['max_loss']:.0f}, "
                          f"Breakeven: ${result['breakeven']:.2f}")
                logger.info(f"    Kelly: {result['kelly_fraction']*100:.1f}%, "
                          f"Contracts: {result['recommended_contracts']}")
                
            except Exception as e:
                logger.error(f"  ✗ Error in deep analysis for {candidate['symbol']}: {str(e)}")
                continue
        
        # Sort by total score
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        logger.info(f"\nTier 3 Complete: {len(results)} high-quality opportunities found")
        return results
    
    def scan(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Run full 3-tier scan and return top opportunities
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"OPTIONS SCANNER - Finding Best Long Call Opportunities")
        logger.info(f"Target: Short to mid-term (2-12 weeks), Medium delta (0.30-0.70)")
        logger.info(f"{'#'*80}")
        
        # Tier 1: Quick filter
        tier1_candidates = self.tier1_quick_filter(self.universe)
        
        if not tier1_candidates:
            logger.warning("No candidates passed Tier 1 filter")
            return []
        
        # Tier 2: Medium analysis
        tier2_candidates = self.tier2_medium_analysis(tier1_candidates)
        
        if not tier2_candidates:
            logger.warning("No candidates passed Tier 2 analysis")
            return []
        
        # Tier 3: Deep analysis
        final_results = self.tier3_deep_analysis(tier2_candidates)
        
        # Return top N results
        top_results = final_results[:max_results]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SCAN COMPLETE - Top {len(top_results)} Opportunities:")
        logger.info(f"{'='*80}")
        
        for i, result in enumerate(top_results, 1):
            logger.info(f"\n{i}. {result['symbol']} - Score: {result['total_score']:.1f}/100")
            logger.info(f"   ${result['strike']:.2f} call @ ${result['option_price']:.2f} "
                       f"exp {result['expiration']} ({result['days_to_expiry']} days)")
            logger.info(f"   Delta: {result['delta']:.3f}, IV: {result['implied_vol']:.1f}%, "
                       f"Momentum: +{result['momentum']:.1f}%")
            logger.info(f"   Max Loss: ${result['max_loss']:.0f}, "
                       f"Breakeven: ${result['breakeven']:.2f}")
        
        return top_results


def main():
    """Main entry point"""
    try:
        # Get max results from command line (default 10)
        max_results = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        
        # Run scan
        scanner = OptionsScanner()
        results = scanner.scan(max_results=max_results)
        
        # Output JSON for backend
        print(json.dumps({
            'success': True,
            'opportunities': results,
            'total_scanned': len(scanner.universe),
            'timestamp': datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Scanner failed: {str(e)}", exc_info=True)
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
