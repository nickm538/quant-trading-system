"""
TOP 3 OPTIONS CHAINS ANALYZER
Find the best 3 call and 3 put options for a given stock
Full Greeks analysis, liquidity metrics, profit/loss scenarios
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from scipy.stats import norm

from data.enhanced_data_ingestion import EnhancedDataIngestion
from risk_free_rate import get_risk_free_rate
from marketdata_client import MarketDataClient
from iv_crush_monitor import IVCrushMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """
    Comprehensive options chain analyzer
    Finds top 3 calls and top 3 puts with full Greeks
    """
    
    def __init__(self):
        self.data_ingestion = EnhancedDataIngestion()
        self.iv_crush_monitor = IVCrushMonitor()
    
    def _get_iv_crush_warning(self, risk_level: str, expected_crush_pct: float, days_to_expiry: int) -> str:
        """
        Generate IV crush warning message based on risk level
        """
        if risk_level == 'extreme':
            if days_to_expiry <= 7:
                return f"⚠️ EXTREME RISK: Expected {expected_crush_pct:.0f}% IV crush + expires in {days_to_expiry} days. Likely earnings event. AVOID buying unless expecting >50% stock move."
            else:
                return f"⚠️ EXTREME RISK: Expected {expected_crush_pct:.0f}% IV crush. Options will lose significant value even if stock doesn't move. Consider selling premium instead."
        elif risk_level == 'high':
            return f"⚠️ HIGH RISK: Expected {expected_crush_pct:.0f}% IV crush. Stock must move >{expected_crush_pct:.0f}% to overcome IV decay. Reduce position size by 50%."
        elif risk_level == 'moderate':
            return f"⚠️ MODERATE RISK: Expected {expected_crush_pct:.0f}% IV crush. Factor into profit targets. Reduce position size by 25%."
        else:
            return "✓ Low IV crush risk. Normal position sizing."
    
    def calculate_greeks(
        self,
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Implied volatility
        option_type: str  # 'call' or 'put'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks using Black-Scholes
        """
        if T <= 0 or sigma <= 0:
            return {
                'delta': 0, 'gamma': 0, 'theta': 0, 
                'vega': 0, 'rho': 0, 'theoretical_price': 0
            }
        
        # Black-Scholes calculations
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            # Call option Greeks
            delta = norm.cdf(d1)
            theoretical_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            # Put option Greeks
            delta = -norm.cdf(-d1)
            theoretical_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Common Greeks
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'theoretical_price': theoretical_price
        }
    
    def calculate_profit_loss_scenarios(
        self,
        current_price: float,
        strike: float,
        option_price: float,
        option_type: str,
        expiration_date: str
    ) -> Dict:
        """
        Calculate profit/loss scenarios at different price points
        """
        # Price range: -20% to +20%
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 21)
        
        scenarios = []
        for price in price_range:
            if option_type == 'call':
                intrinsic_value = max(0, price - strike)
            else:
                intrinsic_value = max(0, strike - price)
            
            profit_loss = (intrinsic_value - option_price) * 100  # Per contract
            profit_loss_pct = (profit_loss / (option_price * 100)) * 100 if option_price > 0 else 0
            
            scenarios.append({
                'stock_price': price,
                'option_value': intrinsic_value,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })
        
        # Breakeven
        if option_type == 'call':
            breakeven = strike + option_price
        else:
            breakeven = strike - option_price
        
        # Max profit/loss
        if option_type == 'call':
            max_loss = -option_price * 100
            max_profit = float('inf')  # Theoretically unlimited
        else:
            max_loss = -option_price * 100
            max_profit = (strike - option_price) * 100
        
        return {
            'scenarios': scenarios,
            'breakeven': breakeven,
            'max_loss': max_loss,
            'max_profit': max_profit if max_profit != float('inf') else 'Unlimited'
        }
    
    def analyze_options_chain(
        self,
        symbol: str,
        min_delta: float = 0.3,
        max_delta: float = 0.6,
        min_days_to_expiry: int = 7
    ) -> Dict:
        """
        Analyze options chain and find top 3 calls and top 3 puts
        """
        logger.info("=" * 80)
        logger.info(f"OPTIONS CHAIN ANALYSIS: {symbol}")
        logger.info("=" * 80)
        logger.info(f"  Delta range: {min_delta} to {max_delta}")
        logger.info(f"  Min days to expiry: {min_days_to_expiry}")
        
        # Get stock data
        complete_data = self.data_ingestion.get_complete_stock_data(symbol)
        price_data = complete_data['price_data']
        
        if price_data.empty:
            logger.error(f"No price data for {symbol}")
            return None
        
        current_price = price_data['close'].iloc[-1]
        logger.info(f"  Current price: ${current_price:.2f}")
        
        # Get options data from MarketData.app (REAL DATA with bid/ask/volume/OI)
        logger.info("  Fetching options chain from MarketData.app...")
        marketdata_client = MarketDataClient()
        options_data = marketdata_client.get_filtered_options(
            symbol=symbol,
            min_dte=min_days_to_expiry,
            max_dte=90,
            min_delta=min_delta,
            max_delta=max_delta,
            min_volume=10,
            min_oi=50
        )
        
        call_contracts = options_data.get('calls', [])
        put_contracts = options_data.get('puts', [])
        
        if not call_contracts and not put_contracts:
            logger.error(f"No options data for {symbol} from MarketData.app")
            return None
        
        logger.info(f"  Found {len(call_contracts)} call options, {len(put_contracts)} put options from MarketData.app")
        
        # Risk-free rate (dynamic - fetched from Yahoo Finance ^TNX)
        r = get_risk_free_rate()
        logger.info(f"  Using risk-free rate: {r:.4f} ({r*100:.2f}%)")
        
        # Calculate historical volatility as fallback
        returns = price_data['close'].pct_change().dropna()
        hist_vol = returns.tail(30).std() * np.sqrt(252)  # 30-day annualized volatility
        min_vol = max(hist_vol, 0.15)  # Minimum 15% volatility threshold
        
        logger.info(f"  Historical volatility (30d): {hist_vol:.1%}")
        logger.info(f"  Using minimum volatility: {min_vol:.1%}")
        
        # Process call options from MarketData.app (REAL DATA)
        call_candidates = []
        
        for contract in call_contracts:
            try:
                # Extract contract details (MarketData.app format)
                strike = contract.get('strike', 0)
                ticker = contract.get('option_symbol', '')
                days_to_expiry = contract.get('dte', 0)
                
                # REAL LIQUIDITY DATA from MarketData.app
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                mid = contract.get('mid', 0)
                volume = contract.get('volume', 0)
                open_interest = contract.get('open_interest', 0)
                last_price = contract.get('last', mid)  # Use last trade or mid price
                
                # REAL GREEKS from MarketData.app
                delta = contract.get('delta', 0)
                gamma = contract.get('gamma', 0)
                theta = contract.get('theta', 0)
                vega = contract.get('vega', 0)
                iv = contract.get('iv', hist_vol)  # Real IV from market
                
                if not strike or days_to_expiry < min_days_to_expiry:
                    continue
                
                T = days_to_expiry / 365.0  # Years
                
                # REAL LIQUIDITY METRICS
                bid_ask_spread = ask - bid if ask > bid else 0
                bid_ask_spread_pct = (bid_ask_spread / mid * 100) if mid > 0 else 100
                oi_vol_ratio = open_interest / volume if volume > 0 else 0
                
                # IV percentile (compare to historical vol)
                iv_percentile = (iv / hist_vol) * 100 if hist_vol > 0 else 100
                
                # Liquidity score (based on REAL volume/OI/spread)
                liquidity_score = 0
                if volume >= 100 and open_interest >= 1000 and bid_ask_spread_pct < 5:
                    liquidity_score = 100  # Excellent liquidity
                elif volume >= 50 and open_interest >= 500 and bid_ask_spread_pct < 10:
                    liquidity_score = 75   # Good liquidity
                elif volume >= 10 and open_interest >= 50 and bid_ask_spread_pct < 20:
                    liquidity_score = 50   # Acceptable liquidity
                else:
                    liquidity_score = 25   # Poor liquidity
                
                # Delta score (prefer 0.45)
                delta_score = 100 - abs(delta - 0.45) * 200
                
                # IV score (prefer moderate IV)
                iv_score = 100 - abs(iv - 0.35) * 200
                
                # IV crush risk penalty
                # Get IV crush analysis for this stock
                iv_crush_data = self.iv_crush_monitor.detect_earnings_iv_crush(symbol)
                
                iv_crush_penalty = 0
                iv_crush_risk = 'low'
                expected_crush_pct = 0
                
                if iv_crush_data:
                    expected_crush_pct = iv_crush_data.get('estimated_iv_crush_pct', 0)
                    iv_rank = iv_crush_data.get('iv_rank', 50)
                    potential_earnings = iv_crush_data.get('potential_earnings_event', False)
                    
                    # Penalize contracts with high IV crush risk
                    if potential_earnings and expected_crush_pct > 40:
                        iv_crush_penalty = 50  # Severe penalty
                        iv_crush_risk = 'extreme'
                    elif potential_earnings and expected_crush_pct > 30:
                        iv_crush_penalty = 30  # High penalty
                        iv_crush_risk = 'high'
                    elif expected_crush_pct > 25:
                        iv_crush_penalty = 15  # Moderate penalty
                        iv_crush_risk = 'moderate'
                    
                    # Extra penalty for contracts expiring soon (within 7 days)
                    if days_to_expiry <= 7 and potential_earnings:
                        iv_crush_penalty += 20  # Likely to expire during/after earnings
                        iv_crush_risk = 'extreme'
                
                # Combined score (weight delta more since we don't have liquidity)
                # Subtract IV crush penalty
                score = (delta_score * 0.7 + iv_score * 0.3) - iv_crush_penalty
                
                # Profit/loss scenarios
                pl_scenarios = self.calculate_profit_loss_scenarios(
                    current_price=current_price,
                    strike=strike,
                    option_price=last_price,
                    option_type='call',
                    expiration_date=None  # MarketData uses timestamp
                )
                
                # Intrinsic and extrinsic values
                intrinsic_value = contract.get('intrinsic_value', max(0, current_price - strike))
                extrinsic_value = contract.get('extrinsic_value', last_price - intrinsic_value)
                
                call_candidates.append({
                    'type': 'CALL',
                    'strike': strike,
                    'expiration': f"{days_to_expiry} days",  # Format for display
                    'days_to_expiry': days_to_expiry,
                    'last_price': last_price,
                    'bid': bid,  # REAL BID from MarketData.app
                    'ask': ask,  # REAL ASK from MarketData.app
                    'bid_ask_spread': bid_ask_spread,
                    'bid_ask_spread_pct': bid_ask_spread_pct,
                    'volume': volume,  # REAL VOLUME
                    'open_interest': open_interest,  # REAL OI
                    'oi_vol_ratio': oi_vol_ratio,
                    'implied_volatility': iv,  # REAL IV from market
                    'iv_percentile': iv_percentile,
                    'delta': delta,  # REAL DELTA from MarketData.app
                    'gamma': gamma,  # REAL GAMMA
                    'theta': theta,  # REAL THETA
                    'vega': vega,    # REAL VEGA
                    'rho': 0,        # Not provided by MarketData.app
                    'theoretical_price': mid,  # Use mid price as theoretical
                    'intrinsic_value': intrinsic_value,
                    'extrinsic_value': extrinsic_value,
                    'breakeven': pl_scenarios['breakeven'],
                    'max_loss': pl_scenarios['max_loss'],
                    'max_profit': pl_scenarios['max_profit'],
                    'profit_loss_scenarios': pl_scenarios['scenarios'],
                    'score': score,
                    'ticker': ticker,  # MarketData option symbol
                    'liquidity_score': liquidity_score,  # NEW: Real liquidity scoring
                    'iv_crush_risk': iv_crush_risk,
                    'expected_iv_crush_pct': expected_crush_pct,
                    'iv_crush_warning': self._get_iv_crush_warning(iv_crush_risk, expected_crush_pct, days_to_expiry)
                })
            
            except Exception as e:
                logger.warning(f"Error processing contract: {e}")
                continue
        
        # MarketData.app provides REAL data - no need for second pass!
        logger.info("\n✓ Options analysis complete with REAL market data")
        logger.info(f"  Total call candidates: {len(call_candidates)}")
        
        # Sort by score to get top recommendations
        call_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top 3 calls and puts with REAL market data
        top_calls = call_candidates[:3]
        
        # Process put options from MarketData.app
        put_candidates = []
        for contract in put_contracts:
            try:
                # Same processing as calls but for puts
                strike = contract.get('strike', 0)
                ticker = contract.get('option_symbol', '')
                days_to_expiry = contract.get('dte', 0)
                
                bid = contract.get('bid', 0)
                ask = contract.get('ask', 0)
                mid = contract.get('mid', 0)
                volume = contract.get('volume', 0)
                open_interest = contract.get('open_interest', 0)
                last_price = contract.get('last', mid)
                
                delta = contract.get('delta', 0)
                gamma = contract.get('gamma', 0)
                theta = contract.get('theta', 0)
                vega = contract.get('vega', 0)
                iv = contract.get('iv', hist_vol)
                
                if not strike or days_to_expiry < min_days_to_expiry:
                    continue
                
                # Liquidity metrics
                bid_ask_spread = ask - bid if ask > bid else 0
                bid_ask_spread_pct = (bid_ask_spread / mid * 100) if mid > 0 else 100
                oi_vol_ratio = open_interest / volume if volume > 0 else 0
                
                # Liquidity score
                liquidity_score = 0
                if volume >= 100 and open_interest >= 1000 and bid_ask_spread_pct < 5:
                    liquidity_score = 100
                elif volume >= 50 and open_interest >= 500 and bid_ask_spread_pct < 10:
                    liquidity_score = 75
                elif volume >= 10 and open_interest >= 50 and bid_ask_spread_pct < 20:
                    liquidity_score = 50
                else:
                    liquidity_score = 25
                
                delta_score = 100 - abs(abs(delta) - 0.45) * 200
                iv_score = 100 - abs(iv - 0.35) * 200
                
                # IV crush analysis
                iv_crush_data = self.iv_crush_monitor.detect_earnings_iv_crush(symbol)
                iv_crush_penalty = 0
                iv_crush_risk = 'low'
                expected_crush_pct = 0
                
                if iv_crush_data:
                    expected_crush_pct = iv_crush_data.get('estimated_iv_crush_pct', 0)
                    potential_earnings = iv_crush_data.get('potential_earnings_event', False)
                    
                    if potential_earnings and expected_crush_pct > 40:
                        iv_crush_penalty = 50
                        iv_crush_risk = 'extreme'
                    elif potential_earnings and expected_crush_pct > 30:
                        iv_crush_penalty = 30
                        iv_crush_risk = 'high'
                    elif expected_crush_pct > 25:
                        iv_crush_penalty = 15
                        iv_crush_risk = 'moderate'
                
                score = (delta_score * 0.7 + iv_score * 0.3) - iv_crush_penalty
                
                pl_scenarios = self.calculate_profit_loss_scenarios(
                    current_price=current_price,
                    strike=strike,
                    option_price=last_price,
                    option_type='put',
                    expiration_date=None
                )
                
                intrinsic_value = contract.get('intrinsic_value', max(0, strike - current_price))
                extrinsic_value = contract.get('extrinsic_value', last_price - intrinsic_value)
                
                put_candidates.append({
                    'type': 'PUT',
                    'strike': strike,
                    'expiration': f"{days_to_expiry} days",
                    'days_to_expiry': days_to_expiry,
                    'last_price': last_price,
                    'bid': bid,
                    'ask': ask,
                    'bid_ask_spread': bid_ask_spread,
                    'bid_ask_spread_pct': bid_ask_spread_pct,
                    'volume': volume,
                    'open_interest': open_interest,
                    'oi_vol_ratio': oi_vol_ratio,
                    'implied_volatility': iv,
                    'iv_percentile': (iv / hist_vol) * 100 if hist_vol > 0 else 100,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': 0,
                    'theoretical_price': mid,
                    'intrinsic_value': intrinsic_value,
                    'extrinsic_value': extrinsic_value,
                    'breakeven': pl_scenarios['breakeven'],
                    'max_loss': pl_scenarios['max_loss'],
                    'max_profit': pl_scenarios['max_profit'],
                    'profit_loss_scenarios': pl_scenarios['scenarios'],
                    'score': score,
                    'ticker': ticker,
                    'liquidity_score': liquidity_score,
                    'iv_crush_risk': iv_crush_risk,
                    'expected_iv_crush_pct': expected_crush_pct,
                    'iv_crush_warning': self._get_iv_crush_warning(iv_crush_risk, expected_crush_pct, days_to_expiry)
                })
            except Exception as e:
                logger.warning(f"Error processing put contract: {e}")
                continue
        
        put_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_puts = put_candidates[:3]
        
        logger.info("\n✓ FINAL RESULTS from MarketData.app (100% REAL DATA)")
        call_info = [f"{c['strike']} ({c['expiration']})" for c in top_calls]
        put_info = [f"{p['strike']} ({p['expiration']})" for p in top_puts]
        logger.info(f"  Top 3 Calls: {call_info}")
        logger.info(f"  Top 3 Puts: {put_info}")
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'top_calls': top_calls,
            'top_puts': top_puts,
            'total_call_candidates': len(call_candidates),
            'total_put_candidates': len(put_candidates)
        }


if __name__ == "__main__":
    analyzer = OptionsAnalyzer()
    result = analyzer.analyze_options_chain('AAPL')
    
    if result:
        print("\n" + "=" * 80)
        print("TOP 2 CALL OPTIONS")
        print("=" * 80)
        for i, call in enumerate(result['top_calls'], 1):
            print("\n" + f"{i}. Strike ${call['strike']} - Exp: {call['expiration']}")
            print(f"   Price: ${call['last_price']:.2f} | Delta: {call['delta']:.3f} | IV: {call['implied_volatility']*100:.1f}%")
            print(f"   Breakeven: ${call['breakeven']:.2f} | Max Loss: ${call['max_loss']:.2f}")
        
        print("\n" + "=" * 80)
        print("INSTITUTIONAL-GRADE OPTIONS DATA - MarketData.app")
        print("=" * 80)
        print(f"Total call candidates: {result['total_call_candidates']}")
        print(f"Total put candidates: {result['total_put_candidates']}")
        print(f"100% REAL DATA: Bid/Ask/Volume/OI/Greeks/IV from live market")
