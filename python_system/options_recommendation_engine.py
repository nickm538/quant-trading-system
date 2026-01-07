"""
World-Class Options Recommendation Engine
Analyzes entire options chain and ranks contracts using sophisticated multi-factor scoring.
Returns TOP 3 long calls and puts with precise entry/exit prices and profit targets.
"""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from functools import partial

logger = logging.getLogger(__name__)

class OptionsRecommendationEngine:
    """
    Master options analyzer that ranks every contract and recommends the best opportunities.
    
    Scoring Factors (weighted):
    1. Probability of Profit (POP) - 25%
    2. Risk/Reward Ratio - 20%
    3. Liquidity Score - 15%
    4. IV Rank (relative cheapness) - 15%
    5. Greeks Optimization (delta, gamma, theta balance) - 15%
    6. Expected Move Alignment - 10%
    """
    
    def __init__(self):
        self.weights = {
            'pop': 0.25,
            'risk_reward': 0.20,
            'liquidity': 0.15,
            'iv_rank': 0.15,
            'greeks': 0.15,
            'expected_move': 0.10
        }
    
    def analyze_and_rank(
        self,
        options_data: dict,
        current_price: float,
        expected_move_pct: float,
        historical_vol: float,
        stock_signal: str = "NEUTRAL",
        technical_levels: dict = None
    ) -> dict:
        """
        Analyze entire options chain and return TOP 3 ranked calls and puts.
        
        Args:
            options_data: Raw options chain from yfinance
            current_price: Current stock price
            expected_move_pct: Expected price move percentage
            historical_vol: Historical volatility
            stock_signal: Overall stock signal (BUY/SELL/HOLD)
            technical_levels: Support/resistance levels
            
        Returns:
            dict with top_calls, top_puts, and analysis metadata
        """
        try:
            calls = options_data.get('calls', [])
            puts = options_data.get('puts', [])
            
            if not calls or not puts:
                return self._empty_result()
            
            # Score all calls in parallel
            logger.info(f"Scoring {len(calls)} call options in parallel using {cpu_count()} CPUs...")
            score_func_calls = partial(
                self._score_option,
                current_price=current_price,
                option_type='call',
                expected_move_pct=expected_move_pct,
                historical_vol=historical_vol,
                stock_signal=stock_signal,
                technical_levels=technical_levels
            )
            
            with Pool(processes=cpu_count()) as pool:
                scored_calls_raw = pool.map(score_func_calls, calls)
            scored_calls = [s for s in scored_calls_raw if s is not None]
            logger.info(f"Scored {len(scored_calls)} valid call options")
            
            # Score all puts in parallel
            logger.info(f"Scoring {len(puts)} put options in parallel using {cpu_count()} CPUs...")
            score_func_puts = partial(
                self._score_option,
                current_price=current_price,
                option_type='put',
                expected_move_pct=expected_move_pct,
                historical_vol=historical_vol,
                stock_signal=stock_signal,
                technical_levels=technical_levels
            )
            
            with Pool(processes=cpu_count()) as pool:
                scored_puts_raw = pool.map(score_func_puts, puts)
            scored_puts = [s for s in scored_puts_raw if s is not None]
            logger.info(f"Scored {len(scored_puts)} valid put options")
            
            # Sort by composite score (descending)
            scored_calls.sort(key=lambda x: x['composite_score'], reverse=True)
            scored_puts.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Get TOP 3
            top_calls = scored_calls[:3]
            top_puts = scored_puts[:3]
            
            return {
                'top_calls': top_calls,
                'top_puts': top_puts,
                'current_price': current_price,
                'total_calls_analyzed': len(scored_calls),
                'total_puts_analyzed': len(scored_puts),
                'analysis_timestamp': datetime.now().isoformat(),
                'scoring_methodology': {
                    'weights': self.weights,
                    'description': 'Multi-factor weighted scoring: POP (25%), Risk/Reward (20%), Liquidity (15%), IV Rank (15%), Greeks (15%), Expected Move (10%)'
                }
            }
            
        except Exception as e:
            logger.error(f"Options ranking failed: {e}")
            return self._empty_result()
    
    def _score_option(
        self,
        option: dict,
        current_price: float,
        option_type: str,
        expected_move_pct: float,
        historical_vol: float,
        stock_signal: str,
        technical_levels: dict
    ) -> Dict[str, Any]:
        """
        Score a single option contract using multi-factor analysis.
        """
        try:
            strike = option.get('strike', 0)
            last_price = option.get('lastPrice', 0)
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)
            bid_ask_spread = ask - bid if ask and bid else 0
            iv = option.get('impliedVolatility', 0)
            volume = option.get('volume', 0)
            open_interest = option.get('openInterest', 0)
            dte = option.get('daysToExpiration', 0)
            delta = option.get('delta', 0)
            gamma = option.get('gamma', 0)
            theta = option.get('theta', 0)
            vega = option.get('vega', 0)
            
            # Filter criteria
            if dte < 7:  # Must have at least 1 week
                return None
            if last_price <= 0:
                return None
            if volume < 10 and open_interest < 50:  # Minimum liquidity
                return None
            
            # Calculate delta filter (0.3-0.6 range for balanced risk/reward)
            abs_delta = abs(delta)
            if abs_delta < 0.25 or abs_delta > 0.65:
                return None
            
            # 1. Probability of Profit (POP)
            pop_score, pop_value = self._calculate_pop(
                strike=strike,
                current_price=current_price,
                option_type=option_type,
                iv=iv,
                dte=dte
            )
            
            # 2. Risk/Reward Ratio
            rr_score, rr_data = self._calculate_risk_reward(
                strike=strike,
                current_price=current_price,
                last_price=last_price,
                option_type=option_type,
                expected_move_pct=expected_move_pct
            )
            
            # 3. Liquidity Score (includes real bid/ask spread)
            liq_score = self._calculate_liquidity_score(
                volume=volume,
                open_interest=open_interest,
                last_price=last_price,
                bid_ask_spread=bid_ask_spread
            )
            
            # 4. IV Rank (relative cheapness)
            iv_rank_score = self._calculate_iv_rank_score(
                iv=iv,
                historical_vol=historical_vol
            )
            
            # 5. Greeks Optimization
            greeks_score = self._calculate_greeks_score(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                dte=dte
            )
            
            # 6. Expected Move Alignment
            em_score = self._calculate_expected_move_score(
                strike=strike,
                current_price=current_price,
                option_type=option_type,
                expected_move_pct=expected_move_pct
            )
            
            # Composite Score (weighted average)
            composite_score = (
                pop_score * self.weights['pop'] +
                rr_score * self.weights['risk_reward'] +
                liq_score * self.weights['liquidity'] +
                iv_rank_score * self.weights['iv_rank'] +
                greeks_score * self.weights['greeks'] +
                em_score * self.weights['expected_move']
            ) * 100  # Scale to 0-100
            
            # Adjust for stock signal alignment
            if stock_signal == "BUY" and option_type == "call":
                composite_score *= 1.1
            elif stock_signal == "SELL" and option_type == "put":
                composite_score *= 1.1
            elif stock_signal == "STRONG_BUY" and option_type == "call":
                composite_score *= 1.2
            elif stock_signal == "STRONG_SELL" and option_type == "put":
                composite_score *= 1.2
            
            # Calculate entry/exit prices
            entry_price = last_price
            profit_target = self._calculate_profit_target(
                entry_price=entry_price,
                strike=strike,
                current_price=current_price,
                option_type=option_type,
                expected_move_pct=expected_move_pct
            )
            stop_loss = entry_price * 0.5  # 50% max loss rule
            
            return {
                'strike': strike,
                'expiration': self._format_expiration(dte),
                'days_to_expiry': dte,
                'last_price': last_price,
                'bid_ask_spread': round(bid_ask_spread, 2),  # Real bid/ask spread from options data
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'implied_volatility': iv,
                'volume': volume,
                'open_interest': open_interest,
                'oi_vol_ratio': open_interest / max(volume, 1),
                'breakeven': strike + last_price if option_type == 'call' else strike - last_price,
                'max_loss': -last_price,
                'probability_of_profit': pop_value,
                'expected_return_pct': rr_data['expected_return_pct'],
                'risk_reward_ratio': rr_data['risk_reward_ratio'],
                'entry_price': entry_price,
                'profit_target': profit_target,
                'stop_loss': stop_loss,
                'composite_score': composite_score,
                'score_breakdown': {
                    'pop': pop_score * 100,
                    'risk_reward': rr_score * 100,
                    'liquidity': liq_score * 100,
                    'iv_rank': iv_rank_score * 100,
                    'greeks': greeks_score * 100,
                    'expected_move': em_score * 100
                },
                'recommendation_reason': self._generate_recommendation_reason(
                    composite_score=composite_score,
                    pop_value=pop_value,
                    rr_data=rr_data,
                    option_type=option_type
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to score option: {e}")
            return None
    
    def _calculate_pop(self, strike: float, current_price: float, option_type: str, iv: float, dte: int) -> Tuple[float, float]:
        """Calculate Probability of Profit using Black-Scholes."""
        if iv <= 0 or dte <= 0:
            return 0.0, 0.0
        
        try:
            # Time to expiration in years
            t = dte / 365.0
            
            # Calculate d2 from Black-Scholes
            d2 = (np.log(current_price / strike) - (0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
            
            # POP for calls: N(d2), for puts: N(-d2)
            if option_type == 'call':
                pop = norm.cdf(d2)
            else:
                pop = norm.cdf(-d2)
            
            # Score: Higher POP = better (0-1 scale)
            score = pop
            
            return score, pop
        except:
            return 0.0, 0.0
    
    def _calculate_risk_reward(self, strike: float, current_price: float, last_price: float, 
                               option_type: str, expected_move_pct: float) -> Tuple[float, dict]:
        """Calculate risk/reward ratio based on expected move."""
        try:
            max_loss = last_price
            
            # Expected profit if stock moves as predicted
            if option_type == 'call':
                expected_price = current_price * (1 + expected_move_pct / 100)
                expected_profit = max(0, expected_price - strike) - last_price
            else:
                expected_price = current_price * (1 - expected_move_pct / 100)
                expected_profit = max(0, strike - expected_price) - last_price
            
            if max_loss <= 0:
                return 0.0, {'risk_reward_ratio': 0, 'expected_return_pct': 0}
            
            rr_ratio = expected_profit / max_loss if max_loss > 0 else 0
            expected_return_pct = (expected_profit / last_price) * 100 if last_price > 0 else 0
            
            # Score: Higher R/R = better (normalize to 0-1)
            # R/R of 2:1 or better gets max score
            score = min(rr_ratio / 2.0, 1.0)
            
            return score, {
                'risk_reward_ratio': rr_ratio,
                'expected_return_pct': expected_return_pct
            }
        except:
            return 0.0, {'risk_reward_ratio': 0, 'expected_return_pct': 0}
    
    def _calculate_liquidity_score(self, volume: int, open_interest: int, last_price: float, bid_ask_spread: float = 0) -> float:
        """Score based on volume, open interest, and bid-ask spread."""
        try:
            # Volume score (0-0.4)
            vol_score = min(volume / 1000, 1.0) * 0.4
            
            # Open interest score (0-0.4)
            oi_score = min(open_interest / 5000, 1.0) * 0.4
            
            # Bid-ask spread score (0-0.2) - calculated from real bid/ask data
            # Tighter spread = better score. Spread as % of last price
            if last_price > 0 and bid_ask_spread >= 0:
                spread_pct = bid_ask_spread / last_price
                # Score: 0.2 for <1% spread, scales down to 0 for >10% spread
                spread_score = max(0, 0.2 * (1 - min(spread_pct / 0.10, 1.0)))
            else:
                spread_score = 0.1  # Neutral if no spread data
            
            return vol_score + oi_score + spread_score
        except:
            return 0.0
    
    def _calculate_iv_rank_score(self, iv: float, historical_vol: float) -> float:
        """Score based on IV relative to historical volatility."""
        try:
            if historical_vol <= 0:
                return 0.5  # Neutral if no historical data
            
            iv_ratio = iv / historical_vol
            
            # Prefer IV close to historical (not too high or too low)
            # Optimal range: 0.8x to 1.2x historical vol
            if 0.8 <= iv_ratio <= 1.2:
                score = 1.0
            elif iv_ratio < 0.8:
                # IV too low - less attractive
                score = iv_ratio / 0.8
            else:
                # IV too high - overpriced
                score = 1.2 / iv_ratio
            
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
    
    def _calculate_greeks_score(self, delta: float, gamma: float, theta: float, vega: float, dte: int) -> float:
        """Score based on optimal Greeks balance."""
        try:
            abs_delta = abs(delta)
            
            # Delta score (0-0.4): Prefer 0.4-0.5 delta (balanced)
            if 0.4 <= abs_delta <= 0.5:
                delta_score = 1.0
            else:
                delta_score = 1.0 - abs(abs_delta - 0.45) / 0.45
            delta_score = max(delta_score, 0.0) * 0.4
            
            # Gamma score (0-0.2): Higher gamma = more leverage
            gamma_score = min(abs(gamma) * 100, 1.0) * 0.2
            
            # Theta score (0-0.2): Lower theta decay = better
            # Normalize theta by DTE
            theta_per_day = abs(theta)
            theta_score = max(1.0 - theta_per_day / 0.1, 0.0) * 0.2
            
            # Vega score (0-0.2): Higher vega = more IV sensitivity
            vega_score = min(abs(vega) / 0.5, 1.0) * 0.2
            
            return delta_score + gamma_score + theta_score + vega_score
        except:
            return 0.0
    
    def _calculate_expected_move_score(self, strike: float, current_price: float, 
                                      option_type: str, expected_move_pct: float) -> float:
        """Score based on alignment with expected move."""
        try:
            expected_price = current_price * (1 + expected_move_pct / 100 if option_type == 'call' else 1 - expected_move_pct / 100)
            
            # Distance from strike to expected price
            distance_pct = abs(strike - expected_price) / current_price * 100
            
            # Prefer strikes near expected price (within 5%)
            if distance_pct <= 5:
                score = 1.0
            else:
                score = max(1.0 - (distance_pct - 5) / 10, 0.0)
            
            return score
        except:
            return 0.0
    
    def _calculate_profit_target(self, entry_price: float, strike: float, current_price: float,
                                option_type: str, expected_move_pct: float) -> float:
        """Calculate profit target price for the option."""
        try:
            # Target 100% return (2x entry price)
            return entry_price * 2.0
        except:
            return entry_price * 1.5
    
    def _format_expiration(self, dte: int) -> str:
        """Format expiration date."""
        from datetime import datetime, timedelta
        exp_date = datetime.now() + timedelta(days=dte)
        return exp_date.strftime('%Y-%m-%d')
    
    def _generate_recommendation_reason(self, composite_score: float, pop_value: float, 
                                       rr_data: dict, option_type: str) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        if composite_score >= 70:
            reasons.append("EXCELLENT opportunity")
        elif composite_score >= 60:
            reasons.append("STRONG opportunity")
        elif composite_score >= 50:
            reasons.append("GOOD opportunity")
        else:
            reasons.append("MODERATE opportunity")
        
        if pop_value >= 0.6:
            reasons.append(f"{pop_value*100:.0f}% probability of profit")
        
        if rr_data['risk_reward_ratio'] >= 2:
            reasons.append(f"{rr_data['risk_reward_ratio']:.1f}:1 risk/reward")
        
        if rr_data['expected_return_pct'] >= 50:
            reasons.append(f"{rr_data['expected_return_pct']:.0f}% expected return")
        
        return "; ".join(reasons)
    
    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            'top_calls': [],
            'top_puts': [],
            'current_price': 0,
            'total_calls_analyzed': 0,
            'total_puts_analyzed': 0,
            'analysis_timestamp': datetime.now().isoformat(),
            'scoring_methodology': {
                'weights': self.weights,
                'description': 'No options data available'
            }
        }
