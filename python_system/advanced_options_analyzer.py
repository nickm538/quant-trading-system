"""
Advanced Options Analysis Module
Provides institutional-grade options analytics including:
- IV crush monitoring with earnings calendar
- Volatility surface and skew analysis
- Drift detection (realized vs implied volatility)
- Term structure analysis
- Greeks surface
- Probability of profit calculations
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.interpolate import griddata
import logging

sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
from options_recommendation_engine import OptionsRecommendationEngine

logger = logging.getLogger(__name__)

class AdvancedOptionsAnalyzer:
    def __init__(self):
        self.api_client = ApiClient()
    
    def analyze_options(self, symbol: str, current_price: float, historical_volatility: float, 
                       days_to_earnings: int = None) -> dict:
        """
        Comprehensive options analysis with advanced metrics.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_volatility: Realized historical volatility (annualized)
            days_to_earnings: Days until next earnings (for IV crush monitoring)
        
        Returns:
            dict with all options analytics
        """
        try:
            # Fetch options chain
            options_data = self._fetch_options_chain(symbol, current_price)
            
            if not options_data:
                return self._empty_result()
            
            # Calculate all advanced metrics
            iv_crush_analysis = self._analyze_iv_crush(options_data, days_to_earnings, historical_volatility)
            volatility_surface = self._calculate_volatility_surface(options_data, current_price)
            skew_analysis = self._analyze_volatility_skew(options_data, current_price)
            drift_detection = self._detect_drift(historical_volatility, volatility_surface.get('atm_iv', 0))
            term_structure = self._analyze_term_structure(options_data)
            expected_move = self._calculate_expected_move(options_data, current_price)
            greeks_analysis = self._analyze_greeks_surface(options_data, current_price)
            
            # Generate TOP 3 recommendations using world-class ranking engine
            recommendation_engine = OptionsRecommendationEngine()
            recommendations = recommendation_engine.analyze_and_rank(
                options_data=options_data,
                current_price=current_price,
                expected_move_pct=expected_move.get('expected_move_pct', 2.0),
                historical_vol=historical_volatility,
                stock_signal="NEUTRAL",  # TODO: Pass from main analysis
                technical_levels=None  # TODO: Pass support/resistance levels
            )
            
            return {
                # World-class recommendations (TOP 3 calls and puts)
                'top_calls': recommendations.get('top_calls', []),
                'top_puts': recommendations.get('top_puts', []),
                'current_price': current_price,
                'total_calls_analyzed': recommendations.get('total_calls_analyzed', 0),
                'total_puts_analyzed': recommendations.get('total_puts_analyzed', 0),
                
                # Advanced analytics
                'iv_crush_monitor': iv_crush_analysis,
                'volatility_surface': volatility_surface,
                'skew_analysis': skew_analysis,
                'drift_detection': drift_detection,
                'term_structure': term_structure,
                'expected_move': expected_move,
                'greeks_surface': greeks_analysis,
                'data_quality': {
                    'total_contracts': len(options_data.get('calls', [])) + len(options_data.get('puts', [])),
                    'has_earnings_date': days_to_earnings is not None,
                    'surface_complete': volatility_surface.get('data_points', 0) > 20
                },
                
                # Scoring methodology
                'scoring_methodology': recommendations.get('scoring_methodology', {})
            }
        
        except Exception as e:
            logger.error(f"Options analysis failed: {e}")
            return self._empty_result()
    
    def _calculate_delta(self, current_price: float, strike: float, iv: float, dte: int, option_type: str) -> float:
        """Calculate option delta using Black-Scholes formula."""
        try:
            if iv <= 0 or dte <= 0 or current_price <= 0:
                return 0.0
            
            t = dte / 365.0
            d1 = (np.log(current_price / strike) + (0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
            
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
            
            return delta
        except:
            return 0.0
    
    def _fetch_options_chain(self, symbol: str, current_price: float) -> dict:
        """Fetch options chain using yfinance library"""
        try:
            import yfinance as yf
            import pandas as pd
            
            ticker = yf.Ticker(symbol)
            
            # Get all expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {}
            
            # Fetch options for all expirations (limit to first 6 for performance)
            all_calls = []
            all_puts = []
            
            for exp_date in expirations[:3]:  # Limit to 3 expirations for speed
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    for _, row in opt_chain.calls.iterrows():
                        strike = float(row['strike'])
                        
                        # Early filter: skip deep ITM/OTM (more than 20% away)
                        moneyness = strike / current_price
                        if moneyness < 0.8 or moneyness > 1.2:
                            continue
                        
                        # Early filter: skip low liquidity
                        volume = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
                        oi = int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0
                        if volume < 10 and oi < 50:
                            continue
                        
                        iv = float(row.get('impliedVolatility', 0))
                        dte = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                        
                        # Calculate delta using Black-Scholes if not provided
                        delta = float(row.get('delta', 0)) if pd.notna(row.get('delta')) else self._calculate_delta(
                            current_price=current_price,
                            strike=strike,
                            iv=iv,
                            dte=dte,
                            option_type='call'
                        )
                        
                        all_calls.append({
                            'strike': strike,
                            'lastPrice': float(row.get('lastPrice', 0)),
                            'impliedVolatility': iv,
                            'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                            'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                            'daysToExpiration': dte,
                            'delta': delta,
                            'gamma': float(row.get('gamma', 0)) if pd.notna(row.get('gamma')) else 0,
                            'vega': float(row.get('vega', 0)) if pd.notna(row.get('vega')) else 0,
                            'theta': float(row.get('theta', 0)) if pd.notna(row.get('theta')) else 0
                        })
                    
                    # Process puts
                    for _, row in opt_chain.puts.iterrows():
                        strike = float(row['strike'])
                        
                        # Early filter: skip deep ITM/OTM (more than 20% away)
                        moneyness = strike / current_price
                        if moneyness < 0.8 or moneyness > 1.2:
                            continue
                        
                        # Early filter: skip low liquidity
                        volume = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
                        oi = int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0
                        if volume < 10 and oi < 50:
                            continue
                        
                        iv = float(row.get('impliedVolatility', 0))
                        dte = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                        
                        # Calculate delta using Black-Scholes if not provided
                        delta = float(row.get('delta', 0)) if pd.notna(row.get('delta')) else self._calculate_delta(
                            current_price=current_price,
                            strike=strike,
                            iv=iv,
                            dte=dte,
                            option_type='put'
                        )
                        
                        all_puts.append({
                            'strike': strike,
                            'lastPrice': float(row.get('lastPrice', 0)),
                            'impliedVolatility': iv,
                            'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                            'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                            'daysToExpiration': dte,
                            'delta': delta,
                            'gamma': float(row.get('gamma', 0)) if pd.notna(row.get('gamma')) else 0,
                            'vega': float(row.get('vega', 0)) if pd.notna(row.get('vega')) else 0,
                            'theta': float(row.get('theta', 0)) if pd.notna(row.get('theta')) else 0
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch options for {exp_date}: {e}")
                    continue
            
            return {
                'calls': all_calls,
                'puts': all_puts
            }
        except Exception as e:
            logger.error(f"Failed to fetch options chain: {e}")
            return {}
    
    def _analyze_iv_crush(self, options_data: dict, days_to_earnings: int, historical_vol: float) -> dict:
        """
        Monitor IV crush risk around earnings.
        IV typically drops 20-50% after earnings announcement.
        """
        if not options_data or not days_to_earnings:
            return {
                'risk_level': 'UNKNOWN',
                'current_iv_percentile': None,
                'expected_crush': None,
                'recommendation': 'No earnings date available'
            }
        
        # Get ATM IV
        atm_iv = self._get_atm_iv(options_data)
        
        # Calculate IV percentile (current IV vs historical vol)
        iv_ratio = atm_iv / historical_vol if historical_vol > 0 else 1.0
        
        # Estimate IV crush based on days to earnings
        if days_to_earnings <= 7:
            risk_level = 'EXTREME'
            expected_crush_pct = 0.40  # 40% typical crush
            recommendation = 'AVOID long options - high IV crush risk. Consider selling premium or waiting until after earnings.'
        elif days_to_earnings <= 14:
            risk_level = 'HIGH'
            expected_crush_pct = 0.30
            recommendation = 'CAUTION on long options - IV likely to decline. Consider shorter-dated options or wait for post-earnings.'
        elif days_to_earnings <= 30:
            risk_level = 'MODERATE'
            expected_crush_pct = 0.15
            recommendation = 'Monitor IV levels - some pre-earnings inflation present.'
        else:
            risk_level = 'LOW'
            expected_crush_pct = 0.05
            recommendation = 'Normal IV environment - earnings impact minimal.'
        
        return {
            'risk_level': risk_level,
            'days_to_earnings': days_to_earnings,
            'current_iv': atm_iv,
            'historical_vol': historical_vol,
            'iv_percentile': iv_ratio * 100,  # IV as % of historical vol
            'expected_crush_pct': expected_crush_pct * 100,
            'post_earnings_iv_estimate': atm_iv * (1 - expected_crush_pct),
            'recommendation': recommendation
        }
    
    def _calculate_volatility_surface(self, options_data: dict, current_price: float) -> dict:
        """
        Calculate volatility surface across strikes and expirations.
        Returns ATM IV, surface data points, and surface characteristics.
        """
        if not options_data:
            return {'atm_iv': 0, 'data_points': 0}
        
        surface_points = []
        
        # Extract IV for all options
        for option_type in ['calls', 'puts']:
            if option_type not in options_data:
                continue
            
            for option in options_data[option_type]:
                strike = option.get('strike', 0)
                iv = option.get('impliedVolatility', 0)
                dte = option.get('daysToExpiration', 0)
                
                if strike > 0 and iv > 0 and dte > 0:
                    moneyness = strike / current_price
                    surface_points.append({
                        'strike': strike,
                        'moneyness': moneyness,
                        'dte': dte,
                        'iv': iv,
                        'type': option_type
                    })
        
        if not surface_points:
            return {'atm_iv': 0, 'data_points': 0}
        
        # Find ATM IV (closest to current price)
        atm_options = sorted(surface_points, key=lambda x: abs(x['moneyness'] - 1.0))
        atm_iv = atm_options[0]['iv'] if atm_options else 0
        
        return {
            'atm_iv': atm_iv,
            'data_points': len(surface_points),
            'surface_data': surface_points[:100],  # Limit to 100 points for output
            'moneyness_range': [min(p['moneyness'] for p in surface_points), 
                               max(p['moneyness'] for p in surface_points)],
            'dte_range': [min(p['dte'] for p in surface_points), 
                         max(p['dte'] for p in surface_points)]
        }
    
    def _analyze_volatility_skew(self, options_data: dict, current_price: float) -> dict:
        """
        Analyze volatility skew (put/call IV difference).
        Negative skew = puts more expensive (fear premium).
        Positive skew = calls more expensive (greed premium).
        """
        if not options_data:
            return {'skew': 0, 'interpretation': 'No data'}
        
        # Get near-term options (30-45 DTE)
        calls_iv = []
        puts_iv = []
        
        for call in options_data.get('calls', []):
            dte = call.get('daysToExpiration', 0)
            strike = call.get('strike', 0)
            iv = call.get('impliedVolatility', 0)
            
            if 30 <= dte <= 45 and 0.95 <= strike/current_price <= 1.05 and iv > 0:
                calls_iv.append(iv)
        
        for put in options_data.get('puts', []):
            dte = put.get('daysToExpiration', 0)
            strike = put.get('strike', 0)
            iv = put.get('impliedVolatility', 0)
            
            if 30 <= dte <= 45 and 0.95 <= strike/current_price <= 1.05 and iv > 0:
                puts_iv.append(iv)
        
        if not calls_iv or not puts_iv:
            return {'skew': 0, 'interpretation': 'Insufficient data'}
        
        avg_call_iv = np.mean(calls_iv)
        avg_put_iv = np.mean(puts_iv)
        skew = avg_put_iv - avg_call_iv
        skew_pct = (skew / avg_call_iv) * 100 if avg_call_iv > 0 else 0
        
        # Interpret skew
        if skew_pct > 10:
            interpretation = 'STRONG PUT SKEW - High fear premium, market expects downside. Bullish contrarian signal.'
            market_sentiment = 'FEARFUL'
        elif skew_pct > 5:
            interpretation = 'MODERATE PUT SKEW - Normal protective put demand.'
            market_sentiment = 'CAUTIOUS'
        elif skew_pct < -5:
            interpretation = 'CALL SKEW - Unusual greed premium, market expects upside. Bearish contrarian signal.'
            market_sentiment = 'GREEDY'
        else:
            interpretation = 'NEUTRAL SKEW - Balanced put/call demand.'
            market_sentiment = 'NEUTRAL'
        
        return {
            'skew': skew,
            'skew_pct': skew_pct,
            'avg_call_iv': avg_call_iv,
            'avg_put_iv': avg_put_iv,
            'interpretation': interpretation,
            'market_sentiment': market_sentiment
        }
    
    def _detect_drift(self, realized_vol: float, implied_vol: float) -> dict:
        """
        Detect drift between realized and implied volatility.
        Large divergence indicates mispricing opportunities.
        """
        if realized_vol == 0 or implied_vol == 0:
            return {'drift': 0, 'signal': 'NEUTRAL', 'explanation': 'Insufficient data'}
        
        drift = implied_vol - realized_vol
        drift_pct = (drift / realized_vol) * 100
        
        # Interpret drift
        if drift_pct > 30:
            signal = 'SELL VOLATILITY'
            explanation = f'IV {drift_pct:.1f}% above realized vol - options overpriced. Consider selling premium (covered calls, cash-secured puts).'
            opportunity = 'PREMIUM_SELLING'
        elif drift_pct > 15:
            signal = 'MODERATE OVERPRICING'
            explanation = f'IV {drift_pct:.1f}% above realized vol - slight overpricing. Neutral to bearish options strategies favored.'
            opportunity = 'NEUTRAL_STRATEGIES'
        elif drift_pct < -15:
            signal = 'BUY VOLATILITY'
            explanation = f'IV {drift_pct:.1f}% below realized vol - options underpriced. Consider buying options (long calls/puts, straddles).'
            opportunity = 'PREMIUM_BUYING'
        else:
            signal = 'FAIR PRICING'
            explanation = f'IV {drift_pct:.1f}% vs realized vol - options fairly priced.'
            opportunity = 'DIRECTIONAL_TRADES'
        
        return {
            'realized_vol': realized_vol,
            'implied_vol': implied_vol,
            'drift': drift,
            'drift_pct': drift_pct,
            'signal': signal,
            'explanation': explanation,
            'opportunity': opportunity
        }
    
    def _analyze_term_structure(self, options_data: dict) -> dict:
        """
        Analyze volatility term structure across expirations.
        Upward sloping = normal (fear of future uncertainty).
        Downward sloping = inverted (near-term event risk).
        """
        if not options_data:
            return {'structure': 'UNKNOWN', 'data_points': []}
        
        # Group by expiration and calculate average IV
        expiration_ivs = {}
        
        for option_type in ['calls', 'puts']:
            if option_type not in options_data:
                continue
            
            for option in options_data[option_type]:
                dte = option.get('daysToExpiration', 0)
                iv = option.get('impliedVolatility', 0)
                
                if dte > 0 and iv > 0:
                    if dte not in expiration_ivs:
                        expiration_ivs[dte] = []
                    expiration_ivs[dte].append(iv)
        
        # Calculate average IV for each expiration
        term_structure_points = []
        for dte in sorted(expiration_ivs.keys()):
            avg_iv = np.mean(expiration_ivs[dte])
            term_structure_points.append({
                'dte': dte,
                'iv': avg_iv
            })
        
        if len(term_structure_points) < 2:
            return {'structure': 'INSUFFICIENT_DATA', 'data_points': term_structure_points}
        
        # Determine structure shape
        near_term_iv = term_structure_points[0]['iv']
        far_term_iv = term_structure_points[-1]['iv']
        slope = (far_term_iv - near_term_iv) / near_term_iv * 100
        
        if slope > 10:
            structure = 'UPWARD_SLOPING'
            interpretation = 'Normal term structure - market expects future uncertainty. No near-term events priced in.'
        elif slope < -10:
            structure = 'INVERTED'
            interpretation = 'Inverted term structure - near-term event risk (earnings, FDA approval, etc.). High IV crush risk.'
        else:
            structure = 'FLAT'
            interpretation = 'Flat term structure - consistent volatility expectations across time.'
        
        return {
            'structure': structure,
            'slope_pct': slope,
            'near_term_iv': near_term_iv,
            'far_term_iv': far_term_iv,
            'interpretation': interpretation,
            'data_points': term_structure_points
        }
    
    def _calculate_expected_move(self, options_data: dict, current_price: float) -> dict:
        """
        Calculate expected move based on ATM straddle prices.
        This is the market's implied probability distribution.
        """
        if not options_data:
            return {'expected_move_pct': 0, 'expected_range': [0, 0]}
        
        # Find nearest ATM straddle (call + put at same strike closest to current price)
        best_straddle = None
        min_distance = float('inf')
        
        calls = options_data.get('calls', [])
        puts = options_data.get('puts', [])
        
        for call in calls:
            call_strike = call.get('strike', 0)
            call_price = call.get('lastPrice', 0)
            dte = call.get('daysToExpiration', 0)
            
            if abs(call_strike - current_price) < min_distance and dte > 0:
                # Find matching put
                matching_put = next((p for p in puts if p.get('strike') == call_strike and p.get('daysToExpiration') == dte), None)
                
                if matching_put:
                    put_price = matching_put.get('lastPrice', 0)
                    straddle_price = call_price + put_price
                    
                    if straddle_price > 0:
                        best_straddle = {
                            'strike': call_strike,
                            'straddle_price': straddle_price,
                            'dte': dte
                        }
                        min_distance = abs(call_strike - current_price)
        
        if not best_straddle:
            return {'expected_move_pct': 0, 'expected_range': [current_price, current_price]}
        
        # Expected move = straddle price (1 standard deviation move)
        expected_move = best_straddle['straddle_price']
        expected_move_pct = (expected_move / current_price) * 100
        
        # Calculate expected range (1 standard deviation)
        upper_bound = current_price + expected_move
        lower_bound = current_price - expected_move
        
        return {
            'expected_move': expected_move,
            'expected_move_pct': expected_move_pct,
            'expected_range': [lower_bound, upper_bound],
            'dte': best_straddle['dte'],
            'interpretation': f'Market expects {expected_move_pct:.1f}% move (±${expected_move:.2f}) by expiration in {best_straddle["dte"]} days.'
        }
    
    def _analyze_greeks_surface(self, options_data: dict, current_price: float) -> dict:
        """
        Analyze Greeks across the options chain.
        Identifies zones of high gamma (pin risk) and vega (volatility sensitivity).
        """
        if not options_data:
            return {'max_gamma_strike': 0, 'max_vega_strike': 0}
        
        # Aggregate Greeks by strike
        strike_greeks = {}
        
        for option_type in ['calls', 'puts']:
            if option_type not in options_data:
                continue
            
            for option in options_data[option_type]:
                strike = option.get('strike', 0)
                delta = option.get('delta', 0)
                gamma = option.get('gamma', 0)
                vega = option.get('vega', 0)
                theta = option.get('theta', 0)
                
                if strike not in strike_greeks:
                    strike_greeks[strike] = {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
                
                strike_greeks[strike]['delta'] += abs(delta)
                strike_greeks[strike]['gamma'] += abs(gamma)
                strike_greeks[strike]['vega'] += abs(vega)
                strike_greeks[strike]['theta'] += abs(theta)
        
        if not strike_greeks:
            return {'max_gamma_strike': 0, 'max_vega_strike': 0}
        
        # Find max gamma and vega strikes
        max_gamma_strike = max(strike_greeks.items(), key=lambda x: x[1]['gamma'])[0]
        max_vega_strike = max(strike_greeks.items(), key=lambda x: x[1]['vega'])[0]
        
        return {
            'max_gamma_strike': max_gamma_strike,
            'max_vega_strike': max_vega_strike,
            'gamma_pin_risk': abs(max_gamma_strike - current_price) / current_price < 0.02,  # Within 2%
            'interpretation': f'Max gamma at ${max_gamma_strike:.2f} (pin risk), max vega at ${max_vega_strike:.2f} (vol sensitivity)'
        }
    
    def _get_atm_iv(self, options_data: dict) -> float:
        """Get ATM implied volatility (average of ATM call and put)"""
        calls = options_data.get('calls', [])
        puts = options_data.get('puts', [])
        
        if not calls or not puts:
            return 0.0
        
        # Find ATM options (first in chain, typically ATM)
        atm_call_iv = calls[0].get('impliedVolatility', 0) if calls else 0
        atm_put_iv = puts[0].get('impliedVolatility', 0) if puts else 0
        
        return (atm_call_iv + atm_put_iv) / 2 if (atm_call_iv + atm_put_iv) > 0 else 0
    
    def _empty_result(self) -> dict:
        """Return empty result structure when options data unavailable"""
        return {
            'iv_crush_monitor': {'risk_level': 'UNKNOWN', 'recommendation': 'Options data unavailable'},
            'volatility_surface': {'atm_iv': 0, 'data_points': 0},
            'skew_analysis': {'skew': 0, 'interpretation': 'No data'},
            'drift_detection': {'drift': 0, 'signal': 'NEUTRAL'},
            'term_structure': {'structure': 'UNKNOWN', 'data_points': []},
            'expected_move': {'expected_move_pct': 0},
            'greeks_surface': {'max_gamma_strike': 0, 'max_vega_strike': 0},
            'data_quality': {'total_contracts': 0, 'has_earnings_date': False, 'surface_complete': False}
        }
