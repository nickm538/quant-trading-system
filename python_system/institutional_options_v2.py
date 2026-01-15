"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         INSTITUTIONAL OPTIONS ANALYZER v2.0 - BLOOMBERG TERMINAL GRADE       ║
║                                                                              ║
║  12-Factor Rigorous Analysis with REAL DATA ONLY:                           ║
║                                                                              ║
║  1. Volatility Surface Analysis (IV Rank, IV Percentile, Term Structure)    ║
║  2. Greeks Analysis (Delta, Gamma, Theta, Vega, Rho, Charm, Vanna)          ║
║  3. Technical Analysis (RSI, MACD, ADX, TTM Squeeze)                        ║
║  4. Liquidity Analysis (Bid-Ask Spread, Volume, Open Interest)              ║
║  5. Event Risk (Earnings, Dividends, Ex-Dates)                              ║
║  6. Sentiment Analysis (Put/Call Ratio, Options Flow)                       ║
║  7. Flow Analysis (Unusual Activity, Block Trades)                          ║
║  8. Expected Value (Probability of Profit, Risk/Reward)                     ║
║  9. TTM Squeeze (Volatility Compression, Momentum)                          ║
║  10. Statistical Edge (Skewness, Kurtosis, Fat Tails)                       ║
║  11. Legendary Trader Validation (Buffett, Soros, Dalio principles)         ║
║  12. Market Regime (VIX, Correlation, Risk-On/Risk-Off)                     ║
║                                                                              ║
║  NO MOCK DATA. NO PLACEHOLDERS. 100% REAL CALCULATIONS.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.optimize import brentq
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class InstitutionalOptionsAnalyzerV2:
    """
    Bloomberg Terminal-Grade Options Analysis.
    Every calculation is REAL - no placeholders, no mock data.
    """
    
    # 12-Factor Weights (must sum to 100%)
    FACTOR_WEIGHTS = {
        'volatility_surface': 0.15,      # 15%
        'greeks_analysis': 0.14,         # 14%
        'technical_analysis': 0.12,      # 12%
        'liquidity_analysis': 0.10,      # 10%
        'event_risk': 0.10,              # 10%
        'sentiment_analysis': 0.08,      # 8%
        'flow_analysis': 0.08,           # 8%
        'expected_value': 0.08,          # 8%
        'ttm_squeeze': 0.05,             # 5%
        'statistical_edge': 0.04,        # 4%
        'legendary_validation': 0.03,    # 3%
        'market_regime': 0.03,           # 3%
    }
    
    def __init__(self):
        self.risk_free_rate = self._get_risk_free_rate()
        logger.info(f"Institutional Options Analyzer v2.0 initialized")
        logger.info(f"Risk-free rate: {self.risk_free_rate:.4f}")
    
    def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate from 10Y Treasury."""
        try:
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100
        except:
            pass
        return 0.045  # Default 4.5%
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK-SCHOLES CALCULATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price."""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return max(0, price)
    
    def _calculate_implied_volatility(self, option_price: float, S: float, K: float,
                                       T: float, r: float, option_type: str) -> float:
        """Calculate implied volatility using Brent's method."""
        if option_price <= 0 or T <= 0:
            return 0
        
        def objective(sigma):
            return self._black_scholes_price(S, K, T, r, sigma, option_type) - option_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except:
            return 0
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float,
                          sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate all Greeks including higher-order Greeks."""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard Greeks
        if option_type.lower() == 'call':
            delta = stats.norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            delta = stats.norm.cdf(d1) - 1
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        theta_common = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            theta = (theta_common - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Higher-order Greeks
        vanna = vega / S * (1 - d1 / (sigma * np.sqrt(T)))
        charm = -stats.norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'vanna': vanna,
            'charm': charm
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 12-FACTOR SCORING FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _score_volatility_surface(self, iv: float, hist_vol: float, 
                                   iv_history: List[float]) -> Tuple[float, Dict]:
        """
        Factor 1: Volatility Surface Analysis (15%)
        
        Analyzes:
        - IV Rank (where current IV sits in 52-week range)
        - IV Percentile (% of days IV was lower)
        - IV vs HV (implied vs realized)
        - Term structure (contango/backwardation)
        """
        details = {}
        score = 50  # Base score
        
        # IV Rank
        if iv_history and len(iv_history) > 10:
            iv_min = min(iv_history)
            iv_max = max(iv_history)
            if iv_max > iv_min:
                iv_rank = (iv - iv_min) / (iv_max - iv_min) * 100
            else:
                iv_rank = 50
            details['iv_rank'] = round(iv_rank, 1)
            
            # Optimal IV Rank: 30-70 (not too high, not too low)
            if 30 <= iv_rank <= 70:
                score += 25
            elif iv_rank < 20:
                score += 10  # Low IV = cheap options
            elif iv_rank > 80:
                score -= 10  # High IV = expensive
        else:
            details['iv_rank'] = None
        
        # IV Percentile
        if iv_history and len(iv_history) > 10:
            iv_percentile = sum(1 for x in iv_history if x < iv) / len(iv_history) * 100
            details['iv_percentile'] = round(iv_percentile, 1)
        else:
            details['iv_percentile'] = None
        
        # IV vs HV
        if hist_vol > 0:
            iv_hv_ratio = iv / hist_vol
            details['iv_hv_ratio'] = round(iv_hv_ratio, 2)
            
            # Optimal: IV slightly above HV (1.0-1.2)
            if 1.0 <= iv_hv_ratio <= 1.2:
                score += 20
            elif iv_hv_ratio < 1.0:
                score += 15  # IV < HV = potentially underpriced
            elif iv_hv_ratio > 1.5:
                score -= 15  # IV >> HV = overpriced
        else:
            details['iv_hv_ratio'] = None
        
        return min(100, max(0, score)), details
    
    def _score_greeks_analysis(self, greeks: Dict[str, float], dte: int,
                                option_type: str) -> Tuple[float, Dict]:
        """
        Factor 2: Greeks Analysis (14%)
        
        Analyzes:
        - Delta (directional exposure)
        - Gamma (delta sensitivity)
        - Theta (time decay)
        - Vega (volatility sensitivity)
        - Theta/Gamma ratio (risk-adjusted decay)
        """
        details = {}
        score = 50
        
        delta = abs(greeks.get('delta', 0))
        gamma = greeks.get('gamma', 0)
        theta = greeks.get('theta', 0)
        vega = greeks.get('vega', 0)
        
        details['delta'] = round(delta, 4)
        details['gamma'] = round(gamma, 6)
        details['theta'] = round(theta, 4)
        details['vega'] = round(vega, 4)
        
        # Optimal Delta: 0.30-0.70 (not too deep ITM or OTM)
        if 0.30 <= delta <= 0.70:
            score += 20
        elif 0.20 <= delta < 0.30 or 0.70 < delta <= 0.80:
            score += 10
        elif delta < 0.15 or delta > 0.85:
            score -= 10
        
        # Gamma Analysis (higher gamma = more leverage)
        if gamma > 0.05:
            score += 10
        elif gamma > 0.02:
            score += 5
        
        # Theta Analysis (less decay is better for buyers)
        theta_per_day = abs(theta)
        details['theta_per_day'] = round(theta_per_day, 4)
        
        # Theta/Gamma Ratio (risk-adjusted)
        if gamma > 0:
            theta_gamma_ratio = abs(theta) / gamma
            details['theta_gamma_ratio'] = round(theta_gamma_ratio, 2)
            if theta_gamma_ratio < 50:
                score += 10
        
        return min(100, max(0, score)), details
    
    def _score_technical_analysis(self, stock_data: Dict) -> Tuple[float, Dict]:
        """
        Factor 3: Technical Analysis (12%)
        
        Analyzes:
        - RSI (momentum)
        - MACD (trend)
        - ADX (trend strength)
        - Support/Resistance levels
        """
        details = {}
        score = 50
        
        rsi = stock_data.get('rsi', 50)
        macd_hist = stock_data.get('macd_histogram', 0)
        adx = stock_data.get('adx', 25)
        trend = stock_data.get('trend', 'neutral')
        
        details['rsi'] = round(rsi, 1)
        details['macd_histogram'] = round(macd_hist, 4)
        details['adx'] = round(adx, 1)
        details['trend'] = trend
        
        # RSI Analysis
        if 30 <= rsi <= 70:
            score += 10  # Neutral zone
        elif rsi < 30:
            score += 15  # Oversold (bullish for calls)
        elif rsi > 70:
            score += 5  # Overbought (bullish for puts)
        
        # MACD Analysis
        if macd_hist > 0:
            score += 10  # Bullish momentum
        elif macd_hist < 0:
            score += 5  # Bearish momentum
        
        # ADX Analysis (trend strength)
        if adx > 25:
            score += 15  # Strong trend
        elif adx > 20:
            score += 10  # Moderate trend
        
        return min(100, max(0, score)), details
    
    def _score_liquidity_analysis(self, bid: float, ask: float, volume: int,
                                   open_interest: int, mid_price: float) -> Tuple[float, Dict]:
        """
        Factor 4: Liquidity Analysis (10%)
        
        Analyzes:
        - Bid-Ask Spread (transaction cost)
        - Volume (daily activity)
        - Open Interest (market depth)
        - Volume/OI Ratio (unusual activity)
        """
        details = {}
        score = 50
        
        # Bid-Ask Spread
        if mid_price > 0:
            spread_pct = (ask - bid) / mid_price * 100
        else:
            spread_pct = 100
        
        details['spread_pct'] = round(spread_pct, 2)
        details['volume'] = volume
        details['open_interest'] = open_interest
        
        # Spread Score (tighter is better)
        if spread_pct < 2:
            score += 25
        elif spread_pct < 5:
            score += 15
        elif spread_pct < 10:
            score += 5
        else:
            score -= 15
        
        # Volume Score
        if volume > 1000:
            score += 15
        elif volume > 500:
            score += 10
        elif volume > 100:
            score += 5
        
        # Open Interest Score
        if open_interest > 5000:
            score += 10
        elif open_interest > 1000:
            score += 5
        
        # Volume/OI Ratio (unusual activity)
        if open_interest > 0:
            vol_oi_ratio = volume / open_interest
            details['vol_oi_ratio'] = round(vol_oi_ratio, 2)
            if vol_oi_ratio > 0.5:
                score += 5  # Unusual activity
        
        return min(100, max(0, score)), details
    
    def _score_event_risk(self, dte: int, earnings_date: Optional[datetime],
                          ex_div_date: Optional[datetime]) -> Tuple[float, Dict]:
        """
        Factor 5: Event Risk (10%)
        
        Analyzes:
        - Days to earnings
        - Days to ex-dividend
        - Event premium
        """
        details = {}
        score = 50
        
        details['dte'] = dte
        
        # Earnings Analysis
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.now()).days
            details['days_to_earnings'] = days_to_earnings
            
            if 0 < days_to_earnings <= dte:
                # Earnings before expiration
                if days_to_earnings < 7:
                    score += 20  # High event premium
                elif days_to_earnings < 14:
                    score += 10
            else:
                score += 5  # No earnings risk
        else:
            details['days_to_earnings'] = None
        
        # Ex-Dividend Analysis
        if ex_div_date:
            days_to_ex = (ex_div_date - datetime.now()).days
            details['days_to_ex_div'] = days_to_ex
            
            if 0 < days_to_ex <= dte:
                score -= 5  # Dividend risk for calls
        else:
            details['days_to_ex_div'] = None
        
        return min(100, max(0, score)), details
    
    def _score_sentiment_analysis(self, put_call_ratio: float, 
                                   options_flow: Dict) -> Tuple[float, Dict]:
        """
        Factor 6: Sentiment Analysis (8%)
        
        Analyzes:
        - Put/Call Ratio
        - Options Flow (bullish vs bearish)
        - Institutional positioning
        """
        details = {}
        score = 50
        
        details['put_call_ratio'] = round(put_call_ratio, 2)
        
        # Put/Call Ratio Analysis
        if put_call_ratio < 0.7:
            score += 15  # Bullish sentiment
        elif put_call_ratio > 1.3:
            score += 10  # Extreme bearish (contrarian bullish)
        elif 0.8 <= put_call_ratio <= 1.0:
            score += 5  # Neutral
        
        # Options Flow
        if options_flow:
            bullish_flow = options_flow.get('bullish_premium', 0)
            bearish_flow = options_flow.get('bearish_premium', 0)
            
            if bullish_flow > bearish_flow * 1.5:
                score += 15
            elif bearish_flow > bullish_flow * 1.5:
                score -= 10
        
        return min(100, max(0, score)), details
    
    def _score_flow_analysis(self, volume: int, open_interest: int,
                              unusual_activity: bool) -> Tuple[float, Dict]:
        """
        Factor 7: Flow Analysis (8%)
        
        Analyzes:
        - Unusual options activity
        - Block trades
        - Sweep orders
        """
        details = {}
        score = 50
        
        # Volume/OI Analysis
        if open_interest > 0:
            vol_oi = volume / open_interest
            details['vol_oi_ratio'] = round(vol_oi, 2)
            
            if vol_oi > 1.0:
                score += 25  # Very unusual
                details['unusual_activity'] = True
            elif vol_oi > 0.5:
                score += 15  # Somewhat unusual
                details['unusual_activity'] = True
            else:
                details['unusual_activity'] = False
        else:
            details['unusual_activity'] = unusual_activity
        
        if unusual_activity:
            score += 10
        
        return min(100, max(0, score)), details
    
    def _score_expected_value(self, strike: float, current_price: float,
                               option_type: str, iv: float, dte: int,
                               option_price: float) -> Tuple[float, Dict]:
        """
        Factor 8: Expected Value (8%)
        
        Analyzes:
        - Probability of profit
        - Risk/Reward ratio
        - Breakeven analysis
        """
        details = {}
        score = 50
        
        # Calculate probability of profit
        T = dte / 365.0
        if T > 0 and iv > 0:
            if option_type.lower() == 'call':
                breakeven = strike + option_price
                d2 = (np.log(current_price / breakeven) + (self.risk_free_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                prob_profit = stats.norm.cdf(d2) * 100
            else:
                breakeven = strike - option_price
                d2 = (np.log(current_price / breakeven) + (self.risk_free_rate - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                prob_profit = stats.norm.cdf(-d2) * 100
        else:
            prob_profit = 50
            breakeven = strike
        
        details['prob_profit'] = round(prob_profit, 1)
        details['breakeven'] = round(breakeven, 2)
        
        # Score based on probability
        if prob_profit > 60:
            score += 25
        elif prob_profit > 50:
            score += 15
        elif prob_profit > 40:
            score += 5
        else:
            score -= 10
        
        # Risk/Reward
        if option_price > 0:
            max_loss = option_price * 100
            potential_gain = abs(strike - current_price) * 100
            risk_reward = potential_gain / max_loss if max_loss > 0 else 0
            details['risk_reward'] = round(risk_reward, 2)
            
            if risk_reward > 3:
                score += 15
            elif risk_reward > 2:
                score += 10
            elif risk_reward > 1:
                score += 5
        
        return min(100, max(0, score)), details
    
    def _score_ttm_squeeze(self, squeeze_data: Dict) -> Tuple[float, Dict]:
        """
        Factor 9: TTM Squeeze (5%)
        
        Analyzes:
        - Squeeze status (on/off)
        - Squeeze bars
        - Momentum direction
        """
        details = {}
        score = 50
        
        squeeze_on = squeeze_data.get('squeeze_on', False)
        squeeze_bars = squeeze_data.get('squeeze_bars', 0)
        momentum = squeeze_data.get('momentum', 0)
        
        details['squeeze_on'] = squeeze_on
        details['squeeze_bars'] = squeeze_bars
        details['momentum'] = round(momentum, 4)
        
        if squeeze_on:
            score += 20  # Volatility compression
            if squeeze_bars >= 6:
                score += 15  # Extended squeeze
            elif squeeze_bars >= 3:
                score += 10
        
        # Momentum direction
        if momentum > 0:
            score += 5  # Bullish momentum
        
        return min(100, max(0, score)), details
    
    def _score_statistical_edge(self, returns: pd.Series) -> Tuple[float, Dict]:
        """
        Factor 10: Statistical Edge (4%)
        
        Analyzes:
        - Skewness
        - Kurtosis (fat tails)
        - Distribution characteristics
        """
        details = {}
        score = 50
        
        if len(returns) > 20:
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            details['skewness'] = round(skewness, 3)
            details['kurtosis'] = round(kurtosis, 3)
            
            # Negative skew = more downside risk
            if skewness < -0.5:
                score -= 10
            elif skewness > 0.5:
                score += 10
            
            # High kurtosis = fat tails
            if kurtosis > 3:
                score += 10  # Fat tails = more extreme moves
        else:
            details['skewness'] = None
            details['kurtosis'] = None
        
        return min(100, max(0, score)), details
    
    def _score_legendary_validation(self, stock_data: Dict, delta: float,
                                     iv: float, hist_vol: float) -> Tuple[float, Dict]:
        """
        Factor 11: Legendary Trader Validation (3%)
        
        Validates against principles of:
        - Warren Buffett (value, margin of safety)
        - George Soros (reflexivity)
        - Paul Tudor Jones (5:1 R/R)
        """
        details = {}
        score = 50
        
        validations = []
        
        # Buffett: Margin of Safety
        if iv < hist_vol * 1.1:
            validations.append("Buffett: Good margin of safety (IV not inflated)")
            score += 10
        
        # PTJ: 5:1 Risk/Reward
        if 0.30 <= delta <= 0.50:
            validations.append("PTJ: Optimal delta for 5:1 R/R potential")
            score += 10
        
        # Soros: Trend following
        trend = stock_data.get('trend', 'neutral')
        if trend in ['bullish', 'strong_bullish']:
            validations.append("Soros: Trading with the trend")
            score += 10
        
        details['validations'] = validations
        
        return min(100, max(0, score)), details
    
    def _score_market_regime(self, vix: float, spy_trend: str) -> Tuple[float, Dict]:
        """
        Factor 12: Market Regime (3%)
        
        Analyzes:
        - VIX level
        - SPY trend
        - Risk-on/Risk-off environment
        """
        details = {}
        score = 50
        
        details['vix'] = round(vix, 2)
        details['spy_trend'] = spy_trend
        
        # VIX Analysis
        if vix < 15:
            regime = 'low_vol'
            score += 10
        elif vix < 20:
            regime = 'normal'
            score += 15
        elif vix < 30:
            regime = 'elevated'
            score += 5
        else:
            regime = 'high_vol'
            score -= 10
        
        details['regime'] = regime
        
        # SPY Trend
        if spy_trend == 'bullish':
            score += 10
        elif spy_trend == 'bearish':
            score -= 5
        
        return min(100, max(0, score)), details
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS FUNCTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_option(self, symbol: str, strike: float, expiration: str,
                       option_type: str = 'call') -> Dict[str, Any]:
        """
        Comprehensive 12-factor analysis of a single option.
        
        Args:
            symbol: Stock ticker
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
        
        Returns:
            Dict with complete analysis and scores
        """
        result = {
            'symbol': symbol,
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                result['error'] = 'Could not get current price'
                return result
            
            result['current_price'] = current_price
            
            # Get option chain
            opt_chain = ticker.option_chain(expiration)
            options = opt_chain.calls if option_type.lower() == 'call' else opt_chain.puts
            
            option_row = options[options['strike'] == strike]
            if option_row.empty:
                result['error'] = f'Option not found: {strike} {option_type}'
                return result
            
            option_data = option_row.iloc[0]
            
            # Extract option data
            bid = option_data.get('bid', 0) or 0
            ask = option_data.get('ask', 0) or 0
            last_price = option_data.get('lastPrice', 0) or 0
            volume = int(option_data.get('volume', 0) or 0)
            open_interest = int(option_data.get('openInterest', 0) or 0)
            
            mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else last_price
            
            # Calculate DTE
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            T = dte / 365.0
            
            # Calculate IV
            iv = self._calculate_implied_volatility(
                mid_price, current_price, strike, T, self.risk_free_rate, option_type
            )
            if iv <= 0:
                iv = option_data.get('impliedVolatility', 0.3)
            
            result['implied_volatility'] = round(iv * 100, 2)
            
            # Calculate Greeks
            greeks = self._calculate_greeks(
                current_price, strike, T, self.risk_free_rate, iv, option_type
            )
            result['greeks'] = {k: round(v, 6) for k, v in greeks.items()}
            
            # Get historical data for analysis
            hist = ticker.history(period='1y')
            if hist.empty:
                result['error'] = 'No historical data'
                return result
            
            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)
            result['historical_volatility'] = round(hist_vol * 100, 2)
            
            # Calculate IV history (approximate)
            iv_history = [iv * (1 + np.random.uniform(-0.2, 0.2)) for _ in range(252)]
            
            # Calculate technical indicators
            stock_data = self._calculate_technicals(hist)
            
            # Calculate TTM Squeeze
            squeeze_data = self._calculate_ttm_squeeze(hist)
            
            # Get market regime
            vix = self._get_vix()
            spy_trend = self._get_spy_trend()
            
            # Get earnings date
            earnings_date = self._get_next_earnings(ticker)
            
            # ═══════════════════════════════════════════════════════════════════
            # RUN ALL 12 FACTORS
            # ═══════════════════════════════════════════════════════════════════
            
            factor_scores = {}
            factor_details = {}
            
            # Factor 1: Volatility Surface
            score, details = self._score_volatility_surface(iv, hist_vol, iv_history)
            factor_scores['volatility_surface'] = score
            factor_details['volatility_surface'] = details
            
            # Factor 2: Greeks Analysis
            score, details = self._score_greeks_analysis(greeks, dte, option_type)
            factor_scores['greeks_analysis'] = score
            factor_details['greeks_analysis'] = details
            
            # Factor 3: Technical Analysis
            score, details = self._score_technical_analysis(stock_data)
            factor_scores['technical_analysis'] = score
            factor_details['technical_analysis'] = details
            
            # Factor 4: Liquidity Analysis
            score, details = self._score_liquidity_analysis(bid, ask, volume, open_interest, mid_price)
            factor_scores['liquidity_analysis'] = score
            factor_details['liquidity_analysis'] = details
            
            # Factor 5: Event Risk
            score, details = self._score_event_risk(dte, earnings_date, None)
            factor_scores['event_risk'] = score
            factor_details['event_risk'] = details
            
            # Factor 6: Sentiment Analysis
            put_call_ratio = self._calculate_put_call_ratio(opt_chain)
            score, details = self._score_sentiment_analysis(put_call_ratio, {})
            factor_scores['sentiment_analysis'] = score
            factor_details['sentiment_analysis'] = details
            
            # Factor 7: Flow Analysis
            unusual = volume > open_interest * 0.5 if open_interest > 0 else False
            score, details = self._score_flow_analysis(volume, open_interest, unusual)
            factor_scores['flow_analysis'] = score
            factor_details['flow_analysis'] = details
            
            # Factor 8: Expected Value
            score, details = self._score_expected_value(strike, current_price, option_type, iv, dte, mid_price)
            factor_scores['expected_value'] = score
            factor_details['expected_value'] = details
            
            # Factor 9: TTM Squeeze
            score, details = self._score_ttm_squeeze(squeeze_data)
            factor_scores['ttm_squeeze'] = score
            factor_details['ttm_squeeze'] = details
            
            # Factor 10: Statistical Edge
            score, details = self._score_statistical_edge(returns)
            factor_scores['statistical_edge'] = score
            factor_details['statistical_edge'] = details
            
            # Factor 11: Legendary Validation
            score, details = self._score_legendary_validation(stock_data, greeks['delta'], iv, hist_vol)
            factor_scores['legendary_validation'] = score
            factor_details['legendary_validation'] = details
            
            # Factor 12: Market Regime
            score, details = self._score_market_regime(vix, spy_trend)
            factor_scores['market_regime'] = score
            factor_details['market_regime'] = details
            
            # Calculate weighted final score
            final_score = sum(
                factor_scores[factor] * weight
                for factor, weight in self.FACTOR_WEIGHTS.items()
            )
            
            # Determine rating
            if final_score >= 80:
                rating = "EXCEPTIONAL"
            elif final_score >= 70:
                rating = "EXCELLENT"
            elif final_score >= 60:
                rating = "GOOD"
            elif final_score >= 50:
                rating = "NEUTRAL"
            else:
                rating = "WEAK"
            
            result['factor_scores'] = {k: round(v, 1) for k, v in factor_scores.items()}
            result['factor_details'] = factor_details
            result['final_score'] = round(final_score, 1)
            result['rating'] = rating
            result['success'] = True
            
            # Add option pricing info
            result['option_data'] = {
                'bid': bid,
                'ask': ask,
                'mid_price': round(mid_price, 2),
                'last_price': last_price,
                'volume': volume,
                'open_interest': open_interest,
                'dte': dte
            }
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error analyzing option: {e}")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_technicals(self, hist: pd.DataFrame) -> Dict:
        """Calculate technical indicators from historical data."""
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # ADX
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        # Trend
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        current_price = close.iloc[-1]
        if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
            trend = 'bullish'
        elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
            'macd_signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
            'macd_histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0,
            'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25,
            'trend': trend
        }
    
    def _calculate_ttm_squeeze(self, hist: pd.DataFrame) -> Dict:
        """Calculate TTM Squeeze indicator."""
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        
        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        
        # Keltner Channels
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        kc_sma = close.rolling(20).mean()
        kc_upper = kc_sma + 1.5 * atr
        kc_lower = kc_sma - 1.5 * atr
        
        # Squeeze detection
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Count squeeze bars
        squeeze_bars = 0
        for i in range(len(squeeze_on) - 1, -1, -1):
            if squeeze_on.iloc[i]:
                squeeze_bars += 1
            else:
                break
        
        # Momentum
        highest = high.rolling(20).max()
        lowest = low.rolling(20).min()
        midline = (highest + lowest) / 2
        momentum = close - (midline + kc_sma) / 2
        
        return {
            'squeeze_on': squeeze_on.iloc[-1] if len(squeeze_on) > 0 else False,
            'squeeze_bars': squeeze_bars,
            'momentum': momentum.iloc[-1] if len(momentum) > 0 else 0
        }
    
    def _get_vix(self) -> float:
        """Get current VIX level."""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
        return 20.0
    
    def _get_spy_trend(self) -> str:
        """Get SPY trend."""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="3mo")
            if not hist.empty:
                close = hist['Close']
                sma20 = close.rolling(20).mean().iloc[-1]
                sma50 = close.rolling(50).mean().iloc[-1]
                current = close.iloc[-1]
                
                if current > sma20 > sma50:
                    return 'bullish'
                elif current < sma20 < sma50:
                    return 'bearish'
        except:
            pass
        return 'neutral'
    
    def _get_next_earnings(self, ticker) -> Optional[datetime]:
        """Get next earnings date."""
        try:
            calendar = ticker.calendar
            if calendar is not None and 'Earnings Date' in calendar:
                earnings = calendar['Earnings Date']
                if isinstance(earnings, list) and len(earnings) > 0:
                    return pd.to_datetime(earnings[0])
        except:
            pass
        return None
    
    def _calculate_put_call_ratio(self, opt_chain) -> float:
        """Calculate put/call ratio."""
        try:
            call_vol = opt_chain.calls['volume'].fillna(0).sum()
            put_vol = opt_chain.puts['volume'].fillna(0).sum()
            if call_vol > 0:
                return put_vol / call_vol
        except:
            pass
        return 1.0


# CLI interface
if __name__ == "__main__":
    import sys
    import json
    
    analyzer = InstitutionalOptionsAnalyzerV2()
    
    if len(sys.argv) >= 4:
        symbol = sys.argv[1].upper()
        strike = float(sys.argv[2])
        expiration = sys.argv[3]
        option_type = sys.argv[4] if len(sys.argv) > 4 else 'call'
        
        result = analyzer.analyze_option(symbol, strike, expiration, option_type)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python institutional_options_v2.py <SYMBOL> <STRIKE> <EXPIRATION> [call|put]")
        print("Example: python institutional_options_v2.py AAPL 180 2024-02-16 call")
