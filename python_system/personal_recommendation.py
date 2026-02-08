"""
PERSONAL TRADING RECOMMENDATION ENGINE
========================================

"If I Were Trading With My Own Money" - Honest, Unfiltered Recommendation

This module synthesizes ALL analysis data (technical, fundamental, pattern recognition,
TAAPI, FinancialDatasets, Monte Carlo, GARCH, expert reasoning) into a single,
honest, plain-English recommendation as if the system were a human trader
risking their own capital.

Key principles:
- Capital preservation first (don't blow up)
- Asymmetric risk/reward (only take trades where upside >> downside)
- Conviction-based sizing (bet big when confident, small when uncertain)
- Time horizon awareness (day trade vs swing vs position)
- Emotional discipline (no FOMO, no revenge trading)
"""

import numpy as np
import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class PersonalRecommendationEngine:
    """
    Generates a personal, "skin in the game" trading recommendation.
    This is the final synthesis layer that combines all analysis into
    a recommendation as if the system were trading with its own money.
    """
    
    # Risk tolerance thresholds
    MAX_SINGLE_POSITION_PCT = 0.10  # Never risk more than 10% on one trade
    MAX_LOSS_PER_TRADE_PCT = 0.02   # Max 2% loss per trade
    MIN_RISK_REWARD = 1.5           # Minimum 1.5:1 risk/reward
    HIGH_CONVICTION_THRESHOLD = 70  # Score above this = high conviction
    LOW_CONVICTION_THRESHOLD = 40   # Score below this = stay away
    
    def __init__(self):
        logger.info("Personal Recommendation Engine initialized")
    
    def generate_recommendation(self, analysis: Dict) -> Dict:
        """
        Generate the personal "If I Were Trading" recommendation.
        
        Args:
            analysis: Complete analysis dict from run_perfect_analysis
            
        Returns:
            Dict with personal recommendation, conviction level, and reasoning
        """
        symbol = analysis.get('symbol', 'UNKNOWN')
        current_price = analysis.get('current_price', 0)
        signal = analysis.get('signal', 'HOLD')
        confidence = analysis.get('confidence', 50)
        bankroll = analysis.get('bankroll', 1000)
        
        # Gather all signal inputs
        signals = self._gather_all_signals(analysis)
        
        # Calculate composite conviction score
        conviction = self._calculate_conviction(signals)
        
        # Determine action
        action = self._determine_action(conviction, signals)
        
        # Calculate position sizing based on conviction
        position = self._calculate_position(
            conviction, current_price, bankroll, 
            analysis.get('stop_loss', current_price * 0.95),
            analysis.get('target_price', current_price * 1.05)
        )
        
        # Generate the narrative recommendation
        narrative = self._generate_narrative(
            symbol, current_price, action, conviction, signals, position, analysis
        )
        
        # Generate time horizon recommendation
        time_horizon = self._recommend_time_horizon(signals, analysis)
        
        # Generate risk warnings
        risk_warnings = self._generate_risk_warnings(signals, analysis)
        
        return {
            'action': action['action'],
            'conviction_score': conviction['total'],
            'conviction_level': conviction['level'],
            'position_size_shares': position['shares'],
            'position_size_dollars': position['dollars'],
            'position_pct_of_bankroll': position['pct_of_bankroll'],
            'risk_per_trade_dollars': position['risk_dollars'],
            'recommended_entry': action.get('entry_strategy', ''),
            'recommended_exit': action.get('exit_strategy', ''),
            'time_horizon': time_horizon,
            'narrative': narrative,
            'risk_warnings': risk_warnings,
            'signal_breakdown': signals,
            'conviction_breakdown': conviction,
            'timestamp': datetime.now().isoformat()
        }
    
    def _gather_all_signals(self, analysis: Dict) -> Dict:
        """Gather and normalize all signals from every data source."""
        signals = {}
        
        # 1. System signal (BUY/SELL/HOLD)
        signal_map = {'STRONG_BUY': 90, 'BUY': 70, 'HOLD': 50, 'SELL': 30, 'STRONG_SELL': 10}
        signals['system_signal'] = {
            'value': signal_map.get(analysis.get('signal', 'HOLD'), 50),
            'raw': analysis.get('signal', 'HOLD'),
            'weight': 0.15
        }
        
        # 2. Technical score
        tech = analysis.get('technical_analysis', {})
        signals['technical'] = {
            'value': tech.get('technical_score', 50),
            'momentum': tech.get('momentum_score', 50),
            'trend': tech.get('trend_score', 50),
            'volatility_score': tech.get('volatility_score', 50),
            'weight': 0.20
        }
        
        # 3. Fundamental score
        signals['fundamental'] = {
            'value': analysis.get('fundamental_score', 50),
            'weight': 0.15
        }
        
        # 4. Sentiment score
        signals['sentiment'] = {
            'value': analysis.get('sentiment_score', 50),
            'weight': 0.10
        }
        
        # 5. Pattern recognition
        pattern = analysis.get('pattern_recognition', {})
        pattern_pred = pattern.get('pattern_prediction', {})
        if pattern_pred:
            prob_up = pattern_pred.get('probability_up', 0.5)
            expected_return = pattern_pred.get('expected_return', 0)
            # Convert to 0-100 scale
            pattern_score = 50 + (prob_up - 0.5) * 60 + expected_return * 100
            pattern_score = max(0, min(100, pattern_score))
        else:
            pattern_score = 50
        signals['pattern_recognition'] = {
            'value': pattern_score,
            'confidence': pattern.get('confidence', 0),
            'matches_found': len(pattern.get('similar_patterns', [])),
            'weight': 0.15
        }
        
        # 6. Risk/Reward assessment
        risk = analysis.get('risk_assessment', {})
        rr_ratio = risk.get('risk_reward_ratio', 1.0)
        # Higher R:R = higher score
        rr_score = min(100, rr_ratio * 30)  # 3.3:1 R:R = 100
        signals['risk_reward'] = {
            'value': rr_score,
            'ratio': rr_ratio,
            'potential_gain_pct': risk.get('potential_gain_pct', 0),
            'potential_loss_pct': risk.get('potential_loss_pct', 0),
            'weight': 0.15
        }
        
        # 7. Monte Carlo / Stochastic
        stochastic = analysis.get('stochastic_analysis', {})
        mc_expected = stochastic.get('expected_return', 0)
        var_95 = stochastic.get('var_95', 0.1)
        # Score based on expected return and VaR
        mc_score = 50 + mc_expected * 200 - var_95 * 100
        mc_score = max(0, min(100, mc_score))
        signals['monte_carlo'] = {
            'value': mc_score,
            'expected_return': mc_expected,
            'var_95': var_95,
            'cvar_95': stochastic.get('cvar_95', 0),
            'weight': 0.10
        }
        
        # 8. TAAPI indicators (if available)
        taapi = analysis.get('taapi_indicators', {})
        if taapi and not taapi.get('error'):
            taapi_score = self._score_taapi_indicators(taapi)
            signals['taapi'] = {
                'value': taapi_score,
                'source': 'TAAPI.io',
                'weight': 0.05  # Supplementary confirmation
            }
        
        # 9. FinancialDatasets fundamentals (if available)
        fd_data = analysis.get('financialdatasets', {})
        if fd_data and not fd_data.get('error'):
            fd_score = self._score_financialdatasets(fd_data)
            signals['financialdatasets'] = {
                'value': fd_score,
                'source': 'FinancialDatasets.ai',
                'weight': 0.05  # Supplementary confirmation
            }
        
        return signals
    
    def _score_taapi_indicators(self, taapi: Dict) -> float:
        """Score TAAPI.io indicators on a 0-100 scale."""
        score = 50.0  # Neutral baseline
        
        # RSI
        rsi = taapi.get('rsi', {})
        if rsi:
            rsi_val = rsi.get('value', 50)
            if rsi_val < 30:
                score += 15  # Oversold = bullish
            elif rsi_val > 70:
                score -= 15  # Overbought = bearish
            elif 40 <= rsi_val <= 60:
                score += 5  # Neutral zone, slight positive
        
        # MACD
        macd = taapi.get('macd', {})
        if macd:
            hist = macd.get('valueMACDHist', 0)
            if hist and hist > 0:
                score += 10
            elif hist and hist < 0:
                score -= 10
        
        # Supertrend
        supertrend = taapi.get('supertrend', {})
        if supertrend:
            if supertrend.get('valueAdvice') == 'buy':
                score += 10
            elif supertrend.get('valueAdvice') == 'sell':
                score -= 10
        
        # ADX (trend strength)
        adx = taapi.get('adx', {})
        if adx:
            adx_val = adx.get('value', 25)
            if adx_val and adx_val > 25:
                score += 5  # Strong trend is tradeable
        
        return max(0, min(100, score))
    
    def _score_financialdatasets(self, fd_data: Dict) -> float:
        """Score FinancialDatasets.ai data on a 0-100 scale."""
        score = 50.0
        
        metrics = fd_data.get('financial_metrics', {})
        fm = {}
        if isinstance(metrics, dict):
            fm_raw = metrics.get('financial_metrics', metrics)
            # financial_metrics can be a list of quarterly metrics - take the most recent
            if isinstance(fm_raw, list) and len(fm_raw) > 0:
                fm = fm_raw[0] if isinstance(fm_raw[0], dict) else {}
            elif isinstance(fm_raw, dict):
                fm = fm_raw
        elif isinstance(metrics, list) and len(metrics) > 0:
            # Handle case where financial_metrics is a flat list
            fm = metrics[0] if isinstance(metrics[0], dict) else {}
        
        if fm:
            
            # ROE > 15% is good (FD API uses 'return_on_equity', fallback to 'roe')
            roe = fm.get('return_on_equity') or fm.get('roe')
            if roe and roe > 0.15:
                score += 10
            elif roe and roe < 0:
                score -= 15
            
            # Debt/Equity < 1 is healthy (FD API uses 'debt_to_equity_ratio', fallback to 'debt_to_equity')
            de = fm.get('debt_to_equity_ratio') or fm.get('debt_to_equity')
            if de and de < 0.5:
                score += 10
            elif de and de > 2.0:
                score -= 10
            
            # Current ratio > 1.5 is healthy
            cr = fm.get('current_ratio')
            if cr and cr > 1.5:
                score += 5
            elif cr and cr < 1.0:
                score -= 10
            
            # Revenue growth
            rev_growth = fm.get('revenue_growth')
            if rev_growth and rev_growth > 0.10:
                score += 10
            elif rev_growth and rev_growth < -0.05:
                score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_conviction(self, signals: Dict) -> Dict:
        """Calculate composite conviction score from all signals."""
        weighted_sum = 0.0
        total_weight = 0.0
        breakdown = {}
        
        for name, signal in signals.items():
            value = signal.get('value', 50)
            weight = signal.get('weight', 0.1)
            
            # Handle None/NaN
            try:
                value = float(value)
                if np.isnan(value) or np.isinf(value):
                    value = 50
            except (TypeError, ValueError):
                value = 50
            
            weighted_sum += value * weight
            total_weight += weight
            breakdown[name] = {
                'score': round(value, 1),
                'weight': weight,
                'contribution': round(value * weight, 1)
            }
        
        total = weighted_sum / total_weight if total_weight > 0 else 50
        total = max(0, min(100, total))
        
        # Determine conviction level
        if total >= 80:
            level = 'VERY_HIGH'
        elif total >= 65:
            level = 'HIGH'
        elif total >= 50:
            level = 'MODERATE'
        elif total >= 35:
            level = 'LOW'
        else:
            level = 'VERY_LOW'
        
        # Check for signal agreement (confluence)
        values = [s.get('value', 50) for s in signals.values()]
        bullish_count = sum(1 for v in values if v > 60)
        bearish_count = sum(1 for v in values if v < 40)
        total_signals = len(values)
        
        confluence = 'STRONG' if (bullish_count >= total_signals * 0.7 or bearish_count >= total_signals * 0.7) else \
                     'MODERATE' if (bullish_count >= total_signals * 0.5 or bearish_count >= total_signals * 0.5) else \
                     'MIXED'
        
        return {
            'total': round(total, 1),
            'level': level,
            'confluence': confluence,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': total_signals - bullish_count - bearish_count,
            'breakdown': breakdown
        }
    
    def _determine_action(self, conviction: Dict, signals: Dict) -> Dict:
        """Determine the specific trading action based on conviction."""
        score = conviction['total']
        level = conviction['level']
        confluence = conviction['confluence']
        
        # Risk/reward check
        rr = signals.get('risk_reward', {})
        rr_ratio = rr.get('ratio', 1.0)
        
        if score >= 75 and confluence == 'STRONG' and rr_ratio >= self.MIN_RISK_REWARD:
            return {
                'action': 'STRONG_BUY',
                'entry_strategy': 'Enter now at market. Strong confluence across all signals. Consider scaling in with 60% now, 40% on any pullback to support.',
                'exit_strategy': f'Set stop loss at system-calculated level. Trail stop by 1 ATR once in profit. Take partial profits (50%) at first target, let rest ride with trailing stop.'
            }
        elif score >= 65 and rr_ratio >= self.MIN_RISK_REWARD:
            return {
                'action': 'BUY',
                'entry_strategy': 'Enter on next pullback to support or use limit order slightly below current price. Scale in: 50% now, 50% on confirmation.',
                'exit_strategy': 'Set hard stop loss. Take profits at target price. If momentum fades before target, consider exiting at breakeven.'
            }
        elif score >= 55:
            return {
                'action': 'CAUTIOUS_BUY',
                'entry_strategy': 'Wait for a clear pullback to support before entering. Use a small position (half normal size). Require confirmation from volume.',
                'exit_strategy': 'Tight stop loss (0.75 ATR). Quick profit-taking at first resistance. Do not hold through earnings or major events.'
            }
        elif score >= 45:
            return {
                'action': 'HOLD_CASH',
                'entry_strategy': 'No entry. Signals are mixed. Wait for clearer setup. Capital preservation is paramount.',
                'exit_strategy': 'If already in position, tighten stops to breakeven. Consider reducing position size.'
            }
        elif score >= 35:
            return {
                'action': 'CAUTIOUS_SELL',
                'entry_strategy': 'If holding, consider reducing position by 50%. If not in, stay out. Bearish signals emerging.',
                'exit_strategy': 'Set tight trailing stop. Exit on any break below key support. Do not add to losing position.'
            }
        elif score >= 25:
            return {
                'action': 'SELL',
                'entry_strategy': 'Exit long positions. Consider short if experienced with shorting and risk management.',
                'exit_strategy': 'Close all long exposure. If shorting, use tight stop above recent resistance.'
            }
        else:
            return {
                'action': 'STRONG_SELL',
                'entry_strategy': 'Exit all positions immediately. Multiple bearish signals confirming. Capital preservation mode.',
                'exit_strategy': 'Market order to close. Do not wait for better prices. Protect capital.'
            }
    
    def _calculate_position(self, conviction: Dict, current_price: float, 
                           bankroll: float, stop_loss: float, target_price: float) -> Dict:
        """Calculate position size based on conviction and risk management."""
        score = conviction['total']
        
        # Base risk per trade: 1-2% of bankroll depending on conviction
        if score >= 75:
            risk_pct = 0.02  # 2% risk for high conviction
        elif score >= 60:
            risk_pct = 0.015  # 1.5% risk
        elif score >= 50:
            risk_pct = 0.01  # 1% risk
        else:
            risk_pct = 0.005  # 0.5% risk for low conviction
        
        risk_dollars = bankroll * risk_pct
        price_risk = abs(current_price - stop_loss)
        
        if price_risk > 0:
            shares = int(risk_dollars / price_risk)
        else:
            shares = 0
        
        # Cap at max position size
        max_position_value = bankroll * self.MAX_SINGLE_POSITION_PCT
        max_shares = int(max_position_value / current_price) if current_price > 0 else 0
        shares = min(shares, max_shares)
        
        # Minimum 1 share if signal is actionable
        if shares == 0 and score >= 55 and current_price <= bankroll:
            shares = 1
        
        dollars = shares * current_price
        pct_of_bankroll = (dollars / bankroll * 100) if bankroll > 0 else 0
        actual_risk = shares * price_risk
        
        return {
            'shares': shares,
            'dollars': round(dollars, 2),
            'pct_of_bankroll': round(pct_of_bankroll, 1),
            'risk_dollars': round(actual_risk, 2),
            'risk_pct': round(risk_pct * 100, 2),
            'max_loss_if_stopped': round(actual_risk, 2),
            'max_gain_if_target': round(shares * abs(target_price - current_price), 2)
        }
    
    def _recommend_time_horizon(self, signals: Dict, analysis: Dict) -> Dict:
        """Recommend optimal holding period based on analysis."""
        tech = signals.get('technical', {})
        trend = tech.get('trend', 50)
        momentum = tech.get('momentum', 50)
        volatility = tech.get('volatility_score', 50)
        
        adx = analysis.get('technical_analysis', {}).get('adx', 25)
        
        # Strong trend + high ADX = longer hold
        if trend > 70 and adx > 30:
            horizon = 'SWING_TRADE'
            days = '5-20 trading days'
            reasoning = 'Strong established trend with high ADX suggests momentum will persist. Swing trade to capture the move.'
        elif trend > 60 and momentum > 60:
            horizon = 'SHORT_SWING'
            days = '3-10 trading days'
            reasoning = 'Good momentum and trend alignment. Short swing trade to capture the immediate move.'
        elif volatility < 40:
            horizon = 'POSITION_TRADE'
            days = '20-60 trading days'
            reasoning = 'Low volatility environment. Position trade with wider stops for a larger move.'
        elif volatility > 70:
            horizon = 'DAY_TRADE'
            days = '1-3 trading days'
            reasoning = 'High volatility. Quick in-and-out to capture the move without overnight risk.'
        else:
            horizon = 'SHORT_SWING'
            days = '3-7 trading days'
            reasoning = 'Mixed signals suggest a short-term trade with defined risk.'
        
        return {
            'horizon': horizon,
            'suggested_days': days,
            'reasoning': reasoning
        }
    
    def _generate_risk_warnings(self, signals: Dict, analysis: Dict) -> List[str]:
        """Generate specific risk warnings based on the analysis."""
        warnings = []
        
        # Volatility warning
        vol = analysis.get('technical_analysis', {}).get('current_volatility', 0.2)
        if vol > 0.40:
            warnings.append(f"HIGH VOLATILITY ({vol*100:.0f}% annualized): Position size should be reduced. Wide intraday swings expected.")
        
        # VaR warning
        var_95 = analysis.get('stochastic_analysis', {}).get('var_95', 0)
        if var_95 > 0.10:
            warnings.append(f"ELEVATED VALUE-AT-RISK: 5% chance of losing >{var_95*100:.1f}% in 30 days based on Monte Carlo simulation.")
        
        # Low confidence warning
        confidence = analysis.get('confidence', 50)
        if confidence < 40:
            warnings.append(f"LOW SYSTEM CONFIDENCE ({confidence:.0f}%): Mixed signals. Consider waiting for clearer setup.")
        
        # Earnings proximity
        days_to_earnings = analysis.get('fundamentals', {}).get('days_to_earnings')
        if days_to_earnings and days_to_earnings < 14:
            warnings.append(f"EARNINGS IN {days_to_earnings} DAYS: IV crush risk for options. Stock could gap significantly. Consider waiting until after earnings.")
        
        # Pattern recognition warning
        pattern = analysis.get('pattern_recognition', {})
        if pattern.get('confidence', 0) < 30:
            warnings.append("LOW PATTERN CONFIDENCE: Historical pattern matching found few reliable matches. Current setup may be unique.")
        
        # Regime warning
        regime = pattern.get('regime_match', {})
        if regime and 'crash' in str(regime.get('regime_name', '')).lower():
            warnings.append(f"REGIME WARNING: Current conditions resemble {regime.get('regime_name', 'a historical crash')}. Extreme caution advised.")
        
        # Mixed signals warning
        bullish = sum(1 for s in signals.values() if s.get('value', 50) > 60)
        bearish = sum(1 for s in signals.values() if s.get('value', 50) < 40)
        if bullish > 0 and bearish > 0 and abs(bullish - bearish) <= 1:
            warnings.append("CONFLICTING SIGNALS: Technical and fundamental signals are diverging. Higher uncertainty in outcome.")
        
        # Risk/reward warning
        rr = analysis.get('risk_assessment', {}).get('risk_reward_ratio', 1.0)
        if rr < 1.0:
            warnings.append(f"POOR RISK/REWARD ({rr:.1f}:1): Potential loss exceeds potential gain. Not a favorable setup.")
        
        if not warnings:
            warnings.append("No major risk warnings. Standard trading risk applies.")
        
        return warnings
    
    def _generate_narrative(self, symbol: str, current_price: float, 
                           action: Dict, conviction: Dict, signals: Dict,
                           position: Dict, analysis: Dict) -> str:
        """
        Generate the plain-English "If I Were Trading" narrative.
        This is the heart of the recommendation - honest, unfiltered advice.
        """
        score = conviction['total']
        level = conviction['level']
        action_str = action['action']
        confluence = conviction['confluence']
        
        # Opening line based on conviction
        if score >= 75:
            opener = f"**I would be buying {symbol} here.** This is one of the stronger setups I'm seeing."
        elif score >= 65:
            opener = f"**I'd take a position in {symbol}, but with discipline.** The setup is favorable but not a slam dunk."
        elif score >= 55:
            opener = f"**I'd consider a small position in {symbol}, but I'd be cautious.** There's enough here to be interested, but not enough to be aggressive."
        elif score >= 45:
            opener = f"**I'd sit this one out on {symbol}.** The signals are too mixed to risk real money. Cash is a position too."
        elif score >= 35:
            opener = f"**I'd be looking to exit or reduce {symbol} here.** The weight of evidence is shifting bearish."
        else:
            opener = f"**I would not touch {symbol} right now.** Multiple signals are flashing danger. Protect your capital."
        
        # Build the narrative
        narrative = f"{opener}\n\n"
        
        # Technical picture
        tech = signals.get('technical', {})
        narrative += f"**The Technical Picture**: "
        trend = tech.get('trend', 50)
        momentum = tech.get('momentum', 50)
        if trend > 65 and momentum > 60:
            narrative += f"Trend and momentum are both bullish. The stock is moving in the right direction with conviction. "
        elif trend > 55:
            narrative += f"The trend is mildly positive but momentum is not overwhelming. "
        elif trend < 40:
            narrative += f"The trend is bearish. Price is below key moving averages. Fighting the trend is a losing game. "
        else:
            narrative += f"The trend is neutral/choppy. No clear directional bias from the technicals. "
        
        # RSI context
        rsi = analysis.get('technical_indicators', {}).get('rsi', 50)
        if rsi:
            if rsi < 30:
                narrative += f"RSI at {rsi:.0f} suggests the stock is oversold - potential bounce setup. "
            elif rsi > 70:
                narrative += f"RSI at {rsi:.0f} is overbought - could see a pullback before continuing. "
            else:
                narrative += f"RSI at {rsi:.0f} is in neutral territory. "
        narrative += "\n\n"
        
        # Pattern recognition insight
        pattern = analysis.get('pattern_recognition', {})
        pattern_pred = pattern.get('pattern_prediction', {})
        if pattern_pred and pattern_pred.get('sample_size', 0) > 3:
            prob_up = pattern_pred.get('probability_up', 0.5)
            expected_ret = pattern_pred.get('expected_return', 0)
            narrative += f"**Historical Pattern Match**: Looking at {pattern_pred.get('sample_size', 0)} similar historical patterns, "
            narrative += f"{prob_up*100:.0f}% of the time the stock went higher over the next 30 days, "
            narrative += f"with an average return of {expected_ret*100:.1f}%. "
            
            # Multi-horizon if available
            horizons = pattern_pred.get('horizons', {})
            if '5d' in horizons and '30d' in horizons:
                h5 = horizons['5d']
                h30 = horizons['30d']
                narrative += f"Short-term (5 days): {h5.get('probability_up', 0.5)*100:.0f}% up probability. "
                narrative += f"Medium-term (30 days): {h30.get('probability_up', 0.5)*100:.0f}% up probability. "
            narrative += "\n\n"
        
        # Regime context
        regime = pattern.get('regime_match', {})
        if regime and regime.get('regime_name') != 'unique_conditions':
            narrative += f"**Market Regime**: Current conditions most resemble the "
            narrative += f"**{regime.get('regime_name', '').replace('_', ' ').title()}** ({regime.get('period', '')}) - "
            narrative += f"{regime.get('description', '')}. "
            narrative += f"Back then, the outcome was: {regime.get('outcome', 'N/A')}. "
            narrative += f"Match confidence: {regime.get('match_score', 0):.0f}%.\n\n"
        
        # Risk/Reward
        rr = analysis.get('risk_assessment', {})
        rr_ratio = rr.get('risk_reward_ratio', 1.0)
        gain_pct = rr.get('potential_gain_pct', 0)
        loss_pct = rr.get('potential_loss_pct', 0)
        narrative += f"**Risk/Reward**: {rr_ratio:.1f}:1 ratio. "
        narrative += f"Potential gain: {gain_pct:.1f}%, potential loss: {loss_pct:.1f}%. "
        if rr_ratio >= 2.0:
            narrative += "This is a favorable risk/reward setup. "
        elif rr_ratio >= 1.5:
            narrative += "Acceptable risk/reward, but not exceptional. "
        else:
            narrative += "The risk/reward is not great. I'd want better odds before committing capital. "
        narrative += "\n\n"
        
        # Position sizing recommendation
        if position['shares'] > 0:
            narrative += f"**My Position**: I'd buy {position['shares']} shares (${position['dollars']:,.2f}, "
            narrative += f"{position['pct_of_bankroll']:.1f}% of bankroll). "
            narrative += f"Max risk on this trade: ${position['risk_dollars']:,.2f}. "
            narrative += f"If it hits target: +${position.get('max_gain_if_target', 0):,.2f}.\n\n"
        else:
            narrative += f"**My Position**: I would not take a position here. The setup doesn't justify risking capital.\n\n"
        
        # Entry/Exit strategy
        narrative += f"**Entry Strategy**: {action.get('entry_strategy', 'N/A')}\n\n"
        narrative += f"**Exit Strategy**: {action.get('exit_strategy', 'N/A')}\n\n"
        
        # Signal confluence
        narrative += f"**Signal Confluence**: {confluence}. "
        narrative += f"{conviction.get('bullish_signals', 0)} bullish, "
        narrative += f"{conviction.get('bearish_signals', 0)} bearish, "
        narrative += f"{conviction.get('neutral_signals', 0)} neutral signals.\n\n"
        
        # Bottom line
        narrative += "**Bottom Line**: "
        if score >= 70:
            narrative += f"This is a trade I'd take with confidence. The stars are aligning for {symbol}. "
            narrative += "But remember - no trade is guaranteed. Stick to the stop loss and let the trade work."
        elif score >= 55:
            narrative += f"{symbol} has potential, but I'd keep the position small and the stops tight. "
            narrative += "If it works, great. If not, the loss is manageable."
        elif score >= 45:
            narrative += f"I'd pass on {symbol} right now. There's no edge here worth risking money on. "
            narrative += "Better opportunities will come. Patience is a superpower in trading."
        else:
            narrative += f"Stay away from {symbol}. The signals are bearish and the risk is elevated. "
            narrative += "Protecting capital is always the right move when the odds aren't in your favor."
        
        return narrative
