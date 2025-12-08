"""
EXPERT REASONING ENGINE
Provides world-class analysis and deep insights for every trade recommendation.
This is what separates institutional-grade systems from basic calculators.
"""

import numpy as np
from typing import Dict, List
from legendary_trader_wisdom import LegendaryTraderWisdom


class ExpertReasoningEngine:
    """
    Generates expert-level reasoning for trade recommendations.
    Explains WHY, WHAT, WHICH, and HOW for every decision.
    """
    
    def generate_recommendation_reasoning(self, analysis: Dict) -> Dict:
        """
        Generate comprehensive reasoning for a trade recommendation.
        
        Returns:
            reasoning: Dict with detailed explanations
        """
        # Get legendary trader perspectives
        trader_wisdom = LegendaryTraderWisdom()
        trader_perspectives = trader_wisdom.get_trader_perspectives(analysis)
        consensus = trader_wisdom.get_consensus_action(trader_perspectives)
        
        reasoning = {
            'primary_thesis': self._generate_primary_thesis(analysis),
            'supporting_factors': self._identify_supporting_factors(analysis),
            'risk_factors': self._identify_risk_factors(analysis),
            'catalysts': self._identify_catalysts(analysis),
            'market_regime': self._assess_market_regime(analysis),
            'confidence_explanation': self._explain_confidence(analysis),
            'execution_strategy': self._suggest_execution_strategy(analysis),
            'alternative_scenarios': self._generate_scenarios(analysis),
            'legendary_trader_perspectives': trader_perspectives,
            'trader_consensus': consensus
        }
        
        return reasoning
    
    def _generate_primary_thesis(self, analysis: Dict) -> str:
        """Generate the main investment thesis"""
        rec = analysis.get('recommendation', 'HOLD')
        symbol = analysis.get('symbol', 'UNKNOWN')
        price = analysis.get('current_price', 0)
        
        # Extract key metrics
        fundamentals = analysis.get('fundamentals', {})
        technical = analysis.get('technical_analysis', {})
        sentiment = analysis.get('sentiment_score', 50)
        
        pe_ratio = fundamentals.get('pe_ratio', 0)
        roe = fundamentals.get('roe', 0)
        profit_margin = fundamentals.get('profit_margin', 0)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        
        rsi = technical.get('rsi', 50)
        trend_score = technical.get('trend_score', 50)
        momentum_score = technical.get('momentum_score', 50)
        
        # Generate thesis based on recommendation
        if rec in ['STRONG_BUY', 'BUY']:
            thesis = f"{symbol} at ${price:.2f} presents a compelling BUY opportunity. "
            
            # Fundamental strength
            if roe > 0.15 and profit_margin > 0.15:
                thesis += f"The company demonstrates strong profitability with {roe*100:.1f}% ROE and {profit_margin*100:.1f}% profit margins. "
            
            # Growth
            if revenue_growth > 0.1:
                thesis += f"Revenue growth of {revenue_growth*100:.1f}% indicates strong business momentum. "
            
            # Technical setup
            if rsi < 40:
                thesis += f"Technical indicators show oversold conditions (RSI: {rsi:.1f}), suggesting a potential reversal. "
            elif trend_score > 60:
                thesis += f"Strong uptrend confirmed by technical analysis (Trend Score: {trend_score:.1f}/100). "
            
            # Sentiment
            if sentiment > 60:
                thesis += f"Positive market sentiment ({sentiment:.1f}/100) provides additional tailwind. "
            
            thesis += "Risk/reward profile favors long positions."
            
        elif rec in ['SELL', 'STRONG_SELL']:
            thesis = f"{symbol} at ${price:.2f} shows concerning signals warranting caution or short positions. "
            
            # Fundamental weakness
            if roe < 0.05:
                thesis += f"Weak profitability (ROE: {roe*100:.1f}%) raises concerns about capital efficiency. "
            
            # Valuation
            if pe_ratio > 40:
                thesis += f"Elevated valuation (P/E: {pe_ratio:.1f}) suggests limited upside. "
            
            # Technical weakness
            if rsi > 70:
                thesis += f"Overbought conditions (RSI: {rsi:.1f}) increase correction risk. "
            elif trend_score < 40:
                thesis += f"Weakening technical trend (Score: {trend_score:.1f}/100) signals potential downside. "
            
            # Sentiment
            if sentiment < 40:
                thesis += f"Negative sentiment ({sentiment:.1f}/100) could pressure prices further. "
            
            thesis += "Risk/reward favors defensive positioning or shorts."
            
        else:  # HOLD
            thesis = f"{symbol} at ${price:.2f} is fairly valued with mixed signals. "
            thesis += "Current risk/reward does not strongly favor either direction. "
            thesis += "Wait for clearer setup or use for portfolio diversification."
        
        return thesis
    
    def _identify_supporting_factors(self, analysis: Dict) -> List[str]:
        """Identify factors supporting the recommendation"""
        factors = []
        
        fundamentals = analysis.get('fundamentals', {})
        technical = analysis.get('technical_analysis', {})
        rec = analysis.get('recommendation', 'HOLD')
        
        # For BUY recommendations
        if rec in ['STRONG_BUY', 'BUY']:
            if fundamentals.get('roe', 0) > 0.15:
                factors.append(f"Strong ROE of {fundamentals['roe']*100:.1f}% indicates efficient capital allocation")
            
            if fundamentals.get('revenue_growth', 0) > 0.1:
                factors.append(f"Revenue growth of {fundamentals['revenue_growth']*100:.1f}% shows expanding market share")
            
            if fundamentals.get('earnings_growth', 0) > 0.15:
                factors.append(f"Earnings growth of {fundamentals['earnings_growth']*100:.1f}% demonstrates operational leverage")
            
            if technical.get('rsi', 50) < 35:
                factors.append(f"RSI of {technical['rsi']:.1f} suggests oversold conditions with reversal potential")
            
            if technical.get('trend_score', 50) > 65:
                factors.append(f"Strong uptrend (Score: {technical['trend_score']:.1f}/100) confirmed by multiple timeframes")
            
            if technical.get('adx', 0) > 25:
                factors.append(f"ADX of {technical['adx']:.1f} confirms strong trending conditions")
        
        # For SELL recommendations
        elif rec in ['SELL', 'STRONG_SELL']:
            if fundamentals.get('pe_ratio', 0) > 40:
                factors.append(f"P/E ratio of {fundamentals['pe_ratio']:.1f} suggests overvaluation relative to earnings")
            
            if fundamentals.get('debt_to_equity', 0) > 2.0:
                factors.append(f"High debt-to-equity ratio of {fundamentals['debt_to_equity']:.2f} increases financial risk")
            
            if technical.get('rsi', 50) > 70:
                factors.append(f"RSI of {technical['rsi']:.1f} indicates overbought conditions prone to correction")
            
            if technical.get('trend_score', 50) < 35:
                factors.append(f"Weak trend (Score: {technical['trend_score']:.1f}/100) suggests downside momentum")
        
        if not factors:
            factors.append("Mixed signals across fundamental and technical indicators")
        
        return factors
    
    def _identify_risk_factors(self, analysis: Dict) -> List[str]:
        """Identify risks to the trade thesis"""
        risks = []
        
        fundamentals = analysis.get('fundamentals', {})
        technical = analysis.get('technical_analysis', {})
        stochastic = analysis.get('stochastic_analysis', {})
        
        # Volatility risk
        volatility = technical.get('current_volatility', 0)
        if volatility > 0.4:
            risks.append(f"High volatility ({volatility*100:.1f}%) increases position risk and potential slippage")
        
        # VaR risk
        var_95 = stochastic.get('var_95', 0)
        if abs(var_95) > 0.05:
            risks.append(f"95% VaR of {abs(var_95)*100:.1f}% indicates significant downside tail risk")
        
        # Liquidity risk
        if fundamentals.get('market_cap', 0) < 1e9:  # < $1B
            risks.append("Small market cap increases liquidity risk and volatility")
        
        # Sentiment risk
        sentiment = analysis.get('sentiment_score', 50)
        if sentiment < 35:
            risks.append(f"Negative sentiment ({sentiment:.1f}/100) could trigger further selling pressure")
        elif sentiment > 75:
            risks.append(f"Extremely positive sentiment ({sentiment:.1f}/100) may indicate crowded trade")
        
        # Technical risk
        if technical.get('adx', 0) < 20:
            risks.append(f"Low ADX ({technical['adx']:.1f}) suggests weak trend and potential whipsaw")
        
        if not risks:
            risks.append("Standard market risks apply; maintain proper position sizing")
        
        return risks
    
    def _identify_catalysts(self, analysis: Dict) -> List[str]:
        """Identify potential catalysts for the trade"""
        catalysts = []
        
        fundamentals = analysis.get('fundamentals', {})
        
        # Earnings growth catalyst
        if fundamentals.get('earnings_growth', 0) > 0.2:
            catalysts.append("Strong earnings growth could drive multiple expansion")
        
        # Valuation catalyst
        if fundamentals.get('pe_ratio', 0) < 15 and fundamentals.get('roe', 0) > 0.15:
            catalysts.append("Undervalued relative to quality; potential for re-rating")
        
        # Momentum catalyst
        if fundamentals.get('revenue_growth', 0) > 0.15:
            catalysts.append("Revenue acceleration could attract growth investors")
        
        # Dividend catalyst
        if fundamentals.get('dividend_yield', 0) > 0.03:
            catalysts.append(f"Dividend yield of {fundamentals['dividend_yield']*100:.2f}% provides income support")
        
        if not catalysts:
            catalysts.append("Monitor for earnings reports, guidance updates, or sector rotation")
        
        return catalysts
    
    def _assess_market_regime(self, analysis: Dict) -> str:
        """Assess current market regime and its implications"""
        technical = analysis.get('technical_analysis', {})
        
        volatility = technical.get('current_volatility', 0.2)
        trend_score = technical.get('trend_score', 50)
        adx = technical.get('adx', 20)
        
        if volatility > 0.4:
            regime = "HIGH VOLATILITY: "
            if trend_score > 60:
                regime += "Strong trend with high volatility suggests momentum continuation but with increased risk. "
                regime += "Use wider stops and smaller position sizes."
            else:
                regime += "High volatility without clear trend indicates choppy, range-bound conditions. "
                regime += "Consider mean-reversion strategies or wait for clarity."
        
        elif volatility < 0.15:
            regime = "LOW VOLATILITY: "
            if adx > 25:
                regime += "Low volatility with trending conditions ideal for trend-following strategies. "
                regime += "Can use tighter stops and larger positions."
            else:
                regime += "Low volatility and weak trends suggest consolidation phase. "
                regime += "Prepare for potential breakout but avoid premature entries."
        
        else:
            regime = "NORMAL VOLATILITY: "
            if trend_score > 60:
                regime += "Healthy trending market with normal volatility. Standard strategies apply."
            else:
                regime += "Mixed conditions require selective positioning and active risk management."
        
        return regime
    
    def _explain_confidence(self, analysis: Dict) -> str:
        """Explain why confidence is high or low"""
        confidence = analysis.get('confidence', 0)
        
        if confidence > 70:
            explanation = f"HIGH CONFIDENCE ({confidence:.1f}%): "
            explanation += "Multiple independent signals align across fundamental, technical, and sentiment analysis. "
            explanation += "Strong conviction in directional bias with favorable risk/reward."
        
        elif confidence > 50:
            explanation = f"MODERATE CONFIDENCE ({confidence:.1f}%): "
            explanation += "Primary indicators support the thesis but some conflicting signals exist. "
            explanation += "Reasonable conviction but maintain disciplined risk management."
        
        else:
            explanation = f"LOW CONFIDENCE ({confidence:.1f}%): "
            explanation += "Mixed or weak signals across analysis dimensions. "
            explanation += "High uncertainty requires smaller positions or waiting for better setup."
        
        return explanation
    
    def _suggest_execution_strategy(self, analysis: Dict) -> str:
        """Suggest optimal execution strategy"""
        rec = analysis.get('recommendation', 'HOLD')
        volatility = analysis.get('technical_analysis', {}).get('current_volatility', 0.2)
        position_size = analysis.get('position_sizing', {})
        
        if rec in ['STRONG_BUY', 'BUY']:
            if volatility > 0.4:
                strategy = "HIGH VOLATILITY ENTRY: Scale into position over 2-3 days using limit orders. "
                strategy += "Avoid market orders to prevent slippage. "
                strategy += f"Start with 50% of target size, add on pullbacks."
            else:
                strategy = "NORMAL ENTRY: Can use market orders for immediate execution. "
                strategy += f"Enter full position size with stop-loss at calculated level."
            
            strategy += f" Monitor for first 24-48 hours and adjust stops if needed."
        
        elif rec in ['SELL', 'STRONG_SELL']:
            strategy = "EXIT STRATEGY: If holding long, exit on strength to minimize slippage. "
            strategy += "For shorts, scale in gradually and use tight stops above resistance. "
            strategy += "Monitor closely for short squeezes in high short-interest names."
        
        else:  # HOLD
            strategy = "NO ACTION: Current setup does not warrant new positions. "
            strategy += "If already holding, maintain position with existing stops. "
            strategy += "Re-evaluate on next earnings report or significant technical break."
        
        return strategy
    
    def _generate_scenarios(self, analysis: Dict) -> Dict:
        """Generate bull/base/bear scenarios"""
        current_price = analysis.get('current_price', 0)
        target_price = analysis.get('target_price', current_price)
        stop_loss = analysis.get('stop_loss', current_price)
        
        # Calculate scenario targets
        bull_target = target_price * 1.2  # 20% above target
        base_target = target_price
        bear_target = stop_loss * 0.9  # 10% below stop
        
        bull_return = (bull_target / current_price - 1) * 100
        base_return = (base_target / current_price - 1) * 100
        bear_return = (bear_target / current_price - 1) * 100
        
        scenarios = {
            'bull_case': {
                'target': round(bull_target, 2),
                'return_pct': round(bull_return, 2),
                'probability': 25,
                'description': "All catalysts materialize, sentiment improves, technical breakout confirmed"
            },
            'base_case': {
                'target': round(base_target, 2),
                'return_pct': round(base_return, 2),
                'probability': 50,
                'description': "Primary thesis plays out as expected, normal market conditions"
            },
            'bear_case': {
                'target': round(bear_target, 2),
                'return_pct': round(bear_return, 2),
                'probability': 25,
                'description': "Thesis invalidated, stop-loss triggered, adverse market conditions"
            }
        }
        
        return scenarios
