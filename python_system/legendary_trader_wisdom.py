"""
LEGENDARY TRADER WISDOM MODULE
===============================

Applies the investment philosophies of history's greatest traders to current market conditions.
This is what separates institutional-grade systems from basic calculators.

Traders modeled:
- Warren Buffett: Value investing, margin of safety, business quality
- George Soros: Reflexivity, regime changes, macro trends  
- Stanley Druckenmiller: Risk management, position sizing, cutting losses
- Peter Lynch: Growth at reasonable price, understand what you own
- Paul Tudor Jones: Risk/reward, never average down on losers
- Jesse Livermore: Tape reading, trend following, pyramiding winners
"""

from typing import Dict, List


class LegendaryTraderWisdom:
    """
    Applies legendary trader philosophies to current market setups.
    Provides "What would [Trader] do?" insights.
    """
    
    def get_trader_perspectives(self, analysis: Dict) -> Dict[str, str]:
        """
        Get perspectives from legendary traders on this setup.
        
        Returns:
            Dict mapping trader name to their likely action/advice
        """
        perspectives = {
            'warren_buffett': self._buffett_perspective(analysis),
            'george_soros': self._soros_perspective(analysis),
            'stanley_druckenmiller': self._druckenmiller_perspective(analysis),
            'peter_lynch': self._lynch_perspective(analysis),
            'paul_tudor_jones': self._paul_tudor_jones_perspective(analysis),
            'jesse_livermore': self._livermore_perspective(analysis)
        }
        
        return perspectives
    
    def _buffett_perspective(self, analysis: Dict) -> str:
        """Warren Buffett: Value investing, business quality, margin of safety"""
        fundamentals = analysis.get('fundamentals', {})
        price = analysis.get('current_price', 0)
        
        roe = fundamentals.get('roe', 0)
        profit_margin = fundamentals.get('profit_margin', 0)
        pe_ratio = fundamentals.get('pe_ratio', 0)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        
        # Buffett's criteria
        has_moat = roe > 0.15 and profit_margin > 0.15
        reasonable_price = pe_ratio < 25 and pe_ratio > 0
        low_debt = debt_to_equity < 0.5
        growing = revenue_growth > 0.05
        
        if has_moat and reasonable_price and low_debt:
            advice = f"**BUFFETT WOULD BUY**: This is a quality business with strong economics (ROE: {roe*100:.1f}%, Margins: {profit_margin*100:.1f}%). "
            advice += f"Trading at reasonable valuation (P/E: {pe_ratio:.1f}). "
            if growing:
                advice += f"Growing revenue ({revenue_growth*100:.1f}% TTM YoY) adds to the moat. "
            advice += "**Fundamentals are trailing twelve months (TTM)**, not quarterly - this is the full-year picture. "
            advice += "**'Price is what you pay, value is what you get.'** This offers value."
        
        elif has_moat and not reasonable_price:
            advice = f"**BUFFETT WOULD WAIT**: Excellent business (ROE: {roe*100:.1f}%) but overpriced (P/E: {pe_ratio:.1f}). "
            advice += "**'Be fearful when others are greedy.'** Wait for a better entry or market correction."
        
        elif not has_moat:
            advice = f"**BUFFETT WOULD PASS**: Weak economic moat (ROE: {roe*100:.1f}%, Margins: {profit_margin*100:.1f}%). "
            advice += "**'Only buy something you'd be happy to hold if the market shut down for 10 years.'** This doesn't qualify."
        
        else:
            advice = "**BUFFETT WOULD PASS**: Doesn't meet quality + value criteria. Look for better opportunities."
        
        return advice
    
    def _soros_perspective(self, analysis: Dict) -> str:
        """George Soros: Reflexivity, regime changes, macro trends"""
        technical = analysis.get('technical_analysis', {})
        sentiment = analysis.get('sentiment_score', 50)
        trend_score = technical.get('trend_score', 50)
        volatility = technical.get('current_volatility', 0.2)
        
        # Soros looks for regime changes and reflexive feedback loops
        strong_trend = trend_score > 65
        high_sentiment = sentiment > 65
        low_sentiment = sentiment < 35
        high_vol = volatility > 0.4
        
        if strong_trend and high_sentiment:
            advice = "**SOROS WOULD BE CAUTIOUS**: Strong trend with euphoric sentiment suggests reflexive feedback loop nearing exhaustion. "
            advice += f"**Market Regime**: High volatility ({volatility*100:.1f}%) + euphoria = late-stage bull phase. "
            advice += f"**'Markets are constantly in a state of uncertainty and flux.'** Trend Score: {trend_score:.1f}, Sentiment: {sentiment:.1f}. "
            advice += "Consider taking profits or waiting for regime change."
        
        elif strong_trend and low_sentiment:
            advice = "**SOROS WOULD BUY AGGRESSIVELY**: Strong trend with negative sentiment = early stage of new regime. "
            advice += f"**Market Regime**: Volatility {volatility*100:.1f}% with negative sentiment = fear-driven opportunity. "
            advice += f"**'When I see a bubble forming I rush in to buy.'** Trend: {trend_score:.1f}, Sentiment: {sentiment:.1f}. "
            advice += "This is the setup for reflexive feedback loop acceleration."    
        elif high_vol and low_sentiment:
            advice = "**SOROS WOULD LOOK FOR TURNING POINT**: High volatility + fear = potential regime change. "
            advice += f"Vol: {volatility*100:.1f}%, Sentiment: {sentiment:.1f}. "
            advice += "**'When I see a bubble forming, I rush in to buy.'** Watch for reversal signals."
        
        else:
            advice = f"**SOROS WOULD WAIT**: No clear regime change or reflexive opportunity. Trend: {trend_score:.1f}, Sentiment: {sentiment:.1f}. "
            advice += "**'It's not whether you're right or wrong, but how much you make when you're right.'** Wait for high-conviction setup."
        
        return advice
    
    def _druckenmiller_perspective(self, analysis: Dict) -> str:
        """Stanley Druckenmiller: Risk management, position sizing, cutting losses"""
        risk = analysis.get('risk_assessment', {})
        confidence = analysis.get('confidence', 0)
        risk_reward = risk.get('risk_reward_ratio', 1.0)
        
        # Druckenmiller's criteria
        high_conviction = confidence > 60
        asymmetric_rr = risk_reward > 3.0
        
        if high_conviction and asymmetric_rr:
            advice = f"**DRUCKENMILLER WOULD SIZE UP**: High conviction ({confidence:.1f}%) + excellent R/R ({risk_reward:.1f}:1). "
            advice += "**'The way to build long-term returns is through preservation of capital and home runs.'** "
            advice += f"This is a home run setup - consider 2-3x normal position size."
        
        elif high_conviction and not asymmetric_rr:
            advice = f"**DRUCKENMILLER WOULD TAKE NORMAL SIZE**: Good conviction ({confidence:.1f}%) but mediocre R/R ({risk_reward:.1f}:1). "
            advice += "**'If you're right 60% of the time, you can make a fortune.'** Standard position, tight stop."
        
        elif not high_conviction:
            advice = f"**DRUCKENMILLER WOULD PASS OR SCOUT**: Low conviction ({confidence:.1f}%). "
            advice += "**'Never invest in any idea you can't illustrate with a crayon.'** "
            advice += "If you do enter, use 1/4 to 1/2 normal size as a 'scout' position."
        
        else:
            advice = "**DRUCKENMILLER WOULD WAIT**: No edge. **'The first thing I heard when I got in the business was bulls make money, bears make money, and pigs get slaughtered.'**"
        
        return advice
    
    def _lynch_perspective(self, analysis: Dict) -> str:
        """Peter Lynch: Growth at reasonable price, understand what you own"""
        fundamentals = analysis.get('fundamentals', {})
        
        pe_ratio = fundamentals.get('pe_ratio', 0)
        earnings_growth = fundamentals.get('earnings_growth', 0)
        revenue_growth = fundamentals.get('revenue_growth', 0)
        peg_ratio = fundamentals.get('peg_ratio', 999)
        
        # Lynch's PEG ratio (P/E divided by growth rate)
        # PEG < 1 = undervalued, PEG 1-2 = fair, PEG > 2 = overvalued
        
        if peg_ratio < 1.0 and earnings_growth > 0.15:
            advice = f"**LYNCH WOULD BUY**: Classic GARP setup! PEG ratio of {peg_ratio:.2f} with {earnings_growth*100:.1f}% earnings growth. "
            advice += "**'Find a stock that's undervalued and hold it until it's overvalued.'** This is undervalued."
        
        elif peg_ratio < 1.5 and revenue_growth > 0.2:
            advice = f"**LYNCH WOULD BUY**: Reasonable PEG ({peg_ratio:.2f}) with strong revenue growth ({revenue_growth*100:.1f}%). "
            advice += "**'Go for a business that any idiot can run - because sooner or later, any idiot probably is going to run it.'** Strong growth speaks for itself."
        
        elif peg_ratio > 2.5:
            advice = f"**LYNCH WOULD SELL/AVOID**: Overvalued with PEG of {peg_ratio:.2f}. P/E: {pe_ratio:.1f}, Growth: {earnings_growth*100:.1f}%. "
            advice += "**'Never invest in any idea you can't illustrate with a crayon.'** This math doesn't work."
        
        else:
            advice = f"**LYNCH WOULD HOLD/WATCH**: Fair valuation (PEG: {peg_ratio:.2f}). "
            advice += "**'Know what you own, and know why you own it.'** Make sure you understand the business before buying."
        
        return advice
    
    def _paul_tudor_jones_perspective(self, analysis: Dict) -> str:
        """Paul Tudor Jones: Risk management, 5:1 reward/risk minimum"""
        risk = analysis.get('risk_assessment', {})
        rec = analysis.get('recommendation', 'HOLD')
        
        # DEBUG: Print what we're getting
        risk_reward = risk.get('risk_reward_ratio', 1.0)
        potential_gain = risk.get('potential_gain_pct', 0)
        potential_loss = risk.get('potential_loss_pct', 0)
        
        # PTJ's 5:1 rule
        meets_ptj_rule = risk_reward >= 5.0
        
        if meets_ptj_rule:
            advice = f"**JONES WOULD BUY**: Exceptional R/R of {risk_reward:.1f}:1 (Risk: {abs(potential_loss)*100:.1f}%, Reward: {abs(potential_gain)*100:.1f}%). "
            advice += "**Intraday Setup**: Targets based on daily ATR from recent price action. "
            advice += "**'Don't focus on making money; focus on protecting what you have.'** This setup protects capital with asymmetric upside."
        
        elif risk_reward >= 3.0:
            advice = f"**JONES WOULD CONSIDER**: Good R/R of {risk_reward:.1f}:1 (Risk: {abs(potential_loss)*100:.1f}%, Reward: {abs(potential_gain)*100:.1f}%). "
            advice += "**Intraday timeframe**: Using daily ATR for stops, expect moves within 1-3 trading sessions. "
            advice += "**'The secret to being successful is to find a way to lose money as slowly as possible.'** Acceptable risk/reward."
        
        elif risk_reward < 2.0:
            advice = f"**JONES WOULD PASS**: Poor R/R of {risk_reward:.1f}:1 (Risk: {abs(potential_loss)*100:.1f}%, Reward: {abs(potential_gain)*100:.1f}%). "
            advice += "**Timeframe context**: Even with intraday signals, this R/R doesn't justify the risk. "
            advice += "**'Losers average losers.'** Never take a trade where you risk more than you can make."
        
        else:
            advice = f"**JONES WOULD WAIT**: Mediocre R/R ({risk_reward:.1f}:1). "
            advice += "**'Every day I assume every position I have is wrong.'** Wait for better setup."
        
        return advice
    
    def _livermore_perspective(self, analysis: Dict) -> str:
        """Jesse Livermore: Tape reading, trend following, pyramiding winners"""
        technical = analysis.get('technical_analysis', {})
        rec = analysis.get('recommendation', 'HOLD')
        
        trend_score = technical.get('trend_score', 50)
        momentum_score = technical.get('momentum_score', 50)
        adx = technical.get('adx', 20)
        
        # Livermore's trend following
        strong_trend = trend_score > 65 and adx > 25
        weak_trend = trend_score < 45 or adx < 20
        
        if strong_trend and rec in ['BUY', 'STRONG_BUY']:
            advice = f"**LIVERMORE WOULD BUY AND PYRAMID**: Strong trend (Score: {trend_score:.1f}, ADX: {adx:.1f}). "
            advice += "**Intraday tape reading**: 5-minute bars during market hours show clear directional bias. "
            advice += "**'It is literally true that millions come easier to a trader after he knows how to trade than hundreds did in the days of his ignorance.'** "
            advice += "Enter now, add on strength (pyramid winners, never losers)."
        
        elif strong_trend and rec in ['SELL', 'STRONG_SELL']:
            advice = f"**LIVERMORE WOULD SHORT**: Strong downtrend (Score: {trend_score:.1f}). "
            advice += "**'The big money is made in the big swing.'** Ride this trend down."
        
        elif weak_trend:
            advice = f"**LIVERMORE WOULD WAIT**: Weak/choppy trend (Score: {trend_score:.1f}, ADX: {adx:.1f}). "
            advice += "**Intraday context**: 5-minute bars show no clear directional conviction during recent sessions. "
            advice += "**'There is a time for all things, but I didn't know it. And that is precisely what beats so many men in Wall Street.'** "
            advice += "Wait for clear trend before entering."
        
        else:
            advice = f"**LIVERMORE WOULD WATCH**: Developing setup (Trend: {trend_score:.1f}, Momentum: {momentum_score:.1f}). "
            advice += "**'It never was my thinking that made the big money for me. It always was my sitting.'** Be patient."
        
        return advice
    
    def get_consensus_action(self, perspectives: Dict[str, str]) -> str:
        """
        Determine consensus action from all legendary traders.
        
        Returns:
            Consensus recommendation with explanation
        """
        # Count votes
        buy_votes = 0
        sell_votes = 0
        wait_votes = 0
        
        for trader, advice in perspectives.items():
            if 'WOULD BUY' in advice or 'BUY AGGRESSIVELY' in advice or 'SIZE UP' in advice:
                buy_votes += 1
            elif 'WOULD SELL' in advice or 'WOULD SHORT' in advice:
                sell_votes += 1
            elif 'WOULD WAIT' in advice or 'WOULD PASS' in advice or 'BE CAUTIOUS' in advice:
                wait_votes += 1
        
        total_votes = buy_votes + sell_votes + wait_votes
        
        if buy_votes >= 4:
            consensus = f"**STRONG CONSENSUS BUY**: {buy_votes}/{total_votes} legendary traders would buy this setup. "
            consensus += "Multiple independent philosophies align - this is a high-quality opportunity."
        elif buy_votes >= 3:
            consensus = f"**MODERATE CONSENSUS BUY**: {buy_votes}/{total_votes} traders favor buying. "
            consensus += "Good setup but some concerns from value/risk perspectives."
        elif sell_votes >= 3:
            consensus = f"**CONSENSUS SELL/AVOID**: {sell_votes}/{total_votes} traders would sell or avoid. "
            consensus += "Multiple red flags across different frameworks."
        elif wait_votes >= 4:
            consensus = f"**CONSENSUS WAIT**: {wait_votes}/{total_votes} traders would wait for better setup. "
            consensus += "Not enough edge or conviction to justify risk."
        else:
            consensus = f"**MIXED SIGNALS**: Buy: {buy_votes}, Sell: {sell_votes}, Wait: {wait_votes}. "
            consensus += "Different trading philosophies see this differently. Proceed with caution."
        
        return consensus
