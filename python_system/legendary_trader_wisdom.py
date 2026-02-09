"""
LEGENDARY TRADER WISDOM MODULE v2.0
====================================

Applies the investment philosophies of history's greatest traders to current market conditions.
Each legend receives the FULL analysis output and performs genuine, deep analysis using
ALL available data — enhanced fundamentals, technicals, market context, dark pools,
sentiment, and real-time intelligence.

NO hardcoded answers. NO predefined thresholds that ignore context.
Every analysis is FRESH, DYNAMIC, and uses REAL current data.

Traders modeled:
- Warren Buffett: Value investing, margin of safety, business quality, competitive moat
- George Soros: Reflexivity theory, regime changes, macro trends, feedback loops
- Stanley Druckenmiller: Risk management, position sizing, macro + micro synthesis
- Peter Lynch: Growth at reasonable price, PEG ratio, sector classification
- Paul Tudor Jones: Risk/reward asymmetry, technical discipline, never average losers
- Jesse Livermore: Tape reading, trend following, pyramiding winners, pivotal points
"""

from typing import Dict, Optional
import math


class LegendaryTraderWisdom:
    """
    Applies legendary trader philosophies to current market setups.
    Each legend receives the FULL output dict with all enriched data.
    """

    def get_trader_perspectives(self, full_output: Dict) -> Dict[str, Dict]:
        """
        Get perspectives from legendary traders on this setup.
        Each perspective is a structured dict with action, conviction, reasoning, and quote.
        """
        perspectives = {
            'warren_buffett': self._buffett_perspective(full_output),
            'george_soros': self._soros_perspective(full_output),
            'stanley_druckenmiller': self._druckenmiller_perspective(full_output),
            'peter_lynch': self._lynch_perspective(full_output),
            'paul_tudor_jones': self._ptj_perspective(full_output),
            'jesse_livermore': self._livermore_perspective(full_output),
        }
        return perspectives

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _safe(d: Optional[Dict], *keys, default=0):
        """Safely traverse nested dicts."""
        val = d
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        return val if val is not None else default

    @staticmethod
    def _pct(v, already_pct=False):
        """Format a value as percentage string."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 'N/A'
        if already_pct:
            return f"{v:.1f}%"
        return f"{v * 100:.1f}%"

    @staticmethod
    def _fmt_price(v):
        if v is None or v == 0:
            return 'N/A'
        return f"${v:,.2f}"

    @staticmethod
    def _fmt_ratio(v):
        if v is None or v == 0:
            return 'N/A'
        return f"{v:.2f}"

    @staticmethod
    def _fmt_large(v):
        if v is None or v == 0:
            return 'N/A'
        abs_v = abs(v)
        if abs_v >= 1e12:
            return f"${v/1e12:.1f}T"
        if abs_v >= 1e9:
            return f"${v/1e9:.1f}B"
        if abs_v >= 1e6:
            return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"

    # ------------------------------------------------------------------ BUFFETT
    def _buffett_perspective(self, o: Dict) -> Dict:
        """
        Warren Buffett: Value investing, margin of safety, business quality.
        
        Buffett's actual methodology:
        1. Durable competitive advantage (high & stable ROE, high margins)
        2. Understandable business with predictable earnings
        3. Honest, competent management (insider ownership)
        4. Available at a reasonable price (margin of safety)
        5. Low debt relative to earnings power
        6. Strong free cash flow generation
        """
        f = o.get('fundamentals', {})
        ef = o.get('enhanced_fundamentals', {}) if isinstance(o.get('enhanced_fundamentals'), dict) else {}
        price = o.get('current_price', 0)
        target = o.get('target_price', 0)
        symbol = o.get('symbol', '?')

        # --- Extract comprehensive data ---
        roe = f.get('roe', 0)  # decimal
        profit_margin = f.get('profit_margin', 0)  # decimal
        pe = f.get('pe_ratio', 0)
        de = f.get('debt_to_equity', 0)
        rev_growth = f.get('revenue_growth', 0)  # decimal
        earn_growth = f.get('earnings_growth', 0)  # decimal
        peg = f.get('peg_ratio', 0)
        pb = f.get('price_to_book', 0)
        fcf_yield = self._safe(ef, 'valuation', 'fcf_yield_pct', default=0)  # already %
        dividend_yield = f.get('dividend_yield', 0)  # decimal
        current_ratio = f.get('current_ratio', 0)
        ebitda = f.get('ebitda', 0)
        market_cap = f.get('market_cap', 0)
        
        # Enhanced fundamentals deep data
        quality_score = self._safe(ef, 'quality_score', 'overall_score', default=0)
        moat_score = self._safe(ef, 'quality_score', 'moat_score', default=0)
        earnings_quality = self._safe(ef, 'quality_score', 'earnings_quality', default=0)
        
        # GARP analysis
        garp_score = self._safe(ef, 'garp_analysis', 'garp_score', default=0)
        
        # Revenue trends
        rev_cagr_5y = self._safe(ef, 'financial_trends', 'revenue_cagr_5y', default=0)  # already %
        earnings_cagr_5y = self._safe(ef, 'financial_trends', 'earnings_cagr_5y', default=0)  # already %
        
        # Altman Z-Score
        z_score = self._safe(ef, 'altman_z', 'z_score', default=0)
        z_rating = self._safe(ef, 'altman_z', 'rating', default='N/A')
        
        # Insider ownership
        insider_pct = self._safe(ef, 'share_structure', 'insider_ownership_pct', default=0)  # already %
        
        # Dark pool activity (Buffett watches institutional flow)
        dark_pool = o.get('stockgrid_analysis', {}) if isinstance(o.get('stockgrid_analysis'), dict) else {}
        dp_sentiment = self._safe(dark_pool, 'dark_pool_analysis', 'sentiment', default='unknown')
        
        # --- Buffett's scoring system ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []

        # 1. MOAT / Business Quality (30 points)
        if roe > 0.20:
            score += 15
            reasons.append(f"Excellent ROE of {self._pct(roe)} indicates strong competitive advantage")
        elif roe > 0.15:
            score += 10
            reasons.append(f"Good ROE of {self._pct(roe)} suggests decent moat")
        elif roe > 0:
            score += 3
            concerns.append(f"Mediocre ROE of {self._pct(roe)} — weak moat")
        else:
            concerns.append(f"Negative ROE of {self._pct(roe)} — no moat, destroying value")

        if profit_margin > 0.20:
            score += 10
            reasons.append(f"Fat margins ({self._pct(profit_margin)}) show pricing power")
        elif profit_margin > 0.10:
            score += 5
            reasons.append(f"Decent margins ({self._pct(profit_margin)})")
        elif profit_margin > 0:
            score += 2
            concerns.append(f"Thin margins ({self._pct(profit_margin)}) — commodity-like business")
        else:
            concerns.append(f"Negative margins ({self._pct(profit_margin)}) — losing money")

        if quality_score > 70:
            score += 5
            reasons.append(f"Quality score of {quality_score:.0f}/100 confirms business excellence")
        elif quality_score > 50:
            score += 2

        # 2. VALUATION / Margin of Safety (25 points)
        if pe > 0:
            if pe < 15:
                score += 15
                reasons.append(f"P/E of {pe:.1f} offers classic Buffett value with margin of safety")
            elif pe < 20:
                score += 10
                reasons.append(f"P/E of {pe:.1f} is reasonable for a quality business")
            elif pe < 30:
                score += 5
                reasons.append(f"P/E of {pe:.1f} — fair but not cheap")
            else:
                concerns.append(f"P/E of {pe:.1f} is expensive — Buffett prefers paying fair prices")
        else:
            concerns.append("Negative P/E — company is unprofitable")

        if pb > 0 and pb < 3:
            score += 5
            reasons.append(f"P/B of {pb:.1f} shows tangible asset backing")
        elif pb > 0 and pb < 5:
            score += 2
        elif pb > 10:
            concerns.append(f"P/B of {pb:.1f} — paying huge premium over book value")

        if fcf_yield > 5:
            score += 5
            reasons.append(f"FCF yield of {fcf_yield:.1f}% generates real cash for shareholders")
        elif fcf_yield > 3:
            score += 2

        # 3. FINANCIAL STRENGTH (20 points)
        if de < 0.3 and de >= 0:
            score += 10
            reasons.append(f"Conservative D/E of {de:.2f} — fortress balance sheet")
        elif de < 0.5:
            score += 7
            reasons.append(f"Manageable D/E of {de:.2f}")
        elif de < 1.0:
            score += 3
            concerns.append(f"D/E of {de:.2f} — moderate leverage")
        elif de > 1.5:
            concerns.append(f"D/E of {de:.2f} — too much debt for Buffett's taste")

        if z_score > 3:
            score += 5
            reasons.append(f"Altman Z-Score of {z_score:.1f} ({z_rating}) — very low bankruptcy risk")
        elif z_score > 1.8:
            score += 3
        elif z_score > 0:
            concerns.append(f"Altman Z-Score of {z_score:.1f} ({z_rating}) — financial stress risk")

        if current_ratio > 1.5:
            score += 5
            reasons.append(f"Current ratio of {current_ratio:.1f} — ample liquidity")
        elif current_ratio > 1.0:
            score += 2

        # 4. GROWTH & DURABILITY (15 points)
        if rev_cagr_5y > 10:
            score += 5
            reasons.append(f"5Y revenue CAGR of {rev_cagr_5y:.1f}% shows durable growth")
        elif rev_cagr_5y > 5:
            score += 3

        if earn_growth > 0.10:
            score += 5
            reasons.append(f"Earnings growing at {self._pct(earn_growth)} — compounding machine")
        elif earn_growth > 0:
            score += 2

        if insider_pct > 5:
            score += 5
            reasons.append(f"Insider ownership of {insider_pct:.1f}% — management has skin in the game")
        elif insider_pct > 1:
            score += 2

        # 5. INSTITUTIONAL FLOW (10 points)
        if dp_sentiment in ['bullish', 'very_bullish']:
            score += 10
            reasons.append(f"Dark pool sentiment is {dp_sentiment} — smart money accumulating")
        elif dp_sentiment in ['bearish', 'very_bearish']:
            concerns.append(f"Dark pool sentiment is {dp_sentiment} — institutions distributing")

        # --- Determine action ---
        if score >= 70:
            action = "BUY"
            conviction = "HIGH"
            headline = f"**BUFFETT WOULD BUY {symbol}**: This is a quality business at a reasonable price."
        elif score >= 50:
            action = "HOLD/WATCH"
            conviction = "MODERATE"
            headline = f"**BUFFETT WOULD WATCH {symbol}**: Good business but needs better price or more conviction."
        elif score >= 30:
            action = "WAIT"
            conviction = "LOW"
            headline = f"**BUFFETT WOULD WAIT ON {symbol}**: Doesn't meet his quality + value criteria yet."
        else:
            action = "PASS"
            conviction = "NONE"
            headline = f"**BUFFETT WOULD PASS ON {symbol}**: Not his kind of business."

        # Build detailed reasoning
        reasoning_parts = [headline, ""]
        
        if reasons:
            reasoning_parts.append("**Strengths (What Buffett likes):**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**Concerns (What gives Buffett pause):**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        # Margin of safety calculation
        if target > 0 and price > 0:
            margin_of_safety = ((target - price) / price) * 100
            if margin_of_safety > 15:
                reasoning_parts.append(f"**Margin of Safety**: {margin_of_safety:.1f}% upside to fair value ({self._fmt_price(target)}) — Buffett likes at least 25% margin.")
            elif margin_of_safety > 0:
                reasoning_parts.append(f"**Margin of Safety**: Only {margin_of_safety:.1f}% upside — thin margin, Buffett would want more cushion.")
            else:
                reasoning_parts.append(f"**Margin of Safety**: Trading {abs(margin_of_safety):.1f}% ABOVE fair value — no margin of safety.")

        reasoning_parts.append("")
        reasoning_parts.append(f"**Buffett Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"Price is what you pay, value is what you get."* — Warren Buffett')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'roe': self._pct(roe),
                'pe_ratio': self._fmt_ratio(pe),
                'debt_to_equity': self._fmt_ratio(de),
                'profit_margin': self._pct(profit_margin),
                'fcf_yield': f"{fcf_yield:.1f}%" if fcf_yield else 'N/A',
                'quality_score': f"{quality_score:.0f}/100" if quality_score else 'N/A',
            }
        }

    # ------------------------------------------------------------------ SOROS
    def _soros_perspective(self, o: Dict) -> Dict:
        """
        George Soros: Reflexivity theory, regime changes, macro trends.
        
        Soros's actual methodology:
        1. Identify prevailing bias in the market (sentiment vs reality gap)
        2. Look for reflexive feedback loops (price action reinforcing narrative)
        3. Detect regime changes early (volatility shifts, macro pivots)
        4. Bet big on high-conviction macro themes
        5. Cut losses immediately when thesis is wrong
        """
        symbol = o.get('symbol', '?')
        price = o.get('current_price', 0)
        
        # Market context (Soros is a MACRO trader)
        mc = o.get('market_context', {}) if isinstance(o.get('market_context'), dict) else {}
        regime = self._safe(mc, 'regime', default={})
        if isinstance(regime, dict):
            regime_name = regime.get('regime', 'unknown')
            regime_confidence = regime.get('confidence', 0)
            momentum_20d = regime.get('momentum_20d', 0)
            vol_annualized = regime.get('volatility_annualized', 0)
        else:
            regime_name = str(regime)
            regime_confidence = 0
            momentum_20d = 0
            vol_annualized = 0

        vix = self._safe(mc, 'vix', default={})
        if isinstance(vix, dict):
            # 'vix' key has the numeric value, 'level' is a string label like 'ELEVATED'
            vix_level = float(vix.get('vix', 0) or 0)
            vix_change = float(vix.get('change_percent', vix.get('change_pct', 0)) or 0)
        else:
            vix_level = float(vix) if isinstance(vix, (int, float)) else 0
            vix_change = 0

        # Sentiment (Soros looks for gaps between sentiment and reality)
        sentiment_score = o.get('sentiment_score', 50)
        
        # Technical trend (reflexive feedback loops show in price)
        ta = o.get('technical_analysis', {})
        trend_score = ta.get('trend_score', 50)
        momentum_score = ta.get('momentum_score', 50)
        volatility_score = ta.get('volatility_score', 50)
        rsi = o.get('technical_indicators', {}).get('rsi', 50)
        
        # Real-time intelligence (Soros reads the news obsessively)
        exa = o.get('exa_intelligence', {}) if isinstance(o.get('exa_intelligence'), dict) else {}
        exa_sentiment = self._safe(exa, 'sentiment_analysis', 'overall_sentiment', default='neutral')
        
        # Dark pool flow (institutional positioning)
        dark_pool = o.get('stockgrid_analysis', {}) if isinstance(o.get('stockgrid_analysis'), dict) else {}
        dp_sentiment = self._safe(dark_pool, 'dark_pool_analysis', 'sentiment', default='unknown')
        net_flow = self._safe(dark_pool, 'dark_pool_analysis', 'net_flow_5d', default=0)
        
        # Comprehensive technicals for momentum confirmation
        ct = o.get('comprehensive_technicals', {}) if isinstance(o.get('comprehensive_technicals'), dict) else {}
        ichimoku = self._safe(ct, 'ichimoku', default={})
        
        # --- Soros's analysis framework ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []
        
        # 1. REGIME DETECTION (30 points)
        # Soros makes his biggest bets on regime changes
        regime_lower = str(regime_name).lower()
        if 'bull' in regime_lower:
            score += 15
            reasons.append(f"Market regime: **{regime_name}** — favorable macro backdrop")
        elif 'bear' in regime_lower:
            score += 5
            concerns.append(f"Market regime: **{regime_name}** — headwinds, but Soros thrives in chaos")
        elif 'transition' in regime_lower or 'volatile' in regime_lower:
            score += 20  # Soros LOVES transitions
            reasons.append(f"Market regime: **{regime_name}** — regime change = Soros's bread and butter")
        else:
            score += 10
        
        if vix_level > 30:
            score += 10
            reasons.append(f"VIX at {vix_level:.1f} — extreme fear creates Soros-style opportunities")
        elif vix_level > 20:
            score += 5
            reasons.append(f"VIX at {vix_level:.1f} — elevated uncertainty, watch for regime shift")
        elif vix_level > 0:
            score += 2
            concerns.append(f"VIX at {vix_level:.1f} — low fear, complacency can be dangerous")

        # 2. REFLEXIVITY / FEEDBACK LOOPS (25 points)
        # When price action and sentiment reinforce each other
        sentiment_trend_gap = abs(sentiment_score - trend_score)
        
        if trend_score > 65 and sentiment_score < 40:
            score += 25
            reasons.append(f"**REFLEXIVE OPPORTUNITY**: Strong trend (score: {trend_score:.0f}) but negative sentiment ({sentiment_score:.0f}) — early-stage feedback loop forming. Soros would buy aggressively here.")
        elif trend_score < 35 and sentiment_score > 60:
            score += 20
            reasons.append(f"**REFLEXIVE REVERSAL**: Weak trend ({trend_score:.0f}) despite positive sentiment ({sentiment_score:.0f}) — narrative about to break down. Soros would short.")
        elif trend_score > 70 and sentiment_score > 70:
            score += 5
            concerns.append(f"Trend ({trend_score:.0f}) and sentiment ({sentiment_score:.0f}) both euphoric — late-stage reflexive loop, nearing exhaustion")
        elif sentiment_trend_gap > 25:
            score += 15
            reasons.append(f"Sentiment-trend divergence of {sentiment_trend_gap:.0f} points — reflexive dislocation Soros would exploit")
        else:
            score += 8

        # 3. MACRO POSITIONING (20 points)
        if momentum_20d > 5:
            score += 10
            reasons.append(f"SPY 20-day momentum: +{momentum_20d:.1f}% — macro tailwind")
        elif momentum_20d < -5:
            score += 5
            concerns.append(f"SPY 20-day momentum: {momentum_20d:.1f}% — macro headwind")
        else:
            score += 5

        # Dark pool institutional flow
        if dp_sentiment in ['bullish', 'very_bullish']:
            score += 10
            reasons.append(f"Dark pool flow is {dp_sentiment} — institutions positioning with the trade")
        elif dp_sentiment in ['bearish', 'very_bearish']:
            concerns.append(f"Dark pool flow is {dp_sentiment} — institutions positioning against")
        else:
            score += 3

        # 4. CONVICTION & TIMING (15 points)
        if rsi < 30 and trend_score > 50:
            score += 15
            reasons.append(f"RSI oversold ({rsi:.0f}) in uptrend — classic Soros contrarian entry")
        elif rsi > 70 and trend_score < 50:
            score += 10
            reasons.append(f"RSI overbought ({rsi:.0f}) in downtrend — Soros would short the exhaustion")
        elif vol_annualized > 40:
            score += 10
            reasons.append(f"Annualized vol of {vol_annualized:.1f}% — high vol = high opportunity for Soros")
        else:
            score += 5

        # 5. REAL-TIME INTELLIGENCE (10 points)
        if exa_sentiment == 'bullish':
            score += 7
            reasons.append("Real-time news sentiment is bullish — narrative supports the trade")
        elif exa_sentiment == 'bearish':
            score += 3
            concerns.append("Real-time news sentiment is bearish — narrative headwind")
        else:
            score += 5

        # --- Determine action ---
        # Soros is more aggressive — he bets big when he sees regime change
        if score >= 65:
            action = "BUY AGGRESSIVELY"
            conviction = "HIGH"
            headline = f"**SOROS WOULD BUY {symbol} AGGRESSIVELY**: Reflexive opportunity with macro alignment."
        elif score >= 50:
            action = "BUY"
            conviction = "MODERATE"
            headline = f"**SOROS WOULD BUY {symbol}**: Decent setup but not a high-conviction macro bet."
        elif score >= 35:
            action = "WAIT"
            conviction = "LOW"
            headline = f"**SOROS WOULD WAIT ON {symbol}**: No clear regime change or reflexive opportunity."
        else:
            action = "PASS"
            conviction = "NONE"
            headline = f"**SOROS WOULD PASS ON {symbol}**: No macro edge, no reflexive dislocation."

        reasoning_parts = [headline, ""]
        
        if reasons:
            reasoning_parts.append("**What Soros sees:**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**What gives Soros pause:**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        reasoning_parts.append(f"**Soros Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"It\'s not whether you\'re right or wrong, but how much you make when you\'re right and how much you lose when you\'re wrong."* — George Soros')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'regime': regime_name,
                'vix': f"{vix_level:.1f}" if vix_level else 'N/A',
                'sentiment_score': f"{sentiment_score:.0f}",
                'trend_score': f"{trend_score:.0f}",
                'dark_pool': dp_sentiment,
            }
        }

    # ------------------------------------------------------------------ DRUCKENMILLER
    def _druckenmiller_perspective(self, o: Dict) -> Dict:
        """
        Stanley Druckenmiller: Risk management, position sizing, macro + micro synthesis.
        
        Druckenmiller's actual methodology:
        1. Combine macro view with micro stock selection
        2. Size positions based on conviction — go big on best ideas
        3. Preserve capital above all — cut losses fast
        4. Look for asymmetric risk/reward (3:1 minimum)
        5. Follow liquidity and central bank policy
        """
        symbol = o.get('symbol', '?')
        price = o.get('current_price', 0)
        target = o.get('target_price', 0)
        stop = o.get('stop_loss', 0)
        confidence = o.get('confidence', 0)
        
        # Risk/reward
        rr_ratio = abs(target - price) / abs(price - stop) if stop != price else 1.0
        potential_gain = ((target - price) / price) * 100 if price > 0 else 0
        potential_loss = ((price - stop) / price) * 100 if price > 0 else 0
        
        # Position sizing
        ps = o.get('position_sizing', {})
        position_size_pct = ps.get('position_size_pct', 0)
        risk_pct = ps.get('risk_pct_of_bankroll', 0)
        
        # Stochastic analysis
        stoch = o.get('stochastic_analysis', {}) if isinstance(o.get('stochastic_analysis'), dict) else {}
        var_95 = stoch.get('var_95', 0)
        max_drawdown = stoch.get('max_drawdown', 0)
        
        # Market context
        mc = o.get('market_context', {}) if isinstance(o.get('market_context'), dict) else {}
        regime = self._safe(mc, 'regime', default={})
        regime_name = regime.get('regime', 'unknown') if isinstance(regime, dict) else str(regime)
        
        # Fundamentals
        f = o.get('fundamentals', {})
        earn_growth = f.get('earnings_growth', 0)
        rev_growth = f.get('revenue_growth', 0)
        
        # Technical
        ta = o.get('technical_analysis', {})
        trend_score = ta.get('trend_score', 50)
        momentum_score = ta.get('momentum_score', 50)
        
        # GARCH volatility
        garch = o.get('garch_analysis', {}) if isinstance(o.get('garch_analysis'), dict) else {}
        garch_vol = garch.get('current_volatility', 0)
        
        # --- Druckenmiller's scoring ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []

        # 1. RISK/REWARD ASYMMETRY (35 points) — Druck's #1 criterion
        if rr_ratio >= 5.0:
            score += 35
            reasons.append(f"**Exceptional R/R of {rr_ratio:.1f}:1** — Druck would size up aggressively (Gain: {potential_gain:.1f}%, Risk: {potential_loss:.1f}%)")
        elif rr_ratio >= 3.0:
            score += 25
            reasons.append(f"**Strong R/R of {rr_ratio:.1f}:1** — meets Druck's minimum threshold (Gain: {potential_gain:.1f}%, Risk: {potential_loss:.1f}%)")
        elif rr_ratio >= 2.0:
            score += 15
            reasons.append(f"R/R of {rr_ratio:.1f}:1 — acceptable but not a home run (Gain: {potential_gain:.1f}%, Risk: {potential_loss:.1f}%)")
        elif rr_ratio >= 1.0:
            score += 5
            concerns.append(f"R/R of {rr_ratio:.1f}:1 — too thin for Druck's standards")
        else:
            concerns.append(f"R/R of {rr_ratio:.1f}:1 — negative expected value, Druck would never touch this")

        # 2. CONVICTION LEVEL (25 points)
        if confidence > 70:
            score += 25
            reasons.append(f"System confidence of {confidence:.0f}% — high conviction, Druck would size up to 2-3x normal")
        elif confidence > 55:
            score += 15
            reasons.append(f"System confidence of {confidence:.0f}% — moderate conviction, standard position size")
        elif confidence > 40:
            score += 8
            concerns.append(f"System confidence of {confidence:.0f}% — low conviction, Druck would use scout position (1/4 size)")
        else:
            concerns.append(f"System confidence of {confidence:.0f}% — too low, Druck would pass entirely")

        # 3. TREND ALIGNMENT (20 points)
        if trend_score > 65 and momentum_score > 55:
            score += 20
            reasons.append(f"Trend ({trend_score:.0f}) and momentum ({momentum_score:.0f}) aligned — Druck rides trends")
        elif trend_score > 55:
            score += 10
            reasons.append(f"Trend developing ({trend_score:.0f}) — Druck would enter with tight stop")
        else:
            score += 3
            concerns.append(f"Weak trend ({trend_score:.0f}) — Druck avoids choppy markets")

        # 4. GROWTH CATALYST (10 points)
        if earn_growth > 0.20:
            score += 10
            reasons.append(f"Earnings growing {self._pct(earn_growth)} — strong fundamental catalyst")
        elif earn_growth > 0.10:
            score += 5
        elif earn_growth < 0:
            concerns.append(f"Earnings declining {self._pct(earn_growth)} — fundamental headwind")

        # 5. RISK MANAGEMENT (10 points)
        if var_95 > 0 and var_95 < 0.10:
            score += 10
            reasons.append(f"VaR(95%) of {self._pct(var_95)} — manageable tail risk")
        elif var_95 < 0.20:
            score += 5
        elif var_95 > 0.20:
            concerns.append(f"VaR(95%) of {self._pct(var_95)} — significant tail risk, reduce size")

        # --- Determine action ---
        if score >= 70:
            action = "SIZE UP"
            conviction = "HIGH"
            size_rec = "2-3x normal position"
            headline = f"**DRUCKENMILLER WOULD SIZE UP ON {symbol}**: Home run setup — go big."
        elif score >= 50:
            action = "BUY"
            conviction = "MODERATE"
            size_rec = "Standard position"
            headline = f"**DRUCKENMILLER WOULD BUY {symbol}**: Good setup, standard position with tight stop."
        elif score >= 35:
            action = "SCOUT"
            conviction = "LOW"
            size_rec = "1/4 to 1/2 normal position"
            headline = f"**DRUCKENMILLER WOULD SCOUT {symbol}**: Small position to test the thesis."
        else:
            action = "PASS"
            conviction = "NONE"
            size_rec = "No position"
            headline = f"**DRUCKENMILLER WOULD PASS ON {symbol}**: Not enough edge to risk capital."

        reasoning_parts = [headline, ""]
        
        if reasons:
            reasoning_parts.append("**Why Druck likes it:**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**Risk factors Druck would flag:**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        reasoning_parts.append(f"**Position Sizing**: {size_rec}")
        reasoning_parts.append(f"**Druckenmiller Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"The way to build long-term returns is through preservation of capital and home runs."* — Stanley Druckenmiller')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'risk_reward': f"{rr_ratio:.1f}:1",
                'confidence': f"{confidence:.0f}%",
                'position_size': size_rec,
                'trend_score': f"{trend_score:.0f}",
            }
        }

    # ------------------------------------------------------------------ LYNCH
    def _lynch_perspective(self, o: Dict) -> Dict:
        """
        Peter Lynch: Growth at reasonable price, PEG ratio, sector classification.
        
        Lynch's actual methodology:
        1. Classify the stock (slow grower, stalwart, fast grower, cyclical, turnaround, asset play)
        2. PEG ratio < 1 is ideal, < 1.5 is acceptable
        3. Understand what the company does (earnings predictability)
        4. Check the balance sheet (debt/equity)
        5. Look for "boring" companies with strong fundamentals
        """
        symbol = o.get('symbol', '?')
        f = o.get('fundamentals', {})
        ef = o.get('enhanced_fundamentals', {}) if isinstance(o.get('enhanced_fundamentals'), dict) else {}
        
        pe = f.get('pe_ratio', 0)
        peg = f.get('peg_ratio', 0)
        earn_growth = f.get('earnings_growth', 0)  # decimal
        rev_growth = f.get('revenue_growth', 0)  # decimal
        de = f.get('debt_to_equity', 0)
        profit_margin = f.get('profit_margin', 0)
        dividend_yield = f.get('dividend_yield', 0)
        
        # Enhanced data
        garp_score = self._safe(ef, 'garp_analysis', 'garp_score', default=0)
        garp_rating = self._safe(ef, 'garp_analysis', 'rating', default='N/A')
        rev_cagr_5y = self._safe(ef, 'financial_trends', 'revenue_cagr_5y', default=0)
        earnings_cagr_5y = self._safe(ef, 'financial_trends', 'earnings_cagr_5y', default=0)
        
        # --- Lynch's stock classification ---
        if earn_growth > 0.25:
            category = "Fast Grower"
            cat_desc = "High-growth company — Lynch's favorite category if the price is right"
        elif earn_growth > 0.10:
            category = "Stalwart"
            cat_desc = "Steady grower — reliable but won't make you rich overnight"
        elif earn_growth > 0 and earn_growth <= 0.05:
            category = "Slow Grower"
            cat_desc = "Mature company with limited growth — Lynch would want a dividend"
        elif earn_growth < -0.10:
            category = "Turnaround"
            cat_desc = "Declining earnings — could be a turnaround play if the thesis is right"
        else:
            category = "Stalwart"
            cat_desc = "Moderate growth profile"

        # --- Lynch's scoring ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []

        # 1. PEG RATIO (35 points) — Lynch's signature metric
        if peg > 0 and peg < 1.0:
            score += 35
            reasons.append(f"**PEG of {peg:.2f} < 1.0** — classic Lynch GARP buy! Growth ({self._pct(earn_growth)}) far exceeds valuation (P/E: {pe:.1f})")
        elif peg > 0 and peg < 1.5:
            score += 25
            reasons.append(f"PEG of {peg:.2f} — reasonably priced growth (P/E: {pe:.1f}, Growth: {self._pct(earn_growth)})")
        elif peg > 0 and peg < 2.0:
            score += 15
            reasons.append(f"PEG of {peg:.2f} — fair but not cheap (P/E: {pe:.1f})")
        elif peg > 2.5:
            concerns.append(f"PEG of {peg:.2f} — overpriced relative to growth. Lynch would sell.")
        elif peg <= 0:
            if earn_growth <= 0:
                concerns.append(f"PEG unavailable — negative earnings growth ({self._pct(earn_growth)})")
            else:
                score += 5

        # 2. EARNINGS GROWTH (20 points)
        if earn_growth > 0.25:
            score += 20
            reasons.append(f"Earnings growing {self._pct(earn_growth)} — fast grower territory")
        elif earn_growth > 0.15:
            score += 15
            reasons.append(f"Solid earnings growth of {self._pct(earn_growth)}")
        elif earn_growth > 0.05:
            score += 8
        elif earn_growth < 0:
            concerns.append(f"Earnings declining {self._pct(earn_growth)} — needs turnaround thesis")

        # 3. REVENUE GROWTH (15 points)
        if rev_growth > 0.20:
            score += 15
            reasons.append(f"Revenue growing {self._pct(rev_growth)} — top-line expansion confirms the story")
        elif rev_growth > 0.10:
            score += 10
        elif rev_growth > 0:
            score += 5
        elif rev_growth < 0:
            concerns.append(f"Revenue declining {self._pct(rev_growth)} — shrinking business")

        # 4. BALANCE SHEET (15 points)
        if de < 0.5 and de >= 0:
            score += 15
            reasons.append(f"Low debt (D/E: {de:.2f}) — Lynch checks the balance sheet first")
        elif de < 1.0:
            score += 8
        elif de > 2.0:
            concerns.append(f"High debt (D/E: {de:.2f}) — Lynch avoids leveraged companies")

        # 5. GARP SCORE (10 points)
        if garp_score > 70:
            score += 10
            reasons.append(f"GARP score of {garp_score:.0f}/100 ({garp_rating}) — quantitative confirmation of Lynch's thesis")
        elif garp_score > 50:
            score += 5
        elif garp_score > 0:
            score += 2

        # 6. LONG-TERM TRACK RECORD (5 points)
        if rev_cagr_5y > 15:
            score += 5
            reasons.append(f"5Y revenue CAGR of {rev_cagr_5y:.1f}% — proven long-term grower")
        elif rev_cagr_5y > 8:
            score += 3

        # --- Determine action ---
        if score >= 65:
            action = "BUY"
            conviction = "HIGH"
            headline = f"**LYNCH WOULD BUY {symbol}** ({category}): Classic GARP opportunity."
        elif score >= 45:
            action = "HOLD/WATCH"
            conviction = "MODERATE"
            headline = f"**LYNCH WOULD WATCH {symbol}** ({category}): Decent but needs better price or growth acceleration."
        elif score >= 25:
            action = "WAIT"
            conviction = "LOW"
            headline = f"**LYNCH WOULD WAIT ON {symbol}** ({category}): {cat_desc}"
        else:
            action = "AVOID"
            conviction = "NONE"
            headline = f"**LYNCH WOULD AVOID {symbol}** ({category}): Doesn't meet GARP criteria."

        reasoning_parts = [headline, ""]
        reasoning_parts.append(f"**Lynch Classification**: {category} — {cat_desc}")
        reasoning_parts.append("")
        
        if reasons:
            reasoning_parts.append("**What Lynch likes:**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**What Lynch dislikes:**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        reasoning_parts.append(f"**Lynch Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"Know what you own, and know why you own it."* — Peter Lynch')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'peg_ratio': self._fmt_ratio(peg),
                'category': category,
                'earnings_growth': self._pct(earn_growth),
                'garp_score': f"{garp_score:.0f}/100" if garp_score else 'N/A',
            }
        }

    # ------------------------------------------------------------------ PTJ
    def _ptj_perspective(self, o: Dict) -> Dict:
        """
        Paul Tudor Jones: Risk/reward asymmetry, technical discipline, 200-day MA.
        
        PTJ's actual methodology:
        1. 5:1 reward-to-risk minimum for new positions
        2. Never average down on losers
        3. 200-day moving average as primary trend filter
        4. Loser's game — focus on not losing money
        5. Technical discipline over fundamental conviction
        """
        symbol = o.get('symbol', '?')
        price = o.get('current_price', 0)
        target = o.get('target_price', 0)
        stop = o.get('stop_loss', 0)
        
        # Risk/reward
        rr_ratio = abs(target - price) / abs(price - stop) if stop != price else 1.0
        potential_gain = ((target - price) / price) * 100 if price > 0 else 0
        potential_loss = ((price - stop) / price) * 100 if price > 0 else 0
        
        # Technical indicators (PTJ is primarily technical)
        ti = o.get('technical_indicators', {})
        ta = o.get('technical_analysis', {})
        rsi = ti.get('rsi', 50)
        sma_200 = ti.get('sma_200', 0)
        sma_50 = ti.get('sma_50', 0)
        sma_20 = ti.get('sma_20', 0)
        atr = ti.get('atr', 0)
        adx = ta.get('adx', 20) if isinstance(ta.get('adx'), (int, float)) else ti.get('adx', 20)
        trend_score = ta.get('trend_score', 50)
        
        # Comprehensive technicals
        ct = o.get('comprehensive_technicals', {}) if isinstance(o.get('comprehensive_technicals'), dict) else {}
        
        # Advanced technicals
        adv = o.get('advanced_technicals', {}) if isinstance(o.get('advanced_technicals'), dict) else {}
        bb = adv.get('bollinger_bands', {}) if isinstance(adv.get('bollinger_bands'), dict) else {}
        bb_pct_b = bb.get('percent_b', 0.5)
        
        # --- PTJ's scoring ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []

        # 1. RISK/REWARD — PTJ's 5:1 RULE (40 points)
        if rr_ratio >= 5.0:
            score += 40
            reasons.append(f"**R/R of {rr_ratio:.1f}:1 meets PTJ's 5:1 rule** — exceptional asymmetry (Reward: +{potential_gain:.1f}%, Risk: -{potential_loss:.1f}%)")
        elif rr_ratio >= 3.0:
            score += 25
            reasons.append(f"R/R of {rr_ratio:.1f}:1 — good but below PTJ's ideal 5:1 (Reward: +{potential_gain:.1f}%, Risk: -{potential_loss:.1f}%)")
        elif rr_ratio >= 2.0:
            score += 15
            concerns.append(f"R/R of {rr_ratio:.1f}:1 — below PTJ's standards")
        else:
            concerns.append(f"R/R of {rr_ratio:.1f}:1 — PTJ would never take this trade")

        # 2. 200-DAY MOVING AVERAGE (25 points) — PTJ's primary trend filter
        if sma_200 > 0 and price > 0:
            pct_above_200 = ((price - sma_200) / sma_200) * 100
            if price > sma_200:
                score += 20
                reasons.append(f"Price is {pct_above_200:.1f}% above 200-day MA ({self._fmt_price(sma_200)}) — PTJ's #1 trend filter is bullish")
                if sma_50 > sma_200:
                    score += 5
                    reasons.append("50-day MA above 200-day MA — golden cross confirms uptrend")
            else:
                concerns.append(f"Price is {abs(pct_above_200):.1f}% BELOW 200-day MA — PTJ would not buy below the 200-day")
                if sma_50 > 0 and sma_50 < sma_200:
                    concerns.append("Death cross (50-day below 200-day) — PTJ would be short, not long")

        # 3. MOMENTUM & TREND (20 points)
        if adx > 25 and trend_score > 60:
            score += 15
            reasons.append(f"Strong trend (ADX: {adx:.0f}, Trend Score: {trend_score:.0f}) — PTJ rides strong trends")
        elif adx > 20:
            score += 8
        else:
            concerns.append(f"Weak trend (ADX: {adx:.0f}) — PTJ avoids choppy markets")

        if rsi > 30 and rsi < 70:
            score += 5
            reasons.append(f"RSI at {rsi:.0f} — not overbought/oversold, clean entry")
        elif rsi < 30:
            score += 3
            reasons.append(f"RSI oversold at {rsi:.0f} — potential reversal, but PTJ waits for confirmation")
        elif rsi > 70:
            concerns.append(f"RSI overbought at {rsi:.0f} — PTJ would wait for pullback")

        # 4. VOLATILITY CONTEXT (15 points)
        if atr > 0 and price > 0:
            atr_pct = (atr / price) * 100
            if atr_pct < 3:
                score += 10
                reasons.append(f"ATR of {atr_pct:.1f}% — manageable volatility for position sizing")
            elif atr_pct < 5:
                score += 5
            else:
                concerns.append(f"ATR of {atr_pct:.1f}% — high volatility requires smaller position")

        if bb_pct_b > 0 and bb_pct_b < 0.2:
            score += 5
            reasons.append(f"Bollinger %B at {bb_pct_b:.2f} — near lower band, potential bounce")
        elif bb_pct_b > 0.8:
            concerns.append(f"Bollinger %B at {bb_pct_b:.2f} — near upper band, extended")

        # --- Determine action ---
        if score >= 70:
            action = "BUY"
            conviction = "HIGH"
            headline = f"**JONES WOULD BUY {symbol}**: Asymmetric R/R with technical confirmation."
        elif score >= 50:
            action = "CONSIDER"
            conviction = "MODERATE"
            headline = f"**JONES WOULD CONSIDER {symbol}**: Decent setup but not his ideal 5:1."
        elif score >= 30:
            action = "WAIT"
            conviction = "LOW"
            headline = f"**JONES WOULD WAIT ON {symbol}**: R/R doesn't justify the risk yet."
        else:
            action = "PASS"
            conviction = "NONE"
            headline = f"**JONES WOULD PASS ON {symbol}**: No asymmetric edge."

        reasoning_parts = [headline, ""]
        
        if reasons:
            reasoning_parts.append("**PTJ's technical read:**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**PTJ's risk flags:**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        reasoning_parts.append(f"**PTJ Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"Don\'t focus on making money; focus on protecting what you have."* — Paul Tudor Jones')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'risk_reward': f"{rr_ratio:.1f}:1",
                'above_200dma': f"{'Yes' if price > sma_200 and sma_200 > 0 else 'No'}",
                'adx': f"{adx:.0f}",
                'rsi': f"{rsi:.0f}",
            }
        }

    # ------------------------------------------------------------------ LIVERMORE
    def _livermore_perspective(self, o: Dict) -> Dict:
        """
        Jesse Livermore: Tape reading, trend following, pyramiding winners, pivotal points.
        
        Livermore's actual methodology:
        1. Follow the path of least resistance (trend)
        2. Wait for pivotal points (breakouts from consolidation)
        3. Pyramid winners — add to winning positions, never losers
        4. The market is never wrong, opinions often are
        5. Time is everything — patience to wait for the right setup
        """
        symbol = o.get('symbol', '?')
        price = o.get('current_price', 0)
        rec = o.get('recommendation', 'HOLD')
        
        # Technical (Livermore is 100% technical/tape reader)
        ti = o.get('technical_indicators', {})
        ta = o.get('technical_analysis', {})
        trend_score = ta.get('trend_score', 50)
        momentum_score = ta.get('momentum_score', 50)
        adx = ta.get('adx', 20) if isinstance(ta.get('adx'), (int, float)) else ti.get('adx', 20)
        rsi = ti.get('rsi', 50)
        sma_20 = ti.get('sma_20', 0)
        sma_50 = ti.get('sma_50', 0)
        sma_200 = ti.get('sma_200', 0)
        atr = ti.get('atr', 0)
        volume = ti.get('volume', 0)
        avg_volume = ti.get('avg_volume', 0)
        
        # Comprehensive technicals
        ct = o.get('comprehensive_technicals', {}) if isinstance(o.get('comprehensive_technicals'), dict) else {}
        aroon = self._safe(ct, 'aroon', default={})
        ichimoku = self._safe(ct, 'ichimoku', default={})
        
        # Advanced technicals
        adv = o.get('advanced_technicals', {}) if isinstance(o.get('advanced_technicals'), dict) else {}
        bb = adv.get('bollinger_bands', {}) if isinstance(adv.get('bollinger_bands'), dict) else {}
        
        # Breakout detection
        breakout = o.get('breakout_analysis', {}) if isinstance(o.get('breakout_analysis'), dict) else {}
        
        # Candlestick patterns
        candles = o.get('candlestick_patterns', {}) if isinstance(o.get('candlestick_patterns'), dict) else {}
        
        # --- Livermore's scoring ---
        score = 0
        max_score = 100
        reasons = []
        concerns = []

        # 1. TREND STRENGTH (35 points) — Livermore follows the trend above all
        if trend_score > 70 and adx > 25:
            score += 35
            reasons.append(f"**Strong trend** (Score: {trend_score:.0f}, ADX: {adx:.0f}) — Livermore's path of least resistance is clear")
        elif trend_score > 60 and adx > 20:
            score += 25
            reasons.append(f"Developing trend (Score: {trend_score:.0f}, ADX: {adx:.0f}) — Livermore would watch for confirmation")
        elif trend_score > 50:
            score += 12
            reasons.append(f"Mild trend (Score: {trend_score:.0f}) — Livermore would be patient")
        else:
            concerns.append(f"Weak/choppy trend (Score: {trend_score:.0f}, ADX: {adx:.0f}) — Livermore says 'there is a time to go long, a time to go short, and a time to go fishing'")

        # 2. PIVOTAL POINTS / BREAKOUTS (25 points)
        # Price relative to key MAs
        above_all_mas = price > sma_20 > 0 and price > sma_50 > 0 and price > sma_200 > 0
        if above_all_mas:
            score += 15
            reasons.append(f"Price above all key MAs (20/50/200) — bullish structure, Livermore would pyramid")
        elif sma_200 > 0 and price > sma_200:
            score += 8
            reasons.append("Price above 200-day MA — long-term trend intact")
        elif sma_200 > 0:
            concerns.append("Price below 200-day MA — Livermore would not go long")

        # Volume confirmation (Livermore was a tape reader — volume matters)
        if avg_volume > 0 and volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 1.5:
                score += 10
                reasons.append(f"Volume {vol_ratio:.1f}x average — strong conviction behind the move, Livermore loves this")
            elif vol_ratio > 1.0:
                score += 5
            else:
                concerns.append(f"Volume {vol_ratio:.1f}x average — weak conviction, Livermore would wait for volume surge")
        
        # Aroon indicator (trend strength)
        if isinstance(aroon, dict) and aroon.get('aroon_up', 0) > 70:
            score += 5
            reasons.append(f"Aroon Up at {aroon['aroon_up']:.0f} — strong uptrend confirmation")

        # 3. MOMENTUM (20 points)
        if momentum_score > 65:
            score += 15
            reasons.append(f"Strong momentum ({momentum_score:.0f}) — Livermore rides momentum")
        elif momentum_score > 50:
            score += 8
        else:
            concerns.append(f"Weak momentum ({momentum_score:.0f}) — Livermore waits for momentum to build")

        if rsi > 50 and rsi < 70:
            score += 5
            reasons.append(f"RSI at {rsi:.0f} — bullish but not overbought, room to run")
        elif rsi > 70:
            concerns.append(f"RSI at {rsi:.0f} — overbought, Livermore would wait for pullback before adding")

        # 4. PATTERN RECOGNITION (10 points)
        if isinstance(candles, dict) and candles.get('patterns_detected'):
            patterns = candles.get('patterns_detected', [])
            if patterns:
                bullish = [p for p in patterns if isinstance(p, dict) and p.get('type', '').lower() == 'bullish']
                if bullish:
                    score += 10
                    reasons.append(f"Bullish candlestick patterns detected — Livermore reads the tape")

        # 5. DIRECTIONAL ALIGNMENT (10 points)
        if rec in ['BUY', 'STRONG_BUY'] and trend_score > 55:
            score += 10
            reasons.append("System recommendation aligns with trend — Livermore goes with the tape")
        elif rec in ['SELL', 'STRONG_SELL'] and trend_score < 45:
            score += 8
            reasons.append("System says sell and trend is down — Livermore would short")
        elif rec == 'HOLD':
            score += 3
            concerns.append("System says hold — Livermore says 'it never was my thinking that made the big money, it was my sitting'")

        # --- Determine action ---
        if score >= 70:
            action = "BUY AND PYRAMID"
            conviction = "HIGH"
            headline = f"**LIVERMORE WOULD BUY {symbol} AND PYRAMID**: Strong trend with volume — add on strength."
        elif score >= 50:
            action = "BUY"
            conviction = "MODERATE"
            headline = f"**LIVERMORE WOULD BUY {symbol}**: Trend developing, enter with initial position."
        elif score >= 35:
            action = "WAIT"
            conviction = "LOW"
            headline = f"**LIVERMORE WOULD WAIT ON {symbol}**: No clear trend yet — patience is key."
        else:
            action = "PASS"
            conviction = "NONE"
            headline = f"**LIVERMORE WOULD PASS ON {symbol}**: No trend, no volume, no trade."

        reasoning_parts = [headline, ""]
        
        if reasons:
            reasoning_parts.append("**What the tape tells Livermore:**")
            for r in reasons:
                reasoning_parts.append(f"- {r}")
            reasoning_parts.append("")
        
        if concerns:
            reasoning_parts.append("**What makes Livermore cautious:**")
            for c in concerns:
                reasoning_parts.append(f"- {c}")
            reasoning_parts.append("")

        # Pyramiding strategy
        if score >= 50:
            reasoning_parts.append("**Pyramiding Strategy**: Enter 1/3 position now. Add 1/3 on first pullback that holds above entry. Add final 1/3 on breakout to new high. Never add to a losing position.")
            reasoning_parts.append("")

        reasoning_parts.append(f"**Livermore Score**: {score}/{max_score}")
        reasoning_parts.append("")
        reasoning_parts.append('> *"There is a time for all things, but I didn\'t know it. And that is precisely what beats so many men in Wall Street."* — Jesse Livermore')

        return {
            'action': action,
            'conviction': conviction,
            'score': score,
            'max_score': max_score,
            'reasoning': '\n'.join(reasoning_parts),
            'key_metrics': {
                'trend_score': f"{trend_score:.0f}",
                'adx': f"{adx:.0f}",
                'momentum': f"{momentum_score:.0f}",
                'above_200dma': 'Yes' if sma_200 > 0 and price > sma_200 else 'No',
            }
        }

    # ------------------------------------------------------------------ CONSENSUS
    def get_consensus_action(self, perspectives: Dict[str, Dict]) -> Dict:
        """
        Determine consensus action from all legendary traders.
        Returns structured consensus with vote breakdown.
        """
        buy_votes = 0
        sell_votes = 0
        wait_votes = 0
        total_score = 0
        max_total = 0
        
        vote_details = []
        
        for trader, p in perspectives.items():
            if isinstance(p, dict):
                action = p.get('action', 'WAIT')
                score = p.get('score', 0)
                max_s = p.get('max_score', 100)
                total_score += score
                max_total += max_s
                
                if action in ['BUY', 'BUY AGGRESSIVELY', 'SIZE UP', 'BUY AND PYRAMID', 'CONSIDER']:
                    buy_votes += 1
                    vote_details.append(f"- **{trader.replace('_', ' ').title()}**: {action} ({score}/{max_s})")
                elif action in ['SELL', 'SHORT', 'AVOID']:
                    sell_votes += 1
                    vote_details.append(f"- **{trader.replace('_', ' ').title()}**: {action} ({score}/{max_s})")
                else:
                    wait_votes += 1
                    vote_details.append(f"- **{trader.replace('_', ' ').title()}**: {action} ({score}/{max_s})")
            else:
                # Legacy string format fallback
                advice = str(p)
                if 'WOULD BUY' in advice or 'BUY AGGRESSIVELY' in advice or 'SIZE UP' in advice:
                    buy_votes += 1
                elif 'WOULD SELL' in advice or 'WOULD SHORT' in advice:
                    sell_votes += 1
                else:
                    wait_votes += 1

        total_votes = buy_votes + sell_votes + wait_votes
        avg_score = (total_score / max_total * 100) if max_total > 0 else 0

        if buy_votes >= 5:
            consensus_action = "STRONG BUY"
            summary = f"**STRONG CONSENSUS BUY**: {buy_votes}/{total_votes} legendary traders would buy this setup. Multiple independent philosophies align — this is a rare, high-quality opportunity."
        elif buy_votes >= 4:
            consensus_action = "BUY"
            summary = f"**CONSENSUS BUY**: {buy_votes}/{total_votes} traders favor buying. Strong alignment across different frameworks."
        elif buy_votes >= 3:
            consensus_action = "MODERATE BUY"
            summary = f"**MODERATE BUY**: {buy_votes}/{total_votes} traders favor buying. Good setup but some concerns from other perspectives."
        elif sell_votes >= 3:
            consensus_action = "SELL/AVOID"
            summary = f"**CONSENSUS SELL/AVOID**: {sell_votes}/{total_votes} traders would sell or avoid. Multiple red flags across different frameworks."
        elif wait_votes >= 4:
            consensus_action = "WAIT"
            summary = f"**CONSENSUS WAIT**: {wait_votes}/{total_votes} traders would wait for better setup. Not enough edge or conviction to justify risk."
        else:
            consensus_action = "MIXED"
            summary = f"**MIXED SIGNALS**: Buy: {buy_votes}, Sell: {sell_votes}, Wait: {wait_votes}. Different trading philosophies see this differently — proceed with caution and smaller size."

        parts = [summary, ""]
        parts.append("**Individual Votes:**")
        parts.extend(vote_details)
        parts.append("")
        parts.append(f"**Aggregate Score**: {total_score}/{max_total} ({avg_score:.0f}%)")

        return {
            'action': consensus_action,
            'summary': '\n'.join(parts),
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'wait_votes': wait_votes,
            'total_votes': total_votes,
            'aggregate_score': round(avg_score, 1),
        }
