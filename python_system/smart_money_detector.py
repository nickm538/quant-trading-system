#!/usr/bin/env python3
"""
Smart Money Detector - Institutional Activity Analysis
Detects likely dark pool/block trade activity using FREE data sources.

This module analyzes:
1. Volume anomalies (unusual volume spikes)
2. Price-volume divergence (high volume, low price movement = accumulation)
3. Options flow analysis (unusual options activity)
4. Institutional ownership changes
5. Insider transactions (SEC Form 4)
6. Short interest changes
7. Block trade detection via tick analysis

NO PAID APIs REQUIRED - Uses yfinance, SEC EDGAR, and calculated metrics.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SmartMoneyDetector:
    """
    Detects institutional/smart money activity using free data sources.
    Provides proxy analysis for dark pool and block trade activity.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive smart money analysis for a symbol.
        Returns institutional activity signals and block trade proxies.
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'smart_money_score': 0,  # 0-100 scale
            'signal': 'NEUTRAL',
            'confidence': 0,
            'analysis': {}
        }
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            hist_1y = ticker.history(period="1y")
            
            if hist.empty:
                result['error'] = "No price data available"
                return result
            
            # 1. Volume Analysis
            volume_analysis = self._analyze_volume(hist, hist_1y)
            result['analysis']['volume'] = volume_analysis
            
            # 2. Price-Volume Divergence (Accumulation/Distribution)
            accumulation = self._detect_accumulation(hist)
            result['analysis']['accumulation'] = accumulation
            
            # 3. Institutional Ownership
            institutional = self._analyze_institutional(ticker, info)
            result['analysis']['institutional'] = institutional
            
            # 4. Insider Transactions
            insider = self._analyze_insider_activity(ticker)
            result['analysis']['insider'] = insider
            
            # 5. Short Interest Analysis
            short_interest = self._analyze_short_interest(info)
            result['analysis']['short_interest'] = short_interest
            
            # 6. Options Flow Proxy (Put/Call from info)
            options_flow = self._analyze_options_flow(ticker, info)
            result['analysis']['options_flow'] = options_flow
            
            # 7. Block Trade Detection (Large volume bars)
            block_trades = self._detect_block_trades(hist)
            result['analysis']['block_trades'] = block_trades
            
            # 8. Money Flow Analysis
            money_flow = self._calculate_money_flow(hist)
            result['analysis']['money_flow'] = money_flow
            
            # Calculate overall smart money score
            score, signal, confidence = self._calculate_smart_money_score(result['analysis'])
            result['smart_money_score'] = score
            result['signal'] = signal
            result['confidence'] = confidence
            
            # Generate summary
            result['summary'] = self._generate_summary(result)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Smart money analysis failed for {symbol}: {e}")
        
        return result
    
    def _analyze_volume(self, hist: pd.DataFrame, hist_1y: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for unusual activity."""
        try:
            # Recent volume metrics
            recent_vol = hist['Volume'].tail(5).mean()
            avg_vol_20 = hist['Volume'].tail(20).mean()
            avg_vol_50 = hist['Volume'].tail(50).mean() if len(hist) >= 50 else avg_vol_20
            
            # 1-year average for comparison
            yearly_avg = hist_1y['Volume'].mean() if not hist_1y.empty else avg_vol_50
            
            # Volume ratios
            vol_ratio_20 = recent_vol / avg_vol_20 if avg_vol_20 > 0 else 1
            vol_ratio_50 = recent_vol / avg_vol_50 if avg_vol_50 > 0 else 1
            vol_ratio_yearly = recent_vol / yearly_avg if yearly_avg > 0 else 1
            
            # Detect volume spikes (potential block trades)
            vol_std = hist['Volume'].tail(20).std()
            vol_z_score = (hist['Volume'].iloc[-1] - avg_vol_20) / vol_std if vol_std > 0 else 0
            
            # Count unusual volume days in last 10 days
            unusual_days = sum(1 for v in hist['Volume'].tail(10) if v > avg_vol_20 * 1.5)
            
            return {
                'recent_avg_volume': int(recent_vol),
                '20d_avg_volume': int(avg_vol_20),
                '50d_avg_volume': int(avg_vol_50),
                'yearly_avg_volume': int(yearly_avg),
                'volume_ratio_20d': round(vol_ratio_20, 2),
                'volume_ratio_50d': round(vol_ratio_50, 2),
                'volume_ratio_yearly': round(vol_ratio_yearly, 2),
                'volume_z_score': round(vol_z_score, 2),
                'unusual_volume_days_10d': unusual_days,
                'signal': 'HIGH_ACTIVITY' if vol_ratio_20 > 1.5 else ('LOW_ACTIVITY' if vol_ratio_20 < 0.7 else 'NORMAL')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_accumulation(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect accumulation/distribution patterns.
        High volume + small price change = institutional accumulation
        """
        try:
            # Calculate daily metrics
            hist = hist.copy()
            hist['price_change_pct'] = hist['Close'].pct_change().abs() * 100
            hist['vol_normalized'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
            
            # Accumulation score: high volume, low price change
            # Distribution score: high volume, high price change
            recent = hist.tail(10)
            
            accumulation_days = 0
            distribution_days = 0
            
            for _, row in recent.iterrows():
                vol_norm = row.get('vol_normalized', 1)
                price_chg = row.get('price_change_pct', 0)
                
                if pd.notna(vol_norm) and pd.notna(price_chg):
                    if vol_norm > 1.3 and price_chg < 1.0:
                        accumulation_days += 1
                    elif vol_norm > 1.3 and price_chg > 2.0:
                        distribution_days += 1
            
            # Calculate Accumulation/Distribution Line
            ad_line = self._calculate_ad_line(hist)
            
            # Chaikin Money Flow
            cmf = self._calculate_cmf(hist)
            
            signal = 'ACCUMULATION' if accumulation_days > distribution_days + 2 else (
                'DISTRIBUTION' if distribution_days > accumulation_days + 2 else 'NEUTRAL'
            )
            
            return {
                'accumulation_days_10d': accumulation_days,
                'distribution_days_10d': distribution_days,
                'ad_line_trend': ad_line,
                'chaikin_money_flow': cmf,
                'signal': signal
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_ad_line(self, hist: pd.DataFrame) -> str:
        """Calculate Accumulation/Distribution Line trend."""
        try:
            # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
            mfm = ((hist['Close'] - hist['Low']) - (hist['High'] - hist['Close'])) / (hist['High'] - hist['Low'])
            mfm = mfm.fillna(0)
            
            # Money Flow Volume = MFM * Volume
            mfv = mfm * hist['Volume']
            
            # AD Line = cumulative sum of MFV
            ad_line = mfv.cumsum()
            
            # Trend: compare last 5 days to previous 5 days
            recent = ad_line.tail(5).mean()
            previous = ad_line.tail(10).head(5).mean()
            
            if recent > previous * 1.05:
                return 'RISING'
            elif recent < previous * 0.95:
                return 'FALLING'
            else:
                return 'FLAT'
        except:
            return 'UNKNOWN'
    
    def _calculate_cmf(self, hist: pd.DataFrame, period: int = 20) -> float:
        """Calculate Chaikin Money Flow."""
        try:
            recent = hist.tail(period)
            
            # Money Flow Multiplier
            mfm = ((recent['Close'] - recent['Low']) - (recent['High'] - recent['Close'])) / (recent['High'] - recent['Low'])
            mfm = mfm.fillna(0)
            
            # Money Flow Volume
            mfv = mfm * recent['Volume']
            
            # CMF = Sum(MFV) / Sum(Volume)
            cmf = mfv.sum() / recent['Volume'].sum() if recent['Volume'].sum() > 0 else 0
            
            return round(cmf, 4)
        except:
            return 0
    
    def _analyze_institutional(self, ticker, info: Dict) -> Dict[str, Any]:
        """Analyze institutional ownership and changes."""
        try:
            inst_ownership = info.get('heldPercentInstitutions', 0)
            insider_ownership = info.get('heldPercentInsiders', 0)
            
            # Get institutional holders
            inst_holders = ticker.institutional_holders
            top_holders = []
            total_inst_shares = 0
            
            if inst_holders is not None and not inst_holders.empty:
                for _, row in inst_holders.head(5).iterrows():
                    holder = {
                        'name': row.get('Holder', 'Unknown'),
                        'shares': int(row.get('Shares', 0)),
                        'value': row.get('Value', 0),
                        'pct_out': row.get('% Out', 0)
                    }
                    top_holders.append(holder)
                    total_inst_shares += holder['shares']
            
            # Concentration risk
            concentration = 'HIGH' if inst_ownership and inst_ownership > 0.8 else (
                'MODERATE' if inst_ownership and inst_ownership > 0.5 else 'LOW'
            )
            
            return {
                'institutional_ownership_pct': round(inst_ownership * 100, 2) if inst_ownership else 0,
                'insider_ownership_pct': round(insider_ownership * 100, 2) if insider_ownership else 0,
                'top_5_holders': top_holders,
                'concentration': concentration,
                'signal': 'INSTITUTIONAL_HEAVY' if inst_ownership and inst_ownership > 0.7 else 'NORMAL'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_insider_activity(self, ticker) -> Dict[str, Any]:
        """Analyze recent insider transactions."""
        try:
            insider_txns = ticker.insider_transactions
            
            if insider_txns is None or insider_txns.empty:
                return {
                    'recent_transactions': [],
                    'net_activity': 'NO_DATA',
                    'signal': 'NEUTRAL'
                }
            
            transactions = []
            buy_value = 0
            sell_value = 0
            
            for _, row in insider_txns.head(10).iterrows():
                txn_type = str(row.get('Transaction', '')).upper()
                value = row.get('Value', 0) or 0
                shares = row.get('Shares', 0) or 0
                
                txn = {
                    'insider': row.get('Insider', 'Unknown'),
                    'relation': row.get('Relation', 'Unknown'),
                    'transaction': txn_type,
                    'shares': int(shares),
                    'value': value,
                    'date': str(row.get('Start Date', 'Unknown'))
                }
                transactions.append(txn)
                
                if 'BUY' in txn_type or 'PURCHASE' in txn_type:
                    buy_value += value if value else 0
                elif 'SELL' in txn_type or 'SALE' in txn_type:
                    sell_value += value if value else 0
            
            net_activity = 'NET_BUYING' if buy_value > sell_value * 1.2 else (
                'NET_SELLING' if sell_value > buy_value * 1.2 else 'MIXED'
            )
            
            signal = 'BULLISH' if net_activity == 'NET_BUYING' else (
                'BEARISH' if net_activity == 'NET_SELLING' else 'NEUTRAL'
            )
            
            return {
                'recent_transactions': transactions[:5],
                'total_buy_value': buy_value,
                'total_sell_value': sell_value,
                'net_activity': net_activity,
                'signal': signal
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_short_interest(self, info: Dict) -> Dict[str, Any]:
        """Analyze short interest data."""
        try:
            short_pct = info.get('shortPercentOfFloat', 0)
            shares_short = info.get('sharesShort', 0)
            short_ratio = info.get('shortRatio', 0)  # Days to cover
            shares_short_prior = info.get('sharesShortPriorMonth', 0)
            
            # Calculate change
            short_change = 0
            if shares_short and shares_short_prior:
                short_change = (shares_short - shares_short_prior) / shares_short_prior * 100
            
            # Signal based on short interest
            signal = 'HIGH_SHORT' if short_pct and short_pct > 0.2 else (
                'MODERATE_SHORT' if short_pct and short_pct > 0.1 else 'LOW_SHORT'
            )
            
            # Squeeze potential
            squeeze_potential = 'HIGH' if short_pct and short_pct > 0.2 and short_ratio and short_ratio > 5 else (
                'MODERATE' if short_pct and short_pct > 0.15 else 'LOW'
            )
            
            return {
                'short_percent_float': round(short_pct * 100, 2) if short_pct else 0,
                'shares_short': shares_short or 0,
                'days_to_cover': round(short_ratio, 2) if short_ratio else 0,
                'short_change_vs_prior_month': round(short_change, 2),
                'squeeze_potential': squeeze_potential,
                'signal': signal
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_options_flow(self, ticker, info: Dict) -> Dict[str, Any]:
        """Analyze options flow for smart money signals."""
        try:
            # Get options data
            options_dates = ticker.options
            
            if not options_dates:
                return {'signal': 'NO_OPTIONS_DATA'}
            
            # Analyze nearest expiration
            nearest_exp = options_dates[0]
            opt_chain = ticker.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Calculate put/call ratio by volume
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            pc_ratio = put_volume / call_volume if call_volume > 0 else 1
            
            # Calculate put/call ratio by open interest
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            
            pc_ratio_oi = put_oi / call_oi if call_oi > 0 else 1
            
            # Detect unusual options activity
            avg_call_vol = calls['volume'].mean() if 'volume' in calls.columns else 0
            avg_put_vol = puts['volume'].mean() if 'volume' in puts.columns else 0
            
            # Find high volume strikes (potential smart money)
            unusual_calls = calls[calls['volume'] > avg_call_vol * 3] if avg_call_vol > 0 else pd.DataFrame()
            unusual_puts = puts[puts['volume'] > avg_put_vol * 3] if avg_put_vol > 0 else pd.DataFrame()
            
            # Signal interpretation
            # Low P/C ratio = bullish (more calls)
            # High P/C ratio = bearish (more puts)
            signal = 'BULLISH' if pc_ratio < 0.7 else ('BEARISH' if pc_ratio > 1.3 else 'NEUTRAL')
            
            return {
                'put_call_ratio_volume': round(pc_ratio, 2),
                'put_call_ratio_oi': round(pc_ratio_oi, 2),
                'total_call_volume': int(call_volume),
                'total_put_volume': int(put_volume),
                'total_call_oi': int(call_oi),
                'total_put_oi': int(put_oi),
                'unusual_call_strikes': len(unusual_calls),
                'unusual_put_strikes': len(unusual_puts),
                'nearest_expiration': nearest_exp,
                'signal': signal
            }
        except Exception as e:
            return {'error': str(e), 'signal': 'ERROR'}
    
    def _detect_block_trades(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect likely block trades from volume patterns.
        Block trades are typically >10,000 shares executed at once.
        """
        try:
            # Calculate average volume per bar
            avg_vol = hist['Volume'].mean()
            std_vol = hist['Volume'].std()
            
            # Identify potential block trade days (volume > 2 std above mean)
            threshold = avg_vol + (2 * std_vol)
            
            block_days = []
            for idx, row in hist.tail(20).iterrows():
                if row['Volume'] > threshold:
                    price_impact = abs(row['Close'] - row['Open']) / row['Open'] * 100
                    block_days.append({
                        'date': str(idx.date()) if hasattr(idx, 'date') else str(idx),
                        'volume': int(row['Volume']),
                        'volume_ratio': round(row['Volume'] / avg_vol, 2),
                        'price_impact_pct': round(price_impact, 2),
                        'likely_type': 'ACCUMULATION' if price_impact < 1 else 'MOMENTUM'
                    })
            
            # Recent block activity
            recent_blocks = len([b for b in block_days if 'ACCUMULATION' in b.get('likely_type', '')])
            
            return {
                'potential_block_days_20d': len(block_days),
                'accumulation_blocks': recent_blocks,
                'momentum_blocks': len(block_days) - recent_blocks,
                'recent_blocks': block_days[:5],
                'signal': 'HIGH_BLOCK_ACTIVITY' if len(block_days) > 5 else (
                    'MODERATE_BLOCK_ACTIVITY' if len(block_days) > 2 else 'LOW_BLOCK_ACTIVITY'
                )
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_money_flow(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Money Flow Index and related metrics."""
        try:
            # Typical Price
            tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
            
            # Raw Money Flow
            rmf = tp * hist['Volume']
            
            # Positive/Negative Money Flow
            tp_diff = tp.diff()
            pos_mf = rmf.where(tp_diff > 0, 0)
            neg_mf = rmf.where(tp_diff < 0, 0)
            
            # 14-period MFI
            period = 14
            pos_mf_sum = pos_mf.rolling(period).sum()
            neg_mf_sum = neg_mf.rolling(period).sum()
            
            mfi = 100 - (100 / (1 + pos_mf_sum / neg_mf_sum.replace(0, 1)))
            current_mfi = mfi.iloc[-1] if not mfi.empty else 50
            
            # Interpretation
            signal = 'OVERBOUGHT' if current_mfi > 80 else (
                'OVERSOLD' if current_mfi < 20 else 'NEUTRAL'
            )
            
            # Net money flow (last 5 days)
            net_flow_5d = (pos_mf.tail(5).sum() - neg_mf.tail(5).sum())
            
            return {
                'mfi_14': round(current_mfi, 2),
                'net_money_flow_5d': round(net_flow_5d, 0),
                'signal': signal,
                'interpretation': 'SMART_MONEY_BUYING' if current_mfi < 30 else (
                    'SMART_MONEY_SELLING' if current_mfi > 70 else 'NEUTRAL'
                )
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_smart_money_score(self, analysis: Dict) -> tuple:
        """
        Calculate overall smart money score from all analysis components.
        Returns (score, signal, confidence)
        """
        score = 50  # Start neutral
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Volume analysis (+/- 10 points)
        vol = analysis.get('volume', {})
        if vol.get('signal') == 'HIGH_ACTIVITY':
            score += 5
            total_signals += 1
        
        # Accumulation/Distribution (+/- 15 points)
        acc = analysis.get('accumulation', {})
        if acc.get('signal') == 'ACCUMULATION':
            score += 15
            bullish_signals += 1
        elif acc.get('signal') == 'DISTRIBUTION':
            score -= 15
            bearish_signals += 1
        total_signals += 1
        
        # Chaikin Money Flow (+/- 10 points)
        cmf = acc.get('chaikin_money_flow', 0)
        if cmf > 0.1:
            score += 10
            bullish_signals += 1
        elif cmf < -0.1:
            score -= 10
            bearish_signals += 1
        total_signals += 1
        
        # Insider activity (+/- 15 points)
        insider = analysis.get('insider', {})
        if insider.get('signal') == 'BULLISH':
            score += 15
            bullish_signals += 1
        elif insider.get('signal') == 'BEARISH':
            score -= 15
            bearish_signals += 1
        total_signals += 1
        
        # Short interest (+/- 10 points)
        short = analysis.get('short_interest', {})
        squeeze = short.get('squeeze_potential', 'LOW')
        if squeeze == 'HIGH':
            score += 10  # Potential squeeze = bullish
            bullish_signals += 1
        total_signals += 1
        
        # Options flow (+/- 10 points)
        options = analysis.get('options_flow', {})
        if options.get('signal') == 'BULLISH':
            score += 10
            bullish_signals += 1
        elif options.get('signal') == 'BEARISH':
            score -= 10
            bearish_signals += 1
        total_signals += 1
        
        # Block trades (+/- 5 points)
        blocks = analysis.get('block_trades', {})
        if blocks.get('signal') == 'HIGH_BLOCK_ACTIVITY':
            score += 5
        total_signals += 1
        
        # Money flow (+/- 10 points)
        mf = analysis.get('money_flow', {})
        if mf.get('interpretation') == 'SMART_MONEY_BUYING':
            score += 10
            bullish_signals += 1
        elif mf.get('interpretation') == 'SMART_MONEY_SELLING':
            score -= 10
            bearish_signals += 1
        total_signals += 1
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        # Determine signal
        if score >= 70:
            signal = 'STRONG_ACCUMULATION'
        elif score >= 60:
            signal = 'ACCUMULATION'
        elif score <= 30:
            signal = 'STRONG_DISTRIBUTION'
        elif score <= 40:
            signal = 'DISTRIBUTION'
        else:
            signal = 'NEUTRAL'
        
        # Calculate confidence
        signal_agreement = max(bullish_signals, bearish_signals) / total_signals if total_signals > 0 else 0
        confidence = round(signal_agreement * 100, 1)
        
        return score, signal, confidence
    
    def _generate_summary(self, result: Dict) -> str:
        """Generate human-readable summary of smart money analysis."""
        score = result.get('smart_money_score', 50)
        signal = result.get('signal', 'NEUTRAL')
        analysis = result.get('analysis', {})
        
        summary_parts = []
        
        # Overall assessment
        if score >= 70:
            summary_parts.append(f"🟢 STRONG SMART MONEY BUYING DETECTED (Score: {score}/100)")
        elif score >= 60:
            summary_parts.append(f"🟢 Smart money accumulation signals present (Score: {score}/100)")
        elif score <= 30:
            summary_parts.append(f"🔴 STRONG SMART MONEY SELLING DETECTED (Score: {score}/100)")
        elif score <= 40:
            summary_parts.append(f"🔴 Smart money distribution signals present (Score: {score}/100)")
        else:
            summary_parts.append(f"⚪ Neutral smart money activity (Score: {score}/100)")
        
        # Key findings
        acc = analysis.get('accumulation', {})
        if acc.get('signal') == 'ACCUMULATION':
            summary_parts.append("• High volume with low price impact suggests institutional accumulation")
        elif acc.get('signal') == 'DISTRIBUTION':
            summary_parts.append("• Volume patterns suggest institutional distribution")
        
        insider = analysis.get('insider', {})
        if insider.get('net_activity') == 'NET_BUYING':
            summary_parts.append("• Insiders are NET BUYERS - bullish signal")
        elif insider.get('net_activity') == 'NET_SELLING':
            summary_parts.append("• Insiders are NET SELLERS - caution advised")
        
        short = analysis.get('short_interest', {})
        if short.get('squeeze_potential') == 'HIGH':
            summary_parts.append(f"• HIGH SHORT SQUEEZE POTENTIAL ({short.get('short_percent_float', 0)}% short)")
        
        options = analysis.get('options_flow', {})
        pc_ratio = options.get('put_call_ratio_volume', 1)
        if pc_ratio < 0.7:
            summary_parts.append(f"• Bullish options flow (P/C ratio: {pc_ratio})")
        elif pc_ratio > 1.3:
            summary_parts.append(f"• Bearish options flow (P/C ratio: {pc_ratio})")
        
        blocks = analysis.get('block_trades', {})
        if blocks.get('potential_block_days_20d', 0) > 3:
            summary_parts.append(f"• {blocks.get('potential_block_days_20d')} potential block trade days detected")
        
        return "\n".join(summary_parts)


# Standalone test
if __name__ == "__main__":
    detector = SmartMoneyDetector()
    result = detector.analyze("NVDA")
    
    import json
    print(json.dumps(result, indent=2, default=str))
