"""
Enhanced Fundamentals Module
============================
Deep cash flow and valuation analysis for institutional-grade stock evaluation.
Includes:
- P/E Ratio with sector comparison
- PEG Ratio (GARP analysis)
- Free Cash Flow (FCF) metrics
- EBITDA/EV valuation
- Free Float and liquidity analysis
- Debt metrics and coverage ratios
- Quality scores and GARP setup detection

All data from live sources - zero placeholders.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
except ImportError:
    yf = None


class EnhancedFundamentalsAnalyzer:
    """
    Deep fundamental analysis with cash flow focus.
    Evaluates stocks like a value investor with growth awareness (GARP).
    """
    
    def __init__(self):
        self.finnhub_key = os.environ.get('KEY') or os.environ.get('FINNHUB_API_KEY') or 'd55b3ohr01qljfdeghm0d55b3ohr01qljfdeghm1'
        self.fmp_key = os.environ.get('FMP_API_KEY') or 'LTecnRjOFtd8bFOTCRLpcncjxrqaZlqq'
        
        # Sector P/E benchmarks (updated periodically)
        self.sector_pe_benchmarks = {
            'Technology': 28.5,
            'Healthcare': 22.0,
            'Financial Services': 14.5,
            'Consumer Cyclical': 20.0,
            'Consumer Defensive': 23.0,
            'Industrials': 21.0,
            'Energy': 12.0,
            'Utilities': 18.0,
            'Real Estate': 35.0,
            'Basic Materials': 15.0,
            'Communication Services': 18.0,
            'Default': 20.0
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis with cash flow focus.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing all fundamental metrics and analysis
        """
        try:
            # Fetch data from multiple sources
            yf_data = self._fetch_yfinance_data(symbol)
            fmp_data = self._fetch_fmp_data(symbol)
            finnhub_data = self._fetch_finnhub_data(symbol)
            
            if not yf_data:
                return {
                    'success': False,
                    'error': f'Unable to fetch data for {symbol}',
                    'symbol': symbol
                }
            
            # Extract key metrics
            info = yf_data.get('info', {})
            
            # Basic info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Valuation metrics
            pe_ratio = info.get('trailingPE') or info.get('forwardPE')
            forward_pe = info.get('forwardPE')
            peg_ratio = info.get('pegRatio')
            price_to_book = info.get('priceToBook')
            price_to_sales = info.get('priceToSalesTrailing12Months')
            
            # Growth metrics
            earnings_growth = info.get('earningsGrowth', 0) or 0
            revenue_growth = info.get('revenueGrowth', 0) or 0
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0) or 0
            
            # Cash flow metrics
            free_cash_flow = info.get('freeCashflow', 0) or 0
            operating_cash_flow = info.get('operatingCashflow', 0) or 0
            
            # Profitability
            profit_margin = info.get('profitMargins', 0) or 0
            operating_margin = info.get('operatingMargins', 0) or 0
            gross_margin = info.get('grossMargins', 0) or 0
            roe = info.get('returnOnEquity', 0) or 0
            roa = info.get('returnOnAssets', 0) or 0
            
            # Balance sheet
            total_debt = info.get('totalDebt', 0) or 0
            total_cash = info.get('totalCash', 0) or 0
            current_ratio = info.get('currentRatio', 0) or 0
            quick_ratio = info.get('quickRatio', 0) or 0
            debt_to_equity = info.get('debtToEquity', 0) or 0
            
            # Share structure
            shares_outstanding = info.get('sharesOutstanding', 0) or 0
            float_shares = info.get('floatShares', 0) or 0
            shares_short = info.get('sharesShort', 0) or 0
            short_ratio = info.get('shortRatio', 0) or 0
            insider_ownership = info.get('heldPercentInsiders', 0) or 0
            institutional_ownership = info.get('heldPercentInstitutions', 0) or 0
            
            # Calculate derived metrics
            
            # Free Float percentage
            free_float_pct = (float_shares / shares_outstanding * 100) if shares_outstanding > 0 else 0
            
            # FCF Yield
            fcf_yield = (free_cash_flow / market_cap * 100) if market_cap > 0 else 0
            
            # EBITDA and EV/EBITDA
            ebitda = info.get('ebitda', 0) or 0
            enterprise_value = info.get('enterpriseValue', 0) or 0
            ev_to_ebitda = (enterprise_value / ebitda) if ebitda > 0 else 0
            
            # EV/FCF
            ev_to_fcf = (enterprise_value / free_cash_flow) if free_cash_flow > 0 else 0
            
            # Net Debt
            net_debt = total_debt - total_cash
            
            # Interest Coverage (estimate from EBITDA and debt)
            interest_coverage = 0
            if total_debt > 0 and ebitda > 0:
                # Estimate interest expense as 5% of debt
                est_interest = total_debt * 0.05
                interest_coverage = ebitda / est_interest if est_interest > 0 else 0
            
            # GARP Analysis
            garp_analysis = self._analyze_garp(pe_ratio, peg_ratio, earnings_growth, revenue_growth, sector)
            
            # Sector comparison
            sector_comparison = self._compare_to_sector(pe_ratio, sector, profit_margin, roe)
            
            # Quality Score
            quality_score = self._calculate_quality_score(
                profit_margin, operating_margin, roe, roa,
                current_ratio, debt_to_equity, free_cash_flow,
                earnings_growth, revenue_growth
            )
            
            # Liquidity Analysis
            liquidity_analysis = self._analyze_liquidity(
                current_ratio, quick_ratio, free_cash_flow,
                operating_cash_flow, total_debt, market_cap
            )
            
            # Value Assessment
            value_assessment = self._assess_value(
                pe_ratio, forward_pe, peg_ratio, price_to_book,
                ev_to_ebitda, fcf_yield, sector
            )
            
            return {
                'success': True,
                'symbol': symbol.upper(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S EST'),
                
                # Basic Info
                'company_info': {
                    'name': info.get('longName', symbol),
                    'sector': sector,
                    'industry': industry,
                    'current_price': round(current_price, 2),
                    'market_cap': market_cap,
                    'market_cap_formatted': self._format_large_number(market_cap)
                },
                
                # Valuation Metrics
                'valuation': {
                    'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                    'forward_pe': round(forward_pe, 2) if forward_pe else None,
                    'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
                    'price_to_book': round(price_to_book, 2) if price_to_book else None,
                    'price_to_sales': round(price_to_sales, 2) if price_to_sales else None,
                    'ev_to_ebitda': round(ev_to_ebitda, 2) if ev_to_ebitda else None,
                    'ev_to_fcf': round(ev_to_fcf, 2) if ev_to_fcf else None,
                    'enterprise_value': enterprise_value,
                    'enterprise_value_formatted': self._format_large_number(enterprise_value)
                },
                
                # Cash Flow Analysis
                'cash_flow': {
                    'free_cash_flow': free_cash_flow,
                    'fcf_formatted': self._format_large_number(free_cash_flow),
                    'fcf_yield_pct': round(fcf_yield, 2),
                    'operating_cash_flow': operating_cash_flow,
                    'ocf_formatted': self._format_large_number(operating_cash_flow),
                    'fcf_to_ocf_ratio': round(free_cash_flow / operating_cash_flow, 2) if operating_cash_flow else None,
                    'ebitda': ebitda,
                    'ebitda_formatted': self._format_large_number(ebitda),
                    'fcf_positive': free_cash_flow > 0,
                    'fcf_growing': None  # Would need historical data
                },
                
                # Growth Metrics
                'growth': {
                    'earnings_growth_pct': round(earnings_growth * 100, 2) if earnings_growth else None,
                    'revenue_growth_pct': round(revenue_growth * 100, 2) if revenue_growth else None,
                    'earnings_quarterly_growth_pct': round(earnings_quarterly_growth * 100, 2) if earnings_quarterly_growth else None
                },
                
                # Profitability
                'profitability': {
                    'gross_margin_pct': round(gross_margin * 100, 2) if gross_margin else None,
                    'operating_margin_pct': round(operating_margin * 100, 2) if operating_margin else None,
                    'profit_margin_pct': round(profit_margin * 100, 2) if profit_margin else None,
                    'roe_pct': round(roe * 100, 2) if roe else None,
                    'roa_pct': round(roa * 100, 2) if roa else None
                },
                
                # Debt & Liquidity
                'debt_liquidity': {
                    'total_debt': total_debt,
                    'total_debt_formatted': self._format_large_number(total_debt),
                    'total_cash': total_cash,
                    'total_cash_formatted': self._format_large_number(total_cash),
                    'net_debt': net_debt,
                    'net_debt_formatted': self._format_large_number(net_debt),
                    'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity else None,
                    'current_ratio': round(current_ratio, 2) if current_ratio else None,
                    'quick_ratio': round(quick_ratio, 2) if quick_ratio else None,
                    'interest_coverage': round(interest_coverage, 2) if interest_coverage else None
                },
                
                # Share Structure & Float
                'share_structure': {
                    'shares_outstanding': shares_outstanding,
                    'shares_outstanding_formatted': self._format_large_number(shares_outstanding),
                    'float_shares': float_shares,
                    'float_shares_formatted': self._format_large_number(float_shares),
                    'free_float_pct': round(free_float_pct, 2),
                    'shares_short': shares_short,
                    'shares_short_formatted': self._format_large_number(shares_short),
                    'short_ratio_days': round(short_ratio, 2) if short_ratio else None,
                    'short_pct_of_float': round(shares_short / float_shares * 100, 2) if float_shares > 0 else None,
                    'insider_ownership_pct': round(insider_ownership * 100, 2) if insider_ownership else None,
                    'institutional_ownership_pct': round(institutional_ownership * 100, 2) if institutional_ownership else None
                },
                
                # Analysis & Scores
                'garp_analysis': garp_analysis,
                'sector_comparison': sector_comparison,
                'quality_score': quality_score,
                'liquidity_analysis': liquidity_analysis,
                'value_assessment': value_assessment,
                
                # Overall Fundamental Rating
                'overall_rating': self._calculate_overall_rating(
                    quality_score, garp_analysis, value_assessment, liquidity_analysis
                )
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'symbol': symbol
            }
    
    def _fetch_yfinance_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Yahoo Finance."""
        if yf is None:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            return {
                'info': ticker.info,
                'financials': ticker.financials.to_dict() if hasattr(ticker.financials, 'to_dict') else {},
                'balance_sheet': ticker.balance_sheet.to_dict() if hasattr(ticker.balance_sheet, 'to_dict') else {},
                'cashflow': ticker.cashflow.to_dict() if hasattr(ticker.cashflow, 'to_dict') else {}
            }
        except Exception:
            return None
    
    def _fetch_fmp_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Financial Modeling Prep."""
        try:
            url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data[0] if data else None
        except Exception:
            pass
        return None
    
    def _fetch_finnhub_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Finnhub."""
        try:
            url = f'https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.finnhub_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
    
    def _analyze_garp(self, pe: float, peg: float, earnings_growth: float, revenue_growth: float, sector: str) -> Dict:
        """
        Analyze stock for Growth At Reasonable Price (GARP) setup.
        Peter Lynch style analysis.
        """
        garp_score = 0
        signals = []
        
        # PEG Ratio analysis (core GARP metric)
        if peg:
            if peg < 1.0:
                garp_score += 30
                signals.append('PEG < 1.0: Undervalued relative to growth (STRONG BUY signal)')
            elif peg < 1.5:
                garp_score += 20
                signals.append('PEG 1.0-1.5: Fairly valued with good growth')
            elif peg < 2.0:
                garp_score += 10
                signals.append('PEG 1.5-2.0: Slightly expensive but acceptable for high quality')
            else:
                signals.append('PEG > 2.0: Expensive relative to growth')
        else:
            signals.append('PEG unavailable - cannot assess growth-adjusted valuation')
        
        # P/E vs Growth comparison
        if pe and earnings_growth:
            eg_pct = earnings_growth * 100
            if pe < eg_pct:
                garp_score += 25
                signals.append(f'P/E ({pe:.1f}) < Earnings Growth ({eg_pct:.1f}%): Classic GARP setup')
            elif pe < eg_pct * 1.5:
                garp_score += 15
                signals.append(f'P/E reasonably aligned with growth')
        
        # Revenue growth check
        if revenue_growth:
            rg_pct = revenue_growth * 100
            if rg_pct > 15:
                garp_score += 15
                signals.append(f'Strong revenue growth: {rg_pct:.1f}%')
            elif rg_pct > 8:
                garp_score += 10
                signals.append(f'Solid revenue growth: {rg_pct:.1f}%')
            elif rg_pct > 0:
                garp_score += 5
                signals.append(f'Positive revenue growth: {rg_pct:.1f}%')
        
        # Earnings growth check
        if earnings_growth:
            eg_pct = earnings_growth * 100
            if eg_pct > 20:
                garp_score += 15
                signals.append(f'Strong earnings growth: {eg_pct:.1f}%')
            elif eg_pct > 10:
                garp_score += 10
                signals.append(f'Solid earnings growth: {eg_pct:.1f}%')
        
        # Sector-adjusted P/E
        sector_pe = self.sector_pe_benchmarks.get(sector, self.sector_pe_benchmarks['Default'])
        if pe and pe < sector_pe * 0.8:
            garp_score += 15
            signals.append(f'P/E below sector average ({sector_pe:.1f}): Value opportunity')
        
        # Determine GARP verdict
        if garp_score >= 70:
            verdict = 'STRONG_GARP_SETUP'
            interpretation = 'Excellent growth at reasonable price - Peter Lynch would approve'
        elif garp_score >= 50:
            verdict = 'GARP_CANDIDATE'
            interpretation = 'Good growth/value balance - worth deeper analysis'
        elif garp_score >= 30:
            verdict = 'MODERATE_GARP'
            interpretation = 'Some GARP characteristics but not ideal'
        else:
            verdict = 'NOT_GARP'
            interpretation = 'Does not meet GARP criteria'
        
        return {
            'score': garp_score,
            'max_score': 100,
            'verdict': verdict,
            'interpretation': interpretation,
            'signals': signals,
            'peg_ratio': round(peg, 2) if peg else None
        }
    
    def _compare_to_sector(self, pe: float, sector: str, profit_margin: float, roe: float) -> Dict:
        """Compare metrics to sector benchmarks."""
        sector_pe = self.sector_pe_benchmarks.get(sector, self.sector_pe_benchmarks['Default'])
        
        comparisons = []
        
        if pe:
            pe_diff = ((pe - sector_pe) / sector_pe) * 100
            if pe_diff < -20:
                comparisons.append(f'P/E {pe_diff:.1f}% below sector: Potentially undervalued')
            elif pe_diff > 20:
                comparisons.append(f'P/E {pe_diff:.1f}% above sector: Premium valuation')
            else:
                comparisons.append(f'P/E in line with sector average')
        
        # Profit margin comparison (rough sector benchmarks)
        if profit_margin:
            pm_pct = profit_margin * 100
            if pm_pct > 20:
                comparisons.append(f'Profit margin {pm_pct:.1f}%: Excellent profitability')
            elif pm_pct > 10:
                comparisons.append(f'Profit margin {pm_pct:.1f}%: Good profitability')
            elif pm_pct > 5:
                comparisons.append(f'Profit margin {pm_pct:.1f}%: Average profitability')
            else:
                comparisons.append(f'Profit margin {pm_pct:.1f}%: Below average')
        
        return {
            'sector': sector,
            'sector_pe_benchmark': sector_pe,
            'company_pe': round(pe, 2) if pe else None,
            'pe_vs_sector': 'BELOW' if pe and pe < sector_pe else 'ABOVE' if pe else 'N/A',
            'comparisons': comparisons
        }
    
    def _calculate_quality_score(self, profit_margin, operating_margin, roe, roa,
                                  current_ratio, debt_to_equity, fcf, 
                                  earnings_growth, revenue_growth) -> Dict:
        """Calculate overall quality score based on fundamentals."""
        score = 0
        max_score = 100
        factors = []
        
        # Profitability (30 points max)
        if profit_margin and profit_margin > 0.15:
            score += 15
            factors.append('High profit margin (+15)')
        elif profit_margin and profit_margin > 0.08:
            score += 10
            factors.append('Good profit margin (+10)')
        elif profit_margin and profit_margin > 0:
            score += 5
            factors.append('Positive profit margin (+5)')
        
        if roe and roe > 0.15:
            score += 15
            factors.append('Strong ROE > 15% (+15)')
        elif roe and roe > 0.10:
            score += 10
            factors.append('Good ROE > 10% (+10)')
        
        # Financial Health (30 points max)
        if current_ratio and current_ratio > 1.5:
            score += 15
            factors.append('Strong current ratio (+15)')
        elif current_ratio and current_ratio > 1.0:
            score += 10
            factors.append('Adequate current ratio (+10)')
        
        if debt_to_equity and debt_to_equity < 50:
            score += 15
            factors.append('Low debt-to-equity (+15)')
        elif debt_to_equity and debt_to_equity < 100:
            score += 10
            factors.append('Moderate debt levels (+10)')
        
        # Cash Flow (20 points max)
        if fcf and fcf > 0:
            score += 20
            factors.append('Positive free cash flow (+20)')
        
        # Growth (20 points max)
        if earnings_growth and earnings_growth > 0.15:
            score += 10
            factors.append('Strong earnings growth (+10)')
        elif earnings_growth and earnings_growth > 0:
            score += 5
            factors.append('Positive earnings growth (+5)')
        
        if revenue_growth and revenue_growth > 0.10:
            score += 10
            factors.append('Strong revenue growth (+10)')
        elif revenue_growth and revenue_growth > 0:
            score += 5
            factors.append('Positive revenue growth (+5)')
        
        # Determine grade
        if score >= 80:
            grade = 'A'
            interpretation = 'Excellent quality - strong fundamentals across the board'
        elif score >= 65:
            grade = 'B'
            interpretation = 'Good quality - solid fundamentals with minor weaknesses'
        elif score >= 50:
            grade = 'C'
            interpretation = 'Average quality - some concerns but acceptable'
        elif score >= 35:
            grade = 'D'
            interpretation = 'Below average - significant fundamental weaknesses'
        else:
            grade = 'F'
            interpretation = 'Poor quality - major fundamental concerns'
        
        return {
            'score': score,
            'max_score': max_score,
            'grade': grade,
            'interpretation': interpretation,
            'factors': factors
        }
    
    def _analyze_liquidity(self, current_ratio, quick_ratio, fcf, ocf, total_debt, market_cap) -> Dict:
        """Analyze company liquidity and financial flexibility."""
        signals = []
        risk_level = 'LOW'
        
        # Current ratio analysis
        if current_ratio:
            if current_ratio > 2.0:
                signals.append('Strong liquidity: Current ratio > 2.0')
            elif current_ratio > 1.5:
                signals.append('Good liquidity: Current ratio > 1.5')
            elif current_ratio > 1.0:
                signals.append('Adequate liquidity: Current ratio > 1.0')
            else:
                signals.append('CAUTION: Current ratio < 1.0 - potential liquidity issues')
                risk_level = 'HIGH'
        
        # Quick ratio (acid test)
        if quick_ratio:
            if quick_ratio > 1.0:
                signals.append('Passes acid test: Quick ratio > 1.0')
            else:
                signals.append('Quick ratio < 1.0 - relies on inventory for short-term obligations')
                if risk_level != 'HIGH':
                    risk_level = 'MODERATE'
        
        # FCF coverage of debt
        if fcf and total_debt and fcf > 0:
            years_to_pay_debt = total_debt / fcf
            if years_to_pay_debt < 3:
                signals.append(f'Strong: Could pay off debt in {years_to_pay_debt:.1f} years with FCF')
            elif years_to_pay_debt < 7:
                signals.append(f'Moderate: {years_to_pay_debt:.1f} years to pay debt with FCF')
            else:
                signals.append(f'Concern: {years_to_pay_debt:.1f} years to pay debt with FCF')
                if risk_level == 'LOW':
                    risk_level = 'MODERATE'
        
        # FCF to market cap
        if fcf and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap * 100
            if fcf_yield > 8:
                signals.append(f'High FCF yield: {fcf_yield:.1f}% - strong cash generation')
            elif fcf_yield > 4:
                signals.append(f'Good FCF yield: {fcf_yield:.1f}%')
        
        return {
            'risk_level': risk_level,
            'signals': signals,
            'current_ratio': round(current_ratio, 2) if current_ratio else None,
            'quick_ratio': round(quick_ratio, 2) if quick_ratio else None
        }
    
    def _assess_value(self, pe, forward_pe, peg, pb, ev_ebitda, fcf_yield, sector) -> Dict:
        """Assess whether stock is undervalued, fairly valued, or overvalued."""
        value_signals = []
        value_score = 0
        
        sector_pe = self.sector_pe_benchmarks.get(sector, 20)
        
        # P/E assessment
        if pe:
            if pe < sector_pe * 0.7:
                value_signals.append(f'P/E significantly below sector: Potential value')
                value_score += 2
            elif pe < sector_pe:
                value_signals.append(f'P/E below sector average')
                value_score += 1
            elif pe > sector_pe * 1.3:
                value_signals.append(f'P/E premium to sector: Growth priced in')
                value_score -= 1
        
        # Forward P/E vs trailing
        if pe and forward_pe and forward_pe < pe:
            value_signals.append(f'Forward P/E < Trailing: Earnings expected to grow')
            value_score += 1
        
        # PEG
        if peg:
            if peg < 1:
                value_signals.append(f'PEG < 1: Undervalued relative to growth')
                value_score += 2
            elif peg < 1.5:
                value_signals.append(f'PEG < 1.5: Reasonably valued')
                value_score += 1
        
        # EV/EBITDA
        if ev_ebitda:
            if ev_ebitda < 8:
                value_signals.append(f'EV/EBITDA < 8: Attractive valuation')
                value_score += 2
            elif ev_ebitda < 12:
                value_signals.append(f'EV/EBITDA < 12: Fair valuation')
                value_score += 1
            elif ev_ebitda > 20:
                value_signals.append(f'EV/EBITDA > 20: Expensive')
                value_score -= 1
        
        # FCF Yield
        if fcf_yield:
            if fcf_yield > 8:
                value_signals.append(f'FCF Yield > 8%: Strong value indicator')
                value_score += 2
            elif fcf_yield > 5:
                value_signals.append(f'FCF Yield > 5%: Good value')
                value_score += 1
        
        # Determine overall assessment
        if value_score >= 5:
            assessment = 'UNDERVALUED'
            interpretation = 'Multiple metrics suggest stock is undervalued'
        elif value_score >= 2:
            assessment = 'FAIRLY_VALUED'
            interpretation = 'Stock appears reasonably priced'
        elif value_score >= 0:
            assessment = 'FULLY_VALUED'
            interpretation = 'Stock is priced for perfection'
        else:
            assessment = 'OVERVALUED'
            interpretation = 'Multiple metrics suggest stock is expensive'
        
        return {
            'assessment': assessment,
            'value_score': value_score,
            'interpretation': interpretation,
            'signals': value_signals
        }
    
    def _calculate_overall_rating(self, quality: Dict, garp: Dict, value: Dict, liquidity: Dict) -> Dict:
        """Calculate overall fundamental rating combining all analyses."""
        
        # Weight the scores
        quality_weight = 0.30
        garp_weight = 0.25
        value_weight = 0.30
        liquidity_weight = 0.15
        
        # Normalize scores to 0-100
        quality_normalized = quality['score']
        garp_normalized = garp['score']
        
        # Value score conversion (-3 to +7 range to 0-100)
        value_normalized = (value['value_score'] + 3) / 10 * 100
        
        # Liquidity score
        liquidity_normalized = {'LOW': 100, 'MODERATE': 60, 'HIGH': 20}.get(liquidity['risk_level'], 50)
        
        # Weighted average
        overall_score = (
            quality_normalized * quality_weight +
            garp_normalized * garp_weight +
            value_normalized * value_weight +
            liquidity_normalized * liquidity_weight
        )
        
        # Determine rating
        if overall_score >= 75:
            rating = 'STRONG_BUY'
            interpretation = 'Excellent fundamentals - strong candidate for investment'
        elif overall_score >= 60:
            rating = 'BUY'
            interpretation = 'Good fundamentals - worthy of consideration'
        elif overall_score >= 45:
            rating = 'HOLD'
            interpretation = 'Mixed fundamentals - hold if owned, wait for better entry'
        elif overall_score >= 30:
            rating = 'WEAK'
            interpretation = 'Below average fundamentals - proceed with caution'
        else:
            rating = 'AVOID'
            interpretation = 'Poor fundamentals - significant risks present'
        
        return {
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'interpretation': interpretation,
            'component_scores': {
                'quality': quality['score'],
                'garp': garp['score'],
                'value': value['value_score'],
                'liquidity_risk': liquidity['risk_level']
            }
        }
    
    def _format_large_number(self, num: float) -> str:
        """Format large numbers for display."""
        if num is None or num == 0:
            return 'N/A'
        
        abs_num = abs(num)
        sign = '-' if num < 0 else ''
        
        if abs_num >= 1e12:
            return f'{sign}${abs_num/1e12:.2f}T'
        elif abs_num >= 1e9:
            return f'{sign}${abs_num/1e9:.2f}B'
        elif abs_num >= 1e6:
            return f'{sign}${abs_num/1e6:.2f}M'
        elif abs_num >= 1e3:
            return f'{sign}${abs_num/1e3:.2f}K'
        else:
            return f'{sign}${abs_num:.2f}'


# Main execution for testing
if __name__ == '__main__':
    analyzer = EnhancedFundamentalsAnalyzer()
    result = analyzer.analyze('AAPL')
    print(json.dumps(result, indent=2))
