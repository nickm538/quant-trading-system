"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE FUNDAMENTALS ANALYZER v1.0                           ║
║                                                                              ║
║  Real-time fundamental analysis with institutional-grade metrics:            ║
║  - PEG Ratio (Price/Earnings to Growth)                                     ║
║  - EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)     ║
║  - Enterprise Value (EV) and EV/EBITDA                                      ║
║  - Free Cash Flow (FCF) and FCF Yield                                       ║
║  - Debt/Equity, Current Ratio, Quick Ratio                                  ║
║  - ROE, ROA, ROIC                                                           ║
║  - Profit Margins (Gross, Operating, Net)                                   ║
║                                                                              ║
║  ALL DATA FROM LIVE APIs - NO MOCK DATA                                     ║
║  Copyright © 2026 SadieAI - All Rights Reserved                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FundamentalsAnalyzer:
    """
    Comprehensive fundamental analysis with real-time data.
    
    Provides institutional-grade metrics with educational definitions
    and pro trader tips for each metric.
    """
    
    # Educational definitions for each metric
    DEFINITIONS = {
        'pe_ratio': {
            'name': 'P/E Ratio (Price-to-Earnings)',
            'definition': 'Stock price divided by earnings per share. Shows how much investors pay per dollar of earnings.',
            'interpretation': 'Lower P/E may indicate undervaluation; higher P/E suggests growth expectations.',
            'pro_tip': 'Compare P/E to industry peers and historical average. A P/E below 15 is traditionally considered "value"; above 25 is "growth".',
            'formula': 'P/E = Stock Price / EPS'
        },
        'peg_ratio': {
            'name': 'PEG Ratio (Price/Earnings to Growth)',
            'definition': 'P/E ratio divided by earnings growth rate. Accounts for growth when valuing a stock.',
            'interpretation': 'PEG < 1.0 suggests undervaluation relative to growth; PEG > 2.0 may indicate overvaluation.',
            'pro_tip': 'Peter Lynch popularized PEG. A PEG of 1.0 means fair value. Best used for growth stocks, not value/dividend stocks.',
            'formula': 'PEG = P/E Ratio / Annual EPS Growth Rate'
        },
        'ebitda': {
            'name': 'EBITDA',
            'definition': 'Earnings Before Interest, Taxes, Depreciation, and Amortization. Measures operating profitability.',
            'interpretation': 'Higher EBITDA indicates stronger operating performance. Used to compare companies with different capital structures.',
            'pro_tip': 'EBITDA removes accounting and financing effects. Good for comparing companies, but ignores capital expenditures.',
            'formula': 'EBITDA = Net Income + Interest + Taxes + Depreciation + Amortization'
        },
        'enterprise_value': {
            'name': 'Enterprise Value (EV)',
            'definition': 'Total company value including debt and excluding cash. The theoretical takeover price.',
            'interpretation': 'EV represents the true cost to acquire a company, accounting for debt obligations.',
            'pro_tip': 'EV is more accurate than market cap for comparing companies with different debt levels.',
            'formula': 'EV = Market Cap + Total Debt - Cash & Equivalents'
        },
        'ev_ebitda': {
            'name': 'EV/EBITDA',
            'definition': 'Enterprise Value divided by EBITDA. Valuation multiple independent of capital structure.',
            'interpretation': 'Lower EV/EBITDA may indicate undervaluation. Typically ranges from 6-15x depending on industry.',
            'pro_tip': 'Warren Buffett prefers EV/EBITDA over P/E. Below 10x is often considered attractive; below 6x may signal distress.',
            'formula': 'EV/EBITDA = Enterprise Value / EBITDA'
        },
        'free_cash_flow': {
            'name': 'Free Cash Flow (FCF)',
            'definition': 'Cash generated after capital expenditures. Money available for dividends, buybacks, or debt repayment.',
            'interpretation': 'Positive FCF indicates the company generates more cash than it needs to maintain operations.',
            'pro_tip': 'FCF is harder to manipulate than earnings. Consistent positive FCF is a sign of a quality business.',
            'formula': 'FCF = Operating Cash Flow - Capital Expenditures'
        },
        'fcf_yield': {
            'name': 'FCF Yield',
            'definition': 'Free Cash Flow per share divided by stock price. Shows cash return relative to price.',
            'interpretation': 'Higher FCF yield indicates better value. Compare to bond yields and dividend yields.',
            'pro_tip': 'FCF yield above 5% is generally attractive. Above 8% may indicate undervaluation or risk.',
            'formula': 'FCF Yield = (Free Cash Flow / Shares Outstanding) / Stock Price'
        },
        'debt_to_equity': {
            'name': 'Debt-to-Equity Ratio',
            'definition': 'Total debt divided by shareholder equity. Measures financial leverage.',
            'interpretation': 'Lower ratio indicates less risk. High debt can amplify returns but increases bankruptcy risk.',
            'pro_tip': 'D/E below 1.0 is conservative. Tech companies often have low D/E; utilities and REITs have higher.',
            'formula': 'D/E = Total Debt / Total Shareholder Equity'
        },
        'current_ratio': {
            'name': 'Current Ratio',
            'definition': 'Current assets divided by current liabilities. Measures short-term liquidity.',
            'interpretation': 'Ratio above 1.0 means company can cover short-term obligations. Below 1.0 may signal liquidity issues.',
            'pro_tip': 'Current ratio between 1.5-3.0 is healthy. Too high may indicate inefficient use of assets.',
            'formula': 'Current Ratio = Current Assets / Current Liabilities'
        },
        'quick_ratio': {
            'name': 'Quick Ratio (Acid Test)',
            'definition': 'Liquid assets divided by current liabilities. Stricter liquidity measure excluding inventory.',
            'interpretation': 'Above 1.0 indicates company can meet obligations without selling inventory.',
            'pro_tip': 'Quick ratio is more conservative than current ratio. Essential for companies with slow-moving inventory.',
            'formula': 'Quick Ratio = (Current Assets - Inventory) / Current Liabilities'
        },
        'roe': {
            'name': 'Return on Equity (ROE)',
            'definition': 'Net income divided by shareholder equity. Measures profitability relative to equity.',
            'interpretation': 'Higher ROE indicates efficient use of equity capital. Top companies often have ROE > 15%.',
            'pro_tip': 'Warren Buffett looks for consistent ROE above 15%. High ROE with low debt is ideal.',
            'formula': 'ROE = Net Income / Shareholder Equity'
        },
        'roa': {
            'name': 'Return on Assets (ROA)',
            'definition': 'Net income divided by total assets. Measures how efficiently assets generate profit.',
            'interpretation': 'Higher ROA indicates better asset utilization. Compare within same industry.',
            'pro_tip': 'ROA above 5% is generally good. Asset-light businesses (tech) have higher ROA than asset-heavy (manufacturing).',
            'formula': 'ROA = Net Income / Total Assets'
        },
        'roic': {
            'name': 'Return on Invested Capital (ROIC)',
            'definition': 'Operating profit after tax divided by invested capital. Measures return on all capital.',
            'interpretation': 'ROIC above cost of capital (WACC) creates shareholder value. Higher is better.',
            'pro_tip': 'ROIC is the ultimate profitability metric. Companies with ROIC > 15% have competitive advantages.',
            'formula': 'ROIC = NOPAT / (Debt + Equity - Cash)'
        },
        'gross_margin': {
            'name': 'Gross Profit Margin',
            'definition': 'Gross profit divided by revenue. Shows profitability after direct costs.',
            'interpretation': 'Higher margin indicates pricing power or cost efficiency. Compare to industry peers.',
            'pro_tip': 'Gross margin above 40% suggests competitive advantage. Software companies often exceed 70%.',
            'formula': 'Gross Margin = (Revenue - COGS) / Revenue'
        },
        'operating_margin': {
            'name': 'Operating Profit Margin',
            'definition': 'Operating income divided by revenue. Shows profitability from core operations.',
            'interpretation': 'Higher margin indicates operational efficiency. Excludes interest and taxes.',
            'pro_tip': 'Operating margin above 15% is strong. Expanding margins over time is a bullish signal.',
            'formula': 'Operating Margin = Operating Income / Revenue'
        },
        'net_margin': {
            'name': 'Net Profit Margin',
            'definition': 'Net income divided by revenue. Shows bottom-line profitability.',
            'interpretation': 'Higher margin means more profit per dollar of sales. Final measure of profitability.',
            'pro_tip': 'Net margin above 10% is healthy. Compare trends over time and vs. competitors.',
            'formula': 'Net Margin = Net Income / Revenue'
        },
        'revenue_growth': {
            'name': 'Revenue Growth (YoY)',
            'definition': 'Year-over-year percentage change in revenue. Measures top-line growth.',
            'interpretation': 'Positive growth indicates expanding business. Compare to industry and GDP growth.',
            'pro_tip': 'Sustainable growth above 10% is excellent. Accelerating growth is a strong bullish signal.',
            'formula': 'Revenue Growth = (Current Revenue - Prior Revenue) / Prior Revenue'
        },
        'earnings_growth': {
            'name': 'Earnings Growth (YoY)',
            'definition': 'Year-over-year percentage change in EPS. Measures profit growth.',
            'interpretation': 'Positive earnings growth drives stock appreciation. Quality of growth matters.',
            'pro_tip': 'Look for earnings growth exceeding revenue growth (margin expansion). Avoid one-time gains.',
            'formula': 'Earnings Growth = (Current EPS - Prior EPS) / Prior EPS'
        }
    }
    
    def __init__(self):
        """Initialize the Fundamentals Analyzer."""
        logger.info("FundamentalsAnalyzer initialized - LIVE DATA ONLY")
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis on a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with all fundamental metrics, definitions, and interpretations
        """
        logger.info(f"Analyzing fundamentals for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial statements for detailed analysis
            income_stmt = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Extract raw data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            # Valuation Metrics
            pe_ratio = info.get('trailingPE') or info.get('forwardPE', 0)
            peg_ratio = info.get('pegRatio', 0)
            price_to_book = info.get('priceToBook', 0)
            price_to_sales = info.get('priceToSalesTrailing12Months', 0)
            
            # Enterprise Value
            enterprise_value = info.get('enterpriseValue', 0)
            ev_to_revenue = info.get('enterpriseToRevenue', 0)
            ev_to_ebitda = info.get('enterpriseToEbitda', 0)
            
            # Profitability
            gross_margin = info.get('grossMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            net_margin = info.get('profitMargins', 0)
            roe = info.get('returnOnEquity', 0)
            roa = info.get('returnOnAssets', 0)
            
            # Growth
            revenue_growth = info.get('revenueGrowth', 0)
            earnings_growth = info.get('earningsGrowth', 0)
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0)
            
            # Financial Health
            total_debt = info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0)
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            
            # Cash Flow
            operating_cash_flow = info.get('operatingCashflow', 0)
            free_cash_flow = info.get('freeCashflow', 0)
            
            # EBITDA
            ebitda = info.get('ebitda', 0)
            
            # Calculate additional metrics
            fcf_yield = 0
            if free_cash_flow and market_cap and market_cap > 0:
                fcf_yield = (free_cash_flow / market_cap) * 100
            
            # Calculate ROIC if we have the data
            roic = 0
            if not income_stmt.empty and not balance_sheet.empty:
                try:
                    operating_income = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else 0
                    tax_rate = 0.21  # Assume 21% corporate tax rate
                    nopat = operating_income * (1 - tax_rate)
                    
                    total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
                    total_debt_bs = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                    cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                    
                    invested_capital = total_equity + total_debt_bs - cash
                    if invested_capital > 0:
                        roic = (nopat / invested_capital) * 100
                except Exception as e:
                    logger.warning(f"Could not calculate ROIC: {e}")
            
            # Build comprehensive result
            result = {
                'success': True,
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'current_price': current_price,
                'market_cap': market_cap,
                'timestamp': datetime.now().isoformat(),
                
                # Valuation Metrics
                'valuation': {
                    'pe_ratio': {
                        'value': round(pe_ratio, 2) if pe_ratio else None,
                        **self.DEFINITIONS['pe_ratio'],
                        'signal': self._get_pe_signal(pe_ratio)
                    },
                    'peg_ratio': {
                        'value': round(peg_ratio, 2) if peg_ratio else None,
                        **self.DEFINITIONS['peg_ratio'],
                        'signal': self._get_peg_signal(peg_ratio)
                    },
                    'price_to_book': round(price_to_book, 2) if price_to_book else None,
                    'price_to_sales': round(price_to_sales, 2) if price_to_sales else None,
                    'enterprise_value': {
                        'value': enterprise_value,
                        **self.DEFINITIONS['enterprise_value']
                    },
                    'ev_ebitda': {
                        'value': round(ev_to_ebitda, 2) if ev_to_ebitda else None,
                        **self.DEFINITIONS['ev_ebitda'],
                        'signal': self._get_ev_ebitda_signal(ev_to_ebitda)
                    }
                },
                
                # Profitability Metrics
                'profitability': {
                    'ebitda': {
                        'value': ebitda,
                        **self.DEFINITIONS['ebitda']
                    },
                    'gross_margin': {
                        'value': round(gross_margin * 100, 2) if gross_margin else None,
                        **self.DEFINITIONS['gross_margin'],
                        'signal': self._get_margin_signal(gross_margin, 'gross')
                    },
                    'operating_margin': {
                        'value': round(operating_margin * 100, 2) if operating_margin else None,
                        **self.DEFINITIONS['operating_margin'],
                        'signal': self._get_margin_signal(operating_margin, 'operating')
                    },
                    'net_margin': {
                        'value': round(net_margin * 100, 2) if net_margin else None,
                        **self.DEFINITIONS['net_margin'],
                        'signal': self._get_margin_signal(net_margin, 'net')
                    },
                    'roe': {
                        'value': round(roe * 100, 2) if roe else None,
                        **self.DEFINITIONS['roe'],
                        'signal': self._get_roe_signal(roe)
                    },
                    'roa': {
                        'value': round(roa * 100, 2) if roa else None,
                        **self.DEFINITIONS['roa'],
                        'signal': self._get_roa_signal(roa)
                    },
                    'roic': {
                        'value': round(roic, 2) if roic else None,
                        **self.DEFINITIONS['roic'],
                        'signal': self._get_roic_signal(roic)
                    }
                },
                
                # Growth Metrics
                'growth': {
                    'revenue_growth': {
                        'value': round(revenue_growth * 100, 2) if revenue_growth else None,
                        **self.DEFINITIONS['revenue_growth'],
                        'signal': self._get_growth_signal(revenue_growth)
                    },
                    'earnings_growth': {
                        'value': round(earnings_growth * 100, 2) if earnings_growth else None,
                        **self.DEFINITIONS['earnings_growth'],
                        'signal': self._get_growth_signal(earnings_growth)
                    },
                    'earnings_quarterly_growth': round(earnings_quarterly_growth * 100, 2) if earnings_quarterly_growth else None
                },
                
                # Financial Health
                'financial_health': {
                    'debt_to_equity': {
                        'value': round(debt_to_equity / 100, 2) if debt_to_equity else None,  # yfinance returns as percentage
                        **self.DEFINITIONS['debt_to_equity'],
                        'signal': self._get_debt_signal(debt_to_equity / 100 if debt_to_equity else 0)
                    },
                    'current_ratio': {
                        'value': round(current_ratio, 2) if current_ratio else None,
                        **self.DEFINITIONS['current_ratio'],
                        'signal': self._get_liquidity_signal(current_ratio)
                    },
                    'quick_ratio': {
                        'value': round(quick_ratio, 2) if quick_ratio else None,
                        **self.DEFINITIONS['quick_ratio'],
                        'signal': self._get_liquidity_signal(quick_ratio)
                    },
                    'total_debt': total_debt,
                    'total_cash': total_cash,
                    'net_debt': total_debt - total_cash if total_debt and total_cash else None
                },
                
                # Cash Flow
                'cash_flow': {
                    'operating_cash_flow': operating_cash_flow,
                    'free_cash_flow': {
                        'value': free_cash_flow,
                        **self.DEFINITIONS['free_cash_flow'],
                        'signal': 'positive' if free_cash_flow and free_cash_flow > 0 else 'negative'
                    },
                    'fcf_yield': {
                        'value': round(fcf_yield, 2) if fcf_yield else None,
                        **self.DEFINITIONS['fcf_yield'],
                        'signal': self._get_fcf_yield_signal(fcf_yield)
                    }
                },
                
                # Overall Assessment
                'overall_assessment': self._generate_overall_assessment(
                    pe_ratio, peg_ratio, ev_to_ebitda, roe, roic, 
                    debt_to_equity, free_cash_flow, revenue_growth
                )
            }
            
            logger.info(f"Fundamental analysis complete for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    def _get_pe_signal(self, pe: float) -> str:
        if not pe or pe <= 0:
            return 'unavailable'
        if pe < 15:
            return 'undervalued'
        if pe < 25:
            return 'fair'
        return 'expensive'
    
    def _get_peg_signal(self, peg: float) -> str:
        if not peg or peg <= 0:
            return 'unavailable'
        if peg < 1.0:
            return 'undervalued'
        if peg < 2.0:
            return 'fair'
        return 'expensive'
    
    def _get_ev_ebitda_signal(self, ev_ebitda: float) -> str:
        if not ev_ebitda or ev_ebitda <= 0:
            return 'unavailable'
        if ev_ebitda < 8:
            return 'undervalued'
        if ev_ebitda < 15:
            return 'fair'
        return 'expensive'
    
    def _get_margin_signal(self, margin: float, margin_type: str) -> str:
        if not margin:
            return 'unavailable'
        thresholds = {
            'gross': (0.30, 0.50),
            'operating': (0.10, 0.20),
            'net': (0.05, 0.15)
        }
        low, high = thresholds.get(margin_type, (0.10, 0.20))
        if margin < low:
            return 'weak'
        if margin < high:
            return 'moderate'
        return 'strong'
    
    def _get_roe_signal(self, roe: float) -> str:
        if not roe:
            return 'unavailable'
        if roe < 0.10:
            return 'weak'
        if roe < 0.15:
            return 'moderate'
        if roe < 0.25:
            return 'strong'
        return 'exceptional'
    
    def _get_roa_signal(self, roa: float) -> str:
        if not roa:
            return 'unavailable'
        if roa < 0.03:
            return 'weak'
        if roa < 0.08:
            return 'moderate'
        return 'strong'
    
    def _get_roic_signal(self, roic: float) -> str:
        if not roic:
            return 'unavailable'
        if roic < 8:
            return 'weak'
        if roic < 15:
            return 'moderate'
        if roic < 25:
            return 'strong'
        return 'exceptional'
    
    def _get_growth_signal(self, growth: float) -> str:
        if not growth:
            return 'unavailable'
        if growth < 0:
            return 'declining'
        if growth < 0.05:
            return 'slow'
        if growth < 0.15:
            return 'moderate'
        if growth < 0.30:
            return 'strong'
        return 'exceptional'
    
    def _get_debt_signal(self, debt_ratio: float) -> str:
        if debt_ratio is None:
            return 'unavailable'
        if debt_ratio < 0.5:
            return 'conservative'
        if debt_ratio < 1.0:
            return 'moderate'
        if debt_ratio < 2.0:
            return 'elevated'
        return 'high_risk'
    
    def _get_liquidity_signal(self, ratio: float) -> str:
        if not ratio:
            return 'unavailable'
        if ratio < 1.0:
            return 'weak'
        if ratio < 1.5:
            return 'adequate'
        if ratio < 3.0:
            return 'strong'
        return 'excess_liquidity'
    
    def _get_fcf_yield_signal(self, fcf_yield: float) -> str:
        if not fcf_yield:
            return 'unavailable'
        if fcf_yield < 0:
            return 'negative'
        if fcf_yield < 3:
            return 'low'
        if fcf_yield < 6:
            return 'moderate'
        if fcf_yield < 10:
            return 'attractive'
        return 'very_attractive'
    
    def _generate_overall_assessment(
        self,
        pe: float,
        peg: float,
        ev_ebitda: float,
        roe: float,
        roic: float,
        debt_equity: float,
        fcf: float,
        revenue_growth: float
    ) -> Dict[str, Any]:
        """Generate overall fundamental assessment with score."""
        
        score = 50  # Start neutral
        strengths = []
        weaknesses = []
        
        # Valuation assessment
        if peg and 0 < peg < 1.0:
            score += 15
            strengths.append("Attractively valued relative to growth (PEG < 1)")
        elif peg and peg > 2.0:
            score -= 10
            weaknesses.append("Expensive relative to growth (PEG > 2)")
        
        if ev_ebitda and ev_ebitda < 10:
            score += 10
            strengths.append("Reasonable EV/EBITDA valuation")
        elif ev_ebitda and ev_ebitda > 20:
            score -= 10
            weaknesses.append("High EV/EBITDA multiple")
        
        # Profitability assessment
        if roe and roe > 0.15:
            score += 10
            strengths.append(f"Strong ROE of {roe*100:.1f}%")
        elif roe and roe < 0.08:
            score -= 5
            weaknesses.append("Below-average ROE")
        
        if roic and roic > 15:
            score += 10
            strengths.append(f"Excellent ROIC of {roic:.1f}%")
        
        # Financial health
        if debt_equity and debt_equity < 0.5:
            score += 5
            strengths.append("Conservative debt levels")
        elif debt_equity and debt_equity > 2.0:
            score -= 10
            weaknesses.append("High debt levels")
        
        # Cash flow
        if fcf and fcf > 0:
            score += 10
            strengths.append("Positive free cash flow")
        elif fcf and fcf < 0:
            score -= 15
            weaknesses.append("Negative free cash flow")
        
        # Growth
        if revenue_growth and revenue_growth > 0.15:
            score += 10
            strengths.append(f"Strong revenue growth of {revenue_growth*100:.1f}%")
        elif revenue_growth and revenue_growth < 0:
            score -= 10
            weaknesses.append("Declining revenue")
        
        # Determine rating
        if score >= 80:
            rating = "EXCELLENT"
            summary = "Strong fundamentals with multiple positive factors."
        elif score >= 65:
            rating = "GOOD"
            summary = "Solid fundamentals with room for improvement."
        elif score >= 50:
            rating = "FAIR"
            summary = "Mixed fundamentals - careful analysis recommended."
        elif score >= 35:
            rating = "WEAK"
            summary = "Several fundamental concerns present."
        else:
            rating = "POOR"
            summary = "Significant fundamental weaknesses identified."
        
        return {
            'score': min(100, max(0, score)),
            'rating': rating,
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses
        }


# Standalone function for easy import
def analyze_fundamentals(symbol: str) -> Dict[str, Any]:
    """Analyze fundamentals for a given symbol."""
    analyzer = FundamentalsAnalyzer()
    return analyzer.analyze(symbol)


if __name__ == "__main__":
    # Test the analyzer
    import json
    
    result = analyze_fundamentals("AAPL")
    print(json.dumps(result, indent=2, default=str))
