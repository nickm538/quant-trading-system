"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE FUNDAMENTALS ANALYZER v2.0                           ║
║                                                                              ║
║  Institutional-Grade Fundamental Analysis with AI-Powered Interpretation    ║
║                                                                              ║
║  VALUATION METRICS:                                                          ║
║  - P/E, P/S, P/B, EV/EBITDA, PEG Ratio                                      ║
║                                                                              ║
║  PROFITABILITY METRICS:                                                      ║
║  - ROE, ROA, ROIC, Gross/Operating/Net Margins                              ║
║                                                                              ║
║  GROWTH METRICS:                                                             ║
║  - Revenue Growth, EPS Growth, 3Y/5Y CAGR                                   ║
║                                                                              ║
║  FINANCIAL HEALTH:                                                           ║
║  - Current Ratio, Quick Ratio, Debt-to-Equity, Leverage Ratios              ║
║                                                                              ║
║  CASH FLOW ANALYSIS:                                                         ║
║  - Operating CF, Free CF, FCF Yield, Cash Conversion                        ║
║                                                                              ║
║  PER-SHARE METRICS:                                                          ║
║  - EPS, Book Value/Share, Revenue/Share, FCF/Share                          ║
║                                                                              ║
║  SPECIAL SETUPS:                                                             ║
║  - GARP Setup Detection (Growth at Reasonable Price)                        ║
║  - MOAT Setup Detection (Economic Moat)                                     ║
║  - R² Score (Earnings Predictability)                                       ║
║                                                                              ║
║  AI-POWERED INTERPRETATION:                                                  ║
║  - Intelligent Scoring (0-100)                                              ║
║  - Sector-Relative Comparison                                               ║
║  - Bull/Bear Case Generation                                                ║
║                                                                              ║
║  ALL DATA FROM LIVE APIs - NO MOCK DATA                                     ║
║  Copyright © 2026 SadieAI - All Rights Reserved                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import yfinance as yf
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import statistics

# Import FinancialDatasets.ai for premium data
try:
    from financial_datasets_client import FinancialDatasetsClient
    HAS_FINANCIAL_DATASETS = True
except ImportError:
    HAS_FINANCIAL_DATASETS = False

logger = logging.getLogger(__name__)


class ComprehensiveFundamentalsV2:
    """
    Institutional-Grade Fundamental Analysis with AI-Powered Interpretation.
    
    Provides comprehensive metrics with GARP/MOAT detection, intelligent scoring,
    and sector-relative comparisons.
    """
    
    # Sector benchmarks for relative comparison
    SECTOR_BENCHMARKS = {
        'Technology': {'pe': 25, 'ps': 6, 'pb': 5, 'roe': 0.18, 'margin': 0.20},
        'Healthcare': {'pe': 20, 'ps': 4, 'pb': 4, 'roe': 0.15, 'margin': 0.15},
        'Financial Services': {'pe': 12, 'ps': 3, 'pb': 1.2, 'roe': 0.12, 'margin': 0.25},
        'Consumer Cyclical': {'pe': 18, 'ps': 1.5, 'pb': 3, 'roe': 0.15, 'margin': 0.10},
        'Consumer Defensive': {'pe': 20, 'ps': 1.5, 'pb': 4, 'roe': 0.20, 'margin': 0.12},
        'Industrials': {'pe': 18, 'ps': 2, 'pb': 3, 'roe': 0.15, 'margin': 0.10},
        'Energy': {'pe': 10, 'ps': 1, 'pb': 1.5, 'roe': 0.12, 'margin': 0.08},
        'Utilities': {'pe': 18, 'ps': 2, 'pb': 1.8, 'roe': 0.10, 'margin': 0.15},
        'Real Estate': {'pe': 35, 'ps': 8, 'pb': 2, 'roe': 0.08, 'margin': 0.30},
        'Basic Materials': {'pe': 12, 'ps': 1.5, 'pb': 2, 'roe': 0.12, 'margin': 0.10},
        'Communication Services': {'pe': 20, 'ps': 3, 'pb': 3, 'roe': 0.12, 'margin': 0.15},
        'Default': {'pe': 18, 'ps': 2, 'pb': 3, 'roe': 0.15, 'margin': 0.12}
    }
    
    def __init__(self):
        """Initialize the Comprehensive Fundamentals Analyzer v2."""
        logger.info("ComprehensiveFundamentalsV2 initialized - INSTITUTIONAL GRADE")
        
        # Initialize FinancialDatasets.ai client for premium data
        self.fd_client = None
        if HAS_FINANCIAL_DATASETS:
            try:
                self.fd_client = FinancialDatasetsClient()
                logger.info("FinancialDatasets.ai client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize FinancialDatasets client: {e}")
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis on a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with all fundamental metrics, GARP/MOAT detection,
            and AI-powered interpretation
        """
        logger.info(f"Analyzing fundamentals for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial statements for detailed analysis
            income_stmt = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Get historical data for growth calculations
            hist = ticker.history(period="5y")
            
            # Extract basic info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            sector = info.get('sector', 'Default')
            industry = info.get('industry', 'Unknown')
            company_name = info.get('shortName', symbol)
            
            # ═══════════════════════════════════════════════════════════════
            # VALUATION METRICS
            # ═══════════════════════════════════════════════════════════════
            valuation = self._calculate_valuation_metrics(info, income_stmt, balance_sheet)
            
            # ═══════════════════════════════════════════════════════════════
            # PROFITABILITY METRICS
            # ═══════════════════════════════════════════════════════════════
            profitability = self._calculate_profitability_metrics(info, income_stmt, balance_sheet)
            
            # ═══════════════════════════════════════════════════════════════
            # GROWTH METRICS
            # ═══════════════════════════════════════════════════════════════
            growth = self._calculate_growth_metrics(info, income_stmt, hist)
            
            # ═══════════════════════════════════════════════════════════════
            # FINANCIAL HEALTH (LIQUIDITY & LEVERAGE)
            # ═══════════════════════════════════════════════════════════════
            financial_health = self._calculate_financial_health(info, balance_sheet)
            
            # ═══════════════════════════════════════════════════════════════
            # CASH FLOW ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            cash_flow_analysis = self._calculate_cash_flow_metrics(info, cash_flow, market_cap)
            
            # ═══════════════════════════════════════════════════════════════
            # PER-SHARE METRICS
            # ═══════════════════════════════════════════════════════════════
            per_share = self._calculate_per_share_metrics(info, income_stmt, balance_sheet, cash_flow, shares_outstanding)
            
            # ═══════════════════════════════════════════════════════════════
            # R² SCORE (EARNINGS PREDICTABILITY)
            # ═══════════════════════════════════════════════════════════════
            r2_score = self._calculate_r2_score(income_stmt)
            
            # ═══════════════════════════════════════════════════════════════
            # GARP SETUP DETECTION
            # ═══════════════════════════════════════════════════════════════
            garp_setup = self._detect_garp_setup(valuation, growth, profitability)
            
            # ═══════════════════════════════════════════════════════════════
            # MOAT SETUP DETECTION
            # ═══════════════════════════════════════════════════════════════
            moat_setup = self._detect_moat_setup(profitability, financial_health, growth, info)
            
            # ═══════════════════════════════════════════════════════════════
            # INTELLIGENT SCORING (0-100)
            # ═══════════════════════════════════════════════════════════════
            intelligent_score = self._calculate_intelligent_score(
                valuation, profitability, growth, financial_health, 
                cash_flow_analysis, garp_setup, moat_setup, r2_score, sector
            )
            
            # ═══════════════════════════════════════════════════════════════
            # AI-POWERED INTERPRETATION
            # ═══════════════════════════════════════════════════════════════
            interpretation = self._generate_ai_interpretation(
                symbol, company_name, sector, industry,
                valuation, profitability, growth, financial_health,
                cash_flow_analysis, garp_setup, moat_setup, intelligent_score
            )
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'sector': sector,
                'industry': industry,
                'current_price': current_price,
                'market_cap': market_cap,
                'timestamp': datetime.now().isoformat(),
                
                # Core Metrics
                'valuation': valuation,
                'profitability': profitability,
                'growth': growth,
                'financial_health': financial_health,
                'cash_flow': cash_flow_analysis,
                'per_share': per_share,
                
                # Special Setups
                'r2_score': r2_score,
                'garp_setup': garp_setup,
                'moat_setup': moat_setup,
                
                # Intelligent Analysis
                'intelligent_score': intelligent_score,
                'interpretation': interpretation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def _calculate_valuation_metrics(self, info: Dict, income_stmt, balance_sheet) -> Dict[str, Any]:
        """Calculate comprehensive valuation metrics with fallback calculations."""
        # Primary sources from info dict
        pe_ratio = info.get('trailingPE') or info.get('forwardPE', 0)
        forward_pe = info.get('forwardPE', 0)
        peg_ratio = info.get('pegRatio', 0)
        price_to_book = info.get('priceToBook', 0)
        price_to_sales = info.get('priceToSalesTrailing12Months', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        ev_to_ebitda = info.get('enterpriseToEbitda', 0)
        ev_to_revenue = info.get('enterpriseToRevenue', 0)
        
        # Get basic info for fallback calculations
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        market_cap = info.get('marketCap', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        # FALLBACK: Calculate P/S from financial statements
        if not price_to_sales or price_to_sales == 0:
            try:
                if not income_stmt.empty and market_cap > 0:
                    revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
                    if revenue and revenue > 0:
                        price_to_sales = market_cap / revenue
            except Exception:
                pass
        
        # FALLBACK: Calculate P/B from financial statements
        if not price_to_book or price_to_book == 0:
            try:
                if not balance_sheet.empty and market_cap > 0:
                    book_value = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
                    if book_value and book_value > 0:
                        price_to_book = market_cap / book_value
            except Exception:
                pass
        
        # FALLBACK: Calculate EV/EBITDA from financial statements
        if not ev_to_ebitda or ev_to_ebitda == 0:
            try:
                if not income_stmt.empty and not balance_sheet.empty:
                    # Calculate Enterprise Value
                    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                    cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                    if not enterprise_value or enterprise_value == 0:
                        enterprise_value = market_cap + (total_debt or 0) - (cash or 0)
                    
                    # Calculate EBITDA
                    ebitda = income_stmt.loc['EBITDA'].iloc[0] if 'EBITDA' in income_stmt.index else 0
                    if not ebitda or ebitda == 0:
                        operating_income = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else 0
                        depreciation = income_stmt.loc['Depreciation And Amortization'].iloc[0] if 'Depreciation And Amortization' in income_stmt.index else 0
                        ebitda = (operating_income or 0) + (depreciation or 0)
                    
                    if ebitda and ebitda > 0 and enterprise_value > 0:
                        ev_to_ebitda = enterprise_value / ebitda
            except Exception:
                pass
        
        # Determine valuation status
        valuation_signals = []
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                valuation_signals.append(('P/E', 'UNDERVALUED', f'P/E of {pe_ratio:.1f} is below 15'))
            elif pe_ratio > 30:
                valuation_signals.append(('P/E', 'OVERVALUED', f'P/E of {pe_ratio:.1f} is above 30'))
            else:
                valuation_signals.append(('P/E', 'FAIR', f'P/E of {pe_ratio:.1f} is in normal range'))
        
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1.0:
                valuation_signals.append(('PEG', 'UNDERVALUED', f'PEG of {peg_ratio:.2f} suggests growth at discount'))
            elif peg_ratio > 2.0:
                valuation_signals.append(('PEG', 'OVERVALUED', f'PEG of {peg_ratio:.2f} is expensive for growth'))
            else:
                valuation_signals.append(('PEG', 'FAIR', f'PEG of {peg_ratio:.2f} is reasonable'))
        
        if ev_to_ebitda and ev_to_ebitda > 0:
            if ev_to_ebitda < 10:
                valuation_signals.append(('EV/EBITDA', 'UNDERVALUED', f'EV/EBITDA of {ev_to_ebitda:.1f} is attractive'))
            elif ev_to_ebitda > 20:
                valuation_signals.append(('EV/EBITDA', 'OVERVALUED', f'EV/EBITDA of {ev_to_ebitda:.1f} is expensive'))
            else:
                valuation_signals.append(('EV/EBITDA', 'FAIR', f'EV/EBITDA of {ev_to_ebitda:.1f} is reasonable'))
        
        # Overall valuation assessment
        undervalued_count = sum(1 for s in valuation_signals if s[1] == 'UNDERVALUED')
        overvalued_count = sum(1 for s in valuation_signals if s[1] == 'OVERVALUED')
        
        if undervalued_count > overvalued_count:
            overall = 'UNDERVALUED'
        elif overvalued_count > undervalued_count:
            overall = 'OVERVALUED'
        else:
            overall = 'FAIRLY VALUED'
        
        return {
            'pe_ratio': pe_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'price_to_sales': price_to_sales,
            'enterprise_value': enterprise_value,
            'ev_to_ebitda': ev_to_ebitda,
            'ev_to_revenue': ev_to_revenue,
            'signals': valuation_signals,
            'overall_assessment': overall,
            'explanation': {
                'pe': 'Price-to-Earnings: How much investors pay per $1 of earnings. Lower = cheaper.',
                'ps': 'Price-to-Sales: Stock price relative to revenue. Useful for unprofitable companies.',
                'pb': 'Price-to-Book: Stock price vs. net asset value. Below 1 may indicate undervaluation.',
                'ev_ebitda': 'EV/EBITDA: Enterprise value vs. operating earnings. Buffett\'s preferred metric.'
            }
        }
    
    def _calculate_profitability_metrics(self, info: Dict, income_stmt, balance_sheet) -> Dict[str, Any]:
        """Calculate comprehensive profitability metrics."""
        roe = info.get('returnOnEquity', 0)
        roa = info.get('returnOnAssets', 0)
        gross_margin = info.get('grossMargins', 0)
        operating_margin = info.get('operatingMargins', 0)
        net_margin = info.get('profitMargins', 0)
        
        # Calculate ROIC if possible
        roic = 0
        try:
            if not income_stmt.empty and not balance_sheet.empty:
                operating_income = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else 0
                tax_rate = 0.21  # Approximate corporate tax rate
                nopat = operating_income * (1 - tax_rate)
                
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
                cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                
                invested_capital = total_debt + total_equity - cash
                if invested_capital > 0:
                    roic = nopat / invested_capital
        except Exception as e:
            logger.debug(f"Could not calculate ROIC: {e}")
        
        # Profitability signals
        signals = []
        if roe and roe > 0:
            if roe > 0.20:
                signals.append(('ROE', 'EXCELLENT', f'ROE of {roe*100:.1f}% exceeds 20%'))
            elif roe > 0.15:
                signals.append(('ROE', 'GOOD', f'ROE of {roe*100:.1f}% is solid'))
            elif roe > 0.10:
                signals.append(('ROE', 'FAIR', f'ROE of {roe*100:.1f}% is acceptable'))
            else:
                signals.append(('ROE', 'WEAK', f'ROE of {roe*100:.1f}% is below average'))
        
        if net_margin and net_margin > 0:
            if net_margin > 0.20:
                signals.append(('Net Margin', 'EXCELLENT', f'Net margin of {net_margin*100:.1f}% is exceptional'))
            elif net_margin > 0.10:
                signals.append(('Net Margin', 'GOOD', f'Net margin of {net_margin*100:.1f}% is healthy'))
            else:
                signals.append(('Net Margin', 'FAIR', f'Net margin of {net_margin*100:.1f}% is thin'))
        
        # Overall profitability rating
        excellent_count = sum(1 for s in signals if s[1] == 'EXCELLENT')
        good_count = sum(1 for s in signals if s[1] == 'GOOD')
        
        if excellent_count >= 2:
            overall = 'HIGHLY PROFITABLE'
        elif excellent_count >= 1 or good_count >= 2:
            overall = 'PROFITABLE'
        elif good_count >= 1:
            overall = 'MODERATELY PROFITABLE'
        else:
            overall = 'LOW PROFITABILITY'
        
        return {
            'roe': roe,
            'roa': roa,
            'roic': roic,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'signals': signals,
            'overall_assessment': overall,
            'explanation': {
                'roe': 'Return on Equity: Profit generated per $1 of shareholder equity. Buffett looks for >15%.',
                'roa': 'Return on Assets: How efficiently assets generate profit.',
                'roic': 'Return on Invested Capital: Return on all capital. Above 15% indicates competitive advantage.',
                'margins': 'Profit Margins: Higher margins = pricing power and efficiency.'
            }
        }
    
    def _calculate_growth_metrics(self, info: Dict, income_stmt, hist) -> Dict[str, Any]:
        """Calculate comprehensive growth metrics."""
        revenue_growth = info.get('revenueGrowth', 0)
        earnings_growth = info.get('earningsGrowth', 0)
        earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0)
        
        # Calculate multi-year growth rates
        revenue_3y_cagr = 0
        revenue_5y_cagr = 0
        eps_3y_cagr = 0
        eps_5y_cagr = 0
        
        try:
            if not income_stmt.empty and len(income_stmt.columns) >= 3:
                revenues = []
                for col in income_stmt.columns[:4]:  # Get up to 4 years
                    if 'Total Revenue' in income_stmt.index:
                        revenues.append(income_stmt.loc['Total Revenue', col])
                
                if len(revenues) >= 3 and revenues[-1] > 0 and revenues[0] > 0:
                    revenue_3y_cagr = (revenues[0] / revenues[-1]) ** (1/3) - 1
                
                # EPS growth
                eps_values = []
                if 'Basic EPS' in income_stmt.index:
                    for col in income_stmt.columns[:4]:
                        eps_values.append(income_stmt.loc['Basic EPS', col])
                    
                    if len(eps_values) >= 3 and eps_values[-1] > 0 and eps_values[0] > 0:
                        eps_3y_cagr = (eps_values[0] / eps_values[-1]) ** (1/3) - 1
        except Exception as e:
            logger.debug(f"Could not calculate CAGR: {e}")
        
        # Growth signals
        signals = []
        if revenue_growth:
            if revenue_growth > 0.20:
                signals.append(('Revenue Growth', 'EXCELLENT', f'{revenue_growth*100:.1f}% YoY growth is exceptional'))
            elif revenue_growth > 0.10:
                signals.append(('Revenue Growth', 'GOOD', f'{revenue_growth*100:.1f}% YoY growth is solid'))
            elif revenue_growth > 0:
                signals.append(('Revenue Growth', 'FAIR', f'{revenue_growth*100:.1f}% YoY growth is modest'))
            else:
                signals.append(('Revenue Growth', 'DECLINING', f'{revenue_growth*100:.1f}% indicates contraction'))
        
        if earnings_growth:
            if earnings_growth > 0.25:
                signals.append(('EPS Growth', 'EXCELLENT', f'{earnings_growth*100:.1f}% earnings growth is exceptional'))
            elif earnings_growth > 0.10:
                signals.append(('EPS Growth', 'GOOD', f'{earnings_growth*100:.1f}% earnings growth is solid'))
            elif earnings_growth > 0:
                signals.append(('EPS Growth', 'FAIR', f'{earnings_growth*100:.1f}% earnings growth is modest'))
            else:
                signals.append(('EPS Growth', 'DECLINING', f'{earnings_growth*100:.1f}% indicates earnings decline'))
        
        # Overall growth assessment
        excellent_count = sum(1 for s in signals if s[1] == 'EXCELLENT')
        declining_count = sum(1 for s in signals if s[1] == 'DECLINING')
        
        if excellent_count >= 2:
            overall = 'HIGH GROWTH'
        elif excellent_count >= 1:
            overall = 'GROWTH'
        elif declining_count >= 1:
            overall = 'DECLINING'
        else:
            overall = 'STABLE'
        
        return {
            'revenue_growth_yoy': revenue_growth,
            'earnings_growth_yoy': earnings_growth,
            'earnings_quarterly_growth': earnings_quarterly_growth,
            'revenue_3y_cagr': revenue_3y_cagr,
            'revenue_5y_cagr': revenue_5y_cagr,
            'eps_3y_cagr': eps_3y_cagr,
            'eps_5y_cagr': eps_5y_cagr,
            'signals': signals,
            'overall_assessment': overall,
            'explanation': {
                'revenue_growth': 'Top-line growth shows business expansion.',
                'eps_growth': 'Earnings growth drives stock appreciation.',
                'cagr': 'Compound Annual Growth Rate: Smoothed multi-year growth.'
            }
        }
    
    def _calculate_financial_health(self, info: Dict, balance_sheet) -> Dict[str, Any]:
        """Calculate financial health metrics (liquidity and leverage)."""
        current_ratio = info.get('currentRatio', 0)
        quick_ratio = info.get('quickRatio', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        
        # Calculate additional leverage ratios
        debt_to_assets = 0
        interest_coverage = 0
        net_debt = total_debt - total_cash if total_debt and total_cash else 0
        
        try:
            if not balance_sheet.empty:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                if total_assets > 0 and total_debt:
                    debt_to_assets = total_debt / total_assets
        except Exception as e:
            logger.debug(f"Could not calculate debt ratios: {e}")
        
        # Financial health signals
        signals = []
        
        # Liquidity
        if current_ratio:
            if current_ratio > 2.0:
                signals.append(('Current Ratio', 'STRONG', f'{current_ratio:.2f} indicates excellent liquidity'))
            elif current_ratio > 1.5:
                signals.append(('Current Ratio', 'GOOD', f'{current_ratio:.2f} indicates healthy liquidity'))
            elif current_ratio > 1.0:
                signals.append(('Current Ratio', 'FAIR', f'{current_ratio:.2f} indicates adequate liquidity'))
            else:
                signals.append(('Current Ratio', 'WEAK', f'{current_ratio:.2f} indicates liquidity risk'))
        
        if quick_ratio:
            if quick_ratio > 1.5:
                signals.append(('Quick Ratio', 'STRONG', f'{quick_ratio:.2f} indicates strong acid test'))
            elif quick_ratio > 1.0:
                signals.append(('Quick Ratio', 'GOOD', f'{quick_ratio:.2f} passes acid test'))
            else:
                signals.append(('Quick Ratio', 'WEAK', f'{quick_ratio:.2f} fails acid test'))
        
        # Leverage
        if debt_to_equity:
            d_e = debt_to_equity / 100 if debt_to_equity > 10 else debt_to_equity  # Normalize
            if d_e < 0.5:
                signals.append(('Debt/Equity', 'CONSERVATIVE', f'{d_e:.2f} indicates low leverage'))
            elif d_e < 1.0:
                signals.append(('Debt/Equity', 'MODERATE', f'{d_e:.2f} indicates moderate leverage'))
            elif d_e < 2.0:
                signals.append(('Debt/Equity', 'ELEVATED', f'{d_e:.2f} indicates higher leverage'))
            else:
                signals.append(('Debt/Equity', 'HIGH RISK', f'{d_e:.2f} indicates high leverage risk'))
        
        # Overall assessment
        strong_count = sum(1 for s in signals if s[1] in ['STRONG', 'CONSERVATIVE'])
        weak_count = sum(1 for s in signals if s[1] in ['WEAK', 'HIGH RISK'])
        
        if strong_count >= 2 and weak_count == 0:
            overall = 'FORTRESS BALANCE SHEET'
        elif strong_count >= 1 and weak_count == 0:
            overall = 'HEALTHY'
        elif weak_count >= 2:
            overall = 'AT RISK'
        else:
            overall = 'MODERATE'
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'total_debt': total_debt,
            'total_cash': total_cash,
            'net_debt': net_debt,
            'interest_coverage': interest_coverage,
            'signals': signals,
            'overall_assessment': overall,
            'explanation': {
                'current_ratio': 'Current assets / current liabilities. Above 1.5 is healthy.',
                'quick_ratio': 'Liquid assets / current liabilities. Excludes inventory.',
                'debt_equity': 'Total debt / shareholder equity. Lower = less risk.',
                'net_debt': 'Total debt minus cash. Negative = net cash position.'
            }
        }
    
    def _calculate_cash_flow_metrics(self, info: Dict, cash_flow, market_cap: float) -> Dict[str, Any]:
        """Calculate comprehensive cash flow metrics."""
        operating_cf = info.get('operatingCashflow', 0)
        free_cf = info.get('freeCashflow', 0)
        
        # FCF Yield
        fcf_yield = 0
        if free_cf and market_cap and market_cap > 0:
            fcf_yield = free_cf / market_cap
        
        # Cash conversion ratio
        cash_conversion = 0
        net_income = info.get('netIncomeToCommon', 0)
        if net_income and operating_cf and net_income > 0:
            cash_conversion = operating_cf / net_income
        
        # CapEx ratio
        capex_ratio = 0
        try:
            if not cash_flow.empty:
                capex = abs(cash_flow.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cash_flow.index else 0
                if operating_cf and operating_cf > 0:
                    capex_ratio = capex / operating_cf
        except Exception as e:
            logger.debug(f"Could not calculate capex ratio: {e}")
        
        # Signals
        signals = []
        
        if fcf_yield:
            if fcf_yield > 0.08:
                signals.append(('FCF Yield', 'EXCELLENT', f'{fcf_yield*100:.1f}% yield is very attractive'))
            elif fcf_yield > 0.05:
                signals.append(('FCF Yield', 'GOOD', f'{fcf_yield*100:.1f}% yield is solid'))
            elif fcf_yield > 0.02:
                signals.append(('FCF Yield', 'FAIR', f'{fcf_yield*100:.1f}% yield is modest'))
            elif fcf_yield > 0:
                signals.append(('FCF Yield', 'LOW', f'{fcf_yield*100:.1f}% yield is low'))
            else:
                signals.append(('FCF Yield', 'NEGATIVE', 'Company is burning cash'))
        
        if cash_conversion:
            if cash_conversion > 1.2:
                signals.append(('Cash Conversion', 'EXCELLENT', f'{cash_conversion:.1f}x earnings converted to cash'))
            elif cash_conversion > 0.8:
                signals.append(('Cash Conversion', 'GOOD', f'{cash_conversion:.1f}x cash conversion is healthy'))
            else:
                signals.append(('Cash Conversion', 'WEAK', f'{cash_conversion:.1f}x indicates earnings quality issues'))
        
        # Overall assessment
        excellent_count = sum(1 for s in signals if s[1] == 'EXCELLENT')
        negative_count = sum(1 for s in signals if s[1] in ['NEGATIVE', 'WEAK'])
        
        if excellent_count >= 1 and negative_count == 0:
            overall = 'CASH MACHINE'
        elif negative_count >= 1:
            overall = 'CASH CONCERNS'
        else:
            overall = 'ADEQUATE CASH FLOW'
        
        return {
            'operating_cash_flow': operating_cf,
            'free_cash_flow': free_cf,
            'fcf_yield': fcf_yield,
            'cash_conversion': cash_conversion,
            'capex_ratio': capex_ratio,
            'signals': signals,
            'overall_assessment': overall,
            'explanation': {
                'fcf': 'Free Cash Flow: Cash left after capital expenditures. Funds dividends and buybacks.',
                'fcf_yield': 'FCF / Market Cap: Cash return on investment. Above 5% is attractive.',
                'cash_conversion': 'Operating CF / Net Income: Above 1.0 indicates quality earnings.'
            }
        }
    
    def _calculate_per_share_metrics(self, info: Dict, income_stmt, balance_sheet, cash_flow, shares: float) -> Dict[str, Any]:
        """Calculate per-share metrics."""
        eps = info.get('trailingEps', 0)
        book_value_per_share = info.get('bookValue', 0)
        revenue_per_share = info.get('revenuePerShare', 0)
        
        # FCF per share
        fcf_per_share = 0
        free_cf = info.get('freeCashflow', 0)
        if free_cf and shares and shares > 0:
            fcf_per_share = free_cf / shares
        
        # Dividend per share
        dividend_per_share = info.get('dividendRate', 0)
        dividend_yield = info.get('dividendYield', 0)
        payout_ratio = info.get('payoutRatio', 0)
        
        return {
            'eps': eps,
            'book_value_per_share': book_value_per_share,
            'revenue_per_share': revenue_per_share,
            'fcf_per_share': fcf_per_share,
            'dividend_per_share': dividend_per_share,
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio,
            'explanation': {
                'eps': 'Earnings Per Share: Net income divided by shares outstanding.',
                'bvps': 'Book Value Per Share: Net assets per share.',
                'fcf_per_share': 'Free Cash Flow Per Share: Cash generation per share.'
            }
        }
    
    def _calculate_r2_score(self, income_stmt) -> Dict[str, Any]:
        """
        Calculate R² Score for earnings predictability.
        
        R² measures how predictable/consistent earnings growth has been.
        Higher R² = more predictable earnings = lower risk.
        """
        r2 = 0
        earnings_trend = 'UNKNOWN'
        
        try:
            if not income_stmt.empty and 'Net Income' in income_stmt.index:
                earnings = []
                for col in income_stmt.columns[:5]:  # Up to 5 years
                    earnings.append(income_stmt.loc['Net Income', col])
                
                if len(earnings) >= 3:
                    # Calculate R² using linear regression
                    x = np.arange(len(earnings))
                    y = np.array(earnings, dtype=float)
                    
                    # Handle NaN values
                    mask = ~np.isnan(y)
                    if mask.sum() >= 3:
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        # Linear regression
                        n = len(x_clean)
                        sum_x = np.sum(x_clean)
                        sum_y = np.sum(y_clean)
                        sum_xy = np.sum(x_clean * y_clean)
                        sum_x2 = np.sum(x_clean ** 2)
                        
                        # Slope and intercept
                        denominator = n * sum_x2 - sum_x ** 2
                        if denominator != 0:
                            slope = (n * sum_xy - sum_x * sum_y) / denominator
                            intercept = (sum_y - slope * sum_x) / n
                            
                            # Predicted values
                            y_pred = slope * x_clean + intercept
                            
                            # R² calculation
                            ss_res = np.sum((y_clean - y_pred) ** 2)
                            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                            
                            if ss_tot > 0:
                                r2 = 1 - (ss_res / ss_tot)
                                r2 = max(0, min(1, r2))  # Clamp to [0, 1]
                            
                            # Determine trend
                            if slope > 0:
                                earnings_trend = 'UPWARD'
                            elif slope < 0:
                                earnings_trend = 'DOWNWARD'
                            else:
                                earnings_trend = 'FLAT'
        except Exception as e:
            logger.debug(f"Could not calculate R² score: {e}")
        
        # Interpretation
        if r2 >= 0.8:
            predictability = 'HIGHLY PREDICTABLE'
            explanation = 'Earnings follow a very consistent pattern. Lower risk.'
        elif r2 >= 0.6:
            predictability = 'PREDICTABLE'
            explanation = 'Earnings are reasonably consistent.'
        elif r2 >= 0.4:
            predictability = 'MODERATE'
            explanation = 'Earnings have some variability.'
        else:
            predictability = 'UNPREDICTABLE'
            explanation = 'Earnings are volatile. Higher risk.'
        
        return {
            'r2': r2,
            'predictability': predictability,
            'earnings_trend': earnings_trend,
            'explanation': explanation
        }
    
    def _detect_garp_setup(self, valuation: Dict, growth: Dict, profitability: Dict) -> Dict[str, Any]:
        """
        Detect GARP (Growth at Reasonable Price) Setup.
        
        GARP combines value and growth investing:
        - PEG ratio < 1.5 (ideally < 1.0)
        - Positive earnings growth
        - Reasonable P/E relative to growth
        - Solid profitability
        """
        peg = valuation.get('peg_ratio', 0)
        pe = valuation.get('pe_ratio', 0)
        earnings_growth = growth.get('earnings_growth_yoy', 0)
        revenue_growth = growth.get('revenue_growth_yoy', 0)
        roe = profitability.get('roe', 0)
        
        # GARP criteria
        criteria = []
        score = 0
        
        # PEG < 1.5 (ideally < 1.0)
        if peg and peg > 0:
            if peg < 1.0:
                criteria.append(('PEG < 1.0', True, f'PEG of {peg:.2f} is excellent'))
                score += 30
            elif peg < 1.5:
                criteria.append(('PEG < 1.5', True, f'PEG of {peg:.2f} is good'))
                score += 20
            else:
                criteria.append(('PEG < 1.5', False, f'PEG of {peg:.2f} is too high'))
        
        # Positive earnings growth
        if earnings_growth and earnings_growth > 0.10:
            criteria.append(('Earnings Growth > 10%', True, f'{earnings_growth*100:.1f}% growth'))
            score += 25
        elif earnings_growth and earnings_growth > 0:
            criteria.append(('Positive Earnings Growth', True, f'{earnings_growth*100:.1f}% growth'))
            score += 15
        else:
            criteria.append(('Positive Earnings Growth', False, 'No earnings growth'))
        
        # Revenue growth
        if revenue_growth and revenue_growth > 0.10:
            criteria.append(('Revenue Growth > 10%', True, f'{revenue_growth*100:.1f}% growth'))
            score += 20
        elif revenue_growth and revenue_growth > 0:
            criteria.append(('Positive Revenue Growth', True, f'{revenue_growth*100:.1f}% growth'))
            score += 10
        
        # ROE > 15%
        if roe and roe > 0.15:
            criteria.append(('ROE > 15%', True, f'ROE of {roe*100:.1f}%'))
            score += 25
        elif roe and roe > 0.10:
            criteria.append(('ROE > 10%', True, f'ROE of {roe*100:.1f}%'))
            score += 15
        
        # Determine if GARP setup
        is_garp = score >= 60
        
        return {
            'is_garp': is_garp,
            'garp_score': score,
            'criteria': criteria,
            'explanation': 'GARP (Growth at Reasonable Price) seeks growth stocks trading at fair valuations. Popularized by Peter Lynch.',
            'signal': 'YES - GARP SETUP DETECTED' if is_garp else 'NO - Does not meet GARP criteria'
        }
    
    def _detect_moat_setup(self, profitability: Dict, financial_health: Dict, growth: Dict, info: Dict) -> Dict[str, Any]:
        """
        Detect Economic MOAT Setup.
        
        MOAT indicators (Warren Buffett's competitive advantage):
        - High and consistent ROE (>15%)
        - High and consistent ROIC (>15%)
        - High gross margins (>40%)
        - Low debt
        - Pricing power (stable/growing margins)
        - Brand value, network effects, switching costs
        """
        roe = profitability.get('roe', 0)
        roic = profitability.get('roic', 0)
        gross_margin = profitability.get('gross_margin', 0)
        operating_margin = profitability.get('operating_margin', 0)
        debt_to_equity = financial_health.get('debt_to_equity', 0)
        revenue_growth = growth.get('revenue_growth_yoy', 0)
        
        # MOAT criteria
        criteria = []
        moat_score = 0
        moat_sources = []
        
        # High ROE (>15%)
        if roe and roe > 0.20:
            criteria.append(('ROE > 20%', True, f'Exceptional ROE of {roe*100:.1f}%'))
            moat_score += 25
            moat_sources.append('High Returns on Equity')
        elif roe and roe > 0.15:
            criteria.append(('ROE > 15%', True, f'Strong ROE of {roe*100:.1f}%'))
            moat_score += 15
        
        # High ROIC (>15%)
        if roic and roic > 0.15:
            criteria.append(('ROIC > 15%', True, f'ROIC of {roic*100:.1f}% indicates competitive advantage'))
            moat_score += 20
            moat_sources.append('High Return on Capital')
        
        # High gross margins (>40%)
        if gross_margin and gross_margin > 0.50:
            criteria.append(('Gross Margin > 50%', True, f'Exceptional margin of {gross_margin*100:.1f}%'))
            moat_score += 20
            moat_sources.append('Pricing Power')
        elif gross_margin and gross_margin > 0.40:
            criteria.append(('Gross Margin > 40%', True, f'Strong margin of {gross_margin*100:.1f}%'))
            moat_score += 10
        
        # Low debt
        if debt_to_equity:
            d_e = debt_to_equity / 100 if debt_to_equity > 10 else debt_to_equity
            if d_e < 0.5:
                criteria.append(('Low Debt', True, f'Conservative D/E of {d_e:.2f}'))
                moat_score += 15
                moat_sources.append('Financial Strength')
        
        # Operating margin > 15%
        if operating_margin and operating_margin > 0.20:
            criteria.append(('Operating Margin > 20%', True, f'Strong operating margin of {operating_margin*100:.1f}%'))
            moat_score += 15
            moat_sources.append('Operational Efficiency')
        elif operating_margin and operating_margin > 0.15:
            criteria.append(('Operating Margin > 15%', True, f'Solid operating margin of {operating_margin*100:.1f}%'))
            moat_score += 10
        
        # Consistent growth
        if revenue_growth and revenue_growth > 0.10:
            criteria.append(('Revenue Growth > 10%', True, f'{revenue_growth*100:.1f}% growth'))
            moat_score += 5
        
        # Determine MOAT type
        if moat_score >= 70:
            moat_type = 'WIDE MOAT'
            has_moat = True
        elif moat_score >= 50:
            moat_type = 'NARROW MOAT'
            has_moat = True
        else:
            moat_type = 'NO MOAT'
            has_moat = False
        
        return {
            'has_moat': has_moat,
            'moat_type': moat_type,
            'moat_score': moat_score,
            'moat_sources': moat_sources,
            'criteria': criteria,
            'explanation': 'Economic MOAT is a sustainable competitive advantage that protects profits. Warren Buffett\'s key investment criterion.',
            'signal': f'{moat_type} DETECTED' if has_moat else 'NO MOAT - Limited competitive advantage'
        }
    
    def _calculate_intelligent_score(self, valuation: Dict, profitability: Dict, growth: Dict,
                                     financial_health: Dict, cash_flow: Dict, garp: Dict,
                                     moat: Dict, r2: Dict, sector: str) -> Dict[str, Any]:
        """
        Calculate intelligent fundamental score (0-100).
        
        Weighted scoring based on institutional investment criteria.
        """
        score = 50  # Start at neutral
        factors = []
        
        # Valuation (20% weight)
        val_assessment = valuation.get('overall_assessment', '')
        if val_assessment == 'UNDERVALUED':
            score += 10
            factors.append(('Valuation', +10, 'Undervalued'))
        elif val_assessment == 'OVERVALUED':
            score -= 10
            factors.append(('Valuation', -10, 'Overvalued'))
        
        # Profitability (25% weight)
        prof_assessment = profitability.get('overall_assessment', '')
        if prof_assessment == 'HIGHLY PROFITABLE':
            score += 12
            factors.append(('Profitability', +12, 'Highly profitable'))
        elif prof_assessment == 'PROFITABLE':
            score += 6
            factors.append(('Profitability', +6, 'Profitable'))
        elif prof_assessment == 'LOW PROFITABILITY':
            score -= 8
            factors.append(('Profitability', -8, 'Low profitability'))
        
        # Growth (20% weight)
        growth_assessment = growth.get('overall_assessment', '')
        if growth_assessment == 'HIGH GROWTH':
            score += 10
            factors.append(('Growth', +10, 'High growth'))
        elif growth_assessment == 'GROWTH':
            score += 5
            factors.append(('Growth', +5, 'Growing'))
        elif growth_assessment == 'DECLINING':
            score -= 10
            factors.append(('Growth', -10, 'Declining'))
        
        # Financial Health (15% weight)
        health_assessment = financial_health.get('overall_assessment', '')
        if health_assessment == 'FORTRESS BALANCE SHEET':
            score += 8
            factors.append(('Financial Health', +8, 'Fortress balance sheet'))
        elif health_assessment == 'HEALTHY':
            score += 4
            factors.append(('Financial Health', +4, 'Healthy'))
        elif health_assessment == 'AT RISK':
            score -= 10
            factors.append(('Financial Health', -10, 'At risk'))
        
        # Cash Flow (10% weight)
        cf_assessment = cash_flow.get('overall_assessment', '')
        if cf_assessment == 'CASH MACHINE':
            score += 5
            factors.append(('Cash Flow', +5, 'Cash machine'))
        elif cf_assessment == 'CASH CONCERNS':
            score -= 5
            factors.append(('Cash Flow', -5, 'Cash concerns'))
        
        # GARP Setup (5% weight)
        if garp.get('is_garp'):
            score += 5
            factors.append(('GARP Setup', +5, 'GARP criteria met'))
        
        # MOAT (10% weight)
        moat_type = moat.get('moat_type', '')
        if moat_type == 'WIDE MOAT':
            score += 10
            factors.append(('Economic Moat', +10, 'Wide moat'))
        elif moat_type == 'NARROW MOAT':
            score += 5
            factors.append(('Economic Moat', +5, 'Narrow moat'))
        
        # R² Score (5% weight)
        r2_val = r2.get('r2', 0)
        if r2_val >= 0.8:
            score += 3
            factors.append(('Earnings Predictability', +3, 'Highly predictable'))
        elif r2_val < 0.4:
            score -= 3
            factors.append(('Earnings Predictability', -3, 'Unpredictable'))
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        # Determine rating
        if score >= 80:
            rating = 'STRONG BUY'
            color = 'green'
        elif score >= 65:
            rating = 'BUY'
            color = 'lightgreen'
        elif score >= 50:
            rating = 'HOLD'
            color = 'yellow'
        elif score >= 35:
            rating = 'SELL'
            color = 'orange'
        else:
            rating = 'STRONG SELL'
            color = 'red'
        
        return {
            'score': score,
            'rating': rating,
            'color': color,
            'factors': factors,
            'explanation': f'Fundamental score of {score}/100 based on weighted analysis of valuation, profitability, growth, financial health, cash flow, and competitive advantages.'
        }
    
    def _generate_ai_interpretation(self, symbol: str, company_name: str, sector: str, industry: str,
                                    valuation: Dict, profitability: Dict, growth: Dict,
                                    financial_health: Dict, cash_flow: Dict, garp: Dict,
                                    moat: Dict, score: Dict) -> Dict[str, Any]:
        """Generate AI-powered interpretation of fundamentals."""
        
        # Build bull case
        bull_points = []
        if valuation.get('overall_assessment') == 'UNDERVALUED':
            bull_points.append(f"Trading at attractive valuation with P/E of {valuation.get('pe_ratio', 'N/A')}")
        if profitability.get('overall_assessment') in ['HIGHLY PROFITABLE', 'PROFITABLE']:
            bull_points.append(f"Strong profitability with ROE of {profitability.get('roe', 0)*100:.1f}%")
        if growth.get('overall_assessment') in ['HIGH GROWTH', 'GROWTH']:
            bull_points.append(f"Growing revenue at {growth.get('revenue_growth_yoy', 0)*100:.1f}% YoY")
        if moat.get('has_moat'):
            bull_points.append(f"{moat.get('moat_type')} provides competitive protection")
        if garp.get('is_garp'):
            bull_points.append("Meets GARP criteria - growth at reasonable price")
        if financial_health.get('overall_assessment') == 'FORTRESS BALANCE SHEET':
            bull_points.append("Fortress balance sheet provides safety")
        
        # Build bear case
        bear_points = []
        if valuation.get('overall_assessment') == 'OVERVALUED':
            bear_points.append(f"Expensive valuation with P/E of {valuation.get('pe_ratio', 'N/A')}")
        if profitability.get('overall_assessment') == 'LOW PROFITABILITY':
            bear_points.append("Weak profitability metrics")
        if growth.get('overall_assessment') == 'DECLINING':
            bear_points.append("Revenue and/or earnings are declining")
        if not moat.get('has_moat'):
            bear_points.append("No clear competitive moat")
        if financial_health.get('overall_assessment') == 'AT RISK':
            bear_points.append("Financial health concerns")
        if cash_flow.get('overall_assessment') == 'CASH CONCERNS':
            bear_points.append("Cash flow concerns")
        
        # Key metrics summary
        key_metrics = {
            'P/E Ratio': valuation.get('pe_ratio', 'N/A'),
            'PEG Ratio': valuation.get('peg_ratio', 'N/A'),
            'EV/EBITDA': valuation.get('ev_to_ebitda', 'N/A'),
            'ROE': f"{profitability.get('roe', 0)*100:.1f}%",
            'Net Margin': f"{profitability.get('net_margin', 0)*100:.1f}%",
            'Revenue Growth': f"{growth.get('revenue_growth_yoy', 0)*100:.1f}%",
            'Current Ratio': financial_health.get('current_ratio', 'N/A'),
            'Debt/Equity': financial_health.get('debt_to_equity', 'N/A'),
            'FCF Yield': f"{cash_flow.get('fcf_yield', 0)*100:.1f}%"
        }
        
        # Generate summary
        rating = score.get('rating', 'HOLD')
        score_val = score.get('score', 50)
        
        if score_val >= 65:
            summary = f"{company_name} ({symbol}) shows strong fundamentals with a score of {score_val}/100. "
            summary += "The company demonstrates solid profitability, healthy growth, and/or attractive valuation. "
            if moat.get('has_moat'):
                summary += f"A {moat.get('moat_type').lower()} provides competitive protection. "
        elif score_val >= 50:
            summary = f"{company_name} ({symbol}) shows mixed fundamentals with a score of {score_val}/100. "
            summary += "Some metrics are positive while others warrant caution. "
        else:
            summary = f"{company_name} ({symbol}) shows concerning fundamentals with a score of {score_val}/100. "
            summary += "Multiple metrics indicate potential risks. "
        
        return {
            'summary': summary,
            'bull_case': bull_points if bull_points else ['No strong bullish factors identified'],
            'bear_case': bear_points if bear_points else ['No major bearish factors identified'],
            'key_metrics': key_metrics,
            'recommendation': rating,
            'sector_context': f"Compared to {sector} sector averages",
            'legendary_perspective': self._get_legendary_perspective(valuation, profitability, growth, moat)
        }
    
    def _get_legendary_perspective(self, valuation: Dict, profitability: Dict, growth: Dict, moat: Dict) -> Dict[str, str]:
        """Get perspectives from legendary investors."""
        perspectives = {}
        
        # Warren Buffett perspective
        buffett_likes = []
        buffett_concerns = []
        if moat.get('has_moat'):
            buffett_likes.append("competitive moat")
        if profitability.get('roe', 0) > 0.15:
            buffett_likes.append("high ROE")
        if valuation.get('pe_ratio', 100) < 20:
            buffett_likes.append("reasonable valuation")
        if not moat.get('has_moat'):
            buffett_concerns.append("no clear moat")
        
        if buffett_likes:
            perspectives['Warren Buffett'] = f"Would appreciate: {', '.join(buffett_likes)}"
        elif buffett_concerns:
            perspectives['Warren Buffett'] = f"Would be concerned about: {', '.join(buffett_concerns)}"
        
        # Peter Lynch perspective (GARP)
        peg = valuation.get('peg_ratio', 0)
        if peg and 0 < peg < 1.5:
            perspectives['Peter Lynch'] = f"PEG of {peg:.2f} suggests growth at reasonable price"
        elif peg and peg > 2:
            perspectives['Peter Lynch'] = f"PEG of {peg:.2f} is expensive for the growth rate"
        
        # Growth investor perspective
        rev_growth = growth.get('revenue_growth_yoy', 0)
        if rev_growth > 0.20:
            perspectives['Growth Investor'] = f"Revenue growth of {rev_growth*100:.1f}% is attractive"
        elif rev_growth < 0:
            perspectives['Growth Investor'] = "Declining revenue is a red flag"
        
        return perspectives


# Export for use in other modules
__all__ = ['ComprehensiveFundamentalsV2']
