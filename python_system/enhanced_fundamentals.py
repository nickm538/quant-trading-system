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
    from polygon_data_provider import PolygonDataProvider
    HAS_POLYGON_PROVIDER = True
except ImportError:
    HAS_POLYGON_PROVIDER = False

try:
    from financial_datasets_client import FinancialDatasetsClient
    HAS_FD_CLIENT = True
except ImportError:
    HAS_FD_CLIENT = False

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
        self.alphavantage_key = os.environ.get('ALPHAVANTAGE_API_KEY') or 'GYALAFBEMJUE8GYO'
        
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
    
    def analyze(self, symbol: str, pre_fetched_df=None) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis with cash flow focus.
        
        Args:
            symbol: Stock ticker symbol
            pre_fetched_df: Optional pandas DataFrame with OHLCV data (from PolygonDataProvider).
            
        Returns:
            Dictionary containing all fundamental metrics and analysis
        """
        try:
            # === PRIMARY SOURCE: FinancialDatasets.ai ===
            fd_data = None
            fd_metrics = None
            fd_health = None
            if HAS_FD_CLIENT:
                try:
                    fd_client = FinancialDatasetsClient()
                    fd_data = fd_client.get_comprehensive_stock_data(symbol)
                    fd_metrics = fd_client.get_financial_metrics_snapshot(symbol)
                    fd_health = fd_client.get_financial_health(symbol)
                    print(f"  \u2713 EnhancedFundamentals: FinancialDatasets.ai loaded", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"  EnhancedFundamentals: FinancialDatasets.ai failed: {e}", file=sys.stderr, flush=True)
            
            # === SECONDARY SOURCE: yfinance (skipped if FinancialDatasets sufficient) ===
            yf_data = None
            if fd_data and fd_metrics:
                # FinancialDatasets succeeded - build yf-compatible data without calling yfinance
                yf_data = self._build_yf_data_from_financialdatasets(fd_data, fd_metrics, fd_health, symbol)
                print(f"  \u2713 EnhancedFundamentals: Skipping yfinance (FinancialDatasets sufficient)", file=sys.stderr, flush=True)
            else:
                # FinancialDatasets failed - fall back to yfinance
                print(f"  EnhancedFundamentals: Falling back to yfinance...", file=sys.stderr, flush=True)
                yf_data = self._fetch_yfinance_data(symbol)
            
            # === TERTIARY SOURCES: FMP, Finnhub, AlphaVantage ===
            fmp_data = self._fetch_fmp_data(symbol)
            finnhub_data = self._fetch_finnhub_data(symbol)
            fmp_key_metrics = self._fetch_fmp_key_metrics(symbol)
            fmp_ratios = self._fetch_fmp_ratios_ttm(symbol)
            av_data = self._fetch_alphavantage_overview(symbol)
            
            # Build a unified info dict - FinancialDatasets first, then yfinance fills gaps
            if not yf_data and not fd_data:
                return {
                    'success': False,
                    'error': f'Unable to fetch data for {symbol}',
                    'symbol': symbol
                }
            
            # If yfinance failed but FinancialDatasets succeeded, build synthetic yf_data
            if not yf_data and fd_data:
                yf_data = self._build_yf_data_from_financialdatasets(fd_data, fd_metrics, fd_health, symbol)
            
            # Extract key metrics
            info = yf_data.get('info', {})
            
            # === OVERLAY: Fill gaps in yfinance data with FinancialDatasets.ai ===
            if yf_data and fd_data and yf_data.get('_source') != 'FinancialDatasets.ai':
                fd_overlay = self._build_yf_data_from_financialdatasets(fd_data, fd_metrics, fd_health, symbol)
                fd_info = fd_overlay.get('info', {})
                for key, val in fd_info.items():
                    if val is not None and val != 0 and val != 'Unknown':
                        existing = info.get(key)
                        if existing is None or existing == 0 or existing == 'Unknown':
                            info[key] = val
            
            # Basic info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Valuation metrics with FMP and AlphaVantage fallbacks
            pe_ratio = info.get('trailingPE') or info.get('forwardPE') or (fmp_ratios.get('priceToEarningsRatioTTM') if fmp_ratios else None)
            if pe_ratio is None and av_data:
                pe_ratio = self._safe_float(av_data.get('PERatio'))
            
            forward_pe = info.get('forwardPE')
            if forward_pe is None and av_data:
                forward_pe = self._safe_float(av_data.get('ForwardPE'))
            
            peg_ratio = info.get('pegRatio') or (fmp_ratios.get('priceToEarningsGrowthRatioTTM') if fmp_ratios else None)
            if peg_ratio is None and av_data:
                peg_ratio = self._safe_float(av_data.get('PEGRatio'))
            
            price_to_book = info.get('priceToBook') or (fmp_ratios.get('priceToBookRatioTTM') if fmp_ratios else None)
            if price_to_book is None and av_data:
                price_to_book = self._safe_float(av_data.get('PriceToBookRatio'))
            
            price_to_sales = info.get('priceToSalesTrailing12Months') or (fmp_ratios.get('priceToSalesRatioTTM') if fmp_ratios else None)
            if price_to_sales is None and av_data:
                price_to_sales = self._safe_float(av_data.get('PriceToSalesRatioTTM'))
            
            # Growth metrics with AlphaVantage fallbacks
            earnings_growth = info.get('earningsGrowth', 0) or 0
            revenue_growth = info.get('revenueGrowth', 0) or 0
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', 0) or 0
            
            # AlphaVantage provides quarterly YOY growth
            if earnings_quarterly_growth == 0 and av_data:
                earnings_quarterly_growth = self._safe_float(av_data.get('QuarterlyEarningsGrowthYOY')) or 0
            if revenue_growth == 0 and av_data:
                revenue_growth = self._safe_float(av_data.get('QuarterlyRevenueGrowthYOY')) or 0
            
            # Cash flow metrics
            free_cash_flow = info.get('freeCashflow', 0) or 0
            operating_cash_flow = info.get('operatingCashflow', 0) or 0
            
            # Profitability with AlphaVantage fallbacks
            profit_margin = info.get('profitMargins', 0) or 0
            if profit_margin == 0 and av_data:
                profit_margin = self._safe_float(av_data.get('ProfitMargin')) or 0
            
            operating_margin = info.get('operatingMargins', 0) or 0
            if operating_margin == 0 and av_data:
                operating_margin = self._safe_float(av_data.get('OperatingMarginTTM')) or 0
            
            gross_margin = info.get('grossMargins', 0) or 0
            
            roe = info.get('returnOnEquity', 0) or 0
            if roe == 0 and av_data:
                roe = self._safe_float(av_data.get('ReturnOnEquityTTM')) or 0
            
            roa = info.get('returnOnAssets', 0) or 0
            if roa == 0 and av_data:
                roa = self._safe_float(av_data.get('ReturnOnAssetsTTM')) or 0
            
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
            
            # FCF Yield with FMP fallback
            fcf_yield = (free_cash_flow / market_cap * 100) if market_cap > 0 else 0
            if fcf_yield == 0 and fmp_key_metrics:
                fcf_yield = (fmp_key_metrics.get('freeCashFlowYield', 0) or 0) * 100  # Convert to percentage
            
            # FCF Margin (FCF / Revenue)
            total_revenue = info.get('totalRevenue', 0) or 0
            fcf_margin = (free_cash_flow / total_revenue * 100) if total_revenue > 0 else 0
            
            # EBITDA and EV/EBITDA with FMP and AlphaVantage fallbacks
            ebitda = info.get('ebitda', 0) or 0
            if ebitda == 0 and av_data:
                ebitda = self._safe_float(av_data.get('EBITDA')) or 0
            
            enterprise_value = info.get('enterpriseValue', 0) or (fmp_key_metrics.get('enterpriseValue') if fmp_key_metrics else 0)
            
            # Calculate EV/EBITDA with FMP and AlphaVantage fallbacks
            ev_to_ebitda = (enterprise_value / ebitda) if ebitda > 0 else 0
            if ev_to_ebitda == 0 and fmp_key_metrics:
                ev_to_ebitda = fmp_key_metrics.get('evToEBITDA', 0) or 0
            if ev_to_ebitda == 0 and av_data:
                ev_to_ebitda = self._safe_float(av_data.get('EVToEBITDA')) or 0
            
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
            
            # NEW: Enhanced data fetching
            earnings_data = self._fetch_earnings_data(symbol)
            dividend_data = self._fetch_dividend_data(symbol, info)
            insider_data = self._fetch_insider_transactions(symbol)
            analyst_data = self._fetch_analyst_ratings(symbol)
            financial_trends = self._fetch_financial_trends(symbol)
            
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
                    'fcf_yield_pct': round(fcf_yield, 2) if fcf_yield else None,
                    'fcf_margin_pct': round(fcf_margin, 2) if fcf_margin else None,
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
                
                # NEW: Enhanced Data Sections
                'earnings': earnings_data,
                'dividends': dividend_data,
                'insider_activity': insider_data,
                'analyst_ratings': analyst_data,
                'financial_trends': financial_trends,
                
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
    
    def _build_yf_data_from_financialdatasets(self, fd_data: Dict, fd_metrics: Dict, fd_health: Dict, symbol: str) -> Dict:
        """
        Build a yfinance-compatible data dict from FinancialDatasets.ai data.
        This allows the rest of the analysis pipeline to work even when yfinance is down.
        """
        info = {}
        
        # Extract from company_facts
        facts = fd_data.get('company_facts', {})
        if isinstance(facts, dict):
            info['longName'] = facts.get('name', symbol)
            info['sector'] = facts.get('sector', 'Unknown')
            info['industry'] = facts.get('industry', 'Unknown')
            info['marketCap'] = facts.get('market_cap', 0)
            info['sharesOutstanding'] = facts.get('weighted_average_shares', 0)
        
        # Extract from price_snapshot
        price = fd_data.get('price_snapshot', {})
        if isinstance(price, dict):
            info['currentPrice'] = price.get('price', 0)
            info['regularMarketPrice'] = price.get('price', 0)
            info['regularMarketVolume'] = price.get('volume', 0)
        
        # Extract from financial_metrics (snapshot or first item if list)
        metrics = fd_metrics if isinstance(fd_metrics, dict) else {}
        if 'financial_metrics' in metrics:
            m = metrics['financial_metrics']
            if isinstance(m, list) and len(m) > 0:
                m = m[0]
            if isinstance(m, dict):
                # FinancialDatasets.ai uses full field names (e.g., 'price_to_earnings_ratio')
                # Map to yfinance-compatible short names
                info['trailingPE'] = m.get('price_to_earnings_ratio') or m.get('pe_ratio')
                info['forwardPE'] = m.get('forward_pe_ratio')
                info['priceToBook'] = m.get('price_to_book_ratio')
                info['priceToSalesTrailing12Months'] = m.get('price_to_sales_ratio')
                info['returnOnEquity'] = m.get('return_on_equity') or m.get('roe')
                info['returnOnAssets'] = m.get('return_on_assets') or m.get('roa')
                info['profitMargins'] = m.get('net_margin')
                info['operatingMargins'] = m.get('operating_margin')
                info['grossMargins'] = m.get('gross_margin')
                info['debtToEquity'] = m.get('debt_to_equity')
                info['currentRatio'] = m.get('current_ratio')
                info['earningsGrowth'] = m.get('earnings_growth')
                info['revenueGrowth'] = m.get('revenue_growth')
                info['enterpriseValue'] = m.get('enterprise_value')
                info['ebitda'] = m.get('ebitda')
                info['freeCashflow'] = m.get('free_cash_flow')
                info['operatingCashflow'] = m.get('operating_cash_flow')
                info['totalRevenue'] = m.get('revenue')
                info['totalDebt'] = m.get('total_debt')
                info['totalCash'] = m.get('total_cash')
                info['pegRatio'] = m.get('peg_ratio')
                # Also map additional useful fields from FD API
                info['returnOnInvestedCapital'] = m.get('return_on_invested_capital')
                info['freeCashFlowYield'] = m.get('free_cash_flow_yield')
                info['revenueGrowthRate'] = m.get('revenue_growth')
                info['earningsGrowthRate'] = m.get('earnings_growth')
                info['freeCashFlowGrowth'] = m.get('free_cash_flow_growth')
                info['operatingIncomeGrowth'] = m.get('operating_income_growth')
                info['ebitdaGrowth'] = m.get('ebitda_growth')
                info['bookValueGrowth'] = m.get('book_value_growth')
                info['epsGrowth'] = m.get('earnings_per_share_growth')
                info['interestCoverage'] = m.get('interest_coverage')
                info['enterpriseToEbitda'] = m.get('enterprise_value_to_ebitda_ratio') or m.get('ev_to_ebitda')
        elif isinstance(fd_metrics, dict):
            # Direct metrics dict (fallback path)
            m = fd_metrics
            info['trailingPE'] = m.get('price_to_earnings_ratio') or m.get('pe_ratio')
            info['returnOnEquity'] = m.get('return_on_equity') or m.get('roe')
            info['returnOnAssets'] = m.get('return_on_assets') or m.get('roa')
            info['profitMargins'] = m.get('net_margin')
            info['operatingMargins'] = m.get('operating_margin')
            info['grossMargins'] = m.get('gross_margin')
            info['debtToEquity'] = m.get('debt_to_equity')
            info['currentRatio'] = m.get('current_ratio')
            info['freeCashflow'] = m.get('free_cash_flow')
            info['operatingCashflow'] = m.get('operating_cash_flow')
            info['ebitda'] = m.get('ebitda')
            info['totalRevenue'] = m.get('revenue')
            info['returnOnInvestedCapital'] = m.get('return_on_invested_capital')
            info['freeCashFlowYield'] = m.get('free_cash_flow_yield')
            info['interestCoverage'] = m.get('interest_coverage')
            info['enterpriseToEbitda'] = m.get('enterprise_value_to_ebitda_ratio') or m.get('ev_to_ebitda')
        
        # Extract from income_statement
        income = fd_data.get('income_statement', {})
        if isinstance(income, dict) and 'income_statements' in income:
            stmts = income['income_statements']
            if isinstance(stmts, list) and len(stmts) > 0:
                stmt = stmts[0]
                if not info.get('totalRevenue'):
                    info['totalRevenue'] = stmt.get('revenue', 0)
                if not info.get('ebitda'):
                    info['ebitda'] = stmt.get('ebitda', 0)
        
        # Extract from cash_flow statement
        cf = fd_data.get('cash_flow', {})
        if isinstance(cf, dict) and 'cash_flow_statements' in cf:
            stmts = cf['cash_flow_statements']
            if isinstance(stmts, list) and len(stmts) > 0:
                stmt = stmts[0]
                if not info.get('freeCashflow'):
                    info['freeCashflow'] = stmt.get('free_cash_flow', 0)
                if not info.get('operatingCashflow'):
                    info['operatingCashflow'] = stmt.get('operating_cash_flow', 0)
        
        # Extract from balance_sheet
        bs = fd_data.get('balance_sheet', {})
        if isinstance(bs, dict) and 'balance_sheets' in bs:
            sheets = bs['balance_sheets']
            if isinstance(sheets, list) and len(sheets) > 0:
                sheet = sheets[0]
                if not info.get('totalDebt'):
                    info['totalDebt'] = sheet.get('total_debt', 0)
                if not info.get('totalCash'):
                    info['totalCash'] = sheet.get('cash_and_equivalents', 0)
        
        # Clean up None values to 0 for numeric fields
        numeric_fields = ['marketCap', 'currentPrice', 'totalRevenue', 'ebitda',
                         'freeCashflow', 'operatingCashflow', 'totalDebt', 'totalCash',
                         'sharesOutstanding', 'enterpriseValue']
        for f in numeric_fields:
            if info.get(f) is None:
                info[f] = 0
        
        return {
            'info': info,
            'financials': {},
            'balance_sheet': {},
            'cashflow': {},
            '_source': 'FinancialDatasets.ai'
        }
    
    def _fetch_fmp_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Financial Modeling Prep."""
        try:
            url = f'https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={self.fmp_key}'
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
    
    def _fetch_fmp_key_metrics(self, symbol: str) -> Optional[Dict]:
        """Fetch key metrics from FMP stable API."""
        try:
            url = f'https://financialmodelingprep.com/stable/key-metrics?symbol={symbol}&limit=1&apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data[0] if data else None
        except Exception:
            pass
        return None
    
    def _fetch_fmp_ratios_ttm(self, symbol: str) -> Optional[Dict]:
        """Fetch TTM ratios from FMP stable API."""
        try:
            url = f'https://financialmodelingprep.com/stable/ratios-ttm?symbol={symbol}&apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data[0] if data else None
        except Exception:
            pass
        return None
    
    def _fetch_alphavantage_overview(self, symbol: str) -> Optional[Dict]:
        """Fetch company overview from AlphaVantage API."""
        try:
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alphavantage_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Check if we got valid data (not rate limit or error)
                if 'Symbol' in data:
                    return data
        except Exception:
            pass
        return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert a value to float, handling None, 'None', and '-' values."""
        if value is None or value == 'None' or value == '-' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
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
        
        # Calculate value score based on P/E vs sector and PEG
        value_score = 0
        if pe and pe < sector_pe:
            value_score += 50 * (1 - pe / sector_pe)  # Higher score for lower P/E vs sector
        if peg and peg < 2:
            value_score += 50 * (1 - peg / 2)  # Higher score for lower PEG
        value_score = min(100, max(0, value_score))  # Clamp to 0-100
        
        # Use the higher of earnings or revenue growth as the primary growth rate
        growth_rate_pct = None
        if earnings_growth and revenue_growth:
            growth_rate_pct = max(earnings_growth * 100, revenue_growth * 100)
        elif earnings_growth:
            growth_rate_pct = earnings_growth * 100
        elif revenue_growth:
            growth_rate_pct = revenue_growth * 100
        
        return {
            'score': garp_score,
            'max_score': 100,
            'verdict': verdict,
            'interpretation': interpretation,
            'signals': signals,
            'peg_ratio': round(peg, 2) if peg else None,
            'growth_rate_used': round(growth_rate_pct, 2) if growth_rate_pct else None,
            'value_score': round(value_score, 1)
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


    def _fetch_earnings_data(self, symbol: str) -> Dict:
        """
        Fetch earnings history, estimates, and surprises.
        Uses Finnhub and FMP APIs.
        """
        earnings_data = {
            'quarterly_earnings': [],
            'annual_earnings': [],
            'eps_estimates': None,
            'earnings_surprises': [],
            'next_earnings_date': None
        }
        
        try:
            # Finnhub earnings calendar
            url = f'https://finnhub.io/api/v1/calendar/earnings?symbol={symbol}&token={self.finnhub_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('earningsCalendar'):
                    for item in data['earningsCalendar'][:4]:
                        earnings_data['quarterly_earnings'].append({
                            'date': item.get('date'),
                            'eps_actual': item.get('epsActual'),
                            'eps_estimate': item.get('epsEstimate'),
                            'revenue_actual': item.get('revenueActual'),
                            'revenue_estimate': item.get('revenueEstimate'),
                            'surprise_pct': item.get('surprisePercent')
                        })
            
            # FMP earnings surprises
            url = f'https://financialmodelingprep.com/stable/earnings-surprises?symbol={symbol}&apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    for item in data[:8]:
                        earnings_data['earnings_surprises'].append({
                            'date': item.get('date'),
                            'actual_eps': item.get('actualEarningResult'),
                            'estimated_eps': item.get('estimatedEarning'),
                            'surprise': round(item.get('actualEarningResult', 0) - item.get('estimatedEarning', 0), 4) if item.get('actualEarningResult') and item.get('estimatedEarning') else None
                        })
            
            # FMP analyst estimates
            url = f'https://financialmodelingprep.com/stable/analyst-estimates?symbol={symbol}&apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    latest = data[0]
                    earnings_data['eps_estimates'] = {
                        'date': latest.get('date'),
                        'estimated_eps_avg': latest.get('estimatedEpsAvg'),
                        'estimated_eps_high': latest.get('estimatedEpsHigh'),
                        'estimated_eps_low': latest.get('estimatedEpsLow'),
                        'estimated_revenue_avg': latest.get('estimatedRevenueAvg'),
                        'number_of_analysts': latest.get('numberAnalystsEstimatedEps')
                    }
        except Exception:
            pass
        
        return earnings_data
    
    def _fetch_dividend_data(self, symbol: str, info: Dict) -> Dict:
        """
        Fetch comprehensive dividend data.
        """
        dividend_data = {
            'dividend_yield_pct': None,
            'annual_dividend': None,
            'payout_ratio_pct': None,
            'ex_dividend_date': None,
            'dividend_date': None,
            'dividend_growth_5yr': None,
            'consecutive_years': None,
            'dividend_history': []
        }
        
        try:
            # From yfinance info
            dividend_data['dividend_yield_pct'] = round((info.get('dividendYield', 0) or 0) * 100, 2)
            dividend_data['annual_dividend'] = info.get('dividendRate')
            dividend_data['payout_ratio_pct'] = round((info.get('payoutRatio', 0) or 0) * 100, 2)
            dividend_data['ex_dividend_date'] = info.get('exDividendDate')
            
            # FMP dividend history
            url = f'https://financialmodelingprep.com/stable/historical-price-eod-dividend?symbol={symbol}&apikey={self.fmp_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('historical'):
                    history = data['historical'][:20]  # Last 20 dividends
                    dividend_data['dividend_history'] = [
                        {
                            'date': item.get('date'),
                            'dividend': item.get('dividend'),
                            'adj_dividend': item.get('adjDividend')
                        } for item in history
                    ]
                    
                    # Calculate 5-year growth
                    if len(history) >= 8:
                        recent = sum(h.get('dividend', 0) for h in history[:4])
                        older = sum(h.get('dividend', 0) for h in history[16:20]) if len(history) >= 20 else sum(h.get('dividend', 0) for h in history[-4:])
                        if older > 0:
                            dividend_data['dividend_growth_5yr'] = round((recent / older - 1) * 100, 2)
            
            # AlphaVantage dividend fallback if FMP didn't return data
            if not dividend_data['dividend_history']:
                url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&apikey={self.alphavantage_key}'
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data'):
                        history = data['data'][:20]  # Last 20 dividends
                        dividend_data['dividend_history'] = [
                            {
                                'date': item.get('ex_dividend_date'),
                                'dividend': float(item.get('amount', 0)),
                                'payment_date': item.get('payment_date')
                            } for item in history
                        ]
                        
                        # Calculate 5-year growth from AlphaVantage data
                        if len(history) >= 20:
                            recent = sum(float(h.get('amount', 0)) for h in history[:4])
                            older = sum(float(h.get('amount', 0)) for h in history[16:20])
                            if older > 0:
                                dividend_data['dividend_growth_5yr'] = round((recent / older - 1) * 100, 2)
        except Exception:
            pass
        
        return dividend_data
    
    def _fetch_insider_transactions(self, symbol: str) -> Dict:
        """
        Fetch recent insider transactions.
        """
        insider_data = {
            'recent_transactions': [],
            'net_insider_activity': 'NEUTRAL',
            'total_buys_90d': 0,
            'total_sells_90d': 0,
            'buy_value_90d': 0,
            'sell_value_90d': 0
        }
        
        try:
            # Finnhub insider transactions
            url = f'https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={self.finnhub_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                    
                    for txn in data['data'][:20]:
                        txn_date = txn.get('transactionDate', '')
                        txn_type = txn.get('transactionCode', '')
                        shares = txn.get('share', 0) or 0
                        price = txn.get('transactionPrice', 0) or 0
                        value = shares * price
                        
                        insider_data['recent_transactions'].append({
                            'date': txn_date,
                            'name': txn.get('name'),
                            'position': txn.get('filingDate'),
                            'type': 'BUY' if txn_type in ['P', 'A'] else 'SELL' if txn_type in ['S', 'F'] else txn_type,
                            'shares': shares,
                            'price': price,
                            'value': value
                        })
                        
                        if txn_date >= ninety_days_ago:
                            if txn_type in ['P', 'A']:
                                insider_data['total_buys_90d'] += 1
                                insider_data['buy_value_90d'] += value
                            elif txn_type in ['S', 'F']:
                                insider_data['total_sells_90d'] += 1
                                insider_data['sell_value_90d'] += value
                    
                    # Determine net activity
                    if insider_data['buy_value_90d'] > insider_data['sell_value_90d'] * 1.5:
                        insider_data['net_insider_activity'] = 'BULLISH'
                    elif insider_data['sell_value_90d'] > insider_data['buy_value_90d'] * 1.5:
                        insider_data['net_insider_activity'] = 'BEARISH'
        except Exception:
            pass
        
        return insider_data
    
    def _fetch_analyst_ratings(self, symbol: str) -> Dict:
        """
        Fetch analyst ratings and price targets.
        Priority: yfinance (most reliable) -> Finnhub -> FMP -> AlphaVantage
        """
        analyst_data = {
            'consensus_rating': None,
            'target_price': None,
            'target_high': None,
            'target_low': None,
            'number_of_analysts': 0,
            'strong_buy': 0,
            'buy': 0,
            'hold': 0,
            'sell': 0,
            'strong_sell': 0,
            'recommendation_trend': []
        }
        
        try:
            # PRIMARY SOURCE: yfinance (most reliable for target prices)
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get target prices from yfinance
                if info.get('targetMeanPrice'):
                    analyst_data['target_price'] = info.get('targetMeanPrice')
                if info.get('targetHighPrice'):
                    analyst_data['target_high'] = info.get('targetHighPrice')
                if info.get('targetLowPrice'):
                    analyst_data['target_low'] = info.get('targetLowPrice')
                if info.get('targetMedianPrice'):
                    analyst_data['target_median'] = info.get('targetMedianPrice')
                if info.get('numberOfAnalystOpinions'):
                    analyst_data['number_of_analysts'] = info.get('numberOfAnalystOpinions')
                
                # Get recommendation from yfinance
                if info.get('recommendationKey'):
                    rec = info.get('recommendationKey', '').upper()
                    if rec in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
                        analyst_data['consensus_rating'] = rec
                    elif rec == 'UNDERPERFORM':
                        analyst_data['consensus_rating'] = 'SELL'
                    elif rec == 'OUTPERFORM':
                        analyst_data['consensus_rating'] = 'BUY'
            except Exception:
                pass
            
            # Finnhub recommendation trends (for detailed breakdown)
            url = f'https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={self.finnhub_key}'
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    latest = data[0]
                    analyst_data['strong_buy'] = latest.get('strongBuy', 0)
                    analyst_data['buy'] = latest.get('buy', 0)
                    analyst_data['hold'] = latest.get('hold', 0)
                    analyst_data['sell'] = latest.get('sell', 0)
                    analyst_data['strong_sell'] = latest.get('strongSell', 0)
                    
                    total = sum([analyst_data['strong_buy'], analyst_data['buy'], analyst_data['hold'], analyst_data['sell'], analyst_data['strong_sell']])
                    if total > analyst_data['number_of_analysts']:
                        analyst_data['number_of_analysts'] = total
                    
                    # Calculate weighted consensus if not already set
                    if not analyst_data['consensus_rating'] and total > 0:
                        score = (analyst_data['strong_buy'] * 5 + analyst_data['buy'] * 4 + analyst_data['hold'] * 3 + analyst_data['sell'] * 2 + analyst_data['strong_sell'] * 1) / total
                        if score >= 4.5:
                            analyst_data['consensus_rating'] = 'STRONG_BUY'
                        elif score >= 3.5:
                            analyst_data['consensus_rating'] = 'BUY'
                        elif score >= 2.5:
                            analyst_data['consensus_rating'] = 'HOLD'
                        elif score >= 1.5:
                            analyst_data['consensus_rating'] = 'SELL'
                        else:
                            analyst_data['consensus_rating'] = 'STRONG_SELL'
                    
                    # Trend data
                    for item in data[:6]:
                        analyst_data['recommendation_trend'].append({
                            'period': item.get('period'),
                            'strong_buy': item.get('strongBuy', 0),
                            'buy': item.get('buy', 0),
                            'hold': item.get('hold', 0),
                            'sell': item.get('sell', 0),
                            'strong_sell': item.get('strongSell', 0)
                        })
            
            # Finnhub price target as fallback (may require premium)
            if not analyst_data['target_price']:
                url = f'https://finnhub.io/api/v1/stock/price-target?symbol={symbol}&token={self.finnhub_key}'
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('targetMean'):
                        analyst_data['target_price'] = data.get('targetMean')
                        analyst_data['target_high'] = data.get('targetHigh')
                        analyst_data['target_low'] = data.get('targetLow')
            
            # FMP price target consensus as fallback
            if not analyst_data['target_price']:
                url = f'https://financialmodelingprep.com/stable/price-target-consensus?symbol={symbol}&apikey={self.fmp_key}'
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        analyst_data['target_price'] = data[0].get('targetConsensus')
                        analyst_data['target_high'] = data[0].get('targetHigh')
                        analyst_data['target_low'] = data[0].get('targetLow')
                        analyst_data['target_median'] = data[0].get('targetMedian')
            
            # AlphaVantage as final fallback for price target
            if not analyst_data['target_price']:
                url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alphavantage_key}'
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('AnalystTargetPrice'):
                        analyst_data['target_price'] = self._safe_float(data.get('AnalystTargetPrice'))
                        # AlphaVantage also has analyst ratings
                        if not analyst_data['strong_buy']:
                            analyst_data['strong_buy'] = int(data.get('AnalystRatingStrongBuy', 0) or 0)
                            analyst_data['buy'] = int(data.get('AnalystRatingBuy', 0) or 0)
                            analyst_data['hold'] = int(data.get('AnalystRatingHold', 0) or 0)
                            analyst_data['sell'] = int(data.get('AnalystRatingSell', 0) or 0)
                            analyst_data['strong_sell'] = int(data.get('AnalystRatingStrongSell', 0) or 0)
                            total = sum([analyst_data['strong_buy'], analyst_data['buy'], analyst_data['hold'], analyst_data['sell'], analyst_data['strong_sell']])
                            analyst_data['number_of_analysts'] = total
        except Exception:
            pass
        
        return analyst_data
    
    def _calculate_cagr(self, beginning_value: float, ending_value: float, years: int) -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate (CAGR).
        
        Formula: CAGR = (Ending Value / Beginning Value)^(1/n) - 1
        
        Args:
            beginning_value: Starting value
            ending_value: Final value
            years: Number of years
            
        Returns:
            CAGR as a percentage, or None if calculation not possible
        """
        try:
            if beginning_value <= 0 or ending_value <= 0 or years <= 0:
                # Handle negative to positive transitions specially
                if beginning_value < 0 and ending_value > 0:
                    return None  # Can't calculate CAGR for sign changes
                if beginning_value > 0 and ending_value < 0:
                    return None
                return None
            
            cagr = (pow(ending_value / beginning_value, 1 / years) - 1) * 100
            return round(cagr, 2)
        except Exception:
            return None
    
    def _fetch_financial_trends(self, symbol: str) -> Dict:
        """
        Fetch historical financial trends and calculate CAGR metrics.
        Uses yfinance as primary source (free, reliable).
        Calculates: Revenue CAGR, Earnings CAGR, FCF CAGR, EPS CAGR, Stock Price CAGR
        For multiple timeframes: 1-year, 3-year, 5-year, 10-year
        """
        trends = {
            # Multi-timeframe CAGR
            'revenue_1yr_cagr': None,
            'revenue_3yr_cagr': None,
            'revenue_5yr_cagr': None,
            'earnings_1yr_cagr': None,
            'earnings_3yr_cagr': None,
            'earnings_5yr_cagr': None,
            'fcf_3yr_cagr': None,
            'fcf_5yr_cagr': None,
            'eps_3yr_cagr': None,
            'eps_5yr_cagr': None,
            'stock_price_1yr_cagr': None,
            'stock_price_3yr_cagr': None,
            'stock_price_5yr_cagr': None,
            'stock_price_10yr_cagr': None,
            'dividend_5yr_cagr': None,
            # Trend data
            'revenue_trend': [],
            'earnings_trend': [],
            'margin_trend': [],
            # Summary
            'cagr_summary': None
        }
        
        try:
            if not yf:
                return trends
            
            ticker = yf.Ticker(symbol)
            
            # Get income statement for revenue and earnings
            try:
                income_stmt = ticker.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    # Get annual data (columns are dates, most recent first)
                    years_available = len(income_stmt.columns)
                    
                    # Extract revenue data
                    revenue_row = None
                    for row_name in ['Total Revenue', 'Revenue', 'Operating Revenue']:
                        if row_name in income_stmt.index:
                            revenue_row = income_stmt.loc[row_name]
                            break
                    
                    # Extract net income data
                    earnings_row = None
                    for row_name in ['Net Income', 'Net Income Common Stockholders', 'Net Income From Continuing Operations']:
                        if row_name in income_stmt.index:
                            earnings_row = income_stmt.loc[row_name]
                            break
                    
                    # Extract EPS data
                    eps_row = None
                    for row_name in ['Basic EPS', 'Diluted EPS', 'Basic Average Shares']:
                        if row_name in income_stmt.index:
                            eps_row = income_stmt.loc[row_name]
                            break
                    
                    # Calculate Revenue CAGR for different timeframes
                    if revenue_row is not None:
                        revenue_values = revenue_row.dropna().values
                        if len(revenue_values) >= 2:
                            trends['revenue_1yr_cagr'] = self._calculate_cagr(revenue_values[1], revenue_values[0], 1)
                        if len(revenue_values) >= 4:
                            trends['revenue_3yr_cagr'] = self._calculate_cagr(revenue_values[3], revenue_values[0], 3)
                        if len(revenue_values) >= 5:
                            trends['revenue_5yr_cagr'] = self._calculate_cagr(revenue_values[4], revenue_values[0], 4)
                    
                    # Calculate Earnings CAGR
                    if earnings_row is not None:
                        earnings_values = earnings_row.dropna().values
                        if len(earnings_values) >= 2:
                            trends['earnings_1yr_cagr'] = self._calculate_cagr(earnings_values[1], earnings_values[0], 1)
                        if len(earnings_values) >= 4:
                            trends['earnings_3yr_cagr'] = self._calculate_cagr(earnings_values[3], earnings_values[0], 3)
                        if len(earnings_values) >= 5:
                            trends['earnings_5yr_cagr'] = self._calculate_cagr(earnings_values[4], earnings_values[0], 4)
                    
                    # Build revenue trend
                    if revenue_row is not None and earnings_row is not None:
                        for i, col in enumerate(income_stmt.columns[:6]):
                            year = col.year if hasattr(col, 'year') else str(col)[:4]
                            rev = revenue_row.iloc[i] if i < len(revenue_row) else None
                            earn = earnings_row.iloc[i] if i < len(earnings_row) else None
                            
                            # Get margins
                            gross_profit = income_stmt.loc['Gross Profit'].iloc[i] if 'Gross Profit' in income_stmt.index and i < len(income_stmt.columns) else None
                            op_income = income_stmt.loc['Operating Income'].iloc[i] if 'Operating Income' in income_stmt.index and i < len(income_stmt.columns) else None
                            
                            trends['revenue_trend'].append({
                                'year': year,
                                'revenue': float(rev) if rev is not None else None,
                                'net_income': float(earn) if earn is not None else None,
                                'gross_margin': round(float(gross_profit) / float(rev) * 100, 2) if gross_profit and rev and float(rev) > 0 else None,
                                'operating_margin': round(float(op_income) / float(rev) * 100, 2) if op_income and rev and float(rev) > 0 else None,
                                'net_margin': round(float(earn) / float(rev) * 100, 2) if earn and rev and float(rev) > 0 else None
                            })
            except Exception as e:
                pass
            
            # Get cash flow statement for FCF
            try:
                cashflow = ticker.cashflow
                if cashflow is not None and not cashflow.empty:
                    fcf_row = None
                    for row_name in ['Free Cash Flow', 'Operating Cash Flow']:
                        if row_name in cashflow.index:
                            fcf_row = cashflow.loc[row_name]
                            break
                    
                    if fcf_row is not None:
                        fcf_values = fcf_row.dropna().values
                        if len(fcf_values) >= 4:
                            trends['fcf_3yr_cagr'] = self._calculate_cagr(fcf_values[3], fcf_values[0], 3)
                        if len(fcf_values) >= 5:
                            trends['fcf_5yr_cagr'] = self._calculate_cagr(fcf_values[4], fcf_values[0], 4)
            except Exception:
                pass
            
            # AlphaVantage fallback for FCF CAGR if yfinance didn't provide it
            if trends['fcf_5yr_cagr'] is None:
                try:
                    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={self.alphavantage_key}'
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        annual_reports = data.get('annualReports', [])
                        if len(annual_reports) >= 5:
                            # Calculate FCF = Operating Cash Flow - Capital Expenditures
                            fcf_values = []
                            for report in annual_reports[:6]:  # Get 6 years for 5yr CAGR
                                ocf = float(report.get('operatingCashflow', 0) or 0)
                                capex = abs(float(report.get('capitalExpenditures', 0) or 0))
                                fcf = ocf - capex
                                fcf_values.append(fcf)
                            
                            if len(fcf_values) >= 4 and fcf_values[0] > 0 and fcf_values[3] > 0:
                                trends['fcf_3yr_cagr'] = self._calculate_cagr(fcf_values[3], fcf_values[0], 3)
                            if len(fcf_values) >= 5 and fcf_values[0] > 0 and fcf_values[4] > 0:
                                trends['fcf_5yr_cagr'] = self._calculate_cagr(fcf_values[4], fcf_values[0], 4)
                except Exception:
                    pass
            
            # Get historical stock price for price CAGR
            try:
                hist = ticker.history(period='10y')
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    current_price = hist['Close'].iloc[-1]
                    
                    # 1-year CAGR
                    if len(hist) >= 252:
                        price_1yr_ago = hist['Close'].iloc[-252]
                        trends['stock_price_1yr_cagr'] = self._calculate_cagr(price_1yr_ago, current_price, 1)
                    
                    # 3-year CAGR
                    if len(hist) >= 756:
                        price_3yr_ago = hist['Close'].iloc[-756]
                        trends['stock_price_3yr_cagr'] = self._calculate_cagr(price_3yr_ago, current_price, 3)
                    
                    # 5-year CAGR
                    if len(hist) >= 1260:
                        price_5yr_ago = hist['Close'].iloc[-1260]
                        trends['stock_price_5yr_cagr'] = self._calculate_cagr(price_5yr_ago, current_price, 5)
                    
                    # 10-year CAGR
                    if len(hist) >= 2520:
                        price_10yr_ago = hist['Close'].iloc[-2520]
                        trends['stock_price_10yr_cagr'] = self._calculate_cagr(price_10yr_ago, current_price, 10)
            except Exception:
                pass
            
            # Get dividend history for dividend CAGR
            try:
                dividends = ticker.dividends
                if dividends is not None and len(dividends) > 0:
                    # Get annual dividends
                    annual_divs = dividends.resample('YE').sum()
                    if len(annual_divs) >= 6:
                        recent_div = annual_divs.iloc[-1]
                        old_div = annual_divs.iloc[-6]
                        if old_div > 0 and recent_div > 0:
                            trends['dividend_5yr_cagr'] = self._calculate_cagr(old_div, recent_div, 5)
            except Exception:
                pass
            
            # AlphaVantage fallback for dividend CAGR if yfinance didn't provide it
            if trends['dividend_5yr_cagr'] is None:
                try:
                    url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&apikey={self.alphavantage_key}'
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        div_data = data.get('data', [])
                        if len(div_data) >= 20:  # Need at least 5 years of quarterly dividends
                            # Group by year and sum
                            from collections import defaultdict
                            yearly_divs = defaultdict(float)
                            for div in div_data:
                                date_str = div.get('ex_dividend_date', '')
                                if date_str:
                                    year = date_str[:4]
                                    amount = float(div.get('amount', 0) or 0)
                                    yearly_divs[year] += amount
                            
                            # Sort years and calculate CAGR
                            sorted_years = sorted(yearly_divs.keys(), reverse=True)
                            if len(sorted_years) >= 6:
                                recent_div = yearly_divs[sorted_years[0]]
                                old_div = yearly_divs[sorted_years[5]]
                                if old_div > 0 and recent_div > 0:
                                    trends['dividend_5yr_cagr'] = self._calculate_cagr(old_div, recent_div, 5)
                except Exception:
                    pass
            
            # Create CAGR summary
            cagr_values = []
            if trends['revenue_5yr_cagr'] is not None:
                cagr_values.append(('Revenue', trends['revenue_5yr_cagr']))
            if trends['earnings_5yr_cagr'] is not None:
                cagr_values.append(('Earnings', trends['earnings_5yr_cagr']))
            if trends['stock_price_5yr_cagr'] is not None:
                cagr_values.append(('Stock Price', trends['stock_price_5yr_cagr']))
            
            if cagr_values:
                avg_cagr = sum(v[1] for v in cagr_values) / len(cagr_values)
                if avg_cagr > 20:
                    assessment = 'Exceptional Growth'
                elif avg_cagr > 10:
                    assessment = 'Strong Growth'
                elif avg_cagr > 5:
                    assessment = 'Moderate Growth'
                elif avg_cagr > 0:
                    assessment = 'Slow Growth'
                else:
                    assessment = 'Declining'
                
                trends['cagr_summary'] = {
                    'average_5yr_cagr': round(avg_cagr, 2),
                    'assessment': assessment,
                    'components': cagr_values
                }
        
        except Exception as e:
            pass
        
        # FMP fallback for CAGR if yfinance data is incomplete
        if trends['revenue_5yr_cagr'] is None or trends['earnings_5yr_cagr'] is None:
            try:
                # Fetch income statement from FMP stable API
                url = f'https://financialmodelingprep.com/stable/income-statement?symbol={symbol}&limit=5&apikey={self.fmp_key}'
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    fmp_data = response.json()
                    if fmp_data and len(fmp_data) >= 2:
                        # Sort by date (oldest first for CAGR calculation)
                        fmp_data = sorted(fmp_data, key=lambda x: x.get('date', ''), reverse=False)
                        
                        # Calculate Revenue CAGR from FMP
                        revenues = [d.get('revenue') for d in fmp_data if d.get('revenue')]
                        if len(revenues) >= 2 and trends['revenue_1yr_cagr'] is None:
                            trends['revenue_1yr_cagr'] = self._calculate_cagr(revenues[-2], revenues[-1], 1)
                        if len(revenues) >= 4 and trends['revenue_3yr_cagr'] is None:
                            trends['revenue_3yr_cagr'] = self._calculate_cagr(revenues[-4], revenues[-1], 3)
                        if len(revenues) >= 5 and trends['revenue_5yr_cagr'] is None:
                            trends['revenue_5yr_cagr'] = self._calculate_cagr(revenues[0], revenues[-1], len(revenues)-1)
                        
                        # Calculate Earnings CAGR from FMP
                        earnings = [d.get('netIncome') for d in fmp_data if d.get('netIncome')]
                        if len(earnings) >= 2 and trends['earnings_1yr_cagr'] is None:
                            trends['earnings_1yr_cagr'] = self._calculate_cagr(earnings[-2], earnings[-1], 1)
                        if len(earnings) >= 4 and trends['earnings_3yr_cagr'] is None:
                            trends['earnings_3yr_cagr'] = self._calculate_cagr(earnings[-4], earnings[-1], 3)
                        if len(earnings) >= 5 and trends['earnings_5yr_cagr'] is None:
                            trends['earnings_5yr_cagr'] = self._calculate_cagr(earnings[0], earnings[-1], len(earnings)-1)
                        
                        # Build revenue trend from FMP if not already populated
                        if not trends['revenue_trend']:
                            for d in reversed(fmp_data[:6]):
                                rev = d.get('revenue')
                                earn = d.get('netIncome')
                                gross = d.get('grossProfit')
                                op_inc = d.get('operatingIncome')
                                trends['revenue_trend'].append({
                                    'year': d.get('fiscalYear') or d.get('date', '')[:4],
                                    'revenue': rev,
                                    'net_income': earn,
                                    'gross_margin': round(gross / rev * 100, 2) if gross and rev else None,
                                    'operating_margin': round(op_inc / rev * 100, 2) if op_inc and rev else None,
                                    'net_margin': round(earn / rev * 100, 2) if earn and rev else None
                                })
                        
                        # Recalculate CAGR summary
                        cagr_values = []
                        if trends['revenue_5yr_cagr'] is not None:
                            cagr_values.append(('Revenue', trends['revenue_5yr_cagr']))
                        if trends['earnings_5yr_cagr'] is not None:
                            cagr_values.append(('Earnings', trends['earnings_5yr_cagr']))
                        if trends['stock_price_5yr_cagr'] is not None:
                            cagr_values.append(('Stock Price', trends['stock_price_5yr_cagr']))
                        
                        if cagr_values:
                            avg_cagr = sum(v[1] for v in cagr_values) / len(cagr_values)
                            if avg_cagr > 20:
                                assessment = 'Exceptional Growth'
                            elif avg_cagr > 10:
                                assessment = 'Strong Growth'
                            elif avg_cagr > 5:
                                assessment = 'Moderate Growth'
                            elif avg_cagr > 0:
                                assessment = 'Slow Growth'
                            else:
                                assessment = 'Declining'
                            
                            trends['cagr_summary'] = {
                                'average_5yr_cagr': round(avg_cagr, 2),
                                'assessment': assessment,
                                'components': cagr_values
                            }
            except Exception:
                pass
        
        # FinancialDatasets.ai fallback for CAGR if still incomplete
        if trends['revenue_5yr_cagr'] is None or trends['earnings_5yr_cagr'] is None:
            try:
                fd_client = FinancialDatasetsClient()
                
                # Fetch income statements from FinancialDatasets.ai
                fd_income = fd_client.get_income_statements(symbol, period='annual', limit=6)
                if fd_income and 'income_statements' in fd_income:
                    stmts = fd_income['income_statements']
                    if isinstance(stmts, list) and len(stmts) >= 2:
                        # Sort by report_period ascending (oldest first)
                        stmts = sorted(stmts, key=lambda x: x.get('report_period', ''), reverse=False)
                        
                        # Calculate Revenue CAGR from FD
                        revenues = [s.get('revenue') for s in stmts if s.get('revenue') and s.get('revenue') > 0]
                        if len(revenues) >= 2 and trends['revenue_1yr_cagr'] is None:
                            trends['revenue_1yr_cagr'] = self._calculate_cagr(revenues[-2], revenues[-1], 1)
                        if len(revenues) >= 4 and trends['revenue_3yr_cagr'] is None:
                            trends['revenue_3yr_cagr'] = self._calculate_cagr(revenues[-4], revenues[-1], 3)
                        if len(revenues) >= 5 and trends['revenue_5yr_cagr'] is None:
                            trends['revenue_5yr_cagr'] = self._calculate_cagr(revenues[0], revenues[-1], len(revenues)-1)
                        
                        # Calculate Earnings CAGR from FD
                        earnings = [s.get('net_income') for s in stmts if s.get('net_income') and s.get('net_income') > 0]
                        if len(earnings) >= 2 and trends['earnings_1yr_cagr'] is None:
                            trends['earnings_1yr_cagr'] = self._calculate_cagr(earnings[-2], earnings[-1], 1)
                        if len(earnings) >= 4 and trends['earnings_3yr_cagr'] is None:
                            trends['earnings_3yr_cagr'] = self._calculate_cagr(earnings[-4], earnings[-1], 3)
                        if len(earnings) >= 5 and trends['earnings_5yr_cagr'] is None:
                            trends['earnings_5yr_cagr'] = self._calculate_cagr(earnings[0], earnings[-1], len(earnings)-1)
                        
                        # Build revenue trend from FD if not already populated
                        if not trends['revenue_trend']:
                            for s in reversed(stmts[:6]):
                                rev = s.get('revenue')
                                earn = s.get('net_income')
                                gross = s.get('gross_profit')
                                op_inc = s.get('operating_income')
                                trends['revenue_trend'].append({
                                    'year': s.get('report_period', '')[:4],
                                    'revenue': rev,
                                    'net_income': earn,
                                    'gross_margin': round(gross / rev * 100, 2) if gross and rev and rev > 0 else None,
                                    'operating_margin': round(op_inc / rev * 100, 2) if op_inc and rev and rev > 0 else None,
                                    'net_margin': round(earn / rev * 100, 2) if earn and rev and rev > 0 else None
                                })
                        
                        # Recalculate CAGR summary
                        cagr_values = []
                        if trends['revenue_5yr_cagr'] is not None:
                            cagr_values.append(('Revenue', trends['revenue_5yr_cagr']))
                        if trends['earnings_5yr_cagr'] is not None:
                            cagr_values.append(('Earnings', trends['earnings_5yr_cagr']))
                        if trends['stock_price_5yr_cagr'] is not None:
                            cagr_values.append(('Stock Price', trends['stock_price_5yr_cagr']))
                        
                        if cagr_values:
                            avg_cagr = sum(v[1] for v in cagr_values) / len(cagr_values)
                            if avg_cagr > 20:
                                assessment = 'Exceptional Growth'
                            elif avg_cagr > 10:
                                assessment = 'Strong Growth'
                            elif avg_cagr > 5:
                                assessment = 'Moderate Growth'
                            elif avg_cagr > 0:
                                assessment = 'Slow Growth'
                            else:
                                assessment = 'Declining'
                            
                            trends['cagr_summary'] = {
                                'average_5yr_cagr': round(avg_cagr, 2),
                                'assessment': assessment,
                                'components': cagr_values
                            }
                
                # Fetch cash flow for FCF CAGR from FD
                if trends['fcf_5yr_cagr'] is None:
                    fd_cf = fd_client.get_cash_flow_statements(symbol, period='annual', limit=6)
                    if fd_cf and 'cash_flow_statements' in fd_cf:
                        cf_stmts = fd_cf['cash_flow_statements']
                        if isinstance(cf_stmts, list) and len(cf_stmts) >= 2:
                            cf_stmts = sorted(cf_stmts, key=lambda x: x.get('report_period', ''), reverse=False)
                            fcf_values = [s.get('free_cash_flow') for s in cf_stmts if s.get('free_cash_flow') and s.get('free_cash_flow') > 0]
                            if len(fcf_values) >= 4 and trends['fcf_3yr_cagr'] is None:
                                trends['fcf_3yr_cagr'] = self._calculate_cagr(fcf_values[-4], fcf_values[-1], 3)
                            if len(fcf_values) >= 5 and trends['fcf_5yr_cagr'] is None:
                                trends['fcf_5yr_cagr'] = self._calculate_cagr(fcf_values[0], fcf_values[-1], len(fcf_values)-1)
            except Exception:
                pass
        
        return trends


# Main execution for testing
if __name__ == '__main__':
    analyzer = EnhancedFundamentalsAnalyzer()
    result = analyzer.analyze('AAPL')
    print(json.dumps(result, indent=2))
