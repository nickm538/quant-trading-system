"""
FINVIZ DATA SCRAPER
====================
Scrapes Finviz quote pages for data not available through free API tiers:
- Insider Ownership %
- Institutional Ownership %
- Short Interest (shares)
- Short Float %
- Short Ratio (days to cover)
- Dividend Yield %
- Payout Ratio
- EPS (ttm)
- P/E, Forward P/E, PEG
- And other fundamental metrics

Uses BeautifulSoup to parse the Finviz quote page table.
Respects rate limits with built-in delays.
"""

import requests
import time
import re
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional


class FinvizScraper:
    """Scrapes Finviz for fundamental data not available through free APIs."""
    
    BASE_URL = "https://finviz.com/quote.ashx"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://finviz.com/',
        'Connection': 'keep-alive',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._last_request_time = 0
        self._min_delay = 1.0  # Minimum 1 second between requests
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request_time = time.time()
    
    def _parse_percentage(self, value: str) -> Optional[float]:
        """Parse a percentage string like '0.10%' or '-2.50%' to a float."""
        if not value or value == '-':
            return None
        try:
            clean = value.replace('%', '').replace(',', '').strip()
            return float(clean)
        except (ValueError, TypeError):
            return None
    
    def _parse_number(self, value: str) -> Optional[float]:
        """Parse a number string with optional suffixes (K, M, B, T)."""
        if not value or value == '-':
            return None
        try:
            clean = value.replace(',', '').strip()
            multiplier = 1
            if clean.endswith('T'):
                multiplier = 1e12
                clean = clean[:-1]
            elif clean.endswith('B'):
                multiplier = 1e9
                clean = clean[:-1]
            elif clean.endswith('M'):
                multiplier = 1e6
                clean = clean[:-1]
            elif clean.endswith('K'):
                multiplier = 1e3
                clean = clean[:-1]
            return float(clean) * multiplier
        except (ValueError, TypeError):
            return None
    
    def _parse_ratio(self, value: str) -> Optional[float]:
        """Parse a ratio/decimal string like '2.36' or '-0.50'."""
        if not value or value == '-':
            return None
        try:
            return float(value.replace(',', '').strip())
        except (ValueError, TypeError):
            return None
    
    def scrape_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape the Finviz quote page for a given symbol.
        
        Returns a dict with all available fundamental data fields.
        """
        result = {
            'success': False,
            'symbol': symbol.upper(),
            'source': 'finviz_scraper',
            'data': {},
            'error': None,
        }
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}?t={symbol.upper()}&ty=c&p=d&b=1"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                result['error'] = f"HTTP {response.status_code}"
                return result
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the snapshot table — Finviz uses a table with class 'snapshot-table2'
            # or we can find all <td> pairs in the fundamentals table
            data = {}
            
            # Method 1: Find the snapshot table by class
            snapshot_table = soup.find('table', class_='snapshot-table2')
            if not snapshot_table:
                # Method 2: Try finding by structure — look for table containing known labels
                tables = soup.find_all('table')
                for table in tables:
                    text = table.get_text()
                    if 'Insider Own' in text and 'Inst Own' in text:
                        snapshot_table = table
                        break
            
            if not snapshot_table:
                result['error'] = 'Could not find snapshot table on page'
                return result
            
            # Parse all rows — each row has alternating label/value cells
            rows = snapshot_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                # Cells come in pairs: label, value, label, value, ...
                for i in range(0, len(cells) - 1, 2):
                    label = cells[i].get_text(strip=True)
                    value = cells[i + 1].get_text(strip=True)
                    if label and value:
                        data[label] = value
            
            if not data:
                result['error'] = 'No data extracted from snapshot table'
                return result
            
            # Map raw Finviz labels to structured fields
            parsed = self._parse_data(data)
            result['data'] = parsed
            result['raw_data'] = data  # Keep raw for debugging
            result['success'] = True
            
        except requests.Timeout:
            result['error'] = 'Request timed out'
        except requests.RequestException as e:
            result['error'] = f'Request failed: {str(e)}'
        except Exception as e:
            result['error'] = f'Parse error: {str(e)}'
        
        return result
    
    def _parse_data(self, raw: Dict[str, str]) -> Dict[str, Any]:
        """Parse raw Finviz label/value pairs into structured data."""
        parsed = {}
        
        # === SHARE STRUCTURE & SHORT INTEREST ===
        parsed['insider_ownership_pct'] = self._parse_percentage(raw.get('Insider Own', ''))
        parsed['institutional_ownership_pct'] = self._parse_percentage(raw.get('Inst Own', ''))
        parsed['shares_outstanding'] = self._parse_number(raw.get('Shs Outstand', ''))
        parsed['shares_float'] = self._parse_number(raw.get('Shs Float', ''))
        parsed['short_float_pct'] = self._parse_percentage(raw.get('Short Float', ''))
        parsed['short_ratio'] = self._parse_ratio(raw.get('Short Ratio', ''))
        parsed['short_interest'] = self._parse_number(raw.get('Short Interest', ''))
        
        # === DIVIDEND ===
        # Finviz format: "Dividend TTM" = "1.03 (0.37%)" or "Dividend Est." = "1.07 (0.39%)"
        div_ttm = raw.get('Dividend TTM', '')
        if div_ttm and '(' in div_ttm:
            parts = div_ttm.split('(')
            parsed['dividend_per_share'] = self._parse_ratio(parts[0].strip())
            parsed['dividend_yield_pct'] = self._parse_percentage(parts[1].replace(')', '').strip())
        else:
            parsed['dividend_per_share'] = None
            parsed['dividend_yield_pct'] = None
        
        div_est = raw.get('Dividend Est.', '')
        if div_est and '(' in div_est:
            parts = div_est.split('(')
            parsed['dividend_est_per_share'] = self._parse_ratio(parts[0].strip())
            parsed['dividend_est_yield_pct'] = self._parse_percentage(parts[1].replace(')', '').strip())
        else:
            parsed['dividend_est_per_share'] = None
            parsed['dividend_est_yield_pct'] = None
        
        # Dividend growth: "4.26% 4.98%" (3Y and 5Y)
        div_growth = raw.get('Dividend Gr. 3/5Y', '')
        if div_growth and ' ' in div_growth:
            growth_parts = div_growth.split()
            parsed['dividend_growth_3y_pct'] = self._parse_percentage(growth_parts[0])
            parsed['dividend_growth_5y_pct'] = self._parse_percentage(growth_parts[1]) if len(growth_parts) > 1 else None
        else:
            parsed['dividend_growth_3y_pct'] = None
            parsed['dividend_growth_5y_pct'] = None
        
        parsed['dividend_ex_date'] = raw.get('Dividend Ex-Date', '')
        parsed['payout_ratio_pct'] = self._parse_percentage(raw.get('Payout', ''))
        
        # === VALUATION ===
        parsed['pe_ratio'] = self._parse_ratio(raw.get('P/E', ''))
        parsed['forward_pe'] = self._parse_ratio(raw.get('Forward P/E', ''))
        parsed['peg_ratio'] = self._parse_ratio(raw.get('PEG', ''))
        parsed['ps_ratio'] = self._parse_ratio(raw.get('P/S', ''))
        parsed['pb_ratio'] = self._parse_ratio(raw.get('P/B', ''))
        parsed['price_to_cash'] = self._parse_ratio(raw.get('P/C', ''))
        parsed['price_to_fcf'] = self._parse_ratio(raw.get('P/FCF', ''))
        parsed['ev_to_ebitda'] = self._parse_ratio(raw.get('EV/EBITDA', ''))
        
        # === PROFITABILITY ===
        parsed['roe_pct'] = self._parse_percentage(raw.get('ROE', ''))
        parsed['roi_pct'] = self._parse_percentage(raw.get('ROI', ''))
        parsed['roa_pct'] = self._parse_percentage(raw.get('ROA', ''))
        parsed['gross_margin_pct'] = self._parse_percentage(raw.get('Gross Margin', ''))
        parsed['operating_margin_pct'] = self._parse_percentage(raw.get('Oper. Margin', ''))
        parsed['profit_margin_pct'] = self._parse_percentage(raw.get('Profit Margin', ''))
        
        # === GROWTH ===
        parsed['eps_ttm'] = self._parse_ratio(raw.get('EPS (ttm)', ''))
        parsed['eps_next_y'] = self._parse_ratio(raw.get('EPS next Y', ''))
        parsed['eps_growth_this_y_pct'] = self._parse_percentage(raw.get('EPS this Y', ''))
        parsed['eps_growth_next_y_pct'] = self._parse_percentage(raw.get('EPS next Y', ''))
        parsed['eps_growth_next_5y_pct'] = self._parse_percentage(raw.get('EPS next 5Y', ''))
        parsed['eps_growth_past_5y_pct'] = self._parse_percentage(raw.get('EPS past 5Y', ''))
        parsed['sales_growth_past_5y_pct'] = self._parse_percentage(raw.get('Sales past 5Y', ''))
        parsed['sales_growth_qoq_pct'] = self._parse_percentage(raw.get('Sales Q/Q', ''))
        parsed['eps_growth_qoq_pct'] = self._parse_percentage(raw.get('EPS Q/Q', ''))
        
        # === FINANCIAL HEALTH ===
        parsed['current_ratio'] = self._parse_ratio(raw.get('Current Ratio', ''))
        parsed['quick_ratio'] = self._parse_ratio(raw.get('Quick Ratio', ''))
        parsed['debt_to_equity'] = self._parse_ratio(raw.get('Debt/Eq', ''))
        parsed['lt_debt_to_equity'] = self._parse_ratio(raw.get('LT Debt/Eq', ''))
        
        # === MARKET DATA ===
        parsed['market_cap'] = self._parse_number(raw.get('Market Cap', ''))
        parsed['income'] = self._parse_number(raw.get('Income', ''))
        parsed['sales'] = self._parse_number(raw.get('Sales', ''))
        parsed['book_value_per_share'] = self._parse_ratio(raw.get('Book/sh', ''))
        parsed['cash_per_share'] = self._parse_ratio(raw.get('Cash/sh', ''))
        
        # === ADDITIONAL GROWTH ===
        parsed['eps_growth_yoy_ttm_pct'] = self._parse_percentage(raw.get('EPS Y/Y TTM', ''))
        parsed['sales_growth_yoy_ttm_pct'] = self._parse_percentage(raw.get('Sales Y/Y TTM', ''))
        # EPS past 3/5Y: "15.36% 17.08%" format
        eps_past = raw.get('EPS past 3/5Y', '')
        if eps_past and ' ' in eps_past:
            eps_parts = eps_past.split()
            parsed['eps_growth_past_3y_pct'] = self._parse_percentage(eps_parts[0])
            parsed['eps_growth_past_5y_pct'] = self._parse_percentage(eps_parts[1]) if len(eps_parts) > 1 else None
        else:
            parsed['eps_growth_past_3y_pct'] = self._parse_percentage(eps_past)
            parsed['eps_growth_past_5y_pct'] = None
        # Sales past 3/5Y
        sales_past = raw.get('Sales past 3/5Y', '')
        if sales_past and ' ' in sales_past:
            sales_parts = sales_past.split()
            parsed['sales_growth_past_3y_pct'] = self._parse_percentage(sales_parts[0])
            parsed['sales_growth_past_5y_pct'] = self._parse_percentage(sales_parts[1]) if len(sales_parts) > 1 else None
        else:
            parsed['sales_growth_past_3y_pct'] = self._parse_percentage(sales_past)
            parsed['sales_growth_past_5y_pct'] = None
        
        parsed['roic_pct'] = self._parse_percentage(raw.get('ROIC', ''))
        parsed['insider_transactions_pct'] = self._parse_percentage(raw.get('Insider Trans', ''))
        parsed['institutional_transactions_pct'] = self._parse_percentage(raw.get('Inst Trans', ''))
        parsed['enterprise_value'] = self._parse_number(raw.get('Enterprise Value', ''))
        parsed['ev_to_sales'] = self._parse_ratio(raw.get('EV/Sales', ''))
        
        # Volatility: "1.22% 1.45%" (week, month)
        vol = raw.get('Volatility', '')
        if vol and ' ' in vol:
            vol_parts = vol.split()
            parsed['volatility_week_pct'] = self._parse_percentage(vol_parts[0])
            parsed['volatility_month_pct'] = self._parse_percentage(vol_parts[1]) if len(vol_parts) > 1 else None
        else:
            parsed['volatility_week_pct'] = None
            parsed['volatility_month_pct'] = None
        
        # === TECHNICAL ===
        parsed['beta'] = self._parse_ratio(raw.get('Beta', ''))
        parsed['atr'] = self._parse_ratio(raw.get('ATR (14)', ''))
        parsed['sma20_pct'] = self._parse_percentage(raw.get('SMA20', ''))
        parsed['sma50_pct'] = self._parse_percentage(raw.get('SMA50', ''))
        parsed['sma200_pct'] = self._parse_percentage(raw.get('SMA200', ''))
        parsed['rsi'] = self._parse_ratio(raw.get('RSI (14)', ''))
        parsed['relative_volume'] = self._parse_ratio(raw.get('Rel Volume', ''))
        parsed['avg_volume'] = self._parse_number(raw.get('Avg Volume', ''))
        parsed['volume'] = self._parse_number(raw.get('Volume', ''))
        
        # === PRICE ===
        parsed['price'] = self._parse_ratio(raw.get('Price', ''))
        parsed['prev_close'] = self._parse_ratio(raw.get('Prev Close', ''))
        parsed['target_price'] = self._parse_ratio(raw.get('Target Price', ''))
        parsed['week_52_high'] = self._parse_ratio(raw.get('52W High', ''))
        parsed['week_52_low'] = self._parse_ratio(raw.get('52W Low', ''))
        parsed['change_pct'] = self._parse_percentage(raw.get('Change', ''))
        
        # === ANALYST ===
        parsed['analyst_recommendation'] = self._parse_ratio(raw.get('Recom', ''))
        
        # === EARNINGS ===
        parsed['earnings_date'] = raw.get('Earnings', '')
        
        # === COMPANY INFO ===
        parsed['sector'] = raw.get('Sector', '')
        parsed['industry'] = raw.get('Industry', '')
        parsed['country'] = raw.get('Country', '')
        parsed['employees'] = self._parse_number(raw.get('Employees', ''))
        
        return parsed
    
    def get_share_structure(self, symbol: str) -> Dict[str, Any]:
        """
        Get share structure data specifically for the enhanced fundamentals module.
        Returns fields matching what the frontend expects.
        """
        result = self.scrape_quote(symbol)
        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
            }
        
        data = result['data']
        return {
            'success': True,
            'source': 'finviz',
            'insider_ownership': data.get('insider_ownership_pct'),
            'institutional_ownership': data.get('institutional_ownership_pct'),
            'insider_transactions_pct': data.get('insider_transactions_pct'),
            'institutional_transactions_pct': data.get('institutional_transactions_pct'),
            'short_interest': data.get('short_interest'),
            'short_float_pct': data.get('short_float_pct'),
            'short_ratio': data.get('short_ratio'),
            'shares_outstanding': data.get('shares_outstanding'),
            'shares_float': data.get('shares_float'),
            'dividend_yield_pct': data.get('dividend_yield_pct'),
            'dividend_per_share': data.get('dividend_per_share'),
            'dividend_est_per_share': data.get('dividend_est_per_share'),
            'dividend_est_yield_pct': data.get('dividend_est_yield_pct'),
            'dividend_growth_3y_pct': data.get('dividend_growth_3y_pct'),
            'dividend_growth_5y_pct': data.get('dividend_growth_5y_pct'),
            'dividend_ex_date': data.get('dividend_ex_date'),
            'payout_ratio_pct': data.get('payout_ratio_pct'),
        }


# Standalone test
if __name__ == '__main__':
    scraper = FinvizScraper()
    result = scraper.scrape_quote('AAPL')
    if result['success']:
        d = result['data']
        print(f"\n=== AAPL Finviz Data ===")
        print(f"Insider Own:    {d.get('insider_ownership_pct')}%")
        print(f"Inst Own:       {d.get('institutional_ownership_pct')}%")
        print(f"Short Interest: {d.get('short_interest'):,.0f}" if d.get('short_interest') else "Short Interest: N/A")
        print(f"Short Float:    {d.get('short_float_pct')}%")
        print(f"Short Ratio:    {d.get('short_ratio')}")
        print(f"Dividend Yield: {d.get('dividend_yield_pct')}%")
        print(f"Dividend/Share: ${d.get('dividend_per_share')}")
        print(f"Payout Ratio:   {d.get('payout_ratio_pct')}%")
        print(f"PEG Ratio:      {d.get('peg_ratio')}")
        print(f"EV/EBITDA:      {d.get('ev_to_ebitda')}")
        print(f"Current Ratio:  {d.get('current_ratio')}")
        print(f"Quick Ratio:    {d.get('quick_ratio')}")
    else:
        print(f"Error: {result['error']}")
