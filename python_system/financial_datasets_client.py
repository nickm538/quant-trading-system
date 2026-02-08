"""
FinancialDatasets.ai API Client
================================
Premium financial data provider integration for the quant-trading-system.

This module provides access to:
- Real-time and historical stock prices
- Financial statements (income, balance sheet, cash flow)
- Financial metrics and ratios
- SEC filings and filing items
- Company facts and segmented revenues
- News articles
- Crypto prices

API Documentation: https://docs.financialdatasets.ai/
"""

import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class FinancialDatasetsClient:
    """
    Client for FinancialDatasets.ai API.
    
    Provides premium financial data including:
    - Stock prices (real-time snapshots and historical)
    - Financial statements (income, balance sheet, cash flow)
    - Financial metrics (P/E, EV/EBITDA, margins, etc.)
    - SEC filings and extracted items
    - Company facts (sector, employees, market cap)
    - Segmented revenues
    - News articles
    - Crypto prices
    """
    
    BASE_URL = "https://api.financialdatasets.ai"
    
    def __init__(self, api_key: str = None):
        """Initialize the client with API key."""
        self.api_key = api_key or os.environ.get('FINANCIAL_DATASETS_API_KEY', '9ca0e09f-df95-4310-b763-7d7c67d5b6c5')
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes cache
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make API request with error handling and caching."""
        cache_key = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        
        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data
        
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self._cache[cache_key] = (datetime.now(), data)
                return data
            elif response.status_code == 401:
                return {"error": "Invalid API key", "status": 401}
            elif response.status_code == 402:
                return {"error": "Payment required - API limit reached", "status": 402}
            elif response.status_code == 404:
                return {"error": "Data not found", "status": 404}
            else:
                return {"error": f"API error: {response.status_code}", "status": response.status_code}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout", "status": 408}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": 500}
    
    # ==================== STOCK PRICES ====================
    
    def get_stock_price_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time price snapshot for a stock.
        
        Returns: open, high, low, close, volume, and other price data.
        """
        return self._make_request("/prices/snapshot", {"ticker": ticker})
    
    def get_stock_prices(self, ticker: str, start_date: str = None, end_date: str = None,
                         interval: str = "day", limit: int = 252) -> Dict[str, Any]:
        """
        Get historical stock prices.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Time interval ('day', 'week', 'month')
            limit: Number of data points (default 252 = 1 year of trading days)
        """
        params = {"ticker": ticker, "interval": interval, "limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._make_request("/prices", params)
    
    # ==================== FINANCIAL STATEMENTS ====================
    
    def get_income_statement(self, ticker: str, period: str = "annual", 
                             limit: int = 5) -> Dict[str, Any]:
        """
        Get income statement data.
        
        Args:
            ticker: Stock symbol
            period: 'annual', 'quarterly', or 'ttm'
            limit: Number of periods to retrieve
        """
        return self._make_request("/financials/income-statements", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })
    
    def get_balance_sheet(self, ticker: str, period: str = "annual",
                          limit: int = 5) -> Dict[str, Any]:
        """
        Get balance sheet data.
        
        Returns: assets, liabilities, shareholders' equity.
        """
        return self._make_request("/financials/balance-sheets", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })
    
    def get_cash_flow_statement(self, ticker: str, period: str = "annual",
                                limit: int = 5) -> Dict[str, Any]:
        """
        Get cash flow statement data.
        
        Returns: operating, investing, financing cash flows.
        """
        return self._make_request("/financials/cash-flow-statements", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })
    
    # ==================== FINANCIAL METRICS ====================
    
    def get_financial_metrics_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Get current financial metrics snapshot.
        
        Returns: P/E ratio, market cap, dividend yield, margins, etc.
        """
        return self._make_request("/financial-metrics/snapshot", {"ticker": ticker})
    
    def get_financial_metrics(self, ticker: str, period: str = "annual",
                              limit: int = 10) -> Dict[str, Any]:
        """
        Get historical financial metrics.
        
        Returns: Historical P/E, EV/EBITDA, margins, ROE, etc.
        """
        return self._make_request("/financial-metrics", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })
    
    # ==================== COMPANY INFO ====================
    
    def get_company_facts(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive company facts.
        
        Returns: Market cap, employees, sector, industry, exchange, location, etc.
        """
        return self._make_request("/company/facts", {"ticker": ticker})
    
    def get_segmented_revenues(self, ticker: str, period: str = "annual",
                               limit: int = 5) -> Dict[str, Any]:
        """
        Get revenue breakdown by segment.
        
        Returns: Revenue by products, services, geographic regions.
        """
        return self._make_request("/financials/segmented-revenues", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })
    
    # ==================== SEC FILINGS ====================
    
    def get_filings(self, ticker: str, filing_type: str = None,
                    limit: int = 20) -> Dict[str, Any]:
        """
        Get SEC filings for a company.
        
        Args:
            ticker: Stock symbol
            filing_type: Filter by type ('10-K', '10-Q', '8-K', 'DEF 14A')
            limit: Number of filings to retrieve
        """
        params = {"ticker": ticker, "limit": limit}
        if filing_type:
            params["filing_type"] = filing_type
        return self._make_request("/filings", params)
    
    def get_filing_items(self, ticker: str, filing_type: str, 
                         year: int = None, items: List[str] = None) -> Dict[str, Any]:
        """
        Extract specific items from SEC filings.
        
        Args:
            ticker: Stock symbol
            filing_type: '10-K', '10-Q', or '8-K'
            year: Year of the filing
            items: List of items to extract (e.g., ['Item 1A', 'Item 7'])
        """
        params = {"ticker": ticker, "filing_type": filing_type}
        if year:
            params["year"] = year
        if items:
            params["item"] = items
        return self._make_request("/filings/items", params)
    
    # ==================== NEWS ====================
    
    def get_news(self, ticker: str, limit: int = 20,
                 start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Get recent news articles for a company.
        
        Returns: News articles with titles, summaries, sources, and dates.
        """
        params = {"ticker": ticker, "limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._make_request("/news", params)
    
    # ==================== CRYPTO ====================
    
    def get_crypto_price_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time crypto price snapshot.
        
        Args:
            ticker: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        """
        return self._make_request("/crypto/prices/snapshot", {"ticker": ticker})
    
    def get_crypto_prices(self, ticker: str, start_date: str = None,
                          end_date: str = None, limit: int = 365) -> Dict[str, Any]:
        """
        Get historical crypto prices.
        """
        params = {"ticker": ticker, "limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._make_request("/crypto/prices", params)
    
    # ==================== COMPREHENSIVE DATA FETCH ====================
    
    def get_comprehensive_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive data for a stock - all available data in one call.
        
        This is the main method for getting complete stock analysis data.
        Combines: price snapshot, financial metrics, company facts, 
        income statement, balance sheet, cash flow, and recent news.
        """
        data = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "source": "FinancialDatasets.ai"
        }
        
        # Price snapshot
        price = self.get_stock_price_snapshot(ticker)
        if "error" not in price:
            data["price_snapshot"] = price
        
        # Financial metrics snapshot
        metrics = self.get_financial_metrics_snapshot(ticker)
        if "error" not in metrics:
            data["financial_metrics"] = metrics
        
        # Company facts
        facts = self.get_company_facts(ticker)
        if "error" not in facts:
            data["company_facts"] = facts
        
        # Recent income statement (TTM)
        income = self.get_income_statement(ticker, period="ttm", limit=1)
        if "error" not in income:
            data["income_statement"] = income
        
        # Recent balance sheet
        balance = self.get_balance_sheet(ticker, period="quarterly", limit=1)
        if "error" not in balance:
            data["balance_sheet"] = balance
        
        # Recent cash flow
        cashflow = self.get_cash_flow_statement(ticker, period="ttm", limit=1)
        if "error" not in cashflow:
            data["cash_flow"] = cashflow
        
        # Recent news (last 10 articles)
        news = self.get_news(ticker, limit=10)
        if "error" not in news:
            data["recent_news"] = news
        
        return data
    
    def get_valuation_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Get valuation-focused data for a stock.
        
        Returns metrics useful for valuation analysis:
        - P/E, P/S, P/B, EV/EBITDA
        - Historical metrics for comparison
        - Revenue and earnings trends
        """
        data = {"ticker": ticker}
        
        # Current metrics
        metrics = self.get_financial_metrics_snapshot(ticker)
        if "error" not in metrics:
            data["current_metrics"] = metrics
        
        # Historical metrics (5 years)
        hist_metrics = self.get_financial_metrics(ticker, period="annual", limit=5)
        if "error" not in hist_metrics:
            data["historical_metrics"] = hist_metrics
        
        # Income trends
        income = self.get_income_statement(ticker, period="annual", limit=5)
        if "error" not in income:
            data["income_trends"] = income
        
        # Segmented revenues
        segments = self.get_segmented_revenues(ticker, limit=3)
        if "error" not in segments:
            data["revenue_segments"] = segments
        
        return data
    
    def get_financial_health(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial health indicators.
        
        Returns data for assessing financial stability:
        - Balance sheet strength
        - Cash flow quality
        - Debt levels
        - Liquidity ratios
        """
        data = {"ticker": ticker}
        
        # Balance sheet
        balance = self.get_balance_sheet(ticker, period="quarterly", limit=4)
        if "error" not in balance:
            data["balance_sheet"] = balance
        
        # Cash flow
        cashflow = self.get_cash_flow_statement(ticker, period="quarterly", limit=4)
        if "error" not in cashflow:
            data["cash_flow"] = cashflow
        
        # Metrics snapshot
        metrics = self.get_financial_metrics_snapshot(ticker)
        if "error" not in metrics:
            data["metrics"] = metrics
        
        return data
    
    def format_for_ai_context(self, ticker: str) -> str:
        """
        Get comprehensive data formatted as a string for AI context.
        
        This method returns a formatted string that can be directly
        injected into an AI prompt for analysis.
        """
        data = self.get_comprehensive_stock_data(ticker)
        
        lines = [f"=== FINANCIALDATASETS.AI DATA FOR {ticker} ==="]
        lines.append(f"Data Timestamp: {data.get('timestamp', 'N/A')}")
        
        # Price snapshot
        if "price_snapshot" in data:
            ps = data["price_snapshot"]
            snapshot = ps.get("snapshot", {})
            lines.append(f"\n--- PRICE SNAPSHOT ---")
            lines.append(f"Price: ${snapshot.get('price', 'N/A')}")
            lines.append(f"Day Change: {snapshot.get('day_change', 'N/A')}%")
            lines.append(f"Day High: ${snapshot.get('day_high', 'N/A')}")
            lines.append(f"Day Low: ${snapshot.get('day_low', 'N/A')}")
            lines.append(f"Volume: {snapshot.get('volume', 'N/A'):,}" if snapshot.get('volume') else "Volume: N/A")
            lines.append(f"52W High: ${snapshot.get('year_high', 'N/A')}")
            lines.append(f"52W Low: ${snapshot.get('year_low', 'N/A')}")
        
        # Financial metrics
        if "financial_metrics" in data:
            fm = data["financial_metrics"]
            metrics = fm.get("financial_metrics", {})
            lines.append(f"\n--- FINANCIAL METRICS ---")
            lines.append(f"Market Cap: ${metrics.get('market_cap', 'N/A'):,.0f}" if metrics.get('market_cap') else "Market Cap: N/A")
            lines.append(f"P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
            lines.append(f"Forward P/E: {metrics.get('forward_pe_ratio', 'N/A')}")
            lines.append(f"P/S Ratio: {metrics.get('ps_ratio', 'N/A')}")
            lines.append(f"P/B Ratio: {metrics.get('pb_ratio', 'N/A')}")
            lines.append(f"EV/EBITDA: {metrics.get('ev_to_ebitda', 'N/A')}")
            lines.append(f"Dividend Yield: {metrics.get('dividend_yield', 'N/A')}%")
            lines.append(f"ROE: {metrics.get('roe', 'N/A')}%")
            lines.append(f"ROA: {metrics.get('roa', 'N/A')}%")
            lines.append(f"Gross Margin: {metrics.get('gross_margin', 'N/A')}%")
            lines.append(f"Operating Margin: {metrics.get('operating_margin', 'N/A')}%")
            lines.append(f"Net Margin: {metrics.get('net_margin', 'N/A')}%")
            lines.append(f"Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}")
            lines.append(f"Current Ratio: {metrics.get('current_ratio', 'N/A')}")
        
        # Company facts
        if "company_facts" in data:
            cf = data["company_facts"]
            facts = cf.get("company_facts", {})
            lines.append(f"\n--- COMPANY FACTS ---")
            lines.append(f"Name: {facts.get('name', 'N/A')}")
            lines.append(f"Sector: {facts.get('sector', 'N/A')}")
            lines.append(f"Industry: {facts.get('industry', 'N/A')}")
            lines.append(f"Employees: {facts.get('employees', 'N/A'):,}" if facts.get('employees') else "Employees: N/A")
            lines.append(f"Exchange: {facts.get('exchange', 'N/A')}")
            lines.append(f"Website: {facts.get('website', 'N/A')}")
        
        # Income statement highlights
        if "income_statement" in data:
            inc = data["income_statement"]
            statements = inc.get("income_statements", [])
            if statements:
                latest = statements[0]
                lines.append(f"\n--- INCOME STATEMENT (TTM) ---")
                lines.append(f"Revenue: ${latest.get('revenue', 'N/A'):,.0f}" if latest.get('revenue') else "Revenue: N/A")
                lines.append(f"Gross Profit: ${latest.get('gross_profit', 'N/A'):,.0f}" if latest.get('gross_profit') else "Gross Profit: N/A")
                lines.append(f"Operating Income: ${latest.get('operating_income', 'N/A'):,.0f}" if latest.get('operating_income') else "Operating Income: N/A")
                lines.append(f"Net Income: ${latest.get('net_income', 'N/A'):,.0f}" if latest.get('net_income') else "Net Income: N/A")
                lines.append(f"EPS: ${latest.get('eps_diluted', 'N/A')}")
        
        # Recent news
        if "recent_news" in data:
            news = data["recent_news"]
            articles = news.get("news", [])
            if articles:
                lines.append(f"\n--- RECENT NEWS ({len(articles)} articles) ---")
                for i, article in enumerate(articles[:5], 1):
                    lines.append(f"{i}. {article.get('title', 'N/A')} ({article.get('date', 'N/A')})")
        
        return "\n".join(lines)


# Singleton instance for easy import
_client = None

def get_client() -> FinancialDatasetsClient:
    """Get or create the singleton client instance."""
    global _client
    if _client is None:
        _client = FinancialDatasetsClient()
    return _client


# Convenience functions for direct use
def get_stock_snapshot(ticker: str) -> Dict[str, Any]:
    """Get real-time stock price snapshot."""
    return get_client().get_stock_price_snapshot(ticker)

def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive stock data."""
    return get_client().get_comprehensive_stock_data(ticker)

def get_ai_context(ticker: str) -> str:
    """Get formatted data for AI context."""
    return get_client().format_for_ai_context(ticker)


if __name__ == "__main__":
    # Test the client
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"Fetching data for {ticker}...")
        print(get_ai_context(ticker))
    else:
        print("Usage: python financial_datasets_client.py TICKER")
        print("Example: python financial_datasets_client.py NVDA")
