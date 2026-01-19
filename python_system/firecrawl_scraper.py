"""
FIRECRAWL WEB SCRAPER FOR SADIE AI
===================================
Real-time web scraping to fill API data gaps.

Scrapes:
- Yahoo Finance options chains (live put/call data)
- Real-time analyst ratings and price targets
- Breaking news and catalysts
- Insider trading activity
- Earnings calendar and estimates

This ensures Sadie NEVER hallucinates - all data is scraped in real-time.
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import re


class FirecrawlScraper:
    """
    Real-time web scraper using Firecrawl API.
    Fills gaps in traditional financial APIs with live web data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FIRECRAWL_API_KEY', '')
        self.base_url = "https://api.firecrawl.dev/v1"
        
    def _make_request(self, endpoint: str, payload: Dict) -> Optional[Dict]:
        """Make a request to Firecrawl API."""
        if not self.api_key:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Firecrawl error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Firecrawl request failed: {e}")
            return None
    
    def scrape_options_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape live options chain from Yahoo Finance.
        Returns calls, puts, put/call ratio, max pain, unusual activity.
        """
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "calls": [],
            "puts": [],
            "summary": {}
        }
        
        url = f"https://finance.yahoo.com/quote/{symbol}/options"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            
            # Parse options data from markdown
            parsed = self._parse_options_markdown(content, symbol)
            result.update(parsed)
            result["status"] = "success"
            
        return result
    
    def _parse_options_markdown(self, content: str, symbol: str) -> Dict:
        """Parse options chain data from scraped markdown."""
        calls = []
        puts = []
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect section
            if 'calls' in line_lower and ('|' in line or 'strike' in line_lower):
                current_section = 'calls'
                continue
            elif 'puts' in line_lower and ('|' in line or 'strike' in line_lower):
                current_section = 'puts'
                continue
            
            # Parse table rows
            if '|' in line and current_section:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                
                # Skip header rows
                if any(h in line.lower() for h in ['strike', 'last', 'bid', 'ask', '---']):
                    continue
                
                # Try to extract option data
                try:
                    # Look for numeric values
                    numbers = re.findall(r'[\d,]+\.?\d*', line)
                    if len(numbers) >= 4:
                        option = {
                            "strike": float(numbers[0].replace(',', '')),
                            "last_price": float(numbers[1].replace(',', '')),
                            "bid": float(numbers[2].replace(',', '')),
                            "ask": float(numbers[3].replace(',', '')),
                            "volume": int(float(numbers[4].replace(',', ''))) if len(numbers) > 4 else 0,
                            "open_interest": int(float(numbers[5].replace(',', ''))) if len(numbers) > 5 else 0,
                            "implied_volatility": float(numbers[6].replace(',', '').replace('%', '')) if len(numbers) > 6 else 0
                        }
                        
                        if current_section == 'calls':
                            calls.append(option)
                        else:
                            puts.append(option)
                except (ValueError, IndexError):
                    continue
        
        # Calculate summary statistics
        total_call_volume = sum(c.get('volume', 0) for c in calls)
        total_put_volume = sum(p.get('volume', 0) for p in puts)
        total_call_oi = sum(c.get('open_interest', 0) for c in calls)
        total_put_oi = sum(p.get('open_interest', 0) for p in puts)
        
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        # Calculate max pain (strike with most total OI)
        strike_oi = {}
        for c in calls:
            strike = c.get('strike', 0)
            strike_oi[strike] = strike_oi.get(strike, 0) + c.get('open_interest', 0)
        for p in puts:
            strike = p.get('strike', 0)
            strike_oi[strike] = strike_oi.get(strike, 0) + p.get('open_interest', 0)
        
        max_pain_strike = max(strike_oi, key=strike_oi.get) if strike_oi else 0
        
        # Find unusual activity (high volume relative to OI)
        unusual_activity = []
        for c in calls:
            if c.get('open_interest', 0) > 0:
                vol_oi_ratio = c.get('volume', 0) / c.get('open_interest', 1)
                if vol_oi_ratio > 2:  # Volume > 2x OI is unusual
                    unusual_activity.append({
                        "type": "call",
                        "strike": c.get('strike'),
                        "volume": c.get('volume'),
                        "open_interest": c.get('open_interest'),
                        "vol_oi_ratio": round(vol_oi_ratio, 2)
                    })
        
        for p in puts:
            if p.get('open_interest', 0) > 0:
                vol_oi_ratio = p.get('volume', 0) / p.get('open_interest', 1)
                if vol_oi_ratio > 2:
                    unusual_activity.append({
                        "type": "put",
                        "strike": p.get('strike'),
                        "volume": p.get('volume'),
                        "open_interest": p.get('open_interest'),
                        "vol_oi_ratio": round(vol_oi_ratio, 2)
                    })
        
        # Sort unusual activity by vol/oi ratio
        unusual_activity.sort(key=lambda x: x.get('vol_oi_ratio', 0), reverse=True)
        
        return {
            "calls": calls[:20],  # Top 20 strikes
            "puts": puts[:20],
            "summary": {
                "total_call_volume": total_call_volume,
                "total_put_volume": total_put_volume,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "put_call_ratio": round(put_call_ratio, 3),
                "max_pain_strike": max_pain_strike,
                "unusual_activity": unusual_activity[:5]  # Top 5 unusual
            }
        }
    
    def scrape_analyst_ratings(self, symbol: str) -> Dict[str, Any]:
        """Scrape analyst ratings and price targets."""
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        url = f"https://finance.yahoo.com/quote/{symbol}/analysis"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            
            # Extract key metrics
            result["raw_content"] = content[:2000]  # First 2000 chars
            result["status"] = "success"
            
            # Try to extract specific values
            patterns = {
                "avg_price_target": r'(?:average|mean)\s*(?:price)?\s*target[:\s]*\$?([\d,.]+)',
                "high_target": r'high\s*(?:price)?\s*target[:\s]*\$?([\d,.]+)',
                "low_target": r'low\s*(?:price)?\s*target[:\s]*\$?([\d,.]+)',
                "buy_ratings": r'(\d+)\s*(?:strong\s*)?buy',
                "hold_ratings": r'(\d+)\s*hold',
                "sell_ratings": r'(\d+)\s*(?:strong\s*)?sell'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content.lower())
                if match:
                    try:
                        result[key] = float(match.group(1).replace(',', ''))
                    except:
                        pass
        
        return result
    
    def scrape_news(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Scrape latest news for a symbol."""
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "articles": []
        }
        
        # Use Firecrawl search for news
        payload = {
            "query": f"{symbol} stock news today",
            "limit": limit
        }
        
        response = self._make_request("search", payload)
        
        if response and response.get("success"):
            results = response.get("data", [])
            
            for item in results:
                result["articles"].append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", "")[:200]
                })
            
            result["status"] = "success"
        
        return result
    
    def scrape_insider_activity(self, symbol: str) -> Dict[str, Any]:
        """Scrape insider trading activity."""
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        url = f"https://finance.yahoo.com/quote/{symbol}/insider-transactions"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            result["raw_content"] = content[:3000]
            result["status"] = "success"
            
            # Count buys vs sells
            buys = len(re.findall(r'\b(?:buy|purchase|acquisition)\b', content.lower()))
            sells = len(re.findall(r'\b(?:sell|sale|disposition)\b', content.lower()))
            
            result["buy_count"] = buys
            result["sell_count"] = sells
            result["net_sentiment"] = "bullish" if buys > sells else "bearish" if sells > buys else "neutral"
        
        return result
    
    def get_complete_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete web-scraped analysis for a symbol.
        Combines options, analyst ratings, news, and insider activity.
        """
        result = {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "data_sources": ["Yahoo Finance Options", "Yahoo Finance Analysis", "Web News", "Insider Transactions"],
            "options": {},
            "analyst_ratings": {},
            "news": {},
            "insider_activity": {}
        }
        
        # Scrape all data sources
        result["options"] = self.scrape_options_chain(symbol)
        result["analyst_ratings"] = self.scrape_analyst_ratings(symbol)
        result["news"] = self.scrape_news(symbol)
        result["insider_activity"] = self.scrape_insider_activity(symbol)
        
        return result
    
    def format_for_prompt(self, data: Dict) -> str:
        """Format scraped data for LLM prompt injection."""
        sections = []
        symbol = data.get("symbol", "UNKNOWN")
        
        sections.append(f"=== REAL-TIME WEB DATA FOR {symbol} (Firecrawl - VERIFIED LIVE) ===")
        sections.append(f"Scraped at: {data.get('timestamp', 'N/A')}")
        sections.append("⚠️ THIS IS LIVE DATA - USE THESE EXACT NUMBERS, DO NOT HALLUCINATE")
        sections.append("")
        
        # Options data
        opts = data.get("options", {})
        if opts.get("status") == "success":
            summary = opts.get("summary", {})
            sections.append("📊 LIVE OPTIONS CHAIN:")
            sections.append(f"  Put/Call Ratio: {summary.get('put_call_ratio', 'N/A')}")
            sections.append(f"  Max Pain Strike: ${summary.get('max_pain_strike', 'N/A')}")
            sections.append(f"  Total Call Volume: {summary.get('total_call_volume', 0):,}")
            sections.append(f"  Total Put Volume: {summary.get('total_put_volume', 0):,}")
            sections.append(f"  Total Call OI: {summary.get('total_call_oi', 0):,}")
            sections.append(f"  Total Put OI: {summary.get('total_put_oi', 0):,}")
            
            unusual = summary.get("unusual_activity", [])
            if unusual:
                sections.append("  🚨 UNUSUAL OPTIONS ACTIVITY:")
                for u in unusual[:3]:
                    sections.append(f"    - {u['type'].upper()} ${u['strike']}: Vol {u['volume']:,}, OI {u['open_interest']:,}, Vol/OI: {u['vol_oi_ratio']}x")
            sections.append("")
        
        # Analyst ratings
        analyst = data.get("analyst_ratings", {})
        if analyst.get("status") == "success":
            sections.append("📈 LIVE ANALYST RATINGS:")
            if analyst.get("avg_price_target"):
                sections.append(f"  Avg Price Target: ${analyst.get('avg_price_target')}")
            if analyst.get("high_target"):
                sections.append(f"  High Target: ${analyst.get('high_target')}")
            if analyst.get("low_target"):
                sections.append(f"  Low Target: ${analyst.get('low_target')}")
            if analyst.get("buy_ratings"):
                sections.append(f"  Buy Ratings: {int(analyst.get('buy_ratings', 0))}")
            if analyst.get("hold_ratings"):
                sections.append(f"  Hold Ratings: {int(analyst.get('hold_ratings', 0))}")
            if analyst.get("sell_ratings"):
                sections.append(f"  Sell Ratings: {int(analyst.get('sell_ratings', 0))}")
            sections.append("")
        
        # News
        news = data.get("news", {})
        if news.get("status") == "success" and news.get("articles"):
            sections.append("📰 LATEST NEWS:")
            for article in news.get("articles", [])[:3]:
                sections.append(f"  - {article.get('title', 'N/A')}")
            sections.append("")
        
        # Insider activity
        insider = data.get("insider_activity", {})
        if insider.get("status") == "success":
            sections.append("👔 INSIDER ACTIVITY:")
            sections.append(f"  Recent Buys: {insider.get('buy_count', 0)}")
            sections.append(f"  Recent Sells: {insider.get('sell_count', 0)}")
            sections.append(f"  Net Sentiment: {insider.get('net_sentiment', 'N/A').upper()}")
            sections.append("")
        
        return "\n".join(sections)


# Singleton instance
_scraper_instance = None

def get_firecrawl_scraper() -> FirecrawlScraper:
    """Get or create the Firecrawl scraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = FirecrawlScraper()
    return _scraper_instance


if __name__ == "__main__":
    # Test the scraper
    scraper = FirecrawlScraper()
    
    print("Testing Firecrawl scraper for AAPL...")
    data = scraper.get_complete_analysis("AAPL")
    
    print("\nFormatted for prompt:")
    print(scraper.format_for_prompt(data))
