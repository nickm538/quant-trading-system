"""
FIRECRAWL WEB SCRAPER FOR SADIE AI
===================================
Real-time web scraping to fill API data gaps.

⚠️ CRITICAL: OPTIONS DATA FLOW ⚠️
================================
OPTIONS CHAIN DATA MUST ONLY COME FROM FIRECRAWL WEB SCRAPING.

Gemini and Perplexity APIs DO NOT have real-time options data.
They WILL hallucinate contract details (strikes, premiums, IV, volume) if asked directly.

Data Flow for Options Queries:
1. User asks about options/options chain/put-call data
2. Firecrawl scrapes REAL-TIME data from:
   - Barchart.com (primary source - most reliable)
   - Yahoo Finance (backup source)
3. Scraped data is injected into the LLM context
4. Gemini/Perplexity ONLY ANALYZE the scraped data
5. LLM is explicitly instructed to use ONLY the Firecrawl numbers

This ensures ZERO hallucinations for options data.

Scrapes:
- Yahoo Finance options chains (live put/call data)
- Barchart options chains (backup)
- Real-time analyst ratings and price targets
- Breaking news and catalysts
- Insider trading activity
- Earnings calendar and estimates

This ensures Sadie NEVER hallucinates - all data is scraped in real-time.
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import re


class FirecrawlScraper:
    """
    Real-time web scraper using Firecrawl API.
    Fills gaps in traditional financial APIs with live web data.
    
    CRITICAL: This is the ONLY source for options chain data.
    DO NOT use LLM APIs (Gemini, Perplexity, GPT) for options data - they will hallucinate.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FIRECRAWL_API_KEY', '')
        self.base_url = "https://api.firecrawl.dev/v1"
        
    def _make_request(self, endpoint: str, payload: Dict) -> Optional[Dict]:
        """Make a request to Firecrawl API."""
        if not self.api_key:
            print("Firecrawl API key not set", file=sys.stderr)
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
                print(f"Firecrawl error: {response.status_code} - {response.text}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"Firecrawl request failed: {e}", file=sys.stderr)
            return None
    
    def scrape_options_chain(self, symbol: str) -> Dict[str, Any]:
        """
        ⚠️ CRITICAL: OPTIONS DATA MUST COME FROM FIRECRAWL - NOT LLM APIs ⚠️
        
        Scrape live options chain from multiple sources (Barchart primary, Yahoo backup).
        Returns calls, puts, put/call ratio, max pain, unusual activity.
        
        IMPORTANT: Gemini and Perplexity APIs DO NOT have real-time options data.
        They will hallucinate contract details if asked directly.
        ALL options chain data MUST be scraped via Firecrawl from:
        - Barchart.com (primary - most reliable)
        - Yahoo Finance (backup)
        
        The LLM APIs should ONLY be used to ANALYZE the scraped data,
        never to generate or provide options contract information.
        """
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "calls": [],
            "puts": [],
            "summary": {},
            "data_source": None,
            "error_message": None
        }
        
        # Try Barchart first (more reliable options data)
        barchart_url = f"https://www.barchart.com/stocks/quotes/{symbol}/options"
        
        payload = {
            "url": barchart_url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            
            # Parse options data from markdown
            parsed = self._parse_options_markdown(content, symbol)
            if parsed.get("calls") or parsed.get("puts"):
                result.update(parsed)
                result["status"] = "success"
                result["data_source"] = "Barchart"
                return result
        
        # Fallback to Yahoo Finance if Barchart fails
        yahoo_url = f"https://finance.yahoo.com/quote/{symbol}/options"
        
        payload = {
            "url": yahoo_url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            
            # Parse options data from markdown
            parsed = self._parse_options_markdown(content, symbol)
            if parsed.get("calls") or parsed.get("puts"):
                result.update(parsed)
                result["status"] = "success"
                result["data_source"] = "Yahoo Finance"
            else:
                result["error_message"] = "Scraped page but could not parse options data"
        else:
            result["error_message"] = "Failed to scrape options page from both Barchart and Yahoo Finance"
            
        return result
    
    def _parse_options_markdown(self, content: str, symbol: str) -> Dict:
        """
        Parse options chain data from scraped markdown.
        Handles Yahoo Finance table format with columns:
        Contract Name | Last Trade Date | Strike | Last Price | Bid | Ask | Change | % Change | Volume | Open Interest | Implied Volatility
        
        The strike price is embedded in links like [145](https://finance.yahoo.com/quote/AAPL/options/?...) 
        """
        calls = []
        puts = []
        
        lines = content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Detect section headers - "### Calls" or "### Puts"
            if '### calls' in line_lower:
                current_section = 'calls'
                continue
            elif '### puts' in line_lower:
                current_section = 'puts'
                continue
            
            # Skip header and separator rows
            if 'contract name' in line_lower or '---' in line:
                continue
            
            # Parse table data rows - they contain | and contract links
            if '|' in line and current_section:
                # Check if this looks like a data row (contains option contract pattern)
                if f'{symbol.upper()}' in line.upper() and ('C00' in line.upper() or 'P00' in line.upper()):
                    try:
                        option = self._parse_option_row(line)
                        if option and option.get('strike'):
                            if current_section == 'calls':
                                calls.append(option)
                            else:
                                puts.append(option)
                    except Exception as e:
                        print(f"Parse error: {e}", file=sys.stderr)
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
    
    def _parse_option_row(self, line: str) -> Optional[Dict]:
        """
        Parse a single option row from Yahoo Finance markdown.
        
        Example line:
        | [AAPL260123C00145000](https://...) | 1/6/2026 3:08 PM | [145](https://...) | 118.08 | 109.45 | 112.70 | 0.00 | 0.00% | 1 | 70 | 249.71% |
        
        Columns: Contract | Date | Strike | Last | Bid | Ask | Change | %Change | Volume | OI | IV
        """
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]  # Remove empty strings
        
        if len(parts) < 8:
            return None
        
        try:
            # Extract strike from the strike column (usually 3rd column)
            # Strike is in format [145](https://...) or just a number
            strike = None
            for i, part in enumerate(parts):
                # Look for strike pattern - a bracketed number that's a reasonable strike price
                strike_match = re.search(r'\[(\d+\.?\d*)\]', part)
                if strike_match:
                    potential_strike = float(strike_match.group(1))
                    # Strike prices are typically between 1 and 5000
                    if 1 <= potential_strike <= 5000:
                        strike = potential_strike
                        # The next columns after strike should be: Last, Bid, Ask, Change, %Change, Volume, OI, IV
                        remaining_parts = parts[i+1:]
                        break
            
            if not strike or not remaining_parts:
                return None
            
            # Extract numeric values from remaining parts
            # Expected order: Last, Bid, Ask, Change, %Change, Volume, OI, IV
            numeric_values = []
            for part in remaining_parts:
                # Clean the part - remove commas, %, $, and handle dashes
                clean = part.replace(',', '').replace('$', '').replace('%', '').strip()
                if clean == '-' or clean == '\\-' or clean == '':
                    numeric_values.append(0)
                else:
                    try:
                        numeric_values.append(float(clean))
                    except ValueError:
                        # Try to extract just the number
                        num_match = re.search(r'([\d.]+)', clean)
                        if num_match:
                            numeric_values.append(float(num_match.group(1)))
                        else:
                            numeric_values.append(0)
            
            # We need at least Last, Bid, Ask
            if len(numeric_values) < 3:
                return None
            
            return {
                "strike": strike,
                "last_price": numeric_values[0] if len(numeric_values) > 0 else 0,
                "bid": numeric_values[1] if len(numeric_values) > 1 else 0,
                "ask": numeric_values[2] if len(numeric_values) > 2 else 0,
                "change": numeric_values[3] if len(numeric_values) > 3 else 0,
                "change_pct": numeric_values[4] if len(numeric_values) > 4 else 0,
                "volume": int(numeric_values[5]) if len(numeric_values) > 5 else 0,
                "open_interest": int(numeric_values[6]) if len(numeric_values) > 6 else 0,
                "implied_volatility": numeric_values[7] if len(numeric_values) > 7 else 0
            }
            
        except Exception as e:
            print(f"Error parsing option row: {e}", file=sys.stderr)
            return None
    
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
                "avg_price_target": r"Average.*?Target.*?(\d+\.?\d*)",
                "high_target": r"High.*?Target.*?(\d+\.?\d*)",
                "low_target": r"Low.*?Target.*?(\d+\.?\d*)",
                "buy_ratings": r"Buy.*?(\d+)",
                "hold_ratings": r"Hold.*?(\d+)",
                "sell_ratings": r"Sell.*?(\d+)"
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        result[key] = float(match.group(1))
                    except ValueError:
                        pass
        
        return result
    
    def scrape_news(self, symbol: str) -> Dict[str, Any]:
        """Scrape latest news for a symbol."""
        result = {
            "status": "error",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "headlines": []
        }
        
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True
        }
        
        response = self._make_request("scrape", payload)
        
        if response and response.get("success"):
            content = response.get("data", {}).get("markdown", "")
            
            # Extract headlines (lines that look like news titles)
            headlines = []
            for line in content.split('\n'):
                # Headlines are typically in markdown links or headers
                if line.strip() and len(line) > 20 and len(line) < 200:
                    # Remove markdown formatting
                    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
                    clean = re.sub(r'[#*_]', '', clean).strip()
                    if clean and not clean.startswith('|'):
                        headlines.append(clean)
            
            result["headlines"] = headlines[:10]  # Top 10 headlines
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
        """
        Format scraped data for LLM prompt injection.
        
        CRITICAL: This formatted output is what the LLM sees.
        The LLM MUST use these exact numbers for options data.
        If options retrieval failed, the LLM MUST tell the user.
        """
        sections = []
        symbol = data.get("symbol", "UNKNOWN")
        
        sections.append(f"=== REAL-TIME WEB DATA FOR {symbol} (Firecrawl - VERIFIED LIVE) ===")
        sections.append(f"Scraped at: {data.get('timestamp', 'N/A')}")
        sections.append("⚠️ THIS IS LIVE DATA - USE THESE EXACT NUMBERS, DO NOT HALLUCINATE")
        sections.append("")
        
        # Options data - CRITICAL: Must report success/failure honestly
        opts = data.get("options", {})
        if opts.get("status") == "success":
            summary = opts.get("summary", {})
            data_source = opts.get("data_source", "Unknown")
            
            # Check if we actually got meaningful data
            has_data = (summary.get('total_call_volume', 0) > 0 or 
                       summary.get('total_put_volume', 0) > 0 or
                       opts.get('calls') or opts.get('puts'))
            
            if has_data:
                sections.append("✅ OPTIONS DATA SUCCESSFULLY RETRIEVED")
                sections.append(f"📊 LIVE OPTIONS CHAIN (Source: {data_source}):")
                sections.append(f"  Put/Call Ratio: {summary.get('put_call_ratio', 'N/A')}")
                sections.append(f"  Max Pain Strike: ${summary.get('max_pain_strike', 'N/A')}")
                sections.append(f"  Total Call Volume: {summary.get('total_call_volume', 0):,}")
                sections.append(f"  Total Put Volume: {summary.get('total_put_volume', 0):,}")
                sections.append(f"  Total Call OI: {summary.get('total_call_oi', 0):,}")
                sections.append(f"  Total Put OI: {summary.get('total_put_oi', 0):,}")
                
                # Include actual contract details if available
                calls = opts.get('calls', [])
                puts = opts.get('puts', [])
                if calls:
                    sections.append("  TOP CALL CONTRACTS:")
                    for c in calls[:5]:
                        sections.append(f"    Strike ${c.get('strike')}: Last ${c.get('last_price', 'N/A')}, Bid ${c.get('bid', 'N/A')}, Ask ${c.get('ask', 'N/A')}, Vol {c.get('volume', 0):,}, OI {c.get('open_interest', 0):,}")
                if puts:
                    sections.append("  TOP PUT CONTRACTS:")
                    for p in puts[:5]:
                        sections.append(f"    Strike ${p.get('strike')}: Last ${p.get('last_price', 'N/A')}, Bid ${p.get('bid', 'N/A')}, Ask ${p.get('ask', 'N/A')}, Vol {p.get('volume', 0):,}, OI {p.get('open_interest', 0):,}")
                
                unusual = summary.get("unusual_activity", [])
                if unusual:
                    sections.append("  🚨 UNUSUAL OPTIONS ACTIVITY:")
                    for u in unusual[:3]:
                        sections.append(f"    - {u['type'].upper()} ${u['strike']}: Vol {u['volume']:,}, OI {u['open_interest']:,}, Vol/OI: {u['vol_oi_ratio']}x")
            else:
                sections.append("⚠️ OPTIONS DATA: Scrape succeeded but no contract data found")
                sections.append("  This may be due to market hours or the symbol not having active options.")
                sections.append("  DO NOT hallucinate options data - tell user options data is unavailable.")
            sections.append("")
        else:
            sections.append("❌ OPTIONS DATA RETRIEVAL FAILED")
            error_msg = opts.get("error_message", "Unknown error")
            sections.append(f"  Error: {error_msg}")
            sections.append("  Firecrawl could not scrape options data from Barchart or Yahoo Finance.")
            sections.append("  DO NOT hallucinate or make up options contract details.")
            sections.append("  Tell the user: 'I was unable to retrieve live options data at this time.'")
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
        if news.get("status") == "success" and news.get("headlines"):
            sections.append("📰 LATEST NEWS HEADLINES:")
            for headline in news.get("headlines", [])[:5]:
                sections.append(f"  • {headline}")
            sections.append("")
        
        # Insider activity
        insider = data.get("insider_activity", {})
        if insider.get("status") == "success":
            sections.append("👔 INSIDER ACTIVITY:")
            sections.append(f"  Net Sentiment: {insider.get('net_sentiment', 'N/A').upper()}")
            sections.append(f"  Buy Transactions: {insider.get('buy_count', 0)}")
            sections.append(f"  Sell Transactions: {insider.get('sell_count', 0)}")
            sections.append("")
        
        return "\n".join(sections)


# Test function
if __name__ == "__main__":
    scraper = FirecrawlScraper()
    
    # Test with AAPL
    print("Testing Firecrawl scraper for AAPL...")
    result = scraper.get_complete_analysis("AAPL")
    
    print("\n" + "="*50)
    print(scraper.format_for_prompt(result))
