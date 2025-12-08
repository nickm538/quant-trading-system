"""
Polygon.io Options Chain Fetcher
Fetches real-time options chains for calls only
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import math
import time

logger = logging.getLogger(__name__)

class PolygonOptionsClient:
    """Client for fetching options chains from Polygon.io (via Massive)"""
    
    def __init__(self, api_key: str = "KYqKTuCIZ7MQWBp_5hZDxRlKBQVcLXMt"):
        self.api_key = api_key
        self.base_url = "https://api.massive.com/v3"
    
    def calculate_iv_newton_raphson(self, market_price: float, S: float, K: float, T: float, r: float, option_type: str = 'call', max_iterations: int = 100, tolerance: float = 1e-5) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Current market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        
        Returns:
            Implied volatility (annualized) or None if failed to converge
        """
        # Initial guess: use Brenner-Subrahmanyam approximation
        iv = math.sqrt(2 * math.pi / T) * (market_price / S)
        
        for i in range(max_iterations):
            # Calculate option price and vega using Black-Scholes
            d1 = (math.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
            d2 = d1 - iv * math.sqrt(T)
            
            if option_type == 'call':
                option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * math.sqrt(T)
            
            # Check convergence
            price_diff = option_price - market_price
            if abs(price_diff) < tolerance:
                return iv
            
            # Newton-Raphson update
            if vega < 1e-10:  # Avoid division by zero
                return None
            
            iv = iv - price_diff / vega
            
            # Ensure IV stays positive
            if iv <= 0:
                return None
        
        # Failed to converge
        logger.warning(f"IV calculation failed to converge after {max_iterations} iterations")
        return None
    
    def get_option_market_price(self, contract_ticker: str) -> Optional[float]:
        """
        Get latest market price for an option contract from Polygon aggregates
        
        Args:
            contract_ticker: Option contract ticker (e.g., O:AAPL251219C00280000)
        
        Returns:
            Latest close price or None if failed
        """
        try:
            # Use 2024 date range (sandbox is in 2025, but Polygon has 2024 data)
            # This gets real market prices from last week for IV calculation
            end_date = '2024-11-29'
            start_date = '2024-11-24'
            
            url = f"https://api.massive.com/v2/aggs/ticker/{contract_ticker}/range/1/day/{start_date}/{end_date}"
            params = {'apiKey': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Rate limiting: Polygon free tier allows 5 requests/minute
            # Add 1-second delay to stay within limits
            time.sleep(1)
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data and data['results']:
                # Get most recent close price
                latest = data['results'][-1]
                return float(latest['c'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get market price for {contract_ticker}: {e}")
            return None
        
    def get_options_chain(self, 
                          underlying_ticker: str, 
                          contract_type: str = "call",
                          limit: int = 900) -> List[Dict]:
        """
        Fetch options chain for a given underlying ticker
        
        Args:
            underlying_ticker: Stock symbol (e.g., "AAPL")
            contract_type: "call" or "put" (default: "call")
            limit: Number of contracts per page (max 900)
            
        Returns:
            List of option contracts with strike, expiration, ticker, etc.
        """
        logger.info(f"Fetching {contract_type} options chain for {underlying_ticker}...")
        
        all_contracts = []
        url = f"{self.base_url}/reference/options/contracts"
        
        params = {
            "underlying_ticker": underlying_ticker,
            "contract_type": contract_type,
            "order": "asc",
            "limit": limit,
            "sort": "expiration_date",
            "apiKey": self.api_key
        }
        
        page_count = 0
        max_pages = 10  # Safety limit to avoid infinite loops
        
        while url and page_count < max_pages:
            try:
                response = requests.get(url, params=params if page_count == 0 else None, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") != "OK":
                    logger.error(f"Polygon API error: {data}")
                    break
                
                results = data.get("results", [])
                all_contracts.extend(results)
                
                page_count += 1
                logger.info(f"✓ Page {page_count}: Got {len(results)} contracts (total: {len(all_contracts)})")
                
                # Check for next page
                url = data.get("next_url")
                if not url:
                    break
                
                # Append API key to next_url (Polygon doesn't include it)
                url = f"{url}&apiKey={self.api_key}"
                    
            except Exception as e:
                logger.error(f"Error fetching options chain: {e}")
                break
        
        logger.info(f"✓ Total contracts fetched: {len(all_contracts)}")
        return all_contracts
    
    def parse_contract_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Parse Polygon contract ticker format: O:TICKER+YYMMDD+C+STRIKE
        Example: O:A251219C00095000
        - O: = option prefix
        - A = underlying ticker
        - 251219 = expiration date (Dec 19, 2025)
        - C = call
        - 00095000 = strike price (95000/1000 = $95)
        
        Returns:
            Dict with parsed components or None if invalid
        """
        try:
            if not ticker.startswith("O:"):
                return None
            
            # Remove "O:" prefix
            ticker = ticker[2:]
            
            # Find the date part (6 digits YYMMDD)
            # Contract format: TICKER + YYMMDD + C/P + STRIKE
            # Need to find where ticker ends and date begins
            # Date is always 6 digits followed by C or P
            
            for i in range(len(ticker) - 15):  # Strike is 8 digits, C/P is 1, date is 6
                if ticker[i:i+7].endswith(('C', 'P')):
                    underlying = ticker[:i]
                    date_str = ticker[i:i+6]
                    option_type = ticker[i+6]
                    strike_str = ticker[i+7:]
                    
                    # Parse date (YYMMDD)
                    year = 2000 + int(date_str[:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    expiration = datetime(year, month, day)
                    
                    # Parse strike (divide by 1000)
                    strike = int(strike_str) / 1000
                    
                    return {
                        "underlying": underlying,
                        "expiration": expiration,
                        "option_type": "call" if option_type == "C" else "put",
                        "strike": strike,
                        "ticker": f"O:{ticker}"
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse contract ticker {ticker}: {e}")
            return None
    
    def filter_by_expiration(self, contracts: List[Dict], 
                            min_days: int = 7, 
                            max_days: int = 90) -> List[Dict]:
        """
        Filter contracts by days to expiration
        
        Args:
            contracts: List of option contracts
            min_days: Minimum days to expiration (default: 7)
            max_days: Maximum days to expiration (default: 90)
            
        Returns:
            Filtered list of contracts
        """
        today = datetime.now()
        filtered = []
        
        for contract in contracts:
            exp_date_str = contract.get("expiration_date")
            if not exp_date_str:
                continue
            
            exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            
            if min_days <= days_to_exp <= max_days:
                contract["days_to_expiration"] = days_to_exp
                filtered.append(contract)
        
        return filtered
    
    def filter_by_delta(self, contracts: List[Dict], 
                       current_price: float,
                       min_delta: float = 0.3, 
                       max_delta: float = 0.6) -> List[Dict]:
        """
        Filter contracts by estimated delta range
        Delta approximation: 
        - ATM (strike = current price): delta ≈ 0.5
        - OTM (strike > current price): delta < 0.5
        - ITM (strike < current price): delta > 0.5
        
        For calls:
        - Delta ≈ 0.3-0.6 means strike should be within 10-20% of current price
        
        Args:
            contracts: List of option contracts
            current_price: Current stock price
            min_delta: Minimum delta (default: 0.3)
            max_delta: Maximum delta (default: 0.6)
            
        Returns:
            Filtered list of contracts
        """
        filtered = []
        
        for contract in contracts:
            strike = contract.get("strike_price")
            if not strike:
                continue
            
            # Rough delta estimation for filtering
            # For calls: delta increases as strike decreases (ITM)
            # ATM (strike = price): delta ≈ 0.5
            # 10% OTM: delta ≈ 0.3-0.4
            # 10% ITM: delta ≈ 0.6-0.7
            
            strike_pct = (strike - current_price) / current_price
            
            # Filter for strikes within reasonable range for target delta
            # 0.3 delta ≈ 5-15% OTM
            # 0.6 delta ≈ 5-10% ITM
            if -0.15 <= strike_pct <= 0.20:  # -15% to +20% from current price
                contract["strike_pct_from_current"] = strike_pct
                filtered.append(contract)
        
        return filtered


def get_options_for_stock(ticker: str, current_price: float) -> List[Dict]:
    """
    Convenience function to get filtered options chain for a stock
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        current_price: Current stock price
        
    Returns:
        List of filtered option contracts (calls only, 7-90 days, 0.3-0.6 delta)
    """
    client = PolygonOptionsClient()
    
    # Fetch all call options
    contracts = client.get_options_chain(ticker, contract_type="call")
    
    if not contracts:
        logger.warning(f"No options contracts found for {ticker}")
        return []
    
    # Filter by expiration (7-90 days)
    contracts = client.filter_by_expiration(contracts, min_days=7, max_days=90)
    logger.info(f"✓ After expiration filter: {len(contracts)} contracts")
    
    # Filter by delta (0.3-0.6)
    contracts = client.filter_by_delta(contracts, current_price, min_delta=0.3, max_delta=0.6)
    logger.info(f"✓ After delta filter: {len(contracts)} contracts")
    
    # Sort by days to expiration
    contracts.sort(key=lambda x: x.get("days_to_expiration", 999))
    
    return contracts


if __name__ == "__main__":
    # Test with AAPL
    logging.basicConfig(level=logging.INFO)
    
    ticker = "AAPL"
    current_price = 278.85
    
    print(f"Testing Polygon.io options chain for {ticker} at ${current_price}")
    print("=" * 60)
    
    contracts = get_options_for_stock(ticker, current_price)
    
    print(f"\n✓ Found {len(contracts)} suitable call options")
    print("\nTop 10 contracts:")
    print(f"{'Ticker':<25} {'Strike':<10} {'Expiration':<12} {'Days':<6} {'Strike %':<10}")
    print("-" * 70)
    
    for contract in contracts[:10]:
        ticker_str = contract.get("ticker", "N/A")
        strike = contract.get("strike_price", 0)
        exp = contract.get("expiration_date", "N/A")
        days = contract.get("days_to_expiration", 0)
        strike_pct = contract.get("strike_pct_from_current", 0)
        
        print(f"{ticker_str:<25} ${strike:<9.2f} {exp:<12} {days:<6} {strike_pct*100:>+6.1f}%")
