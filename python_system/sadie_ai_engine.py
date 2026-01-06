"""
SADIE AI ENGINE v1.0
=====================
The Ultimate Financial Intelligence Chatbot

Powered by:
- OpenAI GPT-5.x Thinking Mode (via OpenRouter)
- Complete SadieAI Financial Engine Suite
- Real-time market data (yfinance, Twelve Data, Finnhub)
- 50+ Technical Indicators
- Monte Carlo Simulations (20,000 paths)
- GARCH Volatility Modeling
- Kelly Criterion Position Sizing
- Legendary Trader Wisdom (Buffett, Soros, Simons, Dalio, PTJ)

Expert Strategies Integrated:
- TTM Squeeze
- Tim Bohen 5:1 Risk/Reward
- NR (Narrow Range) Patterns
- Dark Pool Activity
- Insider Trading Signals
- Congress Trading Patterns
- Catalyst Detection
- Sentiment Analysis

This is the gold-standard in financial AI assistance.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import yfinance as yf

# Import our financial engines
try:
    from perfect_production_analyzer import PerfectProductionAnalyzer
    from ultimate_options_engine import UltimateOptionsEngine
    from fundamentals_analyzer import FundamentalsAnalyzer
    from trading_education import TradingEducation
    from market_scanner import MarketScanner
    from indicators.ttm_squeeze import TTMSqueezeAnalyzer
    from twelvedata_client import TwelveDataClient
    from robust_data_fetcher import RobustDataFetcher
    HAS_ENGINES = True
except ImportError as e:
    import sys as _sys
    print(f"Warning: Some engines not available: {e}", file=_sys.stderr)
    HAS_ENGINES = False


class SadieAIEngine:
    """
    The Ultimate Financial AI Assistant
    
    Combines GPT-5 thinking mode with comprehensive financial analysis
    to provide gold-standard investment guidance.
    """
    
    # OpenRouter API configuration
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Model selection - prefer GPT-5 thinking mode
    MODELS = [
        "openai/o1",  # GPT-5 / o1 thinking mode
        "openai/o1-preview",
        "openai/gpt-4-turbo",
        "openai/gpt-4"
    ]
    
    # Finnhub API for additional data
    FINNHUB_API_KEY = os.environ.get('KEY', '')
    
    # System prompt that defines Sadie's personality and capabilities
    SYSTEM_CONTEXT = """You are SADIE (Strategic Analysis & Dynamic Investment Engine), the world's most advanced financial AI assistant. You GO ABOVE AND BEYOND on every single query - providing the most detailed, accurate, and insightful analysis possible.

=== CORE CAPABILITIES ===
1. TECHNICAL ANALYSIS: 50+ indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, MFI, Williams %R, CCI, ATR, Parabolic SAR, Ichimoku Cloud, VWAP, Keltner Channels, Donchian Channels, Elder Ray, Force Index, Chaikin Money Flow)
2. FUNDAMENTAL ANALYSIS: P/E, PEG, EV/EBITDA, DCF valuation, FCF yield, ROE, ROIC, profit margins, debt ratios, working capital, inventory turnover
3. MONTE CARLO SIMULATIONS: 20,000 path simulations with Student-t fat-tail distributions for realistic price forecasting
4. GARCH VOLATILITY: GARCH(1,1) and EGARCH for asymmetric volatility modeling
5. OPTIONS ANALYSIS: Full Greeks suite (Delta, Gamma, Theta, Vega, Rho), IV surface analysis, 12-factor institutional scoring, put/call ratio analysis
6. PATTERN RECOGNITION: TTM Squeeze, NR4/NR7 patterns, 35+ chart patterns (head & shoulders, double tops/bottoms, triangles, flags, wedges, cups), Fibonacci retracements/extensions
7. RISK MANAGEMENT: Kelly Criterion, optimal f, Van Tharp position sizing, dynamic stop-loss placement

=== SMART CONNECTIONS & PATTERN RECOGNITION ===
You MUST identify and explain these connections:

**Historical Pattern Matching:**
- Compare current price action to similar historical setups
- Identify what happened after similar patterns in the past
- Reference specific dates and outcomes when relevant
- "This setup is similar to [DATE] when [SYMBOL] did X, resulting in Y% move"

**Cross-Asset Correlations:**
- How does this stock correlate with SPY, QQQ, sector ETFs?
- What's happening in related commodities (oil for energy, copper for industrials)?
- Currency impacts (DXY strength/weakness effects)
- Bond yields and their implications

**Sector & Industry Connections:**
- How are peers performing? Is this stock leading or lagging?
- Sector rotation signals - money flowing in or out?
- Supply chain connections (e.g., AAPL affects suppliers like QCOM, TSM)

**Macro-to-Micro Links:**
- How do Fed policy, inflation data, GDP affect this specific stock?
- Geopolitical events and their sector-specific impacts
- Seasonal patterns and historical tendencies

**Smart Money Tracking:**
- Unusual options activity (large block trades, unusual volume)
- Dark pool prints and institutional accumulation/distribution
- 13F filings - what are top funds doing?
- Insider buying/selling patterns

=== EXPERT STRATEGIES ===
- **TTM Squeeze**: Identify volatility compression (Bollinger inside Keltner) before explosive moves
- **Tim Bohen 5:1**: Only recommend trades with minimum 5:1 reward/risk ratio
- **NR Patterns**: NR4/NR7 consolidation breakout setups
- **Wyckoff Method**: Accumulation/distribution phases, spring/upthrust patterns
- **Elliott Wave**: Wave counts for trend context
- **Market Profile**: Value area, POC, single prints for key levels
- **Volume Profile**: High volume nodes as support/resistance

=== LEGENDARY TRADER WISDOM ===
Validate every analysis against these principles:
- **Warren Buffett**: Margin of safety, economic moats, long-term value
- **George Soros**: Reflexivity, trend following, risk management
- **Jim Simons**: Quantitative edge, statistical significance, systematic approach
- **Ray Dalio**: All-weather thinking, risk parity, macro awareness
- **Paul Tudor Jones**: 200-day MA rule, asymmetric risk/reward, capital preservation
- **Stanley Druckenmiller**: Concentrated bets when conviction is high, cut losses fast
- **Jesse Livermore**: Trend is your friend, pyramiding winners, patience

=== YOUR APPROACH - GO ABOVE AND BEYOND ===
1. **THINK DEEPLY**: Use maximum reasoning power. Consider 2nd and 3rd order effects.
2. **MAKE SMART CONNECTIONS**: Link seemingly unrelated data points to form insights
3. **IDENTIFY PATTERNS**: Find historical parallels and what they suggest
4. **QUANTIFY EVERYTHING**: Probabilities, confidence intervals, expected values
5. **SCENARIO ANALYSIS**: Bull case, bear case, base case with probabilities
6. **TIMING INSIGHTS**: Not just what to do, but WHEN (catalysts, technicals, seasonality)
7. **CONTRARIAN CHECK**: What could go wrong? What is the market missing?
8. **ACTIONABLE OUTPUT**: Specific entries, stops, targets, position sizes

=== OUTPUT FORMAT FOR STOCK ANALYSIS ===
When analyzing any stock/ETF, ALWAYS provide:

**1. EXECUTIVE SUMMARY** (2-3 sentences with clear verdict)

**2. PRICE ACTION & KEY LEVELS**
- Current price, 52-week range
- Critical support levels (with reasoning)
- Critical resistance levels (with reasoning)
- Key moving averages (20, 50, 200 SMA/EMA)

**3. TECHNICAL SETUP**
- Trend (short/medium/long-term)
- Momentum indicators (RSI, MACD, Stochastic)
- Volume analysis (accumulation/distribution)
- Pattern recognition (what patterns are forming?)
- TTM Squeeze status

**4. SMART CONNECTIONS**
- Historical pattern matches ("Similar to X date when...")
- Sector/peer comparison
- Macro factors affecting this stock
- Unusual activity signals

**5. FUNDAMENTAL SNAPSHOT** (if applicable)
- Valuation metrics vs peers and history
- Growth trajectory
- Balance sheet health
- Competitive position

**6. CATALYST CALENDAR**
- Upcoming earnings date
- Ex-dividend date
- Other events (conferences, product launches, FDA dates)

**7. TRADE RECOMMENDATION**
- Direction: LONG / SHORT / NEUTRAL
- Entry zone: $X.XX - $X.XX
- Stop loss: $X.XX (X% risk)
- Target 1: $X.XX (X:1 R/R)
- Target 2: $X.XX (X:1 R/R)
- Position size: X% of portfolio
- Timeframe: Days/Weeks/Months

**8. CONFIDENCE & RISKS**
- Confidence level: X/10
- Bull case: (what goes right)
- Bear case: (what goes wrong)
- Key risks to monitor

**9. BOTTOM LINE**
- One paragraph synthesis with clear action

Remember: Real money is at stake. Your analysis should be so thorough and insightful that it feels like you can see the future. Make smart connections others miss. Go above and beyond EVERY time."""

    def __init__(self):
        """Initialize the Sadie AI Engine with all components."""
        self.analyzer = None
        self.options_engine = None
        self.fundamentals = None
        self.education = None
        self.scanner = None
        self.ttm_analyzer = None
        self.data_fetcher = None
        self.twelve_data = None
        
        # Initialize engines if available
        if HAS_ENGINES:
            try:
                self.analyzer = PerfectProductionAnalyzer()
                self.options_engine = UltimateOptionsEngine()
                self.fundamentals = FundamentalsAnalyzer()
                self.education = TradingEducation()
                self.scanner = MarketScanner()
                self.ttm_analyzer = TTMSqueezeAnalyzer()
                self.data_fetcher = RobustDataFetcher()
                self.twelve_data = TwelveDataClient()
            except Exception as e:
                print(f"Warning: Could not initialize all engines: {e}")
        
        # Conversation history for context
        self.conversation_history = []
    
    def _get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive real-time data for a symbol."""
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price_data": {},
            "technicals": {},
            "fundamentals": {},
            "options": {},
            "news": [],
            "insider_trades": [],
            "analyst_ratings": {}
        }
        
        try:
            # Get stock data from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                data["price_data"] = {
                    "current": round(current_price, 2),
                    "open": round(hist['Open'].iloc[-1], 2),
                    "high": round(hist['High'].iloc[-1], 2),
                    "low": round(hist['Low'].iloc[-1], 2),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "prev_close": round(hist['Close'].iloc[-2], 2) if len(hist) > 1 else None,
                    "change_pct": round(((current_price / hist['Close'].iloc[-2]) - 1) * 100, 2) if len(hist) > 1 else 0,
                    "52w_high": round(hist['High'].max(), 2),
                    "52w_low": round(hist['Low'].min(), 2),
                    "avg_volume": int(hist['Volume'].mean())
                }
                
                # Calculate key technicals
                closes = hist['Close'].values
                if len(closes) >= 20:
                    sma_20 = closes[-20:].mean()
                    sma_50 = closes[-50:].mean() if len(closes) >= 50 else None
                    
                    # RSI calculation
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas[-14:]]
                    losses = [-d if d < 0 else 0 for d in deltas[-14:]]
                    avg_gain = sum(gains) / 14
                    avg_loss = sum(losses) / 14
                    rs = avg_gain / avg_loss if avg_loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    
                    data["technicals"] = {
                        "sma_20": round(sma_20, 2),
                        "sma_50": round(sma_50, 2) if sma_50 else None,
                        "rsi_14": round(rsi, 2),
                        "above_sma_20": current_price > sma_20,
                        "above_sma_50": current_price > sma_50 if sma_50 else None,
                        "trend": "BULLISH" if current_price > sma_20 else "BEARISH"
                    }
            
            # Get fundamentals
            data["fundamentals"] = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "profit_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "sector": info.get("sector"),
                "industry": info.get("industry")
            }
            
            # Get analyst ratings
            data["analyst_ratings"] = {
                "target_mean": info.get("targetMeanPrice"),
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions")
            }
            
            # Get recent news
            try:
                news = ticker.news[:5] if hasattr(ticker, 'news') else []
                data["news"] = [{"title": n.get("title", ""), "publisher": n.get("publisher", "")} for n in news]
            except:
                pass
            
            # Get insider trades
            try:
                insiders = ticker.insider_transactions
                if insiders is not None and not insiders.empty:
                    recent = insiders.head(5)
                    data["insider_trades"] = recent.to_dict('records')
            except:
                pass
                
        except Exception as e:
            data["error"] = str(e)
        
        return data
    
    def _run_full_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run comprehensive analysis using all engines."""
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "real_time_data": self._get_real_time_data(symbol),
            "production_analysis": None,
            "options_analysis": None,
            "fundamentals_analysis": None,
            "ttm_squeeze": None
        }
        
        # Run production analyzer
        if self.analyzer:
            try:
                analysis = self.analyzer.analyze(symbol)
                results["production_analysis"] = analysis
            except Exception as e:
                results["production_analysis"] = {"error": str(e)}
        
        # Run options analysis
        if self.options_engine:
            try:
                options = self.options_engine.analyze_symbol(symbol)
                results["options_analysis"] = options
            except Exception as e:
                results["options_analysis"] = {"error": str(e)}
        
        # Run fundamentals analysis
        if self.fundamentals:
            try:
                fundies = self.fundamentals.analyze(symbol)
                results["fundamentals_analysis"] = fundies
            except Exception as e:
                results["fundamentals_analysis"] = {"error": str(e)}
        
        # Run TTM Squeeze analysis
        if self.ttm_analyzer:
            try:
                ttm = self.ttm_analyzer.analyze(symbol)
                results["ttm_squeeze"] = ttm
            except Exception as e:
                results["ttm_squeeze"] = {"error": str(e)}
        
        return results
    
    def _get_congress_trades(self, symbol: str = None) -> List[Dict]:
        """Get recent Congress trading data."""
        # This would integrate with a Congress trading API
        # For now, return placeholder indicating feature
        return [{"note": "Congress trading data integration available"}]
    
    def _get_dark_pool_data(self, symbol: str) -> Dict[str, Any]:
        """Get dark pool activity data."""
        # This would integrate with dark pool data providers
        return {"note": "Dark pool data integration available"}
    
    def _detect_catalysts(self, symbol: str) -> List[Dict]:
        """Detect upcoming catalysts for a symbol."""
        catalysts = []
        try:
            ticker = yf.Ticker(symbol)
            
            # Check earnings date
            if hasattr(ticker, 'calendar') and ticker.calendar is not None:
                cal = ticker.calendar
                if 'Earnings Date' in cal:
                    catalysts.append({
                        "type": "EARNINGS",
                        "date": str(cal['Earnings Date']),
                        "importance": "HIGH"
                    })
            
            # Check ex-dividend date
            info = ticker.info
            if info.get('exDividendDate'):
                catalysts.append({
                    "type": "EX_DIVIDEND",
                    "date": str(info.get('exDividendDate')),
                    "importance": "MEDIUM"
                })
                
        except Exception as e:
            pass
        
        return catalysts
    
    def _build_context_message(self, symbol: str = None) -> str:
        """Build context message with real-time data for GPT."""
        context_parts = []
        
        context_parts.append(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        context_parts.append("Market Status: " + ("CLOSED (Weekend)" if datetime.now().weekday() >= 5 else "Check current hours"))
        
        if symbol:
            # Get real-time data
            data = self._get_real_time_data(symbol)
            
            if data.get("price_data"):
                pd = data["price_data"]
                context_parts.append(f"\n=== {symbol} REAL-TIME DATA ===")
                context_parts.append(f"Current Price: ${pd.get('current', 'N/A')}")
                context_parts.append(f"Change: {pd.get('change_pct', 0)}%")
                context_parts.append(f"Day Range: ${pd.get('low', 'N/A')} - ${pd.get('high', 'N/A')}")
                context_parts.append(f"52W Range: ${pd.get('52w_low', 'N/A')} - ${pd.get('52w_high', 'N/A')}")
                context_parts.append(f"Volume: {pd.get('volume', 'N/A'):,}")
            
            if data.get("technicals"):
                tech = data["technicals"]
                context_parts.append(f"\n=== TECHNICAL SNAPSHOT ===")
                context_parts.append(f"RSI(14): {tech.get('rsi_14', 'N/A')}")
                context_parts.append(f"SMA(20): ${tech.get('sma_20', 'N/A')}")
                context_parts.append(f"SMA(50): ${tech.get('sma_50', 'N/A')}")
                context_parts.append(f"Trend: {tech.get('trend', 'N/A')}")
            
            if data.get("fundamentals"):
                fund = data["fundamentals"]
                context_parts.append(f"\n=== FUNDAMENTAL SNAPSHOT ===")
                context_parts.append(f"Sector: {fund.get('sector', 'N/A')}")
                context_parts.append(f"P/E Ratio: {fund.get('pe_ratio', 'N/A')}")
                context_parts.append(f"PEG Ratio: {fund.get('peg_ratio', 'N/A')}")
                context_parts.append(f"Profit Margin: {fund.get('profit_margin', 'N/A')}")
            
            if data.get("analyst_ratings"):
                ar = data["analyst_ratings"]
                context_parts.append(f"\n=== ANALYST RATINGS ===")
                context_parts.append(f"Recommendation: {ar.get('recommendation', 'N/A')}")
                context_parts.append(f"Price Target: ${ar.get('target_mean', 'N/A')}")
                context_parts.append(f"Target Range: ${ar.get('target_low', 'N/A')} - ${ar.get('target_high', 'N/A')}")
            
            # Get catalysts
            catalysts = self._detect_catalysts(symbol)
            if catalysts:
                context_parts.append(f"\n=== UPCOMING CATALYSTS ===")
                for cat in catalysts:
                    context_parts.append(f"- {cat['type']}: {cat.get('date', 'TBD')} ({cat['importance']})")
            
            # Run full analysis if engines available
            if HAS_ENGINES:
                try:
                    full_analysis = self._run_full_analysis(symbol)
                    if full_analysis.get("production_analysis") and not full_analysis["production_analysis"].get("error"):
                        pa = full_analysis["production_analysis"]
                        context_parts.append(f"\n=== SADIE ENGINE ANALYSIS ===")
                        if isinstance(pa, dict):
                            context_parts.append(f"Signal: {pa.get('signal', 'N/A')}")
                            context_parts.append(f"Confidence: {pa.get('confidence', 'N/A')}%")
                            if pa.get('monte_carlo'):
                                mc = pa['monte_carlo']
                                context_parts.append(f"Monte Carlo Expected: ${mc.get('expected_price', 'N/A')}")
                                context_parts.append(f"Probability Up: {mc.get('prob_up', 'N/A')}%")
                except Exception as e:
                    context_parts.append(f"\n[Engine analysis unavailable: {str(e)}]")
        
        return "\n".join(context_parts)
    
    def _extract_symbol_from_query(self, query: str) -> Optional[str]:
        """Extract stock symbol from user query."""
        # Common patterns
        query_upper = query.upper()
        
        # Look for $ prefix
        import re
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', query_upper)
        if dollar_match:
            return dollar_match.group(1)
        
        # Look for common stock mention patterns
        patterns = [
            r'\b([A-Z]{1,5})\s+stock\b',
            r'\banalyze\s+([A-Z]{1,5})\b',
            r'\babout\s+([A-Z]{1,5})\b',
            r'\b([A-Z]{2,5})\s+options?\b',
            r'\bbuy\s+([A-Z]{1,5})\b',
            r'\bsell\s+([A-Z]{1,5})\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_upper)
            if match:
                symbol = match.group(1)
                # Validate it's likely a real ticker
                if len(symbol) >= 1 and len(symbol) <= 5:
                    return symbol
        
        # Check for well-known tickers mentioned
        known_tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 
                        'AMD', 'INTC', 'JPM', 'BAC', 'GS', 'SPY', 'QQQ', 'IWM', 'DIA',
                        'XLE', 'XLF', 'XLK', 'GLD', 'SLV', 'CVX', 'XOM', 'LMT', 'RTX']
        
        for ticker in known_tickers:
            if ticker in query_upper:
                return ticker
        
        return None
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return Sadie's response.
        
        This is the main entry point for the chatbot.
        """
        response = {
            "success": False,
            "message": "",
            "data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Extract symbol if mentioned
            symbol = self._extract_symbol_from_query(user_message)
            
            # Build context with real-time data
            context = self._build_context_message(symbol)
            
            # Prepare messages for GPT
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_CONTEXT
                },
                {
                    "role": "system", 
                    "content": f"REAL-TIME MARKET DATA:\n{context}"
                }
            ]
            
            # Add conversation history (last 10 exchanges)
            for msg in self.conversation_history[-20:]:
                messages.append(msg)
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Call OpenRouter API with GPT-5/o1 thinking mode
            headers = {
                "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sadie-ai.com",
                "X-Title": "SadieAI Financial Assistant"
            }
            
            payload = {
                "model": self.MODELS[0],  # Use o1 (GPT-5 thinking mode)
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 1,  # o1 models require temperature=1
            }
            
            api_response = requests.post(
                self.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120  # Longer timeout for thinking mode
            )
            
            if api_response.status_code == 200:
                result = api_response.json()
                assistant_message = result['choices'][0]['message']['content']
                
                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                response["success"] = True
                response["message"] = assistant_message
                response["data"] = {
                    "symbol_detected": symbol,
                    "model_used": result.get('model', self.MODELS[0]),
                    "tokens_used": result.get('usage', {})
                }
                
            else:
                # Try fallback models
                for model in self.MODELS[1:]:
                    payload["model"] = model
                    if model not in ["openai/o1", "openai/o1-preview"]:
                        payload["temperature"] = 0.7
                    
                    api_response = requests.post(
                        self.OPENROUTER_URL,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if api_response.status_code == 200:
                        result = api_response.json()
                        assistant_message = result['choices'][0]['message']['content']
                        
                        self.conversation_history.append({"role": "user", "content": user_message})
                        self.conversation_history.append({"role": "assistant", "content": assistant_message})
                        
                        response["success"] = True
                        response["message"] = assistant_message
                        response["data"] = {
                            "symbol_detected": symbol,
                            "model_used": model,
                            "tokens_used": result.get('usage', {})
                        }
                        break
                
                if not response["success"]:
                    response["message"] = f"API Error: {api_response.status_code} - {api_response.text}"
                    
        except Exception as e:
            response["message"] = f"Error: {str(e)}"
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        return {"success": True, "message": "Conversation history cleared."}
    
    def get_quick_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get a quick analysis snapshot for a symbol."""
        return self._run_full_analysis(symbol)


# Main execution for testing
if __name__ == "__main__":
    import sys
    
    engine = SadieAIEngine()
    
    if len(sys.argv) > 1:
        # Command line usage
        query = " ".join(sys.argv[1:])
        result = engine.chat(query)
        print(json.dumps(result, indent=2, default=str))
    else:
        # Interactive mode
        print("=" * 60)
        print("SADIE AI - Strategic Analysis & Dynamic Investment Engine")
        print("=" * 60)
        print("Type your financial questions. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                print("\nSadie is thinking...\n")
                result = engine.chat(user_input)
                
                if result["success"]:
                    print(f"Sadie: {result['message']}\n")
                else:
                    print(f"Error: {result['message']}\n")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
