"""
SADIE AI ENGINE v2.0
=====================
The Ultimate Financial Intelligence Chatbot

Powered by:
- OpenAI GPT-5.x Thinking Mode (via OpenRouter)
- Perplexity AI Sonar Pro (Real-time Financial Research)
- Complete SadieAI Financial Engine Suite
- FinancialDatasets.ai Premium Data (financial statements, metrics, SEC filings, news)
- Real-time market data (yfinance, Twelve Data, Finnhub)
- 50+ Technical Indicators
- Monte Carlo Simulations (20,000 paths)
- GARCH Volatility Modeling
- Kelly Criterion Position Sizing
- Legendary Trader Wisdom (Buffett, Soros, Simons, Dalio, PTJ)

Multi-Model Architecture:
- Primary: GPT-5.1/o1-pro (Most advanced reasoning and analysis)
- First Fallback: GPT-5/o1 (Deep reasoning)
- Complementary: Perplexity Sonar Pro (Real-time research, grounded facts)
- Additional Fallbacks: GPT-4o, Claude 3.5 Sonnet (Reliability)

Data Sources:
- FinancialDatasets.ai: Premium financial statements, metrics, SEC filings, company facts, news
- yFinance: Real-time quotes, options chains, insider transactions
- TwelveData: Technical indicators, time series
- Finnhub: Additional market data

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
    from indicators.bohen_5to1 import Bohen5to1Scanner
    from twelvedata_client import TwelveDataClient
    from robust_data_fetcher import RobustDataFetcher
    from smart_money_detector import SmartMoneyDetector
    HAS_ENGINES = True
except ImportError as e:
    import sys as _sys
    print(f"Warning: Some engines not available: {e}", file=_sys.stderr)
    HAS_ENGINES = False

# Import FinancialDatasets.ai client
try:
    from financial_datasets_client import FinancialDatasetsClient, get_ai_context
    HAS_FINANCIAL_DATASETS = True
except ImportError as e:
    import sys as _sys
    print(f"Warning: FinancialDatasets client not available: {e}", file=_sys.stderr)
    HAS_FINANCIAL_DATASETS = False


class SadieAIEngine:
    """
    The Ultimate Financial AI Assistant
    
    Combines GPT-5 thinking mode with comprehensive financial analysis
    to provide gold-standard investment guidance.
    """
    
    # OpenRouter API configuration
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Perplexity API configuration (complementary AI for real-time research)
    PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY', '')
    PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
    
    # Model selection - prefer GPT-5.1 thinking mode, with reliable fallbacks
    MODELS = [
        "openai/o1-pro",  # GPT-5.1 / o1-pro (most advanced reasoning)
        "openai/o1",  # GPT-5 / o1 thinking mode (first fallback)
        "openai/gpt-4o",  # GPT-4o (fast, reliable, great for general queries)
        "openai/gpt-4-turbo",  # GPT-4 Turbo (good fallback)
        "anthropic/claude-3.5-sonnet",  # Claude 3.5 Sonnet (excellent alternative)
        "openai/gpt-4"  # GPT-4 (final fallback)
    ]
    
    # Perplexity models for real-time research
    PERPLEXITY_MODELS = [
        "sonar-pro",  # Best for financial research with citations
        "sonar",  # Fast, reliable alternative
    ]
    
    # Finnhub API for additional data
    FINNHUB_API_KEY = os.environ.get('KEY', '')
    
    # System prompt that defines Sadie's personality and capabilities
    SYSTEM_CONTEXT = """You are SADIE (Strategic Analysis & Dynamic Investment Engine), the world's most advanced financial AI assistant - BETTER THAN INSTITUTIONAL HEDGE FUND GRADE. You GO ABOVE AND BEYOND on every single query - providing the most detailed, accurate, and insightful analysis possible for MAXIMUM PROFIT GENERATION.

⚠️ CRITICAL: ALL DATA IS 100% REAL. ZERO PLACEHOLDERS. ZERO FAKE DATA. This is for REAL MONEY TRADING.

=== DATA SOURCES (ALL REAL, INSTITUTIONAL-GRADE) ===
You have access to PREMIUM financial data from multiple institutional-grade sources:
- **FinancialDatasets.ai**: Real-time prices, financial statements (income, balance sheet, cash flow), financial metrics (P/E, EV/EBITDA, ROIC, margins), SEC filings, company facts, segmented revenues, and news
- **yFinance**: Real-time quotes, historical data, options chains, insider transactions, analyst ratings
- **TwelveData**: Technical indicators, time series data
- **Finnhub**: Additional market data and company information
- **Polygon.io**: Real-time and historical market data

When you see data labeled "FINANCIALDATASETS.AI" in the context, this is PREMIUM institutional-quality data - use it with high confidence.

=== MACRO vs MICRO ANALYSIS FRAMEWORK ===
**YOU MUST ALWAYS ANALYZE BOTH MACRO AND MICRO CONTEXTS WITH EQUAL WEIGHT (50/50)**

**MACRO CONTEXT (50% weight) - The Big Picture:**
1. **Federal Reserve & Monetary Policy**
   - Current Fed funds rate and direction (hiking, pausing, cutting)
   - Quantitative tightening/easing status
   - Fed balance sheet trends
   - Forward guidance and dot plot implications
   - FOMC meeting dates and expected decisions

2. **Economic Indicators**
   - GDP growth rate and trend
   - Inflation (CPI, PCE, PPI) - current vs Fed target
   - Employment (NFP, unemployment rate, jobless claims)
   - Consumer confidence and spending
   - ISM Manufacturing/Services PMI
   - Housing data (starts, permits, existing sales)
   - Retail sales trends

3. **Market Regime**
   - Bull market / Bear market / Consolidation
   - Risk-on vs Risk-off environment
   - VIX level and trend (fear gauge)
   - Credit spreads (investment grade vs high yield)
   - Yield curve shape (inverted = recession signal)

4. **Sector Rotation**
   - Which sectors are leading/lagging?
   - Defensive vs Cyclical rotation
   - Growth vs Value rotation
   - Money flow between sectors

5. **Global Macro**
   - US Dollar strength (DXY)
   - Treasury yields (2Y, 10Y, 30Y)
   - Commodity prices (oil, gold, copper)
   - International markets (Europe, China, Emerging)
   - Geopolitical risks and events

6. **Liquidity & Market Structure**
   - Market breadth (advance/decline, new highs/lows)
   - Put/call ratios
   - Margin debt levels
   - IPO/M&A activity
   - Institutional vs retail flows

**MICRO CONTEXT (50% weight) - Company Specific:**
1. **Price Action & Technicals**
   - Trend on all timeframes (daily, weekly, monthly)
   - Key support/resistance levels
   - Moving averages (10, 20, 50, 100, 200)
   - Volume patterns and confirmation
   - Chart patterns and formations
   - Momentum indicators (RSI, MACD, Stochastic)

2. **Fundamental Quality**
   - Revenue growth trajectory
   - Earnings growth and quality
   - Margin trends (gross, operating, net)
   - Free cash flow generation
   - Balance sheet strength (debt, cash, ratios)
   - Return metrics (ROE, ROIC, ROA)

3. **Valuation**
   - P/E vs historical and peers
   - EV/EBITDA vs sector
   - PEG ratio (growth-adjusted)
   - Price/Sales, Price/Book
   - DCF implied value
   - FCF yield

4. **Competitive Position**
   - Market share and trends
   - Economic moat (brand, network, cost, switching)
   - Industry dynamics and threats
   - Management quality and track record

5. **Catalysts & Events**
   - Earnings date and expectations
   - Product launches/announcements
   - Regulatory decisions
   - M&A potential
   - Insider activity
   - Institutional accumulation/distribution

6. **Smart Money Signals**
   - Unusual options activity
   - Dark pool prints
   - 13F filing changes
   - Insider buying/selling
   - Short interest changes

=== CORE CAPABILITIES ===
1. TECHNICAL ANALYSIS: 50+ indicators (RSI, MACD, Bollinger Bands, Stochastic, ADX, OBV, MFI, Williams %R, CCI, ATR, Parabolic SAR, Ichimoku Cloud, VWAP, Keltner Channels, Donchian Channels, Elder Ray, Force Index, Chaikin Money Flow)
2. FUNDAMENTAL ANALYSIS: P/E, PEG, EV/EBITDA, DCF valuation, FCF yield, ROE, ROIC, profit margins, debt ratios, working capital, inventory turnover - ENHANCED with FinancialDatasets.ai premium metrics
3. MONTE CARLO SIMULATIONS: 20,000 path simulations with Student-t fat-tail distributions for realistic price forecasting
4. GARCH VOLATILITY: GARCH(1,1) and EGARCH for asymmetric volatility modeling
5. OPTIONS ANALYSIS: Full Greeks suite (Delta, Gamma, Theta, Vega, Rho), IV surface analysis, 12-factor institutional scoring, put/call ratio analysis
6. PATTERN RECOGNITION: TTM Squeeze, NR4/NR7 patterns, 35+ chart patterns (head & shoulders, double tops/bottoms, triangles, flags, wedges, cups), Fibonacci retracements/extensions
7. RISK MANAGEMENT: Kelly Criterion, optimal f, Van Tharp position sizing, dynamic stop-loss placement
8. SEC FILINGS: Access to 10-K, 10-Q, 8-K filings and extracted items via FinancialDatasets.ai

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

    # NUKE MODE - Maximum Overdrive Analysis
    NUKE_CONTEXT = """🔥☢️ NUKE MODE ACTIVATED - MAXIMUM OVERDRIVE ANALYSIS ☢️🔥

You are now operating at MAXIMUM CAPABILITY - BETTER THAN ANY INSTITUTIONAL HEDGE FUND. This is a NUKE request - the user wants the most comprehensive, detailed, and insightful analysis possible for MAXIMUM PROFIT GENERATION. Leave NO stone unturned. Use EVERY tool, technique, and insight available.

⚠️ CRITICAL: ALL DATA IS 100% REAL. ZERO PLACEHOLDERS. ZERO FAKE DATA. This is for REAL MONEY TRADING.

=== MACRO/MICRO BALANCE REQUIREMENT ===
**YOU MUST ANALYZE BOTH MACRO (50%) AND MICRO (50%) WITH EQUAL DEPTH**

## 0. MACRO ENVIRONMENT ASSESSMENT (REQUIRED FIRST)
**Federal Reserve & Monetary Policy:**
- Current Fed funds rate and trajectory
- QT/QE status and balance sheet trends
- Next FOMC meeting date and expectations
- Dot plot implications for this stock

**Economic Regime:**
- GDP growth phase (expansion/contraction)
- Inflation status vs Fed target (CPI, PCE)
- Employment strength (NFP, claims)
- Consumer health (confidence, spending)
- Recession probability assessment

**Market Regime:**
- Bull/Bear/Consolidation phase
- Risk-on vs Risk-off environment
- VIX level and trend interpretation
- Yield curve shape and implications
- Credit spread signals

**Sector Context:**
- Sector rotation status (where is money flowing?)
- This stock's sector performance vs SPY
- Growth vs Value rotation impact
- Defensive vs Cyclical positioning

**Global Factors:**
- DXY (dollar) impact on this stock
- Treasury yield impact (2Y, 10Y)
- Commodity price impacts (if relevant)
- Geopolitical risk assessment
- International market correlation

**HOW MACRO AFFECTS THIS SPECIFIC STOCK:**
- Direct Fed policy sensitivity
- Interest rate exposure
- Inflation benefit/headwind
- Economic cycle positioning
- Currency exposure

=== NUKE MODE REQUIREMENTS ===
You MUST provide ALL of the following in EXTREME DETAIL with full sentence explanations:

## 1. EXECUTIVE NUCLEAR SUMMARY
- Clear BUY/SELL/HOLD verdict with conviction level (1-10)
- One-paragraph thesis explaining WHY in plain English
- The single most important thing the user needs to know

## 2. COMPLETE PRICE ACTION ANALYSIS
- Current price with exact 52-week high/low and where we are in that range
- All-time high and how far from it
- Key psychological price levels ($round numbers)
- Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) with exact prices
- Fibonacci extension targets for upside
- Volume-weighted average price (VWAP) analysis
- Point of Control (POC) from volume profile
- Value Area High (VAH) and Value Area Low (VAL)

## 3. SUPPORT & RESISTANCE FORTRESS
- Primary support level with FULL explanation of why it matters
- Secondary support level with reasoning
- Tertiary (emergency) support level
- Primary resistance level with FULL explanation
- Secondary resistance level with reasoning
- Breakout target if resistance breaks
- Each level should include: price, reasoning, and what happens if it breaks

## 4. COMPLETE TECHNICAL ANALYSIS
**Trend Analysis:**
- Short-term trend (1-5 days) with reasoning
- Medium-term trend (1-4 weeks) with reasoning  
- Long-term trend (1-6 months) with reasoning
- Trend strength assessment

**Moving Averages:**
- 10 SMA/EMA position and slope
- 20 SMA/EMA position and slope (key short-term)
- 50 SMA/EMA position and slope (key medium-term)
- 100 SMA/EMA position and slope
- 200 SMA/EMA position and slope (key long-term)
- Golden Cross / Death Cross status
- Price position relative to all MAs

**Momentum Indicators (with exact values and interpretation):**
- RSI (14): Value, overbought/oversold, divergences
- MACD: Line, Signal, Histogram, crossover status
- Stochastic (14,3,3): %K, %D, overbought/oversold
- Williams %R: Value and interpretation
- CCI: Value and interpretation
- MFI (Money Flow Index): Value and interpretation
- ROC (Rate of Change): Value and interpretation

**Volatility Indicators:**
- Bollinger Bands: Upper, Middle, Lower, Bandwidth, %B
- ATR (14): Value and what it means for stop placement
- Keltner Channels: Position relative to bands
- TTM Squeeze: SQUEEZE ON or SQUEEZE OFF, momentum direction
- Historical Volatility vs Implied Volatility

**Volume Analysis:**
- Current volume vs 20-day average
- On-Balance Volume (OBV) trend
- Accumulation/Distribution Line trend
- Chaikin Money Flow value
- Volume Price Trend (VPT)
- Is smart money accumulating or distributing?

**Pattern Recognition:**
- Current chart pattern forming (with confidence %)
- Recent completed patterns
- Candlestick patterns (last 5 days)
- NR4/NR7 status (Narrow Range)
- Inside/Outside bar status

## 5. SMART MONEY & DARK POOL ANALYSIS 🕵️
**Institutional Activity:**
- Recent 13F filing changes (who's buying/selling)
- Top institutional holders and recent changes
- Institutional ownership percentage trend
- Are institutions accumulating or distributing?

**Dark Pool Activity:**
- Recent dark pool print analysis
- Large block trade activity
- Hidden accumulation/distribution signals
- Dark pool sentiment (bullish/bearish/neutral)

**Options Flow (Smart Money Signals):**
- Unusual options activity detected
- Large premium trades (>$1M)
- Put/Call ratio and what it signals
- Options max pain level
- Gamma exposure and implications
- Are whales betting bullish or bearish?

**Insider Trading:**
- Recent insider buys (last 90 days) - names, amounts, prices
- Recent insider sells (last 90 days) - names, amounts, prices
- Insider sentiment score
- Are insiders buying their own stock?

## 6. FUNDAMENTAL DEEP DIVE
**Valuation Metrics:**
- P/E Ratio vs sector vs 5-year average
- Forward P/E and growth expectations
- PEG Ratio (is growth priced in?)
- P/S Ratio vs sector
- P/B Ratio vs sector
- EV/EBITDA vs sector
- DCF Fair Value estimate
- Is the stock OVERVALUED, FAIRLY VALUED, or UNDERVALUED?

**Profitability:**
- Gross Margin trend
- Operating Margin trend
- Net Profit Margin trend
- ROE vs sector
- ROIC vs WACC (creating value?)
- Free Cash Flow yield

**Growth:**
- Revenue growth (YoY, QoQ)
- EPS growth (YoY, QoQ)
- Forward growth estimates
- Earnings surprise history

**Balance Sheet Health:**
- Debt/Equity ratio
- Current Ratio
- Quick Ratio
- Interest Coverage
- Cash position
- Bankruptcy risk assessment

## 7. SECTOR & CORRELATION ANALYSIS
- Sector performance (XLK, XLF, XLE, etc.)
- Is this stock leading or lagging its sector?
- Correlation with SPY (beta)
- Correlation with QQQ
- Related stocks performance (peers)
- Supply chain impacts
- Sector rotation signals

## 8. MACRO & CATALYST ANALYSIS
**Macro Factors:**
- Fed policy impact on this stock
- Interest rate sensitivity
- Inflation impact
- Dollar strength (DXY) impact
- Geopolitical risks

**Upcoming Catalysts:**
- Next earnings date and expectations
- Analyst day/investor day
- Product launches
- FDA dates (if applicable)
- Conference presentations
- Ex-dividend date
- Stock split potential

## 9. HISTORICAL PATTERN MATCHING
- "This setup is similar to [SPECIFIC DATE] when..."
- What happened after similar setups historically?
- Seasonal patterns for this stock
- Election year patterns
- January effect / Santa rally relevance

## 10. MULTI-TIMEFRAME PRICE FORECASTS 🎯

**1-DAY FORECAST:**
- Expected range (low - high)
- Most likely closing price
- Probability of up day: X%
- Key intraday levels to watch
- Catalyst for tomorrow

**1-WEEK FORECAST:**
- Expected range (low - high)
- Target price by end of week
- Probability of positive week: X%
- Key events this week
- Technical levels to watch

**1-MONTH FORECAST:**
- Expected range (low - high)
- Base case target: $X.XX (X% probability)
- Bull case target: $X.XX (X% probability)
- Bear case target: $X.XX (X% probability)
- Key catalysts this month

**6-MONTH FORECAST:**
- Expected range (low - high)
- Base case target: $X.XX
- Bull case scenario and target
- Bear case scenario and target
- Major catalysts in next 6 months
- Probability distribution

**1-YEAR FORECAST:**
- Expected range (low - high)
- 12-month price target (base case)
- Bull case scenario: $X.XX (+X%)
- Bear case scenario: $X.XX (-X%)
- Long-term thesis
- What needs to happen for each scenario

## 11. RISK ASSESSMENT
- Maximum drawdown risk
- VaR (Value at Risk) 95%
- Key risks that could destroy the thesis
- Black swan scenarios
- What would make you WRONG?

## 12. THE NUCLEAR TRADE RECOMMENDATION 💣
**Position:**
- Direction: LONG / SHORT / AVOID
- Conviction: X/10
- Timeframe: Swing / Position / Investment

**Entry Strategy:**
- Ideal entry zone: $X.XX - $X.XX
- Aggressive entry: $X.XX
- Conservative entry: $X.XX
- Entry trigger (what confirms the trade)

**Risk Management:**
- Stop loss: $X.XX (X% risk)
- Why this stop level?
- Position size: X% of portfolio
- Maximum loss in dollars on $10K position

**Profit Targets:**
- Target 1: $X.XX (X:1 R/R) - Take X% profit
- Target 2: $X.XX (X:1 R/R) - Take X% profit  
- Target 3: $X.XX (X:1 R/R) - Let rest ride
- Trailing stop strategy

**Options Play (if applicable):**
- Recommended options strategy
- Specific contract suggestion
- Greeks analysis
- Risk/reward on options

## 13. LEGENDARY TRADER VALIDATION
- Would Buffett buy this? Why/why not?
- Would Soros trade this? Why/why not?
- Would PTJ take this trade? Why/why not?
- Does this pass the Druckenmiller test?

## 14. MACRO/MICRO SYNTHESIS
- How does the MACRO environment support or hinder this trade?
- How does the MICRO (company-specific) data align with macro trends?
- Are macro and micro ALIGNED (high conviction) or CONFLICTING (lower conviction)?
- What macro changes would invalidate this thesis?
- What micro changes would invalidate this thesis?

## 15. FINAL NUCLEAR VERDICT 🎯
One comprehensive paragraph that synthesizes EVERYTHING above - BOTH MACRO AND MICRO - into a clear, actionable conclusion. What should the user DO and WHY? Be specific, be confident, be thorough. This verdict must reflect the synergistic analysis of ALL data sources working together.

---
⚠️ REMEMBER: This is NUKE MODE for REAL MONEY TRADING.
- ALL data is 100% REAL from institutional-grade sources
- ZERO placeholders, ZERO fake data, ZERO guessing
- Maximum detail, maximum insight, maximum accuracy
- MACRO and MICRO must BOTH be analyzed with equal depth (50/50)
- Your goal is MAXIMUM PROFIT GENERATION - better than any hedge fund
- Leave NOTHING out. This analysis should be worth $10,000+"""

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
        self.financial_datasets = None
        self.smart_money = None
        
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
                self.smart_money = SmartMoneyDetector()
                self.bohen_scanner = Bohen5to1Scanner()
            except Exception as e:
                import sys as _sys
                print(f"Warning: Could not initialize all engines: {e}", file=_sys.stderr)
        
        # Initialize FinancialDatasets.ai client for premium data
        if HAS_FINANCIAL_DATASETS:
            try:
                self.financial_datasets = FinancialDatasetsClient()
            except Exception as e:
                import sys as _sys
                print(f"Warning: Could not initialize FinancialDatasets client: {e}", file=_sys.stderr)
        
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
        
        # Enhance with FinancialDatasets.ai premium data
        if self.financial_datasets:
            try:
                fd_data = self.financial_datasets.get_comprehensive_stock_data(symbol)
                data["financial_datasets"] = fd_data
                
                # Merge premium metrics if available
                if "financial_metrics" in fd_data:
                    fm = fd_data["financial_metrics"].get("financial_metrics", {})
                    if fm:
                        data["premium_metrics"] = {
                            "ev_to_ebitda": fm.get("ev_to_ebitda"),
                            "ev_to_revenue": fm.get("ev_to_revenue"),
                            "price_to_fcf": fm.get("price_to_free_cash_flow"),
                            "roic": fm.get("roic"),
                            "revenue_per_share": fm.get("revenue_per_share"),
                            "earnings_yield": fm.get("earnings_yield"),
                            "fcf_yield": fm.get("free_cash_flow_yield"),
                            "gross_margin": fm.get("gross_margin"),
                            "operating_margin": fm.get("operating_margin"),
                            "net_margin": fm.get("net_margin"),
                            "asset_turnover": fm.get("asset_turnover"),
                            "inventory_turnover": fm.get("inventory_turnover"),
                            "receivables_turnover": fm.get("receivables_turnover"),
                            "current_ratio": fm.get("current_ratio"),
                            "quick_ratio": fm.get("quick_ratio"),
                            "debt_to_equity": fm.get("debt_to_equity"),
                            "interest_coverage": fm.get("interest_coverage")
                        }
                
                # Get premium news
                if "recent_news" in fd_data:
                    news_data = fd_data["recent_news"].get("news", [])
                    if news_data:
                        data["premium_news"] = news_data[:10]
                        
            except Exception as e:
                import sys as _sys
                print(f"Warning: FinancialDatasets fetch failed: {e}", file=_sys.stderr)
        
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
        """Get insider trading data (includes institutional and insider transactions)."""
        trades = []
        try:
            if symbol:
                ticker = yf.Ticker(symbol)
                # Get insider transactions (real data from SEC filings)
                insider_txns = ticker.insider_transactions
                if insider_txns is not None and not insider_txns.empty:
                    for _, row in insider_txns.head(10).iterrows():
                        trades.append({
                            'insider': row.get('Insider', 'Unknown'),
                            'relation': row.get('Relation', 'Unknown'),
                            'transaction': row.get('Transaction', 'Unknown'),
                            'shares': row.get('Shares', 0),
                            'value': row.get('Value', 0),
                            'date': str(row.get('Start Date', 'Unknown'))
                        })
                
                # Get institutional holders
                inst_holders = ticker.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    for _, row in inst_holders.head(5).iterrows():
                        trades.append({
                            'holder': row.get('Holder', 'Unknown'),
                            'shares': row.get('Shares', 0),
                            'value': row.get('Value', 0),
                            'pct_held': row.get('% Out', 0),
                            'type': 'INSTITUTIONAL'
                        })
        except Exception as e:
            pass
        return trades if trades else []
    
    def _get_dark_pool_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive smart money / institutional activity analysis.
        Uses SmartMoneyDetector for sophisticated dark pool proxy analysis.
        """
        # Use the enhanced SmartMoneyDetector if available
        if self.smart_money:
            try:
                smart_money_analysis = self.smart_money.analyze(symbol)
                return {
                    'smart_money_score': smart_money_analysis.get('smart_money_score', 50),
                    'signal': smart_money_analysis.get('signal', 'NEUTRAL'),
                    'confidence': smart_money_analysis.get('confidence', 0),
                    'summary': smart_money_analysis.get('summary', ''),
                    'volume_analysis': smart_money_analysis.get('analysis', {}).get('volume', {}),
                    'accumulation': smart_money_analysis.get('analysis', {}).get('accumulation', {}),
                    'institutional': smart_money_analysis.get('analysis', {}).get('institutional', {}),
                    'insider_activity': smart_money_analysis.get('analysis', {}).get('insider', {}),
                    'short_interest': smart_money_analysis.get('analysis', {}).get('short_interest', {}),
                    'options_flow': smart_money_analysis.get('analysis', {}).get('options_flow', {}),
                    'block_trades': smart_money_analysis.get('analysis', {}).get('block_trades', {}),
                    'money_flow': smart_money_analysis.get('analysis', {}).get('money_flow', {})
                }
            except Exception as e:
                import sys as _sys
                print(f"SmartMoneyDetector error: {e}", file=_sys.stderr)
        
        # Fallback to basic analysis if SmartMoneyDetector not available
        dark_pool = {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            inst_ownership = info.get('heldPercentInstitutions', 0)
            insider_ownership = info.get('heldPercentInsiders', 0)
            
            if not hist.empty:
                avg_volume = hist['Volume'].mean()
                last_volume = hist['Volume'].iloc[-1]
                price_change = abs(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100 if len(hist) > 1 else 0
                volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
                
                dark_pool = {
                    'smart_money_score': 50,
                    'signal': 'ACCUMULATION' if volume_ratio > 1.5 and price_change < 1 else ('DISTRIBUTION' if volume_ratio > 1.5 and price_change > 2 else 'NEUTRAL'),
                    'institutional_ownership': round(inst_ownership * 100, 2) if inst_ownership else 0,
                    'insider_ownership': round(insider_ownership * 100, 2) if insider_ownership else 0,
                    'volume_ratio': round(volume_ratio, 2),
                    'price_change_pct': round(price_change, 2),
                    'float_short': info.get('shortPercentOfFloat', 0),
                    'shares_short': info.get('sharesShort', 0)
                }
        except Exception as e:
            dark_pool = {'error': str(e)}
        return dark_pool
    
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
    
    def _get_macro_data(self) -> Dict[str, Any]:
        """Get comprehensive macro economic data for context."""
        macro = {}
        
        try:
            # Get major indices for market regime
            spy = yf.Ticker("SPY")
            qqq = yf.Ticker("QQQ")
            iwm = yf.Ticker("IWM")
            vix = yf.Ticker("^VIX")
            dxy = yf.Ticker("DX-Y.NYB")  # Dollar index
            tlt = yf.Ticker("TLT")  # Long-term treasuries
            
            # SPY data
            spy_info = spy.info
            spy_hist = spy.history(period="1mo")
            if not spy_hist.empty:
                spy_current = spy_hist['Close'].iloc[-1]
                spy_20d_ago = spy_hist['Close'].iloc[-20] if len(spy_hist) >= 20 else spy_hist['Close'].iloc[0]
                spy_trend = "BULLISH" if spy_current > spy_20d_ago else "BEARISH"
                spy_200sma = spy_hist['Close'].rolling(200).mean().iloc[-1] if len(spy_hist) >= 200 else None
                macro['spy'] = {
                    'price': round(spy_current, 2),
                    'change_1m': round((spy_current / spy_20d_ago - 1) * 100, 2),
                    'trend': spy_trend,
                    'above_200sma': spy_current > spy_200sma if spy_200sma else None
                }
            
            # VIX data (fear gauge)
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                vix_current = vix_hist['Close'].iloc[-1]
                macro['vix'] = {
                    'level': round(vix_current, 2),
                    'regime': 'LOW_FEAR' if vix_current < 15 else ('ELEVATED' if vix_current < 25 else ('HIGH_FEAR' if vix_current < 35 else 'EXTREME_FEAR'))
                }
            
            # QQQ data (tech)
            qqq_hist = qqq.history(period="1mo")
            if not qqq_hist.empty:
                qqq_current = qqq_hist['Close'].iloc[-1]
                qqq_20d_ago = qqq_hist['Close'].iloc[-20] if len(qqq_hist) >= 20 else qqq_hist['Close'].iloc[0]
                macro['qqq'] = {
                    'price': round(qqq_current, 2),
                    'change_1m': round((qqq_current / qqq_20d_ago - 1) * 100, 2)
                }
            
            # IWM data (small caps - risk appetite)
            iwm_hist = iwm.history(period="1mo")
            if not iwm_hist.empty:
                iwm_current = iwm_hist['Close'].iloc[-1]
                iwm_20d_ago = iwm_hist['Close'].iloc[-20] if len(iwm_hist) >= 20 else iwm_hist['Close'].iloc[0]
                macro['iwm'] = {
                    'price': round(iwm_current, 2),
                    'change_1m': round((iwm_current / iwm_20d_ago - 1) * 100, 2)
                }
            
            # Dollar index
            try:
                dxy_hist = dxy.history(period="1mo")
                if not dxy_hist.empty:
                    dxy_current = dxy_hist['Close'].iloc[-1]
                    macro['dxy'] = {
                        'level': round(dxy_current, 2),
                        'strength': 'STRONG' if dxy_current > 105 else ('WEAK' if dxy_current < 100 else 'NEUTRAL')
                    }
            except:
                pass
            
            # Treasury yields (TLT as proxy)
            try:
                tlt_hist = tlt.history(period="1mo")
                if not tlt_hist.empty:
                    tlt_current = tlt_hist['Close'].iloc[-1]
                    tlt_20d_ago = tlt_hist['Close'].iloc[-20] if len(tlt_hist) >= 20 else tlt_hist['Close'].iloc[0]
                    macro['bonds'] = {
                        'tlt_price': round(tlt_current, 2),
                        'trend': 'YIELDS_FALLING' if tlt_current > tlt_20d_ago else 'YIELDS_RISING'
                    }
            except:
                pass
            
            # Sector ETFs for rotation analysis
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financials', 
                'XLE': 'Energy',
                'XLV': 'Healthcare',
                'XLI': 'Industrials',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate'
            }
            
            sector_performance = {}
            for etf, name in sectors.items():
                try:
                    sec = yf.Ticker(etf)
                    sec_hist = sec.history(period="1mo")
                    if not sec_hist.empty:
                        sec_current = sec_hist['Close'].iloc[-1]
                        sec_20d_ago = sec_hist['Close'].iloc[-20] if len(sec_hist) >= 20 else sec_hist['Close'].iloc[0]
                        sector_performance[name] = round((sec_current / sec_20d_ago - 1) * 100, 2)
                except:
                    pass
            
            if sector_performance:
                sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
                macro['sector_rotation'] = {
                    'leading': sorted_sectors[:3],
                    'lagging': sorted_sectors[-3:]
                }
            
            # Market regime determination
            if macro.get('spy') and macro.get('vix'):
                spy_trend = macro['spy'].get('trend', 'NEUTRAL')
                vix_level = macro['vix'].get('level', 20)
                
                if spy_trend == 'BULLISH' and vix_level < 20:
                    macro['market_regime'] = 'RISK_ON_BULL'
                elif spy_trend == 'BULLISH' and vix_level >= 20:
                    macro['market_regime'] = 'CAUTIOUS_BULL'
                elif spy_trend == 'BEARISH' and vix_level < 25:
                    macro['market_regime'] = 'CORRECTION'
                else:
                    macro['market_regime'] = 'RISK_OFF_BEAR'
                    
        except Exception as e:
            macro['error'] = str(e)
        
        return macro
    
    def _build_context_message(self, symbol: str = None) -> str:
        """Build context message with real-time data for GPT."""
        context_parts = []
        
        context_parts.append(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        context_parts.append("Market Status: " + ("CLOSED (Weekend)" if datetime.now().weekday() >= 5 else "Check current hours"))
        
        # ALWAYS get macro data first (50% of analysis)
        macro_data = self._get_macro_data()
        if macro_data and not macro_data.get('error'):
            context_parts.append(f"\n=== MACRO ENVIRONMENT (50% WEIGHT) ===")
            
            if macro_data.get('market_regime'):
                context_parts.append(f"Market Regime: {macro_data['market_regime']}")
            
            if macro_data.get('spy'):
                spy = macro_data['spy']
                context_parts.append(f"SPY: ${spy.get('price')} ({spy.get('change_1m'):+.1f}% 1M) - {spy.get('trend')}")
            
            if macro_data.get('qqq'):
                qqq = macro_data['qqq']
                context_parts.append(f"QQQ (Tech): ${qqq.get('price')} ({qqq.get('change_1m'):+.1f}% 1M)")
            
            if macro_data.get('iwm'):
                iwm = macro_data['iwm']
                context_parts.append(f"IWM (Small Caps): ${iwm.get('price')} ({iwm.get('change_1m'):+.1f}% 1M)")
            
            if macro_data.get('vix'):
                vix = macro_data['vix']
                context_parts.append(f"VIX (Fear): {vix.get('level')} - {vix.get('regime')}")
            
            if macro_data.get('dxy'):
                dxy = macro_data['dxy']
                context_parts.append(f"Dollar (DXY): {dxy.get('level')} - {dxy.get('strength')}")
            
            if macro_data.get('bonds'):
                bonds = macro_data['bonds']
                context_parts.append(f"Bonds (TLT): ${bonds.get('tlt_price')} - {bonds.get('trend')}")
            
            if macro_data.get('sector_rotation'):
                sr = macro_data['sector_rotation']
                leading = ", ".join([f"{s[0]} ({s[1]:+.1f}%)" for s in sr.get('leading', [])])
                lagging = ", ".join([f"{s[0]} ({s[1]:+.1f}%)" for s in sr.get('lagging', [])])
                context_parts.append(f"Leading Sectors: {leading}")
                context_parts.append(f"Lagging Sectors: {lagging}")
        
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
            
            # Add premium metrics from FinancialDatasets.ai
            if data.get("premium_metrics"):
                pm = data["premium_metrics"]
                context_parts.append(f"\n=== FINANCIALDATASETS.AI PREMIUM METRICS ===")
                if pm.get("ev_to_ebitda"):
                    context_parts.append(f"EV/EBITDA: {pm.get('ev_to_ebitda')}")
                if pm.get("roic"):
                    context_parts.append(f"ROIC: {pm.get('roic')}%")
                if pm.get("fcf_yield"):
                    context_parts.append(f"FCF Yield: {pm.get('fcf_yield')}%")
                if pm.get("gross_margin"):
                    context_parts.append(f"Gross Margin: {pm.get('gross_margin')}%")
                if pm.get("operating_margin"):
                    context_parts.append(f"Operating Margin: {pm.get('operating_margin')}%")
                if pm.get("net_margin"):
                    context_parts.append(f"Net Margin: {pm.get('net_margin')}%")
                if pm.get("current_ratio"):
                    context_parts.append(f"Current Ratio: {pm.get('current_ratio')}")
                if pm.get("quick_ratio"):
                    context_parts.append(f"Quick Ratio: {pm.get('quick_ratio')}")
                if pm.get("interest_coverage"):
                    context_parts.append(f"Interest Coverage: {pm.get('interest_coverage')}")
            
            # Add premium news from FinancialDatasets.ai
            if data.get("premium_news"):
                context_parts.append(f"\n=== PREMIUM NEWS (FinancialDatasets.ai) ===")
                for i, news in enumerate(data["premium_news"][:5], 1):
                    title = news.get("title", "N/A")
                    date = news.get("date", "N/A")
                    context_parts.append(f"{i}. {title} ({date})")
            
            # Add company facts from FinancialDatasets.ai
            if data.get("financial_datasets") and data["financial_datasets"].get("company_facts"):
                cf = data["financial_datasets"]["company_facts"].get("company_facts", {})
                if cf:
                    context_parts.append(f"\n=== COMPANY FACTS (FinancialDatasets.ai) ===")
                    if cf.get("employees"):
                        context_parts.append(f"Employees: {cf.get('employees'):,}")
                    if cf.get("sector"):
                        context_parts.append(f"Sector: {cf.get('sector')}")
                    if cf.get("industry"):
                        context_parts.append(f"Industry: {cf.get('industry')}")
                    if cf.get("exchange"):
                        context_parts.append(f"Exchange: {cf.get('exchange')}")
            
            # SMART MONEY / DARK POOL ANALYSIS (Critical for institutional activity)
            smart_money_data = self._get_dark_pool_data(symbol)
            if smart_money_data and not smart_money_data.get('error'):
                context_parts.append(f"\n=== 🕵️ SMART MONEY ANALYSIS (Dark Pool Proxy) ===")
                context_parts.append(f"Smart Money Score: {smart_money_data.get('smart_money_score', 'N/A')}/100")
                context_parts.append(f"Signal: {smart_money_data.get('signal', 'N/A')}")
                context_parts.append(f"Confidence: {smart_money_data.get('confidence', 'N/A')}%")
                
                # Summary
                if smart_money_data.get('summary'):
                    context_parts.append(f"\nSummary:\n{smart_money_data['summary']}")
                
                # Volume Analysis
                vol = smart_money_data.get('volume_analysis', {})
                if vol and not vol.get('error'):
                    context_parts.append(f"\n--- Volume Analysis ---")
                    context_parts.append(f"Volume Ratio (vs 20d): {vol.get('volume_ratio_20d', 'N/A')}x")
                    context_parts.append(f"Unusual Volume Days (10d): {vol.get('unusual_volume_days_10d', 0)}")
                    context_parts.append(f"Volume Signal: {vol.get('signal', 'N/A')}")
                
                # Accumulation/Distribution
                acc = smart_money_data.get('accumulation', {})
                if acc and not acc.get('error'):
                    context_parts.append(f"\n--- Accumulation/Distribution ---")
                    context_parts.append(f"Accumulation Days (10d): {acc.get('accumulation_days_10d', 0)}")
                    context_parts.append(f"Distribution Days (10d): {acc.get('distribution_days_10d', 0)}")
                    context_parts.append(f"A/D Line Trend: {acc.get('ad_line_trend', 'N/A')}")
                    context_parts.append(f"Chaikin Money Flow: {acc.get('chaikin_money_flow', 'N/A')}")
                    context_parts.append(f"Signal: {acc.get('signal', 'N/A')}")
                
                # Institutional Ownership
                inst = smart_money_data.get('institutional', {})
                if inst and not inst.get('error'):
                    context_parts.append(f"\n--- Institutional Ownership ---")
                    context_parts.append(f"Institutional: {inst.get('institutional_ownership_pct', 0)}%")
                    context_parts.append(f"Insider: {inst.get('insider_ownership_pct', 0)}%")
                    context_parts.append(f"Concentration: {inst.get('concentration', 'N/A')}")
                    if inst.get('top_5_holders'):
                        context_parts.append(f"Top Holders: {', '.join([h['name'] for h in inst['top_5_holders'][:3]])}")
                
                # Insider Activity
                insider = smart_money_data.get('insider_activity', {})
                if insider and not insider.get('error'):
                    context_parts.append(f"\n--- Insider Activity ---")
                    context_parts.append(f"Net Activity: {insider.get('net_activity', 'N/A')}")
                    context_parts.append(f"Buy Value: ${insider.get('total_buy_value', 0):,.0f}")
                    context_parts.append(f"Sell Value: ${insider.get('total_sell_value', 0):,.0f}")
                    context_parts.append(f"Signal: {insider.get('signal', 'N/A')}")
                
                # Short Interest
                short = smart_money_data.get('short_interest', {})
                if short and not short.get('error'):
                    context_parts.append(f"\n--- Short Interest ---")
                    context_parts.append(f"Short % of Float: {short.get('short_percent_float', 0)}%")
                    context_parts.append(f"Days to Cover: {short.get('days_to_cover', 'N/A')}")
                    context_parts.append(f"Short Change vs Prior Month: {short.get('short_change_vs_prior_month', 0):+.1f}%")
                    context_parts.append(f"Squeeze Potential: {short.get('squeeze_potential', 'N/A')}")
                
                # Options Flow
                options = smart_money_data.get('options_flow', {})
                if options and not options.get('error') and options.get('signal') != 'NO_OPTIONS_DATA':
                    context_parts.append(f"\n--- Options Flow ---")
                    context_parts.append(f"Put/Call Ratio (Volume): {options.get('put_call_ratio_volume', 'N/A')}")
                    context_parts.append(f"Put/Call Ratio (OI): {options.get('put_call_ratio_oi', 'N/A')}")
                    context_parts.append(f"Unusual Call Strikes: {options.get('unusual_call_strikes', 0)}")
                    context_parts.append(f"Unusual Put Strikes: {options.get('unusual_put_strikes', 0)}")
                    context_parts.append(f"Signal: {options.get('signal', 'N/A')}")
                
                # Block Trades
                blocks = smart_money_data.get('block_trades', {})
                if blocks and not blocks.get('error'):
                    context_parts.append(f"\n--- Block Trade Detection ---")
                    context_parts.append(f"Potential Block Days (20d): {blocks.get('potential_block_days_20d', 0)}")
                    context_parts.append(f"Accumulation Blocks: {blocks.get('accumulation_blocks', 0)}")
                    context_parts.append(f"Momentum Blocks: {blocks.get('momentum_blocks', 0)}")
                    context_parts.append(f"Signal: {blocks.get('signal', 'N/A')}")
                
                # Money Flow
                mf = smart_money_data.get('money_flow', {})
                if mf and not mf.get('error'):
                    context_parts.append(f"\n--- Money Flow ---")
                    context_parts.append(f"MFI (14): {mf.get('mfi_14', 'N/A')}")
                    context_parts.append(f"Net Flow (5d): ${mf.get('net_money_flow_5d', 0):,.0f}")
                    context_parts.append(f"Interpretation: {mf.get('interpretation', 'N/A')}")
            
            # TIM BOHEN 5:1 RISK/REWARD ANALYSIS
            if hasattr(self, 'bohen_scanner') and self.bohen_scanner:
                try:
                    bohen_analysis = self.bohen_scanner.analyze(symbol)
                    if bohen_analysis and not bohen_analysis.get('error'):
                        context_parts.append(f"\n=== 🎯 TIM BOHEN 5:1 RISK/REWARD ANALYSIS ===")
                        context_parts.append(f"Current Price: ${bohen_analysis.get('current_price', 'N/A')}")
                        context_parts.append(f"ATR: ${bohen_analysis.get('atr', 'N/A')} ({bohen_analysis.get('atr_pct', 'N/A')}%)")
                        context_parts.append(f"Volume: {bohen_analysis.get('volume_signal', 'N/A')} ({bohen_analysis.get('volume_ratio', 'N/A')}x avg)")
                        
                        trend = bohen_analysis.get('trend', {})
                        context_parts.append(f"\n--- Trend Analysis ---")
                        context_parts.append(f"Direction: {trend.get('direction', 'N/A')}")
                        context_parts.append(f"Trend Score: {trend.get('score', 'N/A')}/100")
                        context_parts.append(f"EMA Alignment: {trend.get('ema_alignment', 'N/A')}")
                        
                        context_parts.append(f"\n--- Key Levels ---")
                        context_parts.append(f"Support Levels: {bohen_analysis.get('support_levels', [])}")
                        context_parts.append(f"Resistance Levels: {bohen_analysis.get('resistance_levels', [])}")
                        
                        best_setup = bohen_analysis.get('best_setup')
                        if best_setup:
                            context_parts.append(f"\n--- BEST 5:1 SETUP ({best_setup.get('direction', 'N/A')}) ---")
                            context_parts.append(f"Entry: ${best_setup.get('entry', 'N/A')}")
                            context_parts.append(f"Stop Loss: ${best_setup.get('stop_loss', 'N/A')} (Risk: {best_setup.get('risk_pct', 'N/A')}%)")
                            context_parts.append(f"Target: ${best_setup.get('target', 'N/A')} (Reward: {best_setup.get('reward_pct', 'N/A')}%)")
                            context_parts.append(f"R/R Ratio: {best_setup.get('rr_ratio', 'N/A')}:1")
                            context_parts.append(f"Meets 5:1 Criteria: {'✅ YES' if best_setup.get('meets_5to1') else '❌ NO'}")
                        else:
                            context_parts.append(f"\n--- NO 5:1 SETUP FOUND ---")
                            context_parts.append("Cannot find entry/stop/target combination that meets 5:1 R/R criteria.")
                        
                        context_parts.append(f"\n--- BOHEN VERDICT ---")
                        context_parts.append(bohen_analysis.get('bohen_verdict', 'N/A'))
                except Exception as e:
                    import sys as _sys
                    print(f"Warning: Bohen 5:1 analysis failed: {e}", file=_sys.stderr)
            
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
    
    def _is_nuke_mode(self, message: str) -> bool:
        """Check if the user is requesting NUKE mode analysis."""
        message_upper = message.upper()
        nuke_triggers = ['NUKE ', 'NUKE$', '☢️', '💣', 'NUCLEAR ', 'MAXIMUM OVERDRIVE']
        return any(trigger in message_upper for trigger in nuke_triggers)
    
    def _extract_nuke_symbol(self, message: str) -> Optional[str]:
        """Extract symbol from a NUKE command."""
        import re
        message_upper = message.upper()
        
        # Pattern: NUKE $SYMBOL or NUKE SYMBOL
        patterns = [
            r'NUKE\s*\$([A-Z]{1,5})\b',
            r'NUKE\s+([A-Z]{1,5})\b',
            r'☢️\s*\$?([A-Z]{1,5})\b',
            r'💣\s*\$?([A-Z]{1,5})\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_upper)
            if match:
                return match.group(1)
        
        return None
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return Sadie's response.
        
        This is the main entry point for the chatbot.
        Supports NUKE mode for maximum overdrive analysis.
        """
        response = {
            "success": False,
            "message": "",
            "data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check for NUKE mode
            is_nuke = self._is_nuke_mode(user_message)
            
            # Extract symbol - use NUKE-specific extraction if in NUKE mode
            if is_nuke:
                symbol = self._extract_nuke_symbol(user_message)
                if not symbol:
                    symbol = self._extract_symbol_from_query(user_message)
            else:
                symbol = self._extract_symbol_from_query(user_message)
            
            # Build context with real-time data
            context = self._build_context_message(symbol)
            
            # Get Perplexity real-time research (complementary AI)
            perplexity_research = None
            if self.PERPLEXITY_API_KEY:
                perplexity_research = self._get_perplexity_research(user_message, symbol)
            
            # Prepare messages for GPT
            if is_nuke:
                # NUKE MODE - Use enhanced system context
                messages = [
                    {
                        "role": "system",
                        "content": self.SYSTEM_CONTEXT
                    },
                    {
                        "role": "system",
                        "content": self.NUKE_CONTEXT
                    },
                    {
                        "role": "system", 
                        "content": f"REAL-TIME MARKET DATA FOR NUKE ANALYSIS:\n{context}"
                    }
                ]
            else:
                # Normal mode
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
            
            # Add Perplexity research if available (complementary real-time intelligence)
            if perplexity_research:
                messages.append({
                    "role": "system",
                    "content": f"PERPLEXITY AI REAL-TIME RESEARCH (use this for current news, analyst opinions, and recent developments):\n\n{perplexity_research}"
                })
            
            # Add conversation history (last 10 exchanges)
            for msg in self.conversation_history[-20:]:
                messages.append(msg)
            
            # Add current user message with NUKE indicator
            if is_nuke and symbol:
                enhanced_message = f"☢️ NUKE MODE ACTIVATED ☢️\n\nPerform MAXIMUM OVERDRIVE analysis on ${symbol}. Follow the NUKE MODE REQUIREMENTS exactly. Provide ALL 14 sections with EXTREME DETAIL. Include multi-timeframe forecasts (1-day, 1-week, 1-month, 6-month, 1-year). Analyze dark pools, smart money, insider activity, and institutional flows. Leave NOTHING out.\n\nOriginal request: {user_message}"
            else:
                enhanced_message = user_message
            
            messages.append({
                "role": "user",
                "content": enhanced_message
            })
            
            # Call OpenRouter API with GPT-5/o1 thinking mode
            headers = {
                "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sadie-ai.com",
                "X-Title": "SadieAI Financial Assistant"
            }
            
            # NUKE mode gets more tokens for comprehensive analysis
            max_tokens = 16384 if is_nuke else 4096
            
            payload = {
                "model": self.MODELS[0],  # Use o1 (GPT-5 thinking mode)
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 1,  # o1 models require temperature=1
            }
            
            # NUKE mode gets longer timeout for comprehensive analysis
            timeout = 300 if is_nuke else 120  # 5 min for NUKE, 2 min normal
            
            api_response = requests.post(
                self.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if api_response.status_code == 200:
                result = api_response.json()
                assistant_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Check if response is empty - if so, try fallback models
                if not assistant_message or not assistant_message.strip():
                    import sys as _sys
                    print(f"Warning: o1 returned empty response, trying fallback models...", file=_sys.stderr)
                    
                    # Try fallback models for empty response
                    for fallback_model in self.MODELS[1:]:
                        try:
                            fallback_payload = {
                                "model": fallback_model,
                                "messages": messages,
                                "max_tokens": max_tokens,
                                "temperature": 0.7 if fallback_model not in ["openai/o1", "openai/o1-preview"] else 1,
                            }
                            
                            fallback_response = requests.post(
                                self.OPENROUTER_URL,
                                headers=headers,
                                json=fallback_payload,
                                timeout=90
                            )
                            
                            if fallback_response.status_code == 200:
                                fallback_result = fallback_response.json()
                                assistant_message = fallback_result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                
                                if assistant_message and assistant_message.strip():
                                    # Add NUKE mode header if applicable
                                    if is_nuke:
                                        assistant_message = f"☢️ **NUKE MODE ANALYSIS** ☢️\n\n{assistant_message}"
                                    
                                    self.conversation_history.append({"role": "user", "content": user_message})
                                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                                    
                                    response["success"] = True
                                    response["message"] = assistant_message
                                    response["data"] = {
                                        "symbol_detected": symbol,
                                        "model_used": fallback_model,
                                        "tokens_used": fallback_result.get('usage', {}),
                                        "nuke_mode": is_nuke,
                                        "perplexity_used": perplexity_research is not None
                                    }
                                    return response
                        except Exception as fallback_error:
                            print(f"Warning: Fallback model {fallback_model} failed: {fallback_error}", file=_sys.stderr)
                            continue
                    
                    # If all fallbacks failed, return error
                    response["message"] = "I apologize, but I'm having trouble generating a response right now. Please try again or rephrase your question."
                    return response
                
                # Add NUKE mode header if applicable
                if is_nuke:
                    assistant_message = f"☢️ **NUKE MODE ANALYSIS** ☢️\n\n{assistant_message}"
                
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
                    "tokens_used": result.get('usage', {}),
                    "nuke_mode": is_nuke,
                    "perplexity_used": perplexity_research is not None
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
                        
                        # Add NUKE mode header if applicable
                        if is_nuke:
                            assistant_message = f"☢️ **NUKE MODE ANALYSIS** ☢️\n\n{assistant_message}"
                        
                        self.conversation_history.append({"role": "user", "content": user_message})
                        self.conversation_history.append({"role": "assistant", "content": assistant_message})
                        
                        response["success"] = True
                        response["message"] = assistant_message
                        response["data"] = {
                            "symbol_detected": symbol,
                            "model_used": model,
                            "tokens_used": result.get('usage', {}),
                            "nuke_mode": is_nuke,
                            "perplexity_used": perplexity_research is not None
                        }
                        break
                
                if not response["success"]:
                    response["message"] = f"API Error: {api_response.status_code} - {api_response.text}"
                    
        except Exception as e:
            response["message"] = f"Error: {str(e)}"
        
        return response
    
    def _get_perplexity_research(self, query: str, symbol: Optional[str] = None) -> Optional[str]:
        """
        Get real-time financial research from Perplexity AI.
        
        Perplexity excels at:
        - Real-time news and market updates
        - Grounded facts with citations
        - Current events affecting stocks
        - Recent analyst opinions and price targets
        
        This complements GPT's deep reasoning with real-time research.
        """
        if not self.PERPLEXITY_API_KEY:
            return None
        
        try:
            # Build a focused financial research query
            if symbol:
                research_query = f"""Provide current, real-time financial research on {symbol} stock:

1. Latest news and developments (last 24-48 hours)
2. Recent analyst ratings, price targets, and upgrades/downgrades
3. Any upcoming catalysts (earnings, FDA dates, product launches, etc.)
4. Current market sentiment and institutional activity
5. Key risks and concerns being discussed
6. Recent insider trading activity
7. Options flow and unusual activity if notable

Be specific with dates, numbers, and cite sources. Focus on actionable trading information.

Original query: {query}"""
            else:
                research_query = f"""Provide current, real-time financial market research:

{query}

Be specific with dates, numbers, and cite sources. Focus on actionable trading information."""
            
            headers = {
                "Authorization": f"Bearer {self.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.PERPLEXITY_MODELS[0],  # sonar-pro
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional financial research analyst. Provide accurate, current, and actionable market intelligence. Always cite sources and be specific with dates and numbers. Focus on information that would help a trader make profitable decisions."
                    },
                    {
                        "role": "user",
                        "content": research_query
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.2,  # Low temperature for factual accuracy
                "return_citations": True,
                "search_recency_filter": "week"  # Focus on recent information
            }
            
            response = requests.post(
                self.PERPLEXITY_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                citations = result.get('citations', [])
                
                if content:
                    # Format with citations if available
                    research_output = f"📊 **PERPLEXITY REAL-TIME RESEARCH**\n\n{content}"
                    if citations:
                        research_output += "\n\n**Sources:**\n"
                        for i, cite in enumerate(citations[:5], 1):  # Top 5 citations
                            research_output += f"{i}. {cite}\n"
                    return research_output
            else:
                import sys as _sys
                print(f"Perplexity API error: {response.status_code}", file=_sys.stderr)
                
        except Exception as e:
            import sys as _sys
            print(f"Perplexity research error: {e}", file=_sys.stderr)
        
        return None
    
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
