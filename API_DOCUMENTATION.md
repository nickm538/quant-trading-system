# API Documentation Research

## Finnhub API (Primary Source)
**Base URL:** `https://finnhub.io/api/v1`
**API Key:** `d3ul051r01qil4aqj8j0d3ul051r01qil4aqj8jg`
**Authentication:** `token=apiKey` in URL or header `X-Finnhub-Token: apiKey`
**Rate Limit:** 30 API calls/second + plan-specific limits

### Critical Endpoints for Trading System:

#### Real-Time Price Data
- **Quote:** `/quote?symbol={symbol}&token={apiKey}`
  - Returns: current price, change, percent change, high, low, open, previous close
  
- **Candles (OHLCV):** `/stock/candle?symbol={symbol}&resolution={resolution}&from={from}&to={to}&token={apiKey}`
  - Resolutions: 1, 5, 15, 30, 60, D, W, M
  - Returns: open, high, low, close, volume arrays

#### Company Fundamentals
- **Company Profile 2:** `/stock/profile2?symbol={symbol}&token={apiKey}`
  - Returns: marketCap, shares outstanding, industry, sector, logo
  
- **Basic Financials:** `/stock/metric?symbol={symbol}&metric=all&token={apiKey}`
  - Returns: PE, PB, PS ratios, ROE, ROA, profit margin, etc.

#### News & Sentiment
- **Company News:** `/company-news?symbol={symbol}&from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={apiKey}`
  - Returns: headline, summary, url, source, datetime, sentiment
  
- **News Sentiment (PREMIUM):** `/news-sentiment?symbol={symbol}&token={apiKey}`
  - Returns: buzz, sentiment scores, sector sentiment

#### Analyst Data
- **Recommendation Trends:** `/stock/recommendation?symbol={symbol}&token={apiKey}`
  - Returns: buy, hold, sell, strongBuy, strongSell counts
  
- **Price Target (PREMIUM):** `/stock/price-target?symbol={symbol}&token={apiKey}`
  - Returns: targetHigh, targetLow, targetMean, targetMedian

#### Insider & Institutional
- **Insider Transactions:** `/stock/insider-transactions?symbol={symbol}&token={apiKey}`
  - Returns: name, share, change, filingDate, transactionDate
  
- **Institutional Ownership (PREMIUM):** `/institutional/ownership?symbol={symbol}&from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={apiKey}`

#### Earnings & Estimates
- **Earnings Calendar:** `/calendar/earnings?from={YYYY-MM-DD}&to={YYYY-MM-DD}&token={apiKey}`
  - Returns: epsActual, epsEstimate, revenueActual, revenueEstimate
  
- **EPS Surprises:** `/stock/earnings?symbol={symbol}&token={apiKey}`

### Important Notes:
- **Premium endpoints** require paid plan (Price Target, News Sentiment, Institutional Ownership)
- **Free tier** includes: Quote, Candles, Company News, Basic Financials, Recommendations
- **Rate limit:** 429 status code when exceeded
- **Error handling:** Always check for 403 (forbidden), 429 (rate limit), 404 (not found)

---

## AlphaVantage API (Secondary Source)
**Base URL:** `https://www.alphavantage.co/query`
**API Key:** `UDU3WP1A94ETAIME`
**Rate Limit:** 5 API calls/minute (free tier), 75 calls/minute (premium)

### Critical Endpoints:

#### Intraday Data
- **TIME_SERIES_INTRADAY:** `?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={1min|5min|15min|30min|60min}&apikey={apiKey}`
  - Returns: timestamp, open, high, low, close, volume
  
#### Daily Data
- **TIME_SERIES_DAILY:** `?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={apiKey}`
  - Returns: full daily OHLCV data

#### Company Overview
- **OVERVIEW:** `?function=OVERVIEW&symbol={symbol}&apikey={apiKey}`
  - Returns: PE, PEG, EPS, revenue, profit margin, beta, 52-week high/low

#### News Sentiment
- **NEWS_SENTIMENT:** `?function=NEWS_SENTIMENT&tickers={symbol}&apikey={apiKey}`
  - Returns: feed with sentiment scores, relevance scores

#### Technical Indicators
- **RSI, MACD, SMA, EMA, BBANDS, etc.:** `?function={indicator}&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={apiKey}`

#### Treasury Yields
- **TREASURY_YIELD:** `?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={apiKey}`
  - Critical for risk-free rate in options pricing

---

## Yahoo Finance (yfinance) - Tertiary Source
**Library:** `yfinance` Python package
**Rate Limit:** Unofficial API, no documented limits but can be rate-limited

### Available Data:
- Historical OHLCV data
- Options chains with Greeks (delta, gamma, theta, vega)
- Company info (sector, industry, market cap)
- Analyst recommendations
- Earnings dates
- Insider trades

### Limitations:
- No official API, subject to breaking changes
- Rate limiting not documented
- Data quality can vary

---

## Financial Modeling Prep (FMP) - Backup Source
**Base URL:** `https://financialmodelingprep.com/api/v3`
**API Key:** `LTecnRjOFtd8bFOTCRLpcncjxrqaZlqq`
**Rate Limit:** 250 calls/day (free tier)

### Critical Endpoints:

#### Quote
- **Real-Time Quote:** `/quote/{symbol}?apikey={apiKey}`
  - Returns: price, change, dayLow, dayHigh, volume, avgVolume

#### Historical Data
- **Historical Price:** `/historical-price-full/{symbol}?apikey={apiKey}`
  - Returns: full OHLCV history

#### Fundamentals
- **Company Profile:** `/profile/{symbol}?apikey={apiKey}`
  - Returns: sector, industry, marketCap, beta, description

#### Financial Statements
- **Income Statement:** `/income-statement/{symbol}?apikey={apiKey}`
- **Balance Sheet:** `/balance-sheet-statement/{symbol}?apikey={apiKey}`
- **Cash Flow:** `/cash-flow-statement/{symbol}?apikey={apiKey}`

---

## Data Source Hierarchy (Fallback Strategy)

1. **Finnhub (Primary)** - Most comprehensive, real-time data
   - Use for: Real-time quotes, company news, basic fundamentals, insider trades
   - Fallback if: 403 (forbidden), 429 (rate limit), 404 (not found)

2. **AlphaVantage (Secondary)** - Good for intraday and technical indicators
   - Use for: Intraday 1-min data, technical indicators, news sentiment, treasury yields
   - Fallback if: Rate limit exceeded (5 calls/min)

3. **yfinance (Tertiary)** - Reliable for options and historical data
   - Use for: Options chains with Greeks, historical OHLCV, analyst recommendations
   - Fallback if: Connection errors, rate limiting

4. **FMP (Backup)** - Last resort for basic data
   - Use for: Quote, historical price, company profile
   - Fallback if: All other sources fail

---

## Circuit Breaker Thresholds

### Data Integrity Requirements:
- **Price data:** Must have at least 90% of expected data points
- **Options data:** Must have at least 5 strikes per expiration
- **News data:** Sentiment scores must be between -1 and 1
- **Technical indicators:** No NaN values in last 30 days
- **Cross-validation:** Price from multiple sources must agree within 0.5%

### Error Handling:
- **403 Forbidden:** Switch to next data source immediately
- **429 Rate Limit:** Wait and retry with exponential backoff, max 3 retries
- **404 Not Found:** Try alternate symbol format, then fail gracefully
- **Timeout:** Retry once, then switch to next source
- **Invalid Data:** Log warning, use previous valid data point if < 5 minutes old

### Confidence Scoring:
- **100%:** Data from primary source (Finnhub) with cross-validation
- **90%:** Data from primary source without cross-validation
- **80%:** Data from secondary source (AlphaVantage) with validation
- **70%:** Data from tertiary source (yfinance)
- **60%:** Data from backup source (FMP)
- **<60%:** HALT ANALYSIS - Display error to user

**Never display results with confidence < 95% for real-money trading decisions.**
