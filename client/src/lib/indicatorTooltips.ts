/**
 * Indicator Tooltip Definitions
 * 
 * Provides explanations, current values, and normal ranges for all trading indicators
 */

export interface IndicatorTooltip {
  name: string;
  description: string;
  interpretation: string;
  goodRange: string;
  badRange: string;
}

export const INDICATOR_TOOLTIPS: Record<string, IndicatorTooltip> = {
  // === FUNDAMENTAL INDICATORS ===
  pe_ratio: {
    name: "P/E Ratio (Price-to-Earnings)",
    description: "Measures how much investors pay for $1 of earnings. Lower is generally better value.",
    interpretation: "Compares stock price to annual earnings per share",
    goodRange: "15-25 (fair value for most stocks)",
    badRange: ">40 (overvalued), <10 (undervalued or troubled)"
  },
  
  peg_ratio: {
    name: "PEG Ratio (Price/Earnings-to-Growth)",
    description: "P/E ratio divided by earnings growth rate. Accounts for growth expectations.",
    interpretation: "Balances valuation with growth potential",
    goodRange: "<1.0 (undervalued), 1.0-2.0 (fair)",
    badRange: ">2.5 (overvalued relative to growth)"
  },
  
  profit_margin: {
    name: "Profit Margin",
    description: "Percentage of revenue that becomes profit. Higher means more efficient business.",
    interpretation: "Net income / revenue",
    goodRange: ">20% (excellent), 10-20% (good)",
    badRange: "<5% (low profitability)"
  },
  
  roe: {
    name: "ROE (Return on Equity)",
    description: "How efficiently company generates profit from shareholders' equity.",
    interpretation: "Net income / shareholder equity",
    goodRange: ">15% (strong), 20%+ (excellent)",
    badRange: "<10% (weak)"
  },
  
  debt_to_equity: {
    name: "Debt-to-Equity Ratio",
    description: "Company's debt relative to shareholder equity. Lower is safer.",
    interpretation: "Total debt / total equity",
    goodRange: "<0.5 (conservative), 0.5-1.0 (moderate)",
    badRange: ">2.0 (high leverage risk)"
  },
  
  revenue_growth: {
    name: "Revenue Growth (YoY)",
    description: "Year-over-year revenue growth rate. Indicates business expansion.",
    interpretation: "% change in revenue vs last year",
    goodRange: ">10% (strong growth)",
    badRange: "<0% (declining revenue)"
  },
  
  earnings_growth: {
    name: "Earnings Growth (YoY)",
    description: "Year-over-year earnings growth rate. Key driver of stock price.",
    interpretation: "% change in EPS vs last year",
    goodRange: ">15% (strong growth)",
    badRange: "<0% (declining earnings)"
  },
  
  dividend_yield: {
    name: "Dividend Yield",
    description: "Annual dividend payment as % of stock price. Income for investors.",
    interpretation: "Annual dividend / current price",
    goodRange: "2-4% (income stocks)",
    badRange: ">8% (unsustainable?), 0% (no dividend)"
  },
  
  beta: {
    name: "Beta (Market Sensitivity)",
    description: "Stock's volatility relative to overall market. 1.0 = market average.",
    interpretation: "Correlation with market movements",
    goodRange: "0.8-1.2 (moderate volatility)",
    badRange: ">2.0 (very volatile), <0.5 (low growth)"
  },
  
  // === TECHNICAL INDICATORS ===
  rsi: {
    name: "RSI (Relative Strength Index)",
    description: "Momentum oscillator measuring speed/change of price movements.",
    interpretation: "0-100 scale, identifies overbought/oversold conditions",
    goodRange: "40-60 (neutral), 30-40 (buy zone)",
    badRange: ">70 (overbought), <30 (oversold)"
  },
  
  macd: {
    name: "MACD (Moving Average Convergence Divergence)",
    description: "Trend-following momentum indicator showing relationship between two moving averages.",
    interpretation: "Positive = bullish, negative = bearish. Watch for crossovers.",
    goodRange: "Above signal line (bullish)",
    badRange: "Below signal line (bearish)"
  },
  
  adx: {
    name: "ADX (Average Directional Index)",
    description: "Measures trend strength regardless of direction. Does NOT indicate trend direction.",
    interpretation: "0-100 scale, higher = stronger trend",
    goodRange: ">25 (strong trend), >40 (very strong)",
    badRange: "<20 (weak/choppy trend)"
  },
  
  atr: {
    name: "ATR (Average True Range)",
    description: "Volatility indicator showing average price range over period. Used for stop-loss placement.",
    interpretation: "Higher ATR = more volatile, wider stops needed",
    goodRange: "1-3% of price (normal volatility)",
    badRange: ">5% (high volatility risk)"
  },
  
  sma_20: {
    name: "SMA 20 (20-Day Simple Moving Average)",
    description: "Average closing price over last 20 days. Short-term trend indicator.",
    interpretation: "Price above SMA = bullish, below = bearish",
    goodRange: "Price 2-5% above SMA (uptrend)",
    badRange: "Price >10% from SMA (overextended)"
  },
  
  sma_50: {
    name: "SMA 50 (50-Day Simple Moving Average)",
    description: "Average closing price over last 50 days. Medium-term trend indicator.",
    interpretation: "Price above SMA = bullish, below = bearish",
    goodRange: "Price above SMA 50 (uptrend)",
    badRange: "Price below SMA 50 (downtrend)"
  },
  
  sma_200: {
    name: "SMA 200 (200-Day Simple Moving Average)",
    description: "Average closing price over last 200 days. Long-term trend indicator.",
    interpretation: "Price above = bull market, below = bear market",
    goodRange: "Price above SMA 200 (bull market)",
    badRange: "Price below SMA 200 (bear market)"
  },
  
  bollinger_bands: {
    name: "Bollinger Bands",
    description: "Volatility bands around moving average. Shows overbought/oversold levels.",
    interpretation: "Price at upper band = overbought, lower band = oversold",
    goodRange: "Price near middle band (neutral)",
    badRange: "Price at extreme bands (reversal risk)"
  },
  
  vwap: {
    name: "VWAP (Volume-Weighted Average Price)",
    description: "Average price weighted by volume. Institutional benchmark for intraday trading.",
    interpretation: "Price above VWAP = bullish, below = bearish",
    goodRange: "Price near VWAP (fair value)",
    badRange: "Price >5% from VWAP (extended)"
  },
  
  // === RISK METRICS ===
  var_95: {
    name: "VaR 95% (Value at Risk)",
    description: "Maximum expected loss over period with 95% confidence. Risk management metric.",
    interpretation: "5% chance of losing more than this amount",
    goodRange: "<2% (low risk)",
    badRange: ">5% (high risk)"
  },
  
  cvar_95: {
    name: "CVaR 95% (Conditional Value at Risk)",
    description: "Average loss in worst 5% of scenarios. Tail risk measure.",
    interpretation: "Expected loss if VaR threshold is breached",
    goodRange: "<3% (manageable tail risk)",
    badRange: ">7% (severe tail risk)"
  },
  
  sharpe_ratio: {
    name: "Sharpe Ratio",
    description: "Risk-adjusted return. Higher is better. Measures return per unit of risk.",
    interpretation: "(Return - Risk-free rate) / Volatility",
    goodRange: ">1.0 (good), >2.0 (excellent)",
    badRange: "<0.5 (poor risk-adjusted return)"
  },
  
  max_drawdown: {
    name: "Maximum Drawdown",
    description: "Largest peak-to-trough decline. Measures worst-case historical loss.",
    interpretation: "% decline from highest point to lowest",
    goodRange: "<20% (manageable)",
    badRange: ">40% (severe drawdown risk)"
  },
  
  // === VOLATILITY METRICS ===
  implied_volatility: {
    name: "Implied Volatility (IV)",
    description: "Market's expectation of future volatility derived from option prices.",
    interpretation: "Higher IV = higher option prices, more uncertainty",
    goodRange: "20-40% (normal for most stocks)",
    badRange: ">60% (extreme uncertainty)"
  },
  
  historical_volatility: {
    name: "Historical Volatility",
    description: "Actual past price volatility over period. Annualized standard deviation.",
    interpretation: "Measures realized price fluctuations",
    goodRange: "15-30% (moderate volatility)",
    badRange: ">50% (very volatile)"
  },
  
  // === GREEKS (OPTIONS) ===
  delta: {
    name: "Delta",
    description: "Option price change per $1 stock move. Also probability of expiring in-the-money.",
    interpretation: "0-1 for calls, -1-0 for puts",
    goodRange: "0.5-0.7 (at-the-money to slightly ITM)",
    badRange: "<0.3 (far OTM, low probability)"
  },
  
  gamma: {
    name: "Gamma",
    description: "Rate of delta change per $1 stock move. Measures delta acceleration.",
    interpretation: "Higher gamma = delta changes faster",
    goodRange: "0.01-0.05 (moderate sensitivity)",
    badRange: ">0.1 (very sensitive, risky)"
  },
  
  theta: {
    name: "Theta (Time Decay)",
    description: "Option value lost per day due to time passing. Always negative for long options.",
    interpretation: "Daily time decay in dollars",
    goodRange: "-$0.10 to -$0.50 (moderate decay)",
    badRange: "<-$1.00 (rapid decay, near expiration)"
  },
  
  vega: {
    name: "Vega (Volatility Sensitivity)",
    description: "Option price change per 1% change in implied volatility.",
    interpretation: "Higher vega = more sensitive to IV changes",
    goodRange: "0.10-0.50 (moderate IV sensitivity)",
    badRange: ">1.00 (very sensitive to IV)"
  },
  
  rho: {
    name: "Rho (Interest Rate Sensitivity)",
    description: "Option price change per 1% change in interest rates. Usually minor impact.",
    interpretation: "Positive for calls, negative for puts",
    goodRange: "0.05-0.20 (low rate sensitivity)",
    badRange: ">0.50 (high rate sensitivity)"
  },
  
  // === SCORES ===
  fundamental_score: {
    name: "Fundamental Score",
    description: "Composite score of company's financial health and valuation.",
    interpretation: "Weighted average of P/E, ROE, growth, margins, etc.",
    goodRange: "60-80 (strong fundamentals)",
    badRange: "<40 (weak fundamentals)"
  },
  
  technical_score: {
    name: "Technical Score",
    description: "Composite score of price action, momentum, and trend strength.",
    interpretation: "Weighted average of RSI, MACD, moving averages, etc.",
    goodRange: "60-80 (strong technicals)",
    badRange: "<40 (weak technicals)"
  },
  
  sentiment_score: {
    name: "Sentiment Score",
    description: "Market sentiment from news, social media, and analyst ratings.",
    interpretation: "0-100 scale, 50 = neutral",
    goodRange: "55-70 (positive sentiment)",
    badRange: "<35 (negative sentiment), >80 (euphoria)"
  },
  
  momentum_score: {
    name: "Momentum Score",
    description: "Measures strength and direction of price momentum.",
    interpretation: "Based on RSI and rate of change",
    goodRange: "60-80 (strong momentum)",
    badRange: "<40 (weak/negative momentum)"
  },
  
  trend_score: {
    name: "Trend Score",
    description: "Measures trend strength and direction using moving averages.",
    interpretation: "Price vs SMA 20/50/200 + ADX",
    goodRange: "60-90 (strong trend)",
    badRange: "<40 (weak/choppy trend)"
  },
  
  volatility_score: {
    name: "Volatility Score",
    description: "Inverse volatility score. Higher = lower volatility = safer.",
    interpretation: "100 - (volatility × 100)",
    goodRange: ">70 (low volatility, stable)",
    badRange: "<50 (high volatility, risky)"
  },
  
  overall_score: {
    name: "Overall Score",
    description: "Master score combining fundamentals, technicals, and sentiment.",
    interpretation: "Weighted: 30% fundamental, 50% technical, 20% sentiment",
    goodRange: "60-75 (BUY), 75+ (STRONG BUY)",
    badRange: "<40 (SELL), 40-60 (HOLD)"
  },
  
  // === GARCH MODEL ===
  garch_aic: {
    name: "GARCH AIC (Akaike Information Criterion)",
    description: "Model fit quality metric. Lower is better.",
    interpretation: "Measures GARCH model goodness-of-fit",
    goodRange: "Negative values (better fit)",
    badRange: "Positive values (worse fit)"
  },
  
  garch_bic: {
    name: "GARCH BIC (Bayesian Information Criterion)",
    description: "Model fit quality with penalty for complexity. Lower is better.",
    interpretation: "Similar to AIC but penalizes complex models more",
    goodRange: "Negative values (better fit)",
    badRange: "Positive values (worse fit)"
  },
  
  fat_tail_df: {
    name: "Fat-Tail Degrees of Freedom",
    description: "Student-t distribution parameter. Lower = fatter tails = more extreme events.",
    interpretation: "Measures tail risk in return distribution",
    goodRange: ">10 (normal tails)",
    badRange: "<5 (fat tails, high crash risk)"
  }
};

/**
 * Get tooltip for an indicator
 */
export function getIndicatorTooltip(key: string): IndicatorTooltip | null {
  return INDICATOR_TOOLTIPS[key] || null;
}

/**
 * Format tooltip content for display
 */
export function formatTooltipContent(key: string, value: number | string): string {
  const tooltip = getIndicatorTooltip(key);
  if (!tooltip) return "";
  
  return `**${tooltip.name}**\n\n${tooltip.description}\n\n**Current Value:** ${value}\n\n**Interpretation:** ${tooltip.interpretation}\n\n**Good Range:** ${tooltip.goodRange}\n\n**Bad Range:** ${tooltip.badRange}`;
}
