import { int, mediumtext, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Trained ML models for stock prediction
 * Stores model metadata, parameters, and performance metrics
 */
export const trainedModels = mysqlTable("trained_models", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  modelType: mysqlEnum("model_type", ["xgboost", "lightgbm", "lstm", "ensemble"]).notNull(),
  version: varchar("version", { length: 50 }).notNull(),
  
  // Model storage
  modelPath: text("model_path").notNull(), // S3 path to serialized model
  modelData: mediumtext("model_data"), // Base64 encoded serialized model (for direct DB storage)
  
  // Training metadata
  trainedAt: timestamp("trained_at").defaultNow().notNull(),
  trainingStartDate: timestamp("training_start_date").notNull(),
  trainingEndDate: timestamp("training_end_date").notNull(),
  trainingDataPoints: int("training_data_points").notNull(),
  
  // Performance metrics
  trainingAccuracy: int("training_accuracy").notNull(), // Stored as percentage * 100 (e.g., 8532 = 85.32%)
  validationAccuracy: int("validation_accuracy").notNull(),
  testAccuracy: int("test_accuracy").notNull(),
  mse: int("mse").notNull(), // Mean Squared Error * 10000
  mae: int("mae").notNull(), // Mean Absolute Error * 10000
  r2Score: int("r2_score").notNull(), // R² Score * 10000
  
  // Model parameters (JSON)
  hyperparameters: text("hyperparameters").notNull(),
  featureImportance: text("feature_importance"), // JSON of feature importance scores
  
  // Status
  isActive: mysqlEnum("is_active", ["active", "inactive", "deprecated"]).default("active").notNull(),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type TrainedModel = typeof trainedModels.$inferSelect;
export type InsertTrainedModel = typeof trainedModels.$inferInsert;

/**
 * Model predictions and actual outcomes
 * Tracks prediction accuracy over time for continuous improvement
 */
export const modelPredictions = mysqlTable("model_predictions", {
  id: int("id").autoincrement().primaryKey(),
  modelId: int("model_id").notNull(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Prediction details
  predictionDate: timestamp("prediction_date").notNull(),
  targetDate: timestamp("target_date").notNull(), // Date for which prediction was made
  
  // Predicted values (stored as cents, e.g., 15050 = $150.50)
  predictedPrice: int("predicted_price").notNull(),
  predictedLow: int("predicted_low").notNull(),
  predictedHigh: int("predicted_high").notNull(),
  confidence: int("confidence").notNull(), // Confidence score * 100
  
  // Actual values (filled after target date)
  actualPrice: int("actual_price"),
  actualLow: int("actual_low"),
  actualHigh: int("actual_high"),
  
  // Error metrics (calculated after actual values are known)
  priceError: int("price_error"), // Absolute error in cents
  percentageError: int("percentage_error"), // Error percentage * 100
  
  // Status
  status: mysqlEnum("status", ["pending", "validated", "failed"]).default("pending").notNull(),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type ModelPrediction = typeof modelPredictions.$inferSelect;
export type InsertModelPrediction = typeof modelPredictions.$inferInsert;

/**
 * Model retraining history
 * Tracks when and why models were retrained
 */
export const retrainingHistory = mysqlTable("retraining_history", {
  id: int("id").autoincrement().primaryKey(),
  oldModelId: int("old_model_id"),
  newModelId: int("new_model_id").notNull(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Retraining trigger
  triggerReason: mysqlEnum("trigger_reason", [
    "scheduled",
    "accuracy_degradation",
    "regime_change",
    "manual",
    "new_data_available"
  ]).notNull(),
  
  // Performance comparison
  oldAccuracy: int("old_accuracy"),
  newAccuracy: int("new_accuracy").notNull(),
  improvementPct: int("improvement_pct"), // Improvement percentage * 100
  
  // Metadata
  retrainedAt: timestamp("retrained_at").defaultNow().notNull(),
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type RetrainingHistory = typeof retrainingHistory.$inferSelect;
export type InsertRetrainingHistory = typeof retrainingHistory.$inferInsert;

/**
 * Analysis results cache
 * Caches stock analysis results to reduce API calls and improve response time
 */
export const analysisCache = mysqlTable("analysis_cache", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull().unique(),
  
  // Cached analysis result (JSON)
  analysisData: text("analysis_data").notNull(),
  
  // Cache metadata
  cachedAt: timestamp("cached_at").defaultNow().notNull(),
  expiresAt: timestamp("expires_at").notNull(),
  
  // Cache statistics
  hitCount: int("hit_count").default(0).notNull(),
  lastAccessedAt: timestamp("last_accessed_at").defaultNow().notNull(),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type AnalysisCache = typeof analysisCache.$inferSelect;
export type InsertAnalysisCache = typeof analysisCache.$inferInsert;
/**
 * Options data and Greeks
 * Stores options chain data, IV, and Greeks for analysis
 */
export const optionsData = mysqlTable("options_data", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Option details
  optionType: mysqlEnum("option_type", ["call", "put"]).notNull(),
  strikePrice: int("strike_price").notNull(), // Stored as cents
  expirationDate: timestamp("expiration_date").notNull(),
  
  // Pricing
  lastPrice: int("last_price").notNull(), // Stored as cents
  bid: int("bid").notNull(),
  ask: int("ask").notNull(),
  
  // Volume and Open Interest
  volume: int("volume").notNull(),
  openInterest: int("open_interest").notNull(),
  volumeOIRatio: int("volume_oi_ratio").notNull(), // Ratio * 10000
  
  // Greeks (stored as * 10000 for precision)
  delta: int("delta").notNull(),
  gamma: int("gamma").notNull(),
  theta: int("theta").notNull(),
  vega: int("vega").notNull(),
  rho: int("rho").notNull(),
  
  // Implied Volatility
  impliedVolatility: int("implied_volatility").notNull(), // IV * 10000
  ivPercentile: int("iv_percentile"), // IV percentile over 1 year * 100
  ivRank: int("iv_rank"), // IV rank over 1 year * 100
  
  // IV Crush Detection
  preEarningsIV: int("pre_earnings_iv"), // IV before earnings
  postEarningsIV: int("post_earnings_iv"), // IV after earnings
  ivCrushPct: int("iv_crush_pct"), // IV crush percentage * 100
  
  // Timestamps
  dataTimestamp: timestamp("data_timestamp").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type OptionsData = typeof optionsData.$inferSelect;
export type InsertOptionsData = typeof optionsData.$inferInsert;

/**
 * Intraday price data
 * Stores pre-market, post-market, gaps, and VWAP
 */
export const intradayData = mysqlTable("intraday_data", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  tradingDate: timestamp("trading_date").notNull(),
  
  // Pre-market (4am-9:30am ET)
  preMarketOpen: int("pre_market_open"), // Stored as cents
  preMarketHigh: int("pre_market_high"),
  preMarketLow: int("pre_market_low"),
  preMarketClose: int("pre_market_close"),
  preMarketVolume: int("pre_market_volume"),
  preMarketChange: int("pre_market_change"), // Change from previous close * 100
  
  // Regular session (9:30am-4pm ET)
  regularOpen: int("regular_open").notNull(),
  regularHigh: int("regular_high").notNull(),
  regularLow: int("regular_low").notNull(),
  regularClose: int("regular_close").notNull(),
  regularVolume: int("regular_volume").notNull(),
  
  // Post-market (4pm-8pm ET)
  postMarketOpen: int("post_market_open"),
  postMarketHigh: int("post_market_high"),
  postMarketLow: int("post_market_low"),
  postMarketClose: int("post_market_close"),
  postMarketVolume: int("post_market_volume"),
  postMarketChange: int("post_market_change"), // Change from regular close * 100
  
  // Gap Analysis
  gapType: mysqlEnum("gap_type", ["gap_up", "gap_down", "no_gap"]),
  gapSize: int("gap_size"), // Gap size in cents
  gapPct: int("gap_pct"), // Gap percentage * 100
  gapFilled: mysqlEnum("gap_filled", ["yes", "no", "partial"]),
  
  // VWAP
  vwap: int("vwap").notNull(), // Volume Weighted Average Price * 100
  vwapDeviation: int("vwap_deviation"), // Current price deviation from VWAP * 100
  
  // Timestamps
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type IntradayData = typeof intradayData.$inferSelect;
export type InsertIntradayData = typeof intradayData.$inferInsert;

/**
 * Market events calendar
 * Tracks earnings, dividends, splits, Fed meetings, etc.
 */
export const marketEvents = mysqlTable("market_events", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }), // NULL for market-wide events
  
  // Event details
  eventType: mysqlEnum("event_type", [
    "earnings",
    "dividend",
    "split",
    "fed_meeting",
    "economic_data",
    "conference",
    "ipo",
    "merger",
    "other"
  ]).notNull(),
  eventDate: timestamp("event_date").notNull(),
  eventTime: varchar("event_time", { length: 20 }), // e.g., "before_market", "after_market", "10:00 AM ET"
  
  // Event metadata
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  importance: mysqlEnum("importance", ["low", "medium", "high", "critical"]).default("medium").notNull(),
  
  // Earnings-specific
  estimatedEPS: int("estimated_eps"), // EPS estimate * 10000
  actualEPS: int("actual_eps"), // Actual EPS * 10000
  epsSurprise: int("eps_surprise"), // Surprise percentage * 100
  
  // Dividend-specific
  dividendAmount: int("dividend_amount"), // Dividend per share * 10000
  exDividendDate: timestamp("ex_dividend_date"),
  paymentDate: timestamp("payment_date"),
  
  // Split-specific
  splitRatio: varchar("split_ratio", { length: 20 }), // e.g., "2:1", "3:2"
  
  // Impact tracking
  priceImpact: int("price_impact"), // Price change after event * 100
  volumeImpact: int("volume_impact"), // Volume change after event * 100
  
  // Timestamps
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type MarketEvent = typeof marketEvents.$inferSelect;
export type InsertMarketEvent = typeof marketEvents.$inferInsert;

/**
 * Dark pool activity
 * Tracks large block trades and institutional activity
 */
export const darkPoolActivity = mysqlTable("dark_pool_activity", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Trade details
  tradeDate: timestamp("trade_date").notNull(),
  tradeTime: varchar("trade_time", { length: 20 }).notNull(),
  
  // Volume and pricing
  volume: int("volume").notNull(),
  price: int("price").notNull(), // Stored as cents
  totalValue: int("total_value").notNull(), // Total trade value in dollars
  
  // Trade classification
  tradeType: mysqlEnum("trade_type", ["block", "sweep", "split", "unusual"]).notNull(),
  sentiment: mysqlEnum("sentiment", ["bullish", "bearish", "neutral"]).notNull(),
  
  // Context
  percentOfDailyVolume: int("percent_of_daily_volume").notNull(), // Percentage * 100
  priceVsVWAP: int("price_vs_vwap"), // Price relative to VWAP * 100
  
  // Significance
  significance: mysqlEnum("significance", ["low", "medium", "high", "extreme"]).default("medium").notNull(),
  
  // Timestamps
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type DarkPoolActivity = typeof darkPoolActivity.$inferSelect;
export type InsertDarkPoolActivity = typeof darkPoolActivity.$inferInsert;

/**
 * News sentiment cache
 * Caches news articles and sentiment scores
 */
export const newsSentimentCache = mysqlTable("news_sentiment_cache", {
  id: int("id").autoincrement().primaryKey(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Article details
  title: text("title").notNull(),
  source: varchar("source", { length: 255 }).notNull(),
  url: text("url"),
  publishedAt: timestamp("published_at").notNull(),
  
  // Sentiment analysis
  sentimentScore: int("sentiment_score").notNull(), // Score * 10000 (-1 to 1)
  sentimentLabel: mysqlEnum("sentiment_label", ["bearish", "neutral", "bullish"]).notNull(),
  relevanceScore: int("relevance_score").notNull(), // Relevance * 10000 (0 to 1)
  
  // Content
  summary: text("summary"),
  topics: text("topics"), // JSON array of topics
  
  // Cache metadata
  cachedAt: timestamp("cached_at").defaultNow().notNull(),
  expiresAt: timestamp("expires_at").notNull(),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type NewsSentimentCache = typeof newsSentimentCache.$inferSelect;
export type InsertNewsSentimentCache = typeof newsSentimentCache.$inferInsert;

/**
 * Backtesting results
 * Stores walk-forward optimization and out-of-sample testing results
 */
export const backtestingResults = mysqlTable("backtesting_results", {
  id: int("id").autoincrement().primaryKey(),
  modelId: int("model_id").notNull(),
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Test period
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date").notNull(),
  totalDays: int("total_days").notNull(),
  
  // Performance metrics
  totalReturn: int("total_return").notNull(), // Total return * 10000
  annualizedReturn: int("annualized_return").notNull(), // Annualized return * 10000
  sharpeRatio: int("sharpe_ratio").notNull(), // Sharpe ratio * 10000
  sortinoRatio: int("sortino_ratio").notNull(), // Sortino ratio * 10000
  maxDrawdown: int("max_drawdown").notNull(), // Max drawdown * 10000
  
  // Trade statistics
  totalTrades: int("total_trades").notNull(),
  winningTrades: int("winning_trades").notNull(),
  losingTrades: int("losing_trades").notNull(),
  winRate: int("win_rate").notNull(), // Win rate * 10000
  avgWin: int("avg_win").notNull(), // Average win * 10000
  avgLoss: int("avg_loss").notNull(), // Average loss * 10000
  profitFactor: int("profit_factor").notNull(), // Profit factor * 10000
  
  // Risk metrics
  valueAtRisk95: int("value_at_risk_95").notNull(), // VaR 95% * 10000
  conditionalVaR95: int("conditional_var_95").notNull(), // CVaR 95% * 10000
  
  // Test type
  testType: mysqlEnum("test_type", ["in_sample", "out_of_sample", "walk_forward"]).notNull(),
  
  // Detailed results (JSON)
  tradeLog: text("trade_log"), // JSON array of all trades
  equityCurve: text("equity_curve"), // JSON array of equity over time
  
  // Timestamps
  testedAt: timestamp("tested_at").defaultNow().notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type BacktestingResult = typeof backtestingResults.$inferSelect;
export type InsertBacktestingResult = typeof backtestingResults.$inferInsert;
