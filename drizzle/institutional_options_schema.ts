import { mysqlTable, int, varchar, timestamp, text, mysqlEnum } from "drizzle-orm/mysql-core";

/**
 * Institutional Options Analysis Results
 * Stores comprehensive analysis results from the institutional options engine
 * Links to options_data table for raw option data
 */
export const institutionalOptionsAnalysis = mysqlTable("institutional_options_analysis", {
  id: int("id").autoincrement().primaryKey(),
  
  // Reference to raw options data
  optionsDataId: int("options_data_id"), // FK to options_data table (nullable for legacy data)
  
  // Stock and option identification
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  optionType: mysqlEnum("option_type", ["call", "put"]).notNull(),
  strikePrice: int("strike_price").notNull(), // Stored as cents
  expirationDate: timestamp("expiration_date").notNull(),
  daysToExpiry: int("days_to_expiry").notNull(),
  
  // Pricing at analysis time
  lastPrice: int("last_price").notNull(), // Stored as cents
  bid: int("bid").notNull(),
  ask: int("ask").notNull(),
  midPrice: int("mid_price").notNull(),
  currentStockPrice: int("current_stock_price").notNull(),
  
  // Scoring (stored as * 100 for precision, e.g., 7957 = 79.57)
  finalScore: int("final_score").notNull(),
  rating: mysqlEnum("rating", ["EXCEPTIONAL", "EXCELLENT", "GOOD", "ACCEPTABLE", "NEUTRAL"]).notNull(),
  
  // Category scores (stored as * 10 for precision, e.g., 740 = 74.0)
  volatilityScore: int("volatility_score").notNull(),
  greeksScore: int("greeks_score").notNull(),
  technicalScore: int("technical_score").notNull(),
  liquidityScore: int("liquidity_score").notNull(),
  eventRiskScore: int("event_risk_score").notNull(),
  sentimentScore: int("sentiment_score").notNull(),
  flowScore: int("flow_score").notNull(),
  expectedValueScore: int("expected_value_score").notNull(),
  
  // Greeks (stored as * 10000 for precision)
  delta: int("delta").notNull(),
  gamma: int("gamma").notNull(),
  vega: int("vega").notNull(),
  theta: int("theta").notNull(),
  
  // Second-order Greeks (stored as * 10000)
  vanna: int("vanna"),
  charm: int("charm"),
  
  // Key metrics
  impliedVolatility: int("implied_volatility").notNull(), // IV * 100 (e.g., 2229 = 22.29%)
  spreadPct: int("spread_pct").notNull(), // Spread % * 100
  volume: int("volume").notNull(),
  openInterest: int("open_interest").notNull(),
  
  // Risk management (Kelly Criterion)
  kellyPct: int("kelly_pct").notNull(), // Kelly % * 10000 (e.g., 3400 = 34%)
  conservativeKelly: int("conservative_kelly").notNull(),
  maxPositionSizePct: int("max_position_size_pct").notNull(),
  
  // Market context at analysis time
  historicalVolatility: int("historical_volatility"), // HV * 10000
  daysToEarnings: int("days_to_earnings"),
  sentimentScoreValue: int("sentiment_score_value"), // 0-100
  
  // Insights (stored as JSON array)
  insights: text("insights"), // JSON array of insight strings
  
  // Analysis metadata
  analysisTimestamp: timestamp("analysis_timestamp").notNull(),
  engineVersion: varchar("engine_version", { length: 20 }).notNull(), // e.g., "1.0.0"
  
  // Performance tracking (filled later)
  actualOutcome: mysqlEnum("actual_outcome", ["profit", "loss", "breakeven", "pending"]).default("pending"),
  actualReturn: int("actual_return"), // Actual return % * 100
  daysHeld: int("days_held"), // How many days position was held
  exitPrice: int("exit_price"), // Exit price in cents
  exitDate: timestamp("exit_date"),
  
  // Timestamps
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type InstitutionalOptionsAnalysis = typeof institutionalOptionsAnalysis.$inferSelect;
export type InsertInstitutionalOptionsAnalysis = typeof institutionalOptionsAnalysis.$inferInsert;

/**
 * Options Analysis Scan Summary
 * Stores metadata about each full scan/analysis run
 */
export const optionsAnalysisScan = mysqlTable("options_analysis_scan", {
  id: int("id").autoincrement().primaryKey(),
  
  stockSymbol: varchar("stock_symbol", { length: 10 }).notNull(),
  
  // Scan statistics
  totalOptionsAnalyzed: int("total_options_analyzed").notNull(),
  callsAnalyzed: int("calls_analyzed").notNull(),
  putsAnalyzed: int("puts_analyzed").notNull(),
  
  callsPassedFilters: int("calls_passed_filters").notNull(),
  putsPassedFilters: int("puts_passed_filters").notNull(),
  
  callsAboveThreshold: int("calls_above_threshold").notNull(),
  putsAboveThreshold: int("puts_above_threshold").notNull(),
  
  // Top recommendations from this scan
  topCallScore: int("top_call_score"), // Best call score * 100
  topPutScore: int("top_put_score"), // Best put score * 100
  
  // Market conditions at scan time
  stockPrice: int("stock_price").notNull(), // Stock price in cents
  marketVolatility: int("market_volatility"), // Market IV * 10000
  
  // Scan metadata
  scanTimestamp: timestamp("scan_timestamp").notNull(),
  scanDurationMs: int("scan_duration_ms").notNull(), // Scan duration in milliseconds
  engineVersion: varchar("engine_version", { length: 20 }).notNull(),
  
  // Timestamps
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow().notNull(),
});

export type OptionsAnalysisScan = typeof optionsAnalysisScan.$inferSelect;
export type InsertOptionsAnalysisScan = typeof optionsAnalysisScan.$inferInsert;
