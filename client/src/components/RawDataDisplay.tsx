import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface RawDataDisplayProps {
  analysis: any;
}

export function RawDataDisplay({ analysis }: RawDataDisplayProps) {
  const technical = analysis.technical_analysis || {};
  const stochastic = analysis.stochastic_analysis || {};
  const garch = stochastic.garch_analysis || {};
  const monteCarlo = stochastic.monte_carlo || {};
  const positionSizing = analysis.position_sizing || {};
  const recommendation = analysis.recommendation || {};

  return (
    <Card className="bg-card/50 border-border/50">
      <CardHeader>
        <CardTitle className="text-primary">📊 Detailed Raw Data & Calculations</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="technical" className="w-full">
          <TabsList className="grid w-full grid-cols-4 lg:grid-cols-8">
            <TabsTrigger value="technical">Technical</TabsTrigger>
            <TabsTrigger value="advanced">R2/Pivot/Fib</TabsTrigger>
            <TabsTrigger value="candlestick">Candlesticks</TabsTrigger>
            <TabsTrigger value="fundamentals">Cash Flow</TabsTrigger>
            <TabsTrigger value="garch">GARCH</TabsTrigger>
            <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
            <TabsTrigger value="position">Position</TabsTrigger>
            <TabsTrigger value="all">All Data</TabsTrigger>
          </TabsList>

          <TabsContent value="technical" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Technical Score" value={`${technical.technical_score?.toFixed(2) || 'N/A'}/100`} />
              <DataItem label="Momentum Score" value={`${technical.momentum_score?.toFixed(2) || 'N/A'}/100`} />
              <DataItem label="Trend Score" value={`${technical.trend_score?.toFixed(2) || 'N/A'}/100`} />
              <DataItem label="Volatility Score" value={`${technical.volatility_score?.toFixed(2) || 'N/A'}/100`} />
              <DataItem label="RSI (14)" value={technical.rsi?.toFixed(2) || 'N/A'} />
              <DataItem label="MACD" value={technical.macd?.toFixed(4) || 'N/A'} />
              <DataItem label="ADX" value={technical.adx?.toFixed(2) || 'N/A'} />
              <DataItem label="Current Volatility" value={`${(technical.current_volatility * 100)?.toFixed(2) || 'N/A'}%`} />
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Calculation Formula:</h4>
              <code className="text-xs block">
                Technical Score = (Momentum + Trend + Volatility) / 3<br/>
                Momentum = 100 - |RSI - 50| (optimal at RSI=50)<br/>
                Trend = Based on SMA crossovers and ADX strength<br/>
                Volatility = 100 - (Historical Vol * 100)
              </code>
            </div>
          </TabsContent>

          {/* Advanced Technicals - R2, Pivot Points, Fibonacci */}
          <TabsContent value="advanced" className="space-y-4">
            <div className="mb-4 p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <h4 className="font-semibold mb-2 text-indigo-900 dark:text-indigo-100">R2 Score, Pivot Points & Fibonacci Levels</h4>
              <p className="text-sm text-indigo-800 dark:text-indigo-200">
                R2 measures trend predictability (0-1). Pivot points and Fibonacci levels identify key support/resistance zones.
              </p>
            </div>
            
            {analysis.advanced_technicals ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="R2 Score" value={analysis.advanced_technicals.r2_score?.toFixed(4) || 'N/A'} />
                  <DataItem label="Predictability" value={analysis.advanced_technicals.predictability_rating || 'N/A'} />
                  <DataItem label="Trend Direction" value={analysis.advanced_technicals.trend_direction || 'N/A'} />
                  <DataItem label="Slope" value={analysis.advanced_technicals.slope?.toFixed(4) || 'N/A'} />
                </div>
                
                {analysis.advanced_technicals.pivot_points && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Standard Pivot Points</h5>
                    <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
                      <DataItem label="R3" value={`$${analysis.advanced_technicals.pivot_points.standard?.r3?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="R2" value={`$${analysis.advanced_technicals.pivot_points.standard?.r2?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="R1" value={`$${analysis.advanced_technicals.pivot_points.standard?.r1?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="Pivot" value={`$${analysis.advanced_technicals.pivot_points.standard?.pivot?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="S1" value={`$${analysis.advanced_technicals.pivot_points.standard?.s1?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="S2" value={`$${analysis.advanced_technicals.pivot_points.standard?.s2?.toFixed(2) || 'N/A'}`} />
                      <DataItem label="S3" value={`$${analysis.advanced_technicals.pivot_points.standard?.s3?.toFixed(2) || 'N/A'}`} />
                    </div>
                  </div>
                )}
                
                {analysis.advanced_technicals.fibonacci && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Fibonacci Levels</h5>
                    <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
                      {Object.entries(analysis.advanced_technicals.fibonacci.retracement || {}).map(([level, price]: [string, any]) => (
                        <DataItem key={level} label={`${level}`} value={`$${price?.toFixed(2) || 'N/A'}`} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Advanced technicals data not available. Run a full analysis to see R2, Pivot, and Fibonacci levels.
              </div>
            )}
          </TabsContent>

          {/* Candlestick Patterns */}
          <TabsContent value="candlestick" className="space-y-4">
            <div className="mb-4 p-4 bg-amber-50 dark:bg-amber-950 rounded-lg border border-amber-200 dark:border-amber-800">
              <h4 className="font-semibold mb-2 text-amber-900 dark:text-amber-100">Candlestick Pattern Detection</h4>
              <p className="text-sm text-amber-800 dark:text-amber-200">
                Expert-level pattern recognition including Doji, Hammer, Engulfing, Morning/Evening Star, Ichimoku Cloud, and more.
              </p>
            </div>
            
            {analysis.candlestick_patterns ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="Overall Bias" value={analysis.candlestick_patterns.overall_bias || 'N/A'} />
                  <DataItem label="Confidence" value={`${(analysis.candlestick_patterns.confidence * 100)?.toFixed(1) || 'N/A'}%`} />
                  <DataItem label="Patterns Found" value={analysis.candlestick_patterns.patterns_detected?.length || 0} />
                  <DataItem label="Pattern Strength" value={analysis.candlestick_patterns.pattern_strength || 'N/A'} />
                </div>
                
                {analysis.candlestick_patterns.patterns_detected?.length > 0 && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Detected Patterns</h5>
                    <div className="space-y-2">
                      {analysis.candlestick_patterns.patterns_detected.map((pattern: any, idx: number) => (
                        <div key={idx} className={`p-3 rounded-lg border ${pattern.signal === 'bullish' ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800' : pattern.signal === 'bearish' ? 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800' : 'bg-muted/20 border-border'}`}>
                          <div className="flex justify-between items-center">
                            <span className="font-medium">{pattern.name}</span>
                            <span className={`text-sm ${pattern.signal === 'bullish' ? 'text-green-600' : pattern.signal === 'bearish' ? 'text-red-600' : 'text-muted-foreground'}`}>
                              {pattern.signal?.toUpperCase()} ({pattern.reliability || 'N/A'})
                            </span>
                          </div>
                          {pattern.description && <p className="text-xs text-muted-foreground mt-1">{pattern.description}</p>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {analysis.candlestick_patterns.ichimoku && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Ichimoku Cloud</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Trend" value={analysis.candlestick_patterns.ichimoku.trend || 'N/A'} />
                      <DataItem label="TK Cross" value={analysis.candlestick_patterns.ichimoku.tk_cross || 'N/A'} />
                      <DataItem label="Cloud Color" value={analysis.candlestick_patterns.ichimoku.cloud_color || 'N/A'} />
                      <DataItem label="Price vs Cloud" value={analysis.candlestick_patterns.ichimoku.price_vs_cloud || 'N/A'} />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Candlestick pattern data not available. Run a full analysis to see pattern detection.
              </div>
            )}
          </TabsContent>

          {/* Enhanced Fundamentals - Cash Flow */}
          <TabsContent value="fundamentals" className="space-y-4">
            <div className="mb-4 p-4 bg-emerald-50 dark:bg-emerald-950 rounded-lg border border-emerald-200 dark:border-emerald-800">
              <h4 className="font-semibold mb-2 text-emerald-900 dark:text-emerald-100">Enhanced Cash Flow & Valuation Metrics</h4>
              <p className="text-sm text-emerald-800 dark:text-emerald-200">
                Deep fundamental analysis including PE, PEG, GARP scoring, FCF, EBITDA/EV, Free Float, and Liquidity metrics.
              </p>
            </div>
            
            {analysis.enhanced_fundamentals ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="P/E Ratio" value={analysis.enhanced_fundamentals.valuation?.pe_ratio?.toFixed(2) || 'N/A'} />
                  <DataItem label="Forward P/E" value={analysis.enhanced_fundamentals.valuation?.forward_pe?.toFixed(2) || 'N/A'} />
                  <DataItem label="PEG Ratio" value={analysis.enhanced_fundamentals.valuation?.peg_ratio?.toFixed(2) || 'N/A'} />
                  <DataItem label="EV/EBITDA" value={analysis.enhanced_fundamentals.valuation?.ev_ebitda?.toFixed(2) || 'N/A'} />
                </div>
                
                {analysis.enhanced_fundamentals.cash_flow && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Cash Flow Metrics</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Free Cash Flow" value={`$${(analysis.enhanced_fundamentals.cash_flow.free_cash_flow / 1e9)?.toFixed(2) || 'N/A'}B`} />
                      <DataItem label="FCF Yield" value={`${(analysis.enhanced_fundamentals.cash_flow.fcf_yield * 100)?.toFixed(2) || 'N/A'}%`} />
                      <DataItem label="FCF Margin" value={`${(analysis.enhanced_fundamentals.cash_flow.fcf_margin * 100)?.toFixed(2) || 'N/A'}%`} />
                      <DataItem label="Op Cash Flow" value={`$${(analysis.enhanced_fundamentals.cash_flow.operating_cash_flow / 1e9)?.toFixed(2) || 'N/A'}B`} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.garp_analysis && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">GARP Analysis (Growth at Reasonable Price)</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="GARP Score" value={`${analysis.enhanced_fundamentals.garp_analysis.garp_score?.toFixed(1) || 'N/A'}/100`} />
                      <DataItem label="GARP Signal" value={analysis.enhanced_fundamentals.garp_analysis.garp_signal || 'N/A'} />
                      <DataItem label="Growth Rate" value={`${(analysis.enhanced_fundamentals.garp_analysis.earnings_growth * 100)?.toFixed(1) || 'N/A'}%`} />
                      <DataItem label="Value Score" value={`${analysis.enhanced_fundamentals.garp_analysis.value_score?.toFixed(1) || 'N/A'}/100`} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.liquidity && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Liquidity & Float</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Free Float %" value={`${(analysis.enhanced_fundamentals.liquidity.free_float_pct * 100)?.toFixed(1) || 'N/A'}%`} />
                      <DataItem label="Liquidity Score" value={`${analysis.enhanced_fundamentals.liquidity.liquidity_score?.toFixed(1) || 'N/A'}/100`} />
                      <DataItem label="Avg Volume" value={`${(analysis.enhanced_fundamentals.liquidity.avg_volume / 1e6)?.toFixed(2) || 'N/A'}M`} />
                      <DataItem label="Current Ratio" value={analysis.enhanced_fundamentals.liquidity.current_ratio?.toFixed(2) || 'N/A'} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.financial_health && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Financial Health</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Altman Z-Score" value={analysis.enhanced_fundamentals.financial_health.altman_z_score?.toFixed(2) || 'N/A'} />
                      <DataItem label="Z-Score Rating" value={analysis.enhanced_fundamentals.financial_health.z_score_rating || 'N/A'} />
                      <DataItem label="Debt/Equity" value={analysis.enhanced_fundamentals.financial_health.debt_to_equity?.toFixed(2) || 'N/A'} />
                      <DataItem label="Interest Coverage" value={analysis.enhanced_fundamentals.financial_health.interest_coverage?.toFixed(2) || 'N/A'} />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Enhanced fundamentals data not available. Run a full analysis to see cash flow and valuation metrics.
              </div>
            )}
          </TabsContent>

          <TabsContent value="garch" className="space-y-4">
            <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-100">📊 What is GARCH?</h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models time-varying volatility. 
                Unlike simple historical volatility, GARCH adapts to market conditions - volatility clusters during turbulent periods and calms during stable periods.
              </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <DataItem label="Model" value="GARCH(1,1)" />
                <p className="text-xs text-muted-foreground">Standard model: 1 lag for shocks, 1 lag for volatility</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Distribution" value="Student-t" />
                <p className="text-xs text-muted-foreground">Captures extreme moves better than normal distribution</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Fat-Tail DF" value={garch.fat_tail_df?.toFixed(2) || 'N/A'} />
                <p className="text-xs text-muted-foreground">Lower = fatter tails (more extreme events). Typical: 3-10</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="AIC" value={garch.aic?.toFixed(2) || 'N/A'} />
                <p className="text-xs text-muted-foreground">Model quality: Lower is better. Compare across stocks</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="BIC" value={garch.bic?.toFixed(2) || 'N/A'} />
                <p className="text-xs text-muted-foreground">Similar to AIC but penalizes complexity more</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Current Vol" value={`${(garch.current_volatility * 100)?.toFixed(2) || 'N/A'}%`} />
                <p className="text-xs text-muted-foreground">Annualized volatility forecast for next period</p>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">GARCH(1,1) Model:</h4>
              <code className="text-xs block">
                σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁<br/>
                Where:<br/>
                - σ²ₜ = Conditional variance at time t<br/>
                - ω = Long-run variance<br/>
                - α = ARCH parameter (shock impact)<br/>
                - β = GARCH parameter (persistence)<br/>
                - ε²ₜ₋₁ = Previous squared residual<br/>
                <br/>
                Fitted using Maximum Likelihood Estimation (MLE)<br/>
                Student-t distribution captures fat tails (df={garch.fat_tail_df?.toFixed(2) || 'N/A'})
              </code>
            </div>
            
            <div className="mt-4 p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-800">
              <h4 className="font-semibold mb-2 text-green-900 dark:text-green-100">💡 How to Use GARCH</h4>
              <ul className="text-sm text-green-800 dark:text-green-200 space-y-2">
                <li><strong>Current Vol:</strong> Use this for options pricing and risk assessment. Higher vol = higher option premiums.</li>
                <li><strong>Fat-Tail DF:</strong> Lower values (3-5) mean more extreme moves are likely. Be cautious with leverage.</li>
                <li><strong>AIC/BIC:</strong> Compare these across different stocks. Lower scores = model fits better = more reliable forecasts.</li>
                <li><strong>Volatility Clustering:</strong> If recent volatility is high, GARCH predicts it will stay high short-term (and vice versa).</li>
              </ul>
            </div>
          </TabsContent>

          <TabsContent value="montecarlo" className="space-y-4">
            <div className="mb-4 p-4 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-800">
              <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-100">🎲 What is Monte Carlo?</h4>
              <p className="text-sm text-purple-800 dark:text-purple-200">
                Monte Carlo simulation runs 20,000 possible future price paths using GARCH volatility and fat-tail distributions. 
                This shows the range of outcomes and helps quantify risk better than simple predictions.
              </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <DataItem label="Simulations" value="20,000" />
                <p className="text-xs text-muted-foreground">Number of price paths simulated for statistical confidence</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Time Steps" value="30 days" />
                <p className="text-xs text-muted-foreground">Forecast horizon (1 month ahead)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Expected Price" value={`$${stochastic.expected_price?.toFixed(2) || 'N/A'}`} />
                <p className="text-xs text-muted-foreground">Average price across all 20,000 simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Expected Return" value={`${(stochastic.expected_return * 100)?.toFixed(2) || 'N/A'}%`} />
                <p className="text-xs text-muted-foreground">Average return over 30 days (annualized)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="VaR (95%)" value={`${(stochastic.var_95 * 100)?.toFixed(2) || 'N/A'}%`} />
                <p className="text-xs text-muted-foreground">Worst-case loss in 95% of scenarios (5% chance worse)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="CVaR (95%)" value={`${(stochastic.cvar_95 * 100)?.toFixed(2) || 'N/A'}%`} />
                <p className="text-xs text-muted-foreground">Average loss when VaR is exceeded (tail risk)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Max Drawdown" value={`${(stochastic.max_drawdown * 100)?.toFixed(2) || 'N/A'}%`} />
                <p className="text-xs text-muted-foreground">Largest peak-to-trough decline across simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Lower" value={`$${stochastic.confidence_interval_lower?.toFixed(2) || 'N/A'}`} />
                <p className="text-xs text-muted-foreground">Lower bound: 95% chance price stays above this</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Upper" value={`$${stochastic.confidence_interval_upper?.toFixed(2) || 'N/A'}`} />
                <p className="text-xs text-muted-foreground">Upper bound: 95% chance price stays below this</p>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Monte Carlo GBM with Fat Tails:</h4>
              <code className="text-xs block">
                dS/S = μ·dt + σ·dW<br/>
                Log-step: log(Sₜ₊₁/Sₜ) = (μ - 0.5σ²)·dt + σ·√dt·Zₜ<br/>
                <br/>
                Where:<br/>
                - Zₜ ~ Student-t(df=5) for fat tails<br/>
                - μ = Annualized drift (historical mean return)<br/>
                - σ = GARCH-forecasted volatility<br/>
                - Antithetic variates for variance reduction<br/>
                <br/>
                VaR(95%) = 5th percentile of return distribution<br/>
                CVaR(95%) = Mean of returns below VaR<br/>
                Max Drawdown = max((Running Max - Price) / Running Max)
              </code>
            </div>
            
            <div className="mt-4 p-4 bg-orange-50 dark:bg-orange-950 rounded-lg border border-orange-200 dark:border-orange-800">
              <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-100">🎯 How to Use Monte Carlo</h4>
              <ul className="text-sm text-orange-800 dark:text-orange-200 space-y-2">
                <li><strong>Expected Price:</strong> Your best estimate for 30 days out. Don't treat as guaranteed - it's an average.</li>
                <li><strong>VaR (95%):</strong> Your downside risk. If VaR is -15%, you have 5% chance of losing more than 15%.</li>
                <li><strong>CVaR (95%):</strong> Tail risk - how bad it gets when VaR is breached. Critical for position sizing.</li>
                <li><strong>95% CI Range:</strong> Price likely stays within this range. Wider range = more uncertainty = higher risk.</li>
                <li><strong>Max Drawdown:</strong> Worst decline you might see. Use this to set stop losses and manage emotions.</li>
              </ul>
            </div>
          </TabsContent>

          <TabsContent value="position" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Bankroll" value={`$${(analysis.bankroll || 1000).toFixed(2)}`} />
              <DataItem label="Position Size" value={`${((positionSizing.position_size_pct || 0) * 100)?.toFixed(2)}%`} />
              <DataItem label="Shares" value={positionSizing.shares?.toFixed(0) || 'N/A'} />
              <DataItem label="Position Value" value={`$${positionSizing.position_value?.toFixed(2) || 'N/A'}`} />
              <DataItem label="Dollar Risk" value={`$${positionSizing.dollar_risk?.toFixed(2) || 'N/A'}`} />
              <DataItem label="Dollar Reward" value={`$${positionSizing.dollar_reward?.toFixed(2) || 'N/A'}`} />
              <DataItem label="Risk/Reward" value={`1:${positionSizing.risk_reward_ratio?.toFixed(2) || 'N/A'}`} />
              <DataItem label="Risk % of Bankroll" value={`${(positionSizing.risk_pct_of_bankroll * 100)?.toFixed(2) || 'N/A'}%`} />
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Kelly Criterion Position Sizing:</h4>
              <code className="text-xs block">
                Kelly Fraction = (p·b - q) / b<br/>
                Where:<br/>
                - p = Win probability (confidence / 100)<br/>
                - q = Loss probability (1 - p)<br/>
                - b = Win/loss ratio (target gain / stop loss)<br/>
                <br/>
                Position Size = min(Half-Kelly, Risk-Based Max)<br/>
                Risk-Based Max = Max Risk % / Stop Distance %<br/>
                <br/>
                Constraints:<br/>
                - Max 2% risk per trade (moderate risk)<br/>
                - Half-Kelly for safety (0.5 × Kelly)<br/>
                - Position size between 1% and 20% of bankroll<br/>
                <br/>
                Shares = floor(Position Value / Current Price)<br/>
                Dollar Risk = Shares × |Current Price - Stop Loss|
              </code>
            </div>
          </TabsContent>

          <TabsContent value="all" className="space-y-4">
            <div className="p-4 bg-muted/20 rounded-lg max-h-[600px] overflow-auto">
              <pre className="text-xs">
                {JSON.stringify(analysis, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function DataItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="p-3 bg-muted/30 rounded-lg">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}
