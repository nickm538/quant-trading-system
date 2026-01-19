import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface RawDataDisplayProps {
  analysis: any;
}

// Safe number formatting helper - handles null, undefined, strings, and non-numbers
const safeFixed = (value: any, decimals: number = 2): string => {
  if (value === null || value === undefined) return 'N/A';
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (typeof num !== 'number' || isNaN(num)) return 'N/A';
  return num.toFixed(decimals);
};

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
            <TabsTrigger value="advanced">Pivot/Fib</TabsTrigger>
            <TabsTrigger value="candlestick">Candlesticks</TabsTrigger>
            <TabsTrigger value="fundamentals">Cash Flow</TabsTrigger>
            <TabsTrigger value="garch">GARCH</TabsTrigger>
            <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
            <TabsTrigger value="position">Position</TabsTrigger>
            <TabsTrigger value="all">All Data</TabsTrigger>
          </TabsList>

          <TabsContent value="technical" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Technical Score" value={`${safeFixed(technical.technical_score)}/100`} />
              <DataItem label="Momentum Score" value={`${safeFixed(technical.momentum_score)}/100`} />
              <DataItem label="Trend Score" value={`${safeFixed(technical.trend_score)}/100`} />
              <DataItem label="Volatility Score" value={`${safeFixed(technical.volatility_score)}/100`} />
              <DataItem label="RSI (14)" value={safeFixed(technical.rsi)} />
              <DataItem label="MACD" value={safeFixed(technical.macd, 4)} />
              <DataItem label="ADX" value={safeFixed(technical.adx)} />
              <DataItem label="Current Volatility" value={`${safeFixed(technical.current_volatility * 100)}%`} />
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

          {/* Pivot Points & Fibonacci - R2 moved to ML/Quant page */}
          <TabsContent value="advanced" className="space-y-4">
            <div className="mb-4 p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <h4 className="font-semibold mb-2 text-indigo-900 dark:text-indigo-100">Pivot Points & Fibonacci Levels</h4>
              <p className="text-sm text-indigo-800 dark:text-indigo-200">
                Pivot points and Fibonacci levels identify key support/resistance zones. R2 Score is available on the ML/Quant page.
              </p>
            </div>
            
            {analysis.advanced_technicals ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="Trend Direction" value={analysis.advanced_technicals.r2_analysis?.trend_direction || analysis.advanced_technicals.trend_direction || 'N/A'} />
                  <DataItem label="Trend Strength" value={analysis.advanced_technicals.r2_analysis?.trend_strength || 'N/A'} />
                  <DataItem label="Current Price" value={`$${safeFixed(analysis.advanced_technicals.current_price)}`} />
                  <DataItem label="Nearest Support" value={`$${safeFixed(analysis.advanced_technicals.key_levels?.nearest_support)}`} />
                </div>
                
                {analysis.advanced_technicals.pivot_points && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Standard Pivot Points</h5>
                    <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
                      <DataItem label="R3" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r3)}`} />
                      <DataItem label="R2" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r2)}`} />
                      <DataItem label="R1" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r1)}`} />
                      <DataItem label="Pivot" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.pivot)}`} />
                      <DataItem label="S1" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s1)}`} />
                      <DataItem label="S2" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s2)}`} />
                      <DataItem label="S3" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s3)}`} />
                    </div>
                  </div>
                )}
                
                {analysis.advanced_technicals.fibonacci && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Fibonacci Levels</h5>
                    <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
                      {Object.entries(analysis.advanced_technicals.fibonacci.retracement || {}).map(([level, price]: [string, any]) => (
                        <DataItem key={level} label={`${level}`} value={`$${safeFixed(price)}`} />
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
                  <DataItem label="Confidence" value={`${safeFixed(analysis.candlestick_patterns.confidence * 100, 1)}%`} />
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
                
                {/* Vision AI Chart Analysis */}
                {analysis.candlestick_patterns.vision_ai_analysis && (
                  <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-purple-600 dark:text-purple-400">🤖</span>
                      <h5 className="font-semibold text-purple-900 dark:text-purple-100">Vision AI Chart Analysis</h5>
                      <span className="text-xs bg-purple-100 dark:bg-purple-900 px-2 py-0.5 rounded text-purple-700 dark:text-purple-300">
                        {analysis.candlestick_patterns.vision_ai_analysis.chart_source} + {analysis.candlestick_patterns.vision_ai_analysis.ai_model}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <DataItem label="AI Bias" value={analysis.candlestick_patterns.vision_ai_analysis.overall_bias || 'N/A'} />
                      <DataItem label="Trend Direction" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.direction || 'N/A'} />
                      <DataItem label="Trend Strength" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.strength || 'N/A'} />
                      <DataItem label="Momentum" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.momentum || 'N/A'} />
                    </div>
                    
                    {/* AI Recommendation */}
                    {analysis.candlestick_patterns.vision_ai_analysis.recommendation && (
                      <div className="mb-4 p-3 bg-white dark:bg-gray-800 rounded border">
                        <h6 className="font-medium mb-2">AI Trading Recommendation</h6>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Signal: </span>
                            <span className={`font-semibold ${
                              analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal === 'BUY' ? 'text-green-600' :
                              analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                            }`}>
                              {analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal}
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Confidence: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.recommendation.confidence}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">R/R: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.recommendation.risk_reward}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Stop Loss: </span>
                            <span className="font-semibold">${analysis.candlestick_patterns.vision_ai_analysis.recommendation.stop_loss}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Support/Resistance from AI */}
                    {(analysis.candlestick_patterns.vision_ai_analysis.support_levels?.length > 0 || analysis.candlestick_patterns.vision_ai_analysis.resistance_levels?.length > 0) && (
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <h6 className="font-medium text-green-700 dark:text-green-400 mb-1">AI Support Levels</h6>
                          <div className="text-sm space-y-1">
                            {analysis.candlestick_patterns.vision_ai_analysis.support_levels?.map((level: number, idx: number) => (
                              <div key={idx} className="text-green-600">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h6 className="font-medium text-red-700 dark:text-red-400 mb-1">AI Resistance Levels</h6>
                          <div className="text-sm space-y-1">
                            {analysis.candlestick_patterns.vision_ai_analysis.resistance_levels?.map((level: number, idx: number) => (
                              <div key={idx} className="text-red-600">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* AI Key Observations */}
                    {analysis.candlestick_patterns.vision_ai_analysis.key_observations?.length > 0 && (
                      <div>
                        <h6 className="font-medium mb-2">AI Key Observations</h6>
                        <ul className="text-sm space-y-1">
                          {analysis.candlestick_patterns.vision_ai_analysis.key_observations.map((obs: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-purple-500">•</span>
                              <span>{obs}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* AI Detected Patterns */}
                    {analysis.candlestick_patterns.vision_ai_analysis.candlestick_patterns?.length > 0 && (
                      <div className="mt-4">
                        <h6 className="font-medium mb-2">AI Detected Candlestick Patterns</h6>
                        <div className="space-y-2">
                          {analysis.candlestick_patterns.vision_ai_analysis.candlestick_patterns.map((pattern: any, idx: number) => (
                            <div key={idx} className={`p-2 rounded border ${
                              pattern.signal === 'bullish' ? 'bg-green-50 dark:bg-green-950 border-green-300' :
                              pattern.signal === 'bearish' ? 'bg-red-50 dark:bg-red-950 border-red-300' : 'bg-gray-50 dark:bg-gray-900 border-gray-300'
                            }`}>
                              <div className="flex justify-between">
                                <span className="font-medium">{pattern.name}</span>
                                <span className={`text-xs ${
                                  pattern.signal === 'bullish' ? 'text-green-600' : pattern.signal === 'bearish' ? 'text-red-600' : 'text-gray-600'
                                }`}>
                                  {pattern.signal?.toUpperCase()} • {pattern.reliability}
                                </span>
                              </div>
                              {pattern.description && <p className="text-xs text-muted-foreground mt-1">{pattern.description}</p>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
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
                  <DataItem label="P/E Ratio" value={safeFixed(analysis.enhanced_fundamentals.valuation?.pe_ratio)} />
                  <DataItem label="Forward P/E" value={safeFixed(analysis.enhanced_fundamentals.valuation?.forward_pe)} />
                  <DataItem label="PEG Ratio" value={safeFixed(analysis.enhanced_fundamentals.valuation?.peg_ratio)} />
                  <DataItem label="EV/EBITDA" value={safeFixed(analysis.enhanced_fundamentals.valuation?.ev_ebitda)} />
                </div>
                
                {analysis.enhanced_fundamentals.cash_flow && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Cash Flow Metrics</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Free Cash Flow" value={`$${safeFixed(analysis.enhanced_fundamentals.cash_flow.free_cash_flow / 1e9)}B`} />
                      <DataItem label="FCF Yield" value={`${safeFixed(analysis.enhanced_fundamentals.cash_flow.fcf_yield * 100)}%`} />
                      <DataItem label="FCF Margin" value={`${safeFixed(analysis.enhanced_fundamentals.cash_flow.fcf_margin * 100)}%`} />
                      <DataItem label="Op Cash Flow" value={`$${safeFixed(analysis.enhanced_fundamentals.cash_flow.operating_cash_flow / 1e9)}B`} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.garp_analysis && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">GARP Analysis (Growth at Reasonable Price)</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="GARP Score" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.garp_score, 1)}/100`} />
                      <DataItem label="GARP Signal" value={analysis.enhanced_fundamentals.garp_analysis.garp_signal || 'N/A'} />
                      <DataItem label="Growth Rate" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.earnings_growth * 100, 1)}%`} />
                      <DataItem label="Value Score" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.value_score, 1)}/100`} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.liquidity && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Liquidity & Float</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Free Float %" value={`${safeFixed(analysis.enhanced_fundamentals.liquidity.free_float_pct * 100, 1)}%`} />
                      <DataItem label="Liquidity Score" value={`${safeFixed(analysis.enhanced_fundamentals.liquidity.liquidity_score, 1)}/100`} />
                      <DataItem label="Avg Volume" value={`${safeFixed(analysis.enhanced_fundamentals.liquidity.avg_volume / 1e6)}M`} />
                      <DataItem label="Current Ratio" value={safeFixed(analysis.enhanced_fundamentals.liquidity.current_ratio)} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.financial_health && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Financial Health</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Altman Z-Score" value={safeFixed(analysis.enhanced_fundamentals.financial_health.altman_z_score)} />
                      <DataItem label="Z-Score Rating" value={analysis.enhanced_fundamentals.financial_health.z_score_rating || 'N/A'} />
                      <DataItem label="Debt/Equity" value={safeFixed(analysis.enhanced_fundamentals.financial_health.debt_to_equity)} />
                      <DataItem label="Interest Coverage" value={safeFixed(analysis.enhanced_fundamentals.financial_health.interest_coverage)} />
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
                <DataItem label="Fat-Tail DF" value={safeFixed(garch.fat_tail_df)} />
                <p className="text-xs text-muted-foreground">Lower = fatter tails (more extreme events). Typical: 3-10</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="AIC" value={safeFixed(garch.aic)} />
                <p className="text-xs text-muted-foreground">Model quality: Lower is better. Compare across stocks</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="BIC" value={safeFixed(garch.bic)} />
                <p className="text-xs text-muted-foreground">Similar to AIC but penalizes complexity more</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Current Vol" value={`${safeFixed(garch.current_volatility * 100)}%`} />
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
                Student-t distribution captures fat tails (df={safeFixed(garch.fat_tail_df)})
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
                <DataItem label="Expected Price" value={`$${safeFixed(stochastic.expected_price)}`} />
                <p className="text-xs text-muted-foreground">Average price across all 20,000 simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Expected Return" value={`${safeFixed(stochastic.expected_return * 100)}%`} />
                <p className="text-xs text-muted-foreground">Average return over 30 days (annualized)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="VaR (95%)" value={`${safeFixed(stochastic.var_95 * 100)}%`} />
                <p className="text-xs text-muted-foreground">Worst-case loss in 95% of scenarios (5% chance worse)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="CVaR (95%)" value={`${safeFixed(stochastic.cvar_95 * 100)}%`} />
                <p className="text-xs text-muted-foreground">Average loss when VaR is exceeded (tail risk)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Max Drawdown" value={`${safeFixed(stochastic.max_drawdown * 100)}%`} />
                <p className="text-xs text-muted-foreground">Largest peak-to-trough decline across simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Lower" value={`$${safeFixed(stochastic.confidence_interval_lower)}`} />
                <p className="text-xs text-muted-foreground">Lower bound: 95% chance price stays above this</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Upper" value={`$${safeFixed(stochastic.confidence_interval_upper)}`} />
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
              <DataItem label="Bankroll" value={`$${safeFixed(analysis.bankroll || 1000)}`} />
              <DataItem label="Position Size" value={`${safeFixed((positionSizing.position_size_pct || 0) * 100)}%`} />
              <DataItem label="Shares" value={safeFixed(positionSizing.shares, 0)} />
              <DataItem label="Position Value" value={`$${safeFixed(positionSizing.position_value)}`} />
              <DataItem label="Dollar Risk" value={`$${safeFixed(positionSizing.dollar_risk)}`} />
              <DataItem label="Dollar Reward" value={`$${safeFixed(positionSizing.dollar_reward)}`} />
              <DataItem label="Risk/Reward" value={`1:${safeFixed(positionSizing.risk_reward_ratio)}`} />
              <DataItem label="Risk % of Bankroll" value={`${safeFixed((positionSizing.risk_pct_of_bankroll || 0) * 100)}%`} />
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
