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
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="technical">Technical</TabsTrigger>
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
