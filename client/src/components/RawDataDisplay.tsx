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
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Model" value="GARCH(1,1)" />
              <DataItem label="Distribution" value="Student-t" />
              <DataItem label="Fat-Tail DF" value={garch.fat_tail_df?.toFixed(2) || 'N/A'} />
              <DataItem label="AIC" value={garch.aic?.toFixed(2) || 'N/A'} />
              <DataItem label="BIC" value={garch.bic?.toFixed(2) || 'N/A'} />
              <DataItem label="Current Vol" value={`${(garch.current_volatility * 100)?.toFixed(2) || 'N/A'}%`} />
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
          </TabsContent>

          <TabsContent value="montecarlo" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Simulations" value="20,000" />
              <DataItem label="Time Steps" value="30 days" />
              <DataItem label="Expected Price" value={`$${stochastic.expected_price?.toFixed(2) || 'N/A'}`} />
              <DataItem label="Expected Return" value={`${(stochastic.expected_return * 100)?.toFixed(2) || 'N/A'}%`} />
              <DataItem label="VaR (95%)" value={`${(stochastic.var_95 * 100)?.toFixed(2) || 'N/A'}%`} />
              <DataItem label="CVaR (95%)" value={`${(stochastic.cvar_95 * 100)?.toFixed(2) || 'N/A'}%`} />
              <DataItem label="Max Drawdown" value={`${(stochastic.max_drawdown * 100)?.toFixed(2) || 'N/A'}%`} />
              <DataItem label="95% CI Lower" value={`$${stochastic.confidence_interval_lower?.toFixed(2) || 'N/A'}`} />
              <DataItem label="95% CI Upper" value={`$${stochastic.confidence_interval_upper?.toFixed(2) || 'N/A'}`} />
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
