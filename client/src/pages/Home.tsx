import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, BarChart3, Search, AlertCircle, Brain, Zap } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { APP_TITLE } from "@/const";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { MonteCarloChart } from "@/components/MonteCarloChart";
import { TechnicalChart } from "@/components/TechnicalChart";
import { RiskDistributionChart } from "@/components/RiskDistributionChart";
import { RawDataDisplay } from "@/components/RawDataDisplay";
import { ExpertReasoningDisplay } from "@/components/ExpertReasoningDisplay";
import { QuantMLAnalysis } from "@/components/QuantMLAnalysis";
import { GreeksHeatmap } from "@/components/GreeksHeatmap";
import { InstitutionalOptionsAnalysis } from "@/components/InstitutionalOptionsAnalysis";

export default function Home() {
  const [symbol, setSymbol] = useState("");
  const [bankroll, setBankroll] = useState(1000);
  const [analysis, setAnalysis] = useState<any>(null);
  const [optionsAnalysis, setOptionsAnalysis] = useState<any>(null);
  const [institutionalOptions, setInstitutionalOptions] = useState<any>(null);
  const [greeksHeatmapData, setGreeksHeatmapData] = useState<any>(null);
  const [marketScan, setMarketScan] = useState<any>(null);
  const [trainingOutput, setTrainingOutput] = useState<string>("");
  const [mlPrediction, setMlPrediction] = useState<any>(null);

  const analyzeMutation = trpc.trading.analyzeStock.useMutation({
    onSuccess: (data) => {
      setAnalysis(data);
      // Also fetch ML prediction for this stock
      mlPredictionMutation.mutate({ symbol: data.symbol, horizon_days: 30 });
    },
  });

  const mlPredictionMutation = trpc.trading.getMLPrediction.useMutation({
    onSuccess: (data) => {
      setMlPrediction(data);
    },
  });

  const greeksHeatmapMutation = trpc.trading.getGreeksHeatmap.useMutation({
    onSuccess: (data) => {
      if (data.success) {
        setGreeksHeatmapData(data);
      }
    },
  });

  const optionsMutation = trpc.trading.analyzeOptions.useMutation({
    onSuccess: (data) => {
      setOptionsAnalysis(data);
    },
  });

  const institutionalOptionsMutation = trpc.trading.analyzeInstitutionalOptions.useMutation({
    onSuccess: (data) => {
      setInstitutionalOptions(data);
    },
  });

  const scanMutation = trpc.trading.scanMarket.useMutation({
    onSuccess: (data) => {
      setMarketScan(data);
    },
  });

  const trainMutation = trpc.ml.trainModels.useMutation({
    onSuccess: (data) => {
      setTrainingOutput(data.output || "");
    },
  });

  const handleAnalyze = () => {
    if (!symbol) return;
    setAnalysis(null);
    analyzeMutation.mutate({
      symbol: symbol.toUpperCase(),
      monte_carlo_sims: 20000,
      forecast_days: 30,
      bankroll: bankroll,
    });
  };

  const handleAnalyzeOptions = () => {
    if (!symbol) return;
    setOptionsAnalysis(null);
    optionsMutation.mutate({
      symbol: symbol.toUpperCase(),
      min_delta: 0.3,
      max_delta: 0.6,
      min_days: 7,
    });
  };

  const handleAnalyzeInstitutionalOptions = () => {
    if (!symbol) return;
    setInstitutionalOptions(null);
    institutionalOptionsMutation.mutate({
      symbol: symbol.toUpperCase(),
    });
  };

  const handleScanMarket = () => {
    setMarketScan(null);
    scanMutation.mutate({
      top_n: 20,
    });
  };

  const getSignalColor = (signal: string) => {
    if (signal === 'BUY') return 'text-green-500';
    if (signal === 'SELL') return 'text-red-500';
    return 'text-yellow-500';
  };

  const getSignalBadgeVariant = (signal: string) => {
    if (signal === 'BUY') return 'default';
    if (signal === 'SELL') return 'destructive';
    return 'secondary';
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="border-b border-border/40 bg-gradient-to-b from-background to-muted/20">
        <div className="container py-12">
          <div className="flex flex-col items-center text-center space-y-4">
            <Badge variant="outline" className="px-4 py-1">
              Institutional-Grade Quantitative Analysis
            </Badge>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              {APP_TITLE}
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl">
              Advanced stochastic modeling with 20,000 Monte Carlo simulations, GARCH volatility, 
              50+ technical indicators, and real-time market data
            </p>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <span>✓ Fat-Tail Distributions</span>
              <span>✓ Options Greeks</span>
              <span>✓ Position Sizing</span>
              <span>✓ Market Scanner</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container py-8">
        <Tabs defaultValue="stock" className="w-full">
          <TabsList className="grid w-full grid-cols-6 mb-8">
            <TabsTrigger value="stock" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Stock Analysis
            </TabsTrigger>
            <TabsTrigger value="options" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Options Analyzer
            </TabsTrigger>
            <TabsTrigger value="institutional" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Institutional Options
            </TabsTrigger>
            <TabsTrigger value="scanner" className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Market Scanner
            </TabsTrigger>
            <TabsTrigger value="ml" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              ML/Quantum
            </TabsTrigger>
            <TabsTrigger value="train" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Train Models
            </TabsTrigger>
          </TabsList>

          {/* Stock Analysis Tab */}
          <TabsContent value="stock" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Comprehensive Stock Analysis</CardTitle>
                <CardDescription>
                  Full institutional-grade analysis with Monte Carlo simulations, GARCH volatility, and 50+ technical indicators
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="md:col-span-2">
                    <label className="text-sm font-medium mb-2 block">Stock Symbol</label>
                    <Input
                      placeholder="Enter symbol (e.g., AAPL)"
                      value={symbol}
                      onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                      onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                      className="uppercase"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Bankroll ($)</label>
                    <Input
                      type="number"
                      value={bankroll}
                      onChange={(e) => setBankroll(Number(e.target.value))}
                      min={100}
                      max={1000000}
                    />
                  </div>
                </div>
                <Button 
                  onClick={handleAnalyze} 
                  disabled={!symbol || analyzeMutation.isPending}
                  className="w-full"
                  size="lg"
                >
                  {analyzeMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing (30-60s)...
                    </>
                  ) : (
                    'Analyze Stock'
                  )}
                </Button>
              </CardContent>
            </Card>

            {analyzeMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {analyzeMutation.error.message}
                </AlertDescription>
              </Alert>
            )}

            {analysis && (
              <div className="space-y-6">
                {/* Summary Card */}
                <Card className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-3xl">{analysis.symbol}</CardTitle>
                        <CardDescription className="text-lg mt-1">
                          ${analysis.current_price?.toFixed(2)}
                        </CardDescription>
                      </div>
                      <Badge variant={getSignalBadgeVariant(analysis.signal)} className="text-2xl px-6 py-2">
                        {analysis.signal}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-sm text-muted-foreground">Confidence</div>
                        <div className="text-2xl font-bold">{analysis.confidence?.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-sm text-muted-foreground">Target Price</div>
                        <div className="text-2xl font-bold text-green-500">
                          ${analysis.target_price?.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-muted-foreground">Expected Return</div>
                        <div className="text-2xl font-bold text-green-500">
                          {(analysis.stochastic_analysis?.expected_return * 100)?.toFixed(2)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-muted-foreground">Position Size</div>
                        <div className="text-2xl font-bold">
                          {analysis.position_size} shares
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Expert Reasoning */}
                <ExpertReasoningDisplay analysis={analysis} />

                {/* Position Details */}
                <Card>
                  <CardHeader>
                    <CardTitle>Position Sizing & Risk Management</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Entry Price</div>
                        <div className="text-lg font-semibold">${analysis.current_price?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Stop Loss</div>
                        <div className="text-lg font-semibold text-red-500">${analysis.stop_loss?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Dollar Risk</div>
                        <div className="text-lg font-semibold text-red-500">${analysis.risk_assessment?.potential_loss_pct?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Dollar Reward</div>
                        <div className="text-lg font-semibold text-green-500">${analysis.risk_assessment?.potential_gain_pct?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Risk/Reward</div>
                        <div className="text-lg font-semibold">1:{analysis.risk_assessment?.risk_reward_ratio?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Position Value</div>
                        <div className="text-lg font-semibold">${(analysis.current_price * analysis.position_size)?.toFixed(2)}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">VaR (95%)</div>
                        <div className="text-lg font-semibold text-red-500">{(analysis.stochastic_analysis?.var_95 * 100)?.toFixed(2)}%</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">CVaR (95%)</div>
                        <div className="text-lg font-semibold text-red-500">{(analysis.stochastic_analysis?.cvar_95 * 100)?.toFixed(2)}%</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Visualizations */}
                <Card>
                  <CardHeader>
                    <CardTitle>Monte Carlo Price Forecast</CardTitle>
                    <CardDescription>20,000 simulations with fat-tail distributions and GARCH volatility</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {analysis.stochastic_analysis?.monte_carlo && (
                      <MonteCarloChart
                        meanPath={analysis.stochastic_analysis.monte_carlo.mean_path || []}
                        ci95Lower={analysis.stochastic_analysis.monte_carlo.ci_95_lower_path || []}
                        ci95Upper={analysis.stochastic_analysis.monte_carlo.ci_95_upper_path || []}
                        ci68Lower={analysis.stochastic_analysis.monte_carlo.ci_68_lower_path || []}
                        ci68Upper={analysis.stochastic_analysis.monte_carlo.ci_68_upper_path || []}
                        currentPrice={analysis.current_price || 0}
                        forecastDays={30}
                      />
                    )}
                  </CardContent>
                </Card>

                {analysis.stochastic_analysis?.monte_carlo?.final_returns && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Return Distribution & Tail Risk</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <RiskDistributionChart
                        returns={analysis.stochastic_analysis.monte_carlo.final_returns}
                        var95={analysis.stochastic_analysis?.var_95 || 0}
                        cvar95={analysis.stochastic_analysis?.cvar_95 || 0}
                      />
                    </CardContent>
                  </Card>
                )}

                {/* Raw Data Display */}
                <RawDataDisplay analysis={analysis} />
              </div>
            )}
          </TabsContent>

          {/* Options Analyzer Tab */}
          <TabsContent value="options" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Top 3 Options Chains Analyzer</CardTitle>
                <CardDescription>
                  Find the best call and put options with full Greeks analysis (0.3-0.6 delta, greater than 1 week expiry)
                  <div className="mt-2 flex items-center gap-2 text-xs">
                    <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-500/20">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500 mr-1.5 animate-pulse"></span>
                      Real-Time Data: MarketData.app
                    </Badge>
                    <Badge variant="outline" className="bg-blue-500/10 text-blue-600 border-blue-500/20">
                      Cached 30min • 100% Real Bid/Ask/Greeks/IV
                    </Badge>
                  </div>
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Stock Symbol</label>
                  <Input
                    placeholder="Enter symbol (e.g., AAPL)"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyzeOptions()}
                    className="uppercase"
                  />
                </div>
                <Button 
                  onClick={handleAnalyzeOptions} 
                  disabled={!symbol || optionsMutation.isPending}
                  className="w-full"
                  size="lg"
                >
                  {optionsMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing Options Chain...
                    </>
                  ) : (
                    'Analyze Options'
                  )}
                </Button>
              </CardContent>
            </Card>

            {optionsMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {optionsMutation.error.message}
                </AlertDescription>
              </Alert>
            )}

            {optionsAnalysis && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Top 3 Call Options</CardTitle>
                    <CardDescription>Current Price: ${optionsAnalysis.current_price?.toFixed(2) || 'N/A'}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {optionsAnalysis.top_calls?.map((call: any, idx: number) => (
                      <div key={idx} className="p-4 border border-border rounded-lg bg-muted/20">
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <div className="text-2xl font-bold">${call.strike} Strike</div>
                            <div className="text-sm text-muted-foreground">Expires: {call.expiration} ({call.days_to_expiry} days)</div>
                          </div>
                          <Badge variant="default" className="text-lg px-4 py-1">
                            ${call.last_price?.toFixed(2)}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Bid</div>
                            <div className="font-semibold text-green-600">${call.bid?.toFixed(2)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Ask</div>
                            <div className="font-semibold text-red-600">${call.ask?.toFixed(2)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Spread</div>
                            <div className="font-semibold">{call.bid_ask_spread_pct?.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Volume</div>
                            <div className="font-semibold">{call.volume?.toLocaleString()}</div>
                          </div>
                        </div>
                        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Delta</div>
                            <div className="font-semibold">{call.delta?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Gamma</div>
                            <div className="font-semibold">{call.gamma?.toFixed(4)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Theta</div>
                            <div className="font-semibold text-red-500">{call.theta?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Vega</div>
                            <div className="font-semibold">{call.vega?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">IV</div>
                            <div className="font-semibold">{(call.implied_volatility * 100)?.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Open Int</div>
                            <div className="font-semibold">{call.open_interest?.toLocaleString()}</div>
                          </div>
                        </div>
                        <div className="mt-3 pt-3 border-t border-border grid grid-cols-2 gap-3">
                          <div>
                            <div className="text-xs text-muted-foreground">Breakeven</div>
                            <div className="font-semibold">${call.breakeven?.toFixed(2)}</div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">Max Loss</div>
                            <div className="font-semibold text-red-500">${Math.abs(call.max_loss)?.toFixed(2)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Top 3 Put Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {optionsAnalysis.top_puts?.map((put: any, idx: number) => (
                      <div key={idx} className="p-4 border border-border rounded-lg bg-muted/20">
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <div className="text-2xl font-bold">${put.strike} Strike</div>
                            <div className="text-sm text-muted-foreground">Expires: {put.expiration} ({put.days_to_expiry} days)</div>
                          </div>
                          <Badge variant="destructive" className="text-lg px-4 py-1">
                            ${put.last_price?.toFixed(2)}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Bid</div>
                            <div className="font-semibold text-green-600">${put.bid?.toFixed(2)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Ask</div>
                            <div className="font-semibold text-red-600">${put.ask?.toFixed(2)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Spread</div>
                            <div className="font-semibold">{put.bid_ask_spread_pct?.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Volume</div>
                            <div className="font-semibold">{put.volume?.toLocaleString()}</div>
                          </div>
                        </div>
                        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Delta</div>
                            <div className="font-semibold">{put.delta?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Gamma</div>
                            <div className="font-semibold">{put.gamma?.toFixed(4)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Theta</div>
                            <div className="font-semibold text-red-500">{put.theta?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Vega</div>
                            <div className="font-semibold">{put.vega?.toFixed(3)}</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">IV</div>
                            <div className="font-semibold">{(put.implied_volatility * 100)?.toFixed(1)}%</div>
                          </div>
                          <div className="p-2 bg-background rounded">
                            <div className="text-xs text-muted-foreground">Open Int</div>
                            <div className="font-semibold">{put.open_interest?.toLocaleString()}</div>
                          </div>
                        </div>
                        <div className="mt-3 pt-3 border-t border-border grid grid-cols-2 gap-3">
                          <div>
                            <div className="text-xs text-muted-foreground">Breakeven</div>
                            <div className="font-semibold">${put.breakeven?.toFixed(2)}</div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">Max Loss</div>
                            <div className="font-semibold text-red-500">${Math.abs(put.max_loss)?.toFixed(2)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                {/* Greeks Heatmap */}
                <div className="mt-6">
                  <Button
                    onClick={() => greeksHeatmapMutation.mutate({ symbol })}
                    disabled={!symbol || greeksHeatmapMutation.isPending}
                    variant="outline"
                    className="w-full mb-4"
                  >
                    {greeksHeatmapMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Generating Greeks Heatmap...
                      </>
                    ) : (
                      'Show Greeks Heatmap'
                    )}
                  </Button>

                  {greeksHeatmapData && greeksHeatmapData.success && (
                    <GreeksHeatmap data={greeksHeatmapData} />
                  )}

                  {greeksHeatmapData && !greeksHeatmapData.success && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        {greeksHeatmapData.error}
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Institutional Options Tab */}
          <TabsContent value="institutional" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  Institutional-Grade Options Analysis
                </CardTitle>
                <CardDescription>
                  Advanced 8-factor scoring with Black-Scholes Greeks (including Vanna, Charm), Kelly Criterion position sizing, and AI pattern recognition
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Precision over quantity:</strong> This system analyzes 500-1000 options but only returns those scoring ≥60/100.
                    Typical pass rate: 2-5%. Expected analysis time: 10-15 seconds.
                  </AlertDescription>
                </Alert>
                <div className="flex gap-4">
                  <Input
                    placeholder="Enter symbol (e.g., AAPL, TSLA)"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyzeInstitutionalOptions()}
                  />
                  <Button 
                    onClick={handleAnalyzeInstitutionalOptions} 
                    disabled={!symbol || institutionalOptionsMutation.isPending}
                    size="lg"
                    className="whitespace-nowrap"
                  >
                    {institutionalOptionsMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Zap className="mr-2 h-4 w-4" />
                        Analyze Options
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>

            {institutionalOptionsMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {institutionalOptionsMutation.error.message}
                </AlertDescription>
              </Alert>
            )}

            {institutionalOptions && (
              <InstitutionalOptionsAnalysis data={institutionalOptions} />
            )}
          </TabsContent>

          {/* Market Scanner Tab */}
          <TabsContent value="scanner" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>AI-Powered Market Scanner</CardTitle>
                <CardDescription>
                  Deep scan across S&P 500, NASDAQ 100, Dow 30, and Russell 2000 with institutional-grade analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    This scan analyzes hundreds of stocks with full Monte Carlo simulations. 
                    Expected time: 20-30 minutes. Results will show the top 20 opportunities.
                  </AlertDescription>
                </Alert>
                <Button 
                  onClick={handleScanMarket} 
                  disabled={scanMutation.isPending}
                  className="w-full"
                  size="lg"
                >
                  {scanMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Scanning Market (20-30 min)...
                    </>
                  ) : (
                    'Start Market Scan'
                  )}
                </Button>
              </CardContent>
            </Card>

            {scanMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {scanMutation.error.message}
                </AlertDescription>
              </Alert>
            )}

            {marketScan && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Scan Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-sm text-muted-foreground">Symbols Scanned</div>
                        <div className="text-2xl font-bold">{marketScan.symbols_scanned}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-sm text-muted-foreground">Opportunities Found</div>
                        <div className="text-2xl font-bold text-green-500">{marketScan.opportunities?.length || 0}</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-sm text-muted-foreground">Scan Time</div>
                        <div className="text-2xl font-bold">{marketScan.scan_time_minutes?.toFixed(1)} min</div>
                      </div>
                      <div className="p-4 bg-muted/30 rounded-lg">
                        <div className="text-sm text-muted-foreground">Timestamp</div>
                        <div className="text-sm font-semibold">{new Date(marketScan.timestamp).toLocaleString()}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Top Opportunities</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {marketScan.opportunities?.map((opp: any, idx: number) => (
                        <div key={idx} className="p-4 border border-border rounded-lg bg-muted/20 hover:bg-muted/30 transition-colors">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-3">
                              <div className="text-2xl font-bold text-muted-foreground">#{idx + 1}</div>
                              <div>
                                <div className="text-xl font-bold">{opp.symbol}</div>
                                <div className="text-sm text-muted-foreground">${opp.current_price?.toFixed(2)}</div>
                              </div>
                            </div>
                            <Badge variant={getSignalBadgeVariant(opp.signal)} className="text-lg px-4 py-1">
                              {opp.signal}
                            </Badge>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                            <div>
                              <div className="text-xs text-muted-foreground">Opportunity Score</div>
                              <div className="text-lg font-bold text-primary">{opp.opportunity_score?.toFixed(1)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Expected Return</div>
                              <div className="text-lg font-bold text-green-500">{opp.expected_return?.toFixed(2)}%</div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Confidence</div>
                              <div className="text-lg font-bold">{opp.confidence?.toFixed(1)}%</div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Target</div>
                              <div className="text-lg font-bold">${opp.target_price?.toFixed(2)}</div>
                            </div>
                            <div>
                              <div className="text-xs text-muted-foreground">Position</div>
                              <div className="text-lg font-bold">{opp.shares} shares</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* ML/Quantum Predictions Tab */}
          <TabsContent value="ml" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>ML/Quantum Price Predictions</CardTitle>
                <CardDescription>
                  Advanced machine learning predictions using XGBoost and LightGBM ensemble models trained on 50+ technical indicators,
                  stochastic analysis, and historical patterns. Enter a symbol to get AI-powered price forecasts.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex gap-4">
                    <Input
                      placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)"
                      value={symbol}
                      onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                      onKeyPress={(e) => e.key === 'Enter' && handleMLPredict()}
                      className="flex-1"
                    />
                    <Button 
                      onClick={handleMLPredict}
                      disabled={mlPredictionMutation.isPending || !symbol}
                    >
                      {mlPredictionMutation.isPending ? (
                        <>
                          <span className="animate-spin mr-2">⏳</span>
                          Predicting...
                        </>
                      ) : (
                        <>
                          <Brain className="mr-2 h-4 w-4" />
                          Get ML Prediction
                        </>
                      )}
                    </Button>
                  </div>

                  {/* ML Prediction Results */}
                  <QuantMLAnalysis 
                    mlPrediction={mlPrediction} 
                    loading={mlPredictionMutation.isPending}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Train Models Tab */}
          <TabsContent value="train" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Train ML Models</CardTitle>
                <CardDescription>
                  Train XGBoost and LightGBM models on the top 15 selected stocks. Models will be stored in the database for predictions.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-muted/50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Training Dataset:</h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    15 high-quality stocks: BAC, INTC, AAPL, AMZN, GOOG, MSFT, XOM, JPM, DIS, ATVI, EA, CSCO, F, CRM, PFE
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Each model uses 2 years of historical data with walk-forward validation to prevent overfitting.
                  </p>
                </div>

                <Button
                  onClick={() => {
                    setTrainingOutput("");
                    trainMutation.mutate();
                  }}
                  disabled={trainMutation.isPending}
                  size="lg"
                  className="w-full"
                >
                  {trainMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Training Models... (This may take 5-10 minutes)
                    </>
                  ) : (
                    <>
                      <Brain className="mr-2 h-5 w-5" />
                      Start Training
                    </>
                  )}
                </Button>

                {trainMutation.isError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Training failed: {trainMutation.error?.message || "Unknown error"}
                    </AlertDescription>
                  </Alert>
                )}

                {trainMutation.isSuccess && (
                  <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
                    <AlertCircle className="h-4 w-4 text-green-600" />
                    <AlertDescription className="text-green-600 dark:text-green-400">
                      ✅ Training completed successfully! Models saved to database.
                    </AlertDescription>
                  </Alert>
                )}

                {trainingOutput && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Training Details</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-xs bg-muted p-4 rounded-lg overflow-auto max-h-96 whitespace-pre-wrap">
                        {trainingOutput}
                      </pre>
                    </CardContent>
                  </Card>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
