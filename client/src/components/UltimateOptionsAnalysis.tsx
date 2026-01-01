import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { 
  Loader2, 
  Search, 
  Zap, 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Target,
  Shield,
  Brain,
  BarChart3,
  Activity,
  DollarSign,
  Clock,
  Flame,
  Award,
  LineChart,
  PieChart
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { TTMSqueezeIndicator } from "./TTMSqueezeIndicator";

// Score color helper
const getScoreColor = (score: number) => {
  if (score >= 80) return "text-green-500";
  if (score >= 60) return "text-blue-500";
  if (score >= 40) return "text-yellow-500";
  return "text-red-500";
};

const getScoreBg = (score: number) => {
  if (score >= 80) return "bg-green-500/10 border-green-500/30";
  if (score >= 60) return "bg-blue-500/10 border-blue-500/30";
  if (score >= 40) return "bg-yellow-500/10 border-yellow-500/30";
  return "bg-red-500/10 border-red-500/30";
};

const getRatingBadge = (rating: string) => {
  switch (rating) {
    case "EXCEPTIONAL": return "bg-gradient-to-r from-green-500 to-emerald-500";
    case "EXCELLENT": return "bg-gradient-to-r from-blue-500 to-cyan-500";
    case "GOOD": return "bg-gradient-to-r from-yellow-500 to-orange-500";
    case "NEUTRAL": return "bg-gradient-to-r from-gray-500 to-slate-500";
    default: return "bg-gradient-to-r from-red-500 to-pink-500";
  }
};

// 12-Factor Score Display Component
function FactorScoreCard({ name, score, icon: Icon, description }: { 
  name: string; 
  score: number; 
  icon: any;
  description: string;
}) {
  return (
    <div className={`p-3 rounded-lg border ${getScoreBg(score)}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">{name}</span>
        </div>
        <span className={`text-lg font-bold ${getScoreColor(score)}`}>{score.toFixed(0)}</span>
      </div>
      <Progress value={score} className="h-1.5" />
      <p className="text-xs text-muted-foreground mt-1">{description}</p>
    </div>
  );
}

// Greeks Display Component
function GreeksDisplay({ greeks }: { greeks: any }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <div className="p-3 bg-muted/30 rounded-lg text-center">
        <div className="text-xs text-muted-foreground">Delta (Δ)</div>
        <div className="text-xl font-bold text-blue-500">{greeks.delta?.toFixed(4)}</div>
        <div className="text-xs text-muted-foreground">Price sensitivity</div>
      </div>
      <div className="p-3 bg-muted/30 rounded-lg text-center">
        <div className="text-xs text-muted-foreground">Gamma (Γ)</div>
        <div className="text-xl font-bold text-purple-500">{greeks.gamma?.toFixed(4)}</div>
        <div className="text-xs text-muted-foreground">Delta acceleration</div>
      </div>
      <div className="p-3 bg-muted/30 rounded-lg text-center">
        <div className="text-xs text-muted-foreground">Theta (Θ)</div>
        <div className="text-xl font-bold text-red-500">{greeks.theta?.toFixed(4)}</div>
        <div className="text-xs text-muted-foreground">Time decay/day</div>
      </div>
      <div className="p-3 bg-muted/30 rounded-lg text-center">
        <div className="text-xs text-muted-foreground">Vega (ν)</div>
        <div className="text-xl font-bold text-green-500">{greeks.vega?.toFixed(4)}</div>
        <div className="text-xs text-muted-foreground">Vol sensitivity</div>
      </div>
      <div className="p-3 bg-muted/30 rounded-lg text-center">
        <div className="text-xs text-muted-foreground">Rho (ρ)</div>
        <div className="text-xl font-bold text-orange-500">{greeks.rho?.toFixed(4)}</div>
        <div className="text-xs text-muted-foreground">Rate sensitivity</div>
      </div>
    </div>
  );
}

// Single Option Card Component
function OptionCard({ option, rank }: { option: any; rank: number }) {
  const isCall = option.option_type === "CALL";
  
  return (
    <div className="p-5 border-2 border-border rounded-xl bg-gradient-to-br from-muted/20 to-background hover:from-muted/30 transition-all">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className={`text-3xl font-bold ${rank === 1 ? 'text-yellow-500' : rank === 2 ? 'text-gray-400' : rank === 3 ? 'text-amber-600' : 'text-muted-foreground'}`}>
            #{rank}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <Badge className={isCall ? "bg-green-500" : "bg-red-500"}>
                {option.option_type}
              </Badge>
              <span className="text-2xl font-bold">${option.strike}</span>
            </div>
            <div className="text-sm text-muted-foreground">
              Expires: {option.expiration} ({option.dte} days)
            </div>
          </div>
        </div>
        <div className="text-right">
          <Badge className={`${getRatingBadge(option.rating)} text-white px-3 py-1`}>
            {option.rating}
          </Badge>
          <div className={`text-3xl font-bold mt-1 ${getScoreColor(option.final_score)}`}>
            {option.final_score?.toFixed(1)}
          </div>
          <div className="text-xs text-muted-foreground">Total Score</div>
        </div>
      </div>

      {/* Price Info */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4 p-3 bg-muted/20 rounded-lg">
        <div>
          <div className="text-xs text-muted-foreground">Premium</div>
          <div className="text-lg font-bold">${option.last_price?.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Bid / Ask</div>
          <div className="text-lg font-bold">
            <span className="text-green-500">${option.bid?.toFixed(2)}</span>
            {" / "}
            <span className="text-red-500">${option.ask?.toFixed(2)}</span>
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Spread</div>
          <div className={`text-lg font-bold ${option.key_metrics?.spread_pct < 5 ? 'text-green-500' : option.key_metrics?.spread_pct < 10 ? 'text-yellow-500' : 'text-red-500'}`}>
            {option.key_metrics?.spread_pct?.toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Volume / OI</div>
          <div className="text-lg font-bold">
            {option.key_metrics?.volume?.toLocaleString()} / {option.key_metrics?.open_interest?.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Greeks */}
      <div className="mb-4">
        <div className="text-sm font-semibold mb-2 flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          Greeks
        </div>
        <GreeksDisplay greeks={option.key_metrics} />
      </div>

      {/* 12-Factor Scores */}
      <div className="mb-4">
        <div className="text-sm font-semibold mb-2 flex items-center gap-2">
          <PieChart className="h-4 w-4" />
          12-Factor Institutional Scoring
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <FactorScoreCard name="Volatility" score={option.scores?.volatility || 0} icon={Activity} description="IV analysis" />
          <FactorScoreCard name="Greeks" score={option.scores?.greeks || 0} icon={BarChart3} description="Positioning" />
          <FactorScoreCard name="Technical" score={option.scores?.technical || 0} icon={LineChart} description="Momentum" />
          <FactorScoreCard name="Liquidity" score={option.scores?.liquidity || 0} icon={DollarSign} description="Spread/Volume" />
          <FactorScoreCard name="Event Risk" score={option.scores?.event_risk || 0} icon={AlertCircle} description="Earnings" />
          <FactorScoreCard name="Sentiment" score={option.scores?.sentiment || 0} icon={TrendingUp} description="News/Analyst" />
          <FactorScoreCard name="Flow" score={option.scores?.flow || 0} icon={Flame} description="Unusual activity" />
          <FactorScoreCard name="Expected Value" score={option.scores?.expected_value || 0} icon={Target} description="Probability" />
          <FactorScoreCard name="TTM Squeeze" score={option.scores?.ttm_squeeze || 0} icon={Zap} description="Vol compression" />
          <FactorScoreCard name="AI Prediction" score={option.scores?.ai_prediction || 0} icon={Brain} description="ML ensemble" />
          <FactorScoreCard name="Legendary" score={option.scores?.legendary_validation || 0} icon={Award} description="Strategy validation" />
          <FactorScoreCard name="Market Regime" score={option.scores?.market_regime || 0} icon={Shield} description="VIX/Conditions" />
        </div>
      </div>

      {/* Risk Management */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-muted/20 rounded-lg mb-4">
        <div>
          <div className="text-xs text-muted-foreground">IV</div>
          <div className="text-lg font-bold">{option.key_metrics?.iv?.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">IV Rank</div>
          <div className={`text-lg font-bold ${option.iv_rank < 30 ? 'text-green-500' : option.iv_rank > 70 ? 'text-red-500' : 'text-yellow-500'}`}>
            {option.iv_rank?.toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Kelly %</div>
          <div className="text-lg font-bold text-blue-500">{option.risk_management?.kelly_pct?.toFixed(2)}%</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Max Position</div>
          <div className="text-lg font-bold text-purple-500">{option.risk_management?.max_position_size_pct?.toFixed(2)}%</div>
        </div>
      </div>

      {/* AI Reasoning */}
      {option.ai_reasoning && (
        <div className="p-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg border border-blue-500/20 mb-3">
          <div className="flex items-center gap-2 mb-1">
            <Brain className="h-4 w-4 text-blue-500" />
            <span className="text-xs font-semibold text-blue-500">AI/QUANTUM INSIGHT</span>
          </div>
          <p className="text-sm text-muted-foreground italic">"{option.ai_reasoning}"</p>
        </div>
      )}

      {/* Legendary Trader Reasoning */}
      {option.legendary_reasoning && (
        <div className="p-3 bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-lg border border-yellow-500/20">
          <div className="flex items-center gap-2 mb-1">
            <Award className="h-4 w-4 text-yellow-500" />
            <span className="text-xs font-semibold text-yellow-500">LEGENDARY TRADER INSIGHT</span>
          </div>
          <p className="text-sm text-muted-foreground italic">"{option.legendary_reasoning}"</p>
        </div>
      )}
    </div>
  );
}

// Market Scan Result Card
function ScanResultCard({ opp, rank }: { opp: any; rank: number }) {
  return (
    <div className="p-5 border-2 border-border rounded-xl bg-gradient-to-br from-muted/20 to-background hover:from-muted/30 transition-all">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className={`text-3xl font-bold ${rank === 1 ? 'text-yellow-500' : rank === 2 ? 'text-gray-400' : rank === 3 ? 'text-amber-600' : 'text-muted-foreground'}`}>
            #{rank}
          </div>
          <div>
            <div className="text-2xl font-bold">{opp.symbol}</div>
            <div className="text-sm text-muted-foreground">{opp.sector}</div>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-3xl font-bold ${getScoreColor(opp.total_score)}`}>
            {opp.total_score?.toFixed(1)}
          </div>
          <div className="text-xs text-muted-foreground">Total Score</div>
        </div>
      </div>

      {/* Option Details */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-3 p-3 bg-muted/20 rounded-lg">
        <div>
          <div className="text-xs text-muted-foreground">Stock Price</div>
          <div className="text-lg font-bold">${opp.current_price?.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Strike</div>
          <div className="text-lg font-bold">${opp.strike?.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Premium</div>
          <div className="text-lg font-bold">${opp.option_price?.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Expiration</div>
          <div className="text-sm font-bold">{opp.expiration}</div>
          <div className="text-xs text-muted-foreground">{opp.days_to_expiry} days</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Breakeven</div>
          <div className="text-lg font-bold">${opp.breakeven?.toFixed(2)}</div>
        </div>
      </div>

      {/* Greeks Row */}
      <div className="grid grid-cols-4 gap-3 mb-3">
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Delta</div>
          <div className="text-lg font-bold text-blue-500">{opp.delta?.toFixed(3)}</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Gamma</div>
          <div className="text-lg font-bold text-purple-500">{opp.gamma?.toFixed(4)}</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Theta</div>
          <div className="text-lg font-bold text-red-500">{opp.theta?.toFixed(3)}</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Vega</div>
          <div className="text-lg font-bold text-green-500">{opp.vega?.toFixed(3)}</div>
        </div>
      </div>

      {/* Volatility & Risk */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-3">
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">IV</div>
          <div className="font-bold">{opp.implied_vol?.toFixed(1)}%</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">HV</div>
          <div className="font-bold">{opp.hist_vol?.toFixed(1)}%</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">IV Rank</div>
          <div className={`font-bold ${opp.iv_rank < 30 ? 'text-green-500' : opp.iv_rank > 70 ? 'text-red-500' : 'text-yellow-500'}`}>
            {opp.iv_rank?.toFixed(1)}%
          </div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Kelly %</div>
          <div className="font-bold text-blue-500">{(opp.kelly_fraction * 100)?.toFixed(2)}%</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Max Loss</div>
          <div className="font-bold text-red-500">${opp.max_loss?.toFixed(0)}</div>
        </div>
        <div className="p-2 bg-muted/30 rounded text-center">
          <div className="text-xs text-muted-foreground">Momentum</div>
          <div className={`font-bold ${opp.momentum > 0 ? 'text-green-500' : 'text-red-500'}`}>
            {opp.momentum?.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* TTM Squeeze - Only show if there's meaningful data */}
      {opp.squeeze_active !== undefined && opp.squeeze_momentum !== 0 && (
        <div className="mb-3">
          <TTMSqueezeIndicator
            squeeze_active={opp.squeeze_active}
            squeeze_bars={opp.squeeze_bars || 0}
            squeeze_momentum={opp.squeeze_momentum || 0}
            squeeze_signal={opp.squeeze_signal || 'none'}
            compact={true}
          />
        </div>
      )}

      {/* AI Reasoning */}
      {opp.ai_reasoning && (
        <div className="p-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg border border-blue-500/20 mb-3">
          <div className="flex items-center gap-2 mb-1">
            <Brain className="h-4 w-4 text-blue-500" />
            <span className="text-xs font-semibold text-blue-500">AI/QUANTUM INSIGHT</span>
          </div>
          <p className="text-sm text-muted-foreground italic">"{opp.ai_reasoning}"</p>
        </div>
      )}

      {/* Legendary Trader Reasoning */}
      {opp.legendary_reasoning && (
        <div className="p-3 bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-lg border border-yellow-500/20">
          <div className="flex items-center gap-2 mb-1">
            <Award className="h-4 w-4 text-yellow-500" />
            <span className="text-xs font-semibold text-yellow-500">LEGENDARY TRADER INSIGHT</span>
          </div>
          <p className="text-sm text-muted-foreground italic">"{opp.legendary_reasoning}"</p>
        </div>
      )}
    </div>
  );
}

// Main Component
export function UltimateOptionsAnalysis() {
  const [symbol, setSymbol] = useState("");
  const [mode, setMode] = useState<"scan" | "analyze">("scan");
  const [scanResults, setScanResults] = useState<any>(null);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  // tRPC mutations
  const scanMutation = trpc.trading.scanUltimateOptions.useMutation({
    onSuccess: (data) => {
      setScanResults(data);
    },
  });

  const analyzeMutation = trpc.trading.analyzeUltimateOptions.useMutation({
    onSuccess: (data) => {
      setAnalysisResults(data);
    },
  });

  const handleScan = () => {
    setScanResults(null);
    scanMutation.mutate({ max_results: 10, option_type: "both" });
  };

  const handleAnalyze = () => {
    if (!symbol) return;
    setAnalysisResults(null);
    analyzeMutation.mutate({ symbol: symbol.toUpperCase(), option_type: "both" });
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card className="border-2 border-primary/20 bg-gradient-to-br from-primary/5 to-background">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-xl bg-primary/10">
              <Zap className="h-8 w-8 text-primary" />
            </div>
            <div>
              <CardTitle className="text-2xl">Ultimate Options Intelligence Engine</CardTitle>
              <CardDescription className="text-base">
                12-Factor Institutional Scoring • AI/ML Ensemble • Legendary Trader Validation • Full Greeks Suite
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Alert className="mb-4">
            <Brain className="h-4 w-4" />
            <AlertDescription>
              <strong>Proprietary Analysis:</strong> Combines institutional-grade 12-factor scoring, AI/ML predictions 
              (momentum, volatility regime, mean reversion), and validation against legendary trader strategies 
              (Buffett, Soros, Simons, Dalio, PTJ). Scans 200+ stocks across all sectors.
            </AlertDescription>
          </Alert>

          {/* Mode Selector */}
          <Tabs value={mode} onValueChange={(v) => setMode(v as "scan" | "analyze")} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="scan" className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                Market Scan
              </TabsTrigger>
              <TabsTrigger value="analyze" className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                Symbol Analysis
              </TabsTrigger>
            </TabsList>

            {/* Market Scan Tab */}
            <TabsContent value="scan" className="space-y-4">
              <div className="flex flex-col gap-4">
                <p className="text-sm text-muted-foreground">
                  Scan the entire market (200+ stocks) to find the best options opportunities using 
                  3-tier filtering: Quick screen → Medium analysis → Deep institutional scoring.
                </p>
                <Button 
                  onClick={handleScan} 
                  disabled={scanMutation.isPending}
                  className="w-full"
                  size="lg"
                >
                  {scanMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Scanning Market (5-10 min)...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-5 w-5" />
                      Start Market Scan
                    </>
                  )}
                </Button>
              </div>
            </TabsContent>

            {/* Symbol Analysis Tab */}
            <TabsContent value="analyze" className="space-y-4">
              <div className="flex gap-4">
                <Input
                  placeholder="Enter symbol (e.g., AAPL, TSLA, NVDA)"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                  className="text-lg"
                />
                <Button 
                  onClick={handleAnalyze} 
                  disabled={!symbol || analyzeMutation.isPending}
                  size="lg"
                  className="whitespace-nowrap"
                >
                  {analyzeMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Target className="mr-2 h-5 w-5" />
                      Analyze Options
                    </>
                  )}
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Error Display */}
      {(scanMutation.isError || analyzeMutation.isError) && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {scanMutation.error?.message || analyzeMutation.error?.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Scan Results */}
      {scanResults && scanResults.success && (
        <div className="space-y-6">
          {/* Scan Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Market Scan Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Stocks Scanned</div>
                  <div className="text-2xl font-bold">{scanResults.total_scanned}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Tier 1 Passed</div>
                  <div className="text-2xl font-bold">{scanResults.tier1_passed}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Tier 2 Passed</div>
                  <div className="text-2xl font-bold">{scanResults.tier2_passed}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Opportunities</div>
                  <div className="text-2xl font-bold text-green-500">{scanResults.final_opportunities}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Scan Time</div>
                  <div className="text-2xl font-bold">{scanResults.scan_duration_seconds}s</div>
                </div>
              </div>

              {/* Methodology */}
              <div className="p-3 bg-muted/20 rounded-lg">
                <div className="text-sm font-semibold mb-2">Methodology</div>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">{scanResults.methodology?.scoring_factors} Factors</Badge>
                  {scanResults.methodology?.ai_models?.map((model: string) => (
                    <Badge key={model} variant="secondary">{model}</Badge>
                  ))}
                  {scanResults.methodology?.legendary_strategies?.map((strat: string) => (
                    <Badge key={strat} variant="outline" className="border-yellow-500/50">{strat}</Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Opportunities */}
          <Card>
            <CardHeader>
              <CardTitle>Top Options Opportunities</CardTitle>
              <CardDescription>
                Ranked by 12-factor institutional score
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {scanResults.opportunities?.map((opp: any, idx: number) => (
                <ScanResultCard key={idx} opp={opp} rank={idx + 1} />
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Analysis Results */}
      {analysisResults && analysisResults.success && (
        <div className="space-y-6">
          {/* Analysis Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                {analysisResults.symbol} Options Analysis
              </CardTitle>
              <CardDescription>
                Current Price: ${analysisResults.current_price?.toFixed(2)} • 
                Historical Volatility: {analysisResults.historical_volatility?.toFixed(1)}% •
                Analysis Time: {analysisResults.analysis_duration_seconds}s
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Calls Analyzed</div>
                  <div className="text-2xl font-bold">{analysisResults.total_calls_analyzed}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Puts Analyzed</div>
                  <div className="text-2xl font-bold">{analysisResults.total_puts_analyzed}</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">ATM IV</div>
                  <div className="text-2xl font-bold">{analysisResults.volatility_surface?.atm_iv_pct?.toFixed(1)}%</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">IV Rank</div>
                  <div className={`text-2xl font-bold ${analysisResults.iv_rank < 30 ? 'text-green-500' : analysisResults.iv_rank > 70 ? 'text-red-500' : 'text-yellow-500'}`}>
                    {analysisResults.iv_rank?.toFixed(1)}%
                  </div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Expected Move</div>
                  <div className="text-2xl font-bold">{analysisResults.expected_move?.expected_move_pct?.toFixed(1)}%</div>
                </div>
                <div className="p-4 bg-muted/30 rounded-lg text-center">
                  <div className="text-sm text-muted-foreground">Market Regime</div>
                  <div className={`text-xl font-bold ${analysisResults.market_regime?.regime === 'bullish' ? 'text-green-500' : analysisResults.market_regime?.regime === 'bearish' ? 'text-red-500' : 'text-yellow-500'}`}>
                    {analysisResults.market_regime?.regime?.toUpperCase()}
                  </div>
                </div>
              </div>

              {/* AI Prediction */}
              {analysisResults.ai_prediction && (
                <div className="p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg mb-6">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="h-5 w-5 text-purple-500" />
                    <span className="font-semibold">AI/ML Ensemble Prediction</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <Badge className={`text-lg px-4 py-1 ${analysisResults.ai_prediction.direction === 'BULLISH' ? 'bg-green-500' : analysisResults.ai_prediction.direction === 'BEARISH' ? 'bg-red-500' : 'bg-yellow-500'}`}>
                      {analysisResults.ai_prediction.direction}
                    </Badge>
                    <span className="text-lg">
                      Confidence: <strong>{analysisResults.ai_prediction.confidence?.toFixed(1)}%</strong>
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {analysisResults.ai_prediction.signals?.map((signal: string) => (
                      <Badge key={signal} variant="outline">{signal}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Legendary Validation */}
              {analysisResults.legendary_validation && (
                <div className="p-4 bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-lg mb-6">
                  <div className="flex items-center gap-2 mb-3">
                    <Award className="h-5 w-5 text-yellow-500" />
                    <span className="font-semibold">Legendary Trader Strategy Validation</span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(analysisResults.legendary_validation).map(([trader, data]: [string, any]) => (
                      <div key={trader} className={`p-3 rounded-lg border ${data.aligned ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'}`}>
                        <div className="font-semibold capitalize">{trader}</div>
                        <div className="text-xs text-muted-foreground">{data.strategy}</div>
                        <Badge className={`mt-1 ${data.aligned ? 'bg-green-500' : 'bg-red-500'}`}>
                          {data.aligned ? '✓ Aligned' : '✗ Not Aligned'}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TTM Squeeze */}
              {analysisResults.ttm_squeeze && (
                <div className="mb-6">
                  <TTMSqueezeIndicator
                    squeeze_active={analysisResults.ttm_squeeze.active}
                    squeeze_bars={analysisResults.ttm_squeeze.bars}
                    squeeze_momentum={analysisResults.ttm_squeeze.momentum}
                    squeeze_signal={analysisResults.ttm_squeeze.signal}
                    compact={false}
                  />
                </div>
              )}

              {/* Earnings Info */}
              {analysisResults.earnings_data?.next_earnings_date && (
                <Alert className="mb-6">
                  <Clock className="h-4 w-4" />
                  <AlertDescription>
                    <strong>Earnings Alert:</strong> Next earnings on {analysisResults.earnings_data.next_earnings_date}
                    {analysisResults.earnings_data.days_to_earnings !== null && (
                      <> ({analysisResults.earnings_data.days_to_earnings} days away)</>
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Top Calls */}
          {analysisResults.top_calls?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-green-500" />
                  Top Call Options
                </CardTitle>
                <CardDescription>
                  Best bullish opportunities ranked by 12-factor score
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysisResults.top_calls.map((option: any, idx: number) => (
                  <OptionCard key={idx} option={option} rank={idx + 1} />
                ))}
              </CardContent>
            </Card>
          )}

          {/* Top Puts */}
          {analysisResults.top_puts?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingDown className="h-5 w-5 text-red-500" />
                  Top Put Options
                </CardTitle>
                <CardDescription>
                  Best bearish opportunities ranked by 12-factor score
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {analysisResults.top_puts.map((option: any, idx: number) => (
                  <OptionCard key={idx} option={option} rank={idx + 1} />
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* No Results Message */}
      {scanResults && !scanResults.success && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{scanResults.error}</AlertDescription>
        </Alert>
      )}

      {analysisResults && !analysisResults.success && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{analysisResults.error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
