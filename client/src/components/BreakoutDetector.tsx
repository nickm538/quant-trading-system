import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, AlertCircle, Rocket, TrendingUp, TrendingDown, Zap } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export function BreakoutDetector() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  
  const breakoutMutation = trpc.scanners.breakout.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      breakoutMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 75) return 'bg-green-500';
    if (score >= 55) return 'bg-green-400';
    if (score >= 35) return 'bg-yellow-500';
    if (score >= 15) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getProbabilityColor = (prob: string) => {
    if (prob === 'VERY HIGH') return 'bg-green-500';
    if (prob === 'HIGH') return 'bg-green-400';
    if (prob === 'MODERATE') return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getDirectionIcon = (direction: string) => {
    if (direction === 'BULLISH') return '🐂';
    if (direction === 'BEARISH') return '🐻';
    return '↔️';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Rocket className="h-6 w-6 text-orange-500" />
        <h2 className="text-2xl font-bold">Breakout Detector</h2>
        <Badge variant="outline" className="ml-2">Multi-Signal</Badge>
      </div>

      <p className="text-muted-foreground">
        Institutional-grade breakout detection combining 12+ signals: TTM Squeeze, NR4/NR7 patterns, 
        OBV divergence, support/resistance, volume analysis, and more.
      </p>

      {/* Search */}
      <div className="flex gap-2">
        <Input
          placeholder="Enter symbol (e.g., AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === 'Enter' && handleScan()}
          className="max-w-xs"
        />
        <Button onClick={handleScan} disabled={breakoutMutation.isPending || !symbol.trim()}>
          {breakoutMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Search className="h-4 w-4 mr-2" />
          )}
          Detect Breakout
        </Button>
      </div>

      {/* Error */}
      {result?.error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{result.error}</AlertDescription>
        </Alert>
      )}

      {/* Results */}
      {result && !result.error && result.status === 'success' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Overall Score */}
          <Card className="md:col-span-2 lg:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5 text-yellow-500" />
                Breakout Score
              </CardTitle>
              <CardDescription>
                ${result.current_price?.toFixed(2)} • {result.symbol}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-4">
                  <div className={`text-4xl font-bold rounded-full w-20 h-20 flex items-center justify-center text-white ${getScoreColor(result.breakout_score || 0)}`}>
                    {result.breakout_score || 0}
                  </div>
                  <div>
                    <Badge className={`${getProbabilityColor(result.breakout_probability || '')} text-white`}>
                      {result.breakout_probability || 'N/A'}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      {getDirectionIcon(result.direction_bias)} {result.direction_bias || 'Neutral'} Bias
                    </p>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full ${getScoreColor(result.breakout_score || 0)}`}
                    style={{ width: `${result.breakout_score || 0}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  Base: {result.base_score || 0} | Synergy: +{result.synergy_bonus || 0} | Quality: {result.quality_multiplier || 1}x
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Signal Summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Signal Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Active Signals:</span>
                  <Badge className="bg-blue-500 text-white">{result.signal_count || 0}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Synergies:</span>
                  <Badge className="bg-yellow-500 text-white">{result.synergies?.length || 0}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Probability:</span>
                  <Badge className={getProbabilityColor(result.breakout_probability || '')}>
                    {result.breakout_probability || 'N/A'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Patterns - Using actual nested data */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Key Patterns</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span>NR7 Pattern:</span>
                  <Badge variant={result.nr_patterns?.nr7 ? "default" : "outline"}>
                    {result.nr_patterns?.nr7 ? '✅ DETECTED' : 'Not Found'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>NR4 Pattern:</span>
                  <Badge variant={result.nr_patterns?.nr4 ? "default" : "outline"}>
                    {result.nr_patterns?.nr4 ? '✅ DETECTED' : 'Not Found'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>TTM Squeeze:</span>
                  <Badge variant={result.ttm_squeeze?.squeeze_on ? "destructive" : "outline"}>
                    {result.ttm_squeeze?.squeeze_on ? '🔴 ON' : '🟢 OFF'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Squeeze Fired:</span>
                  <Badge variant={result.ttm_squeeze?.squeeze_fired ? "default" : "outline"}>
                    {result.ttm_squeeze?.squeeze_fired ? '🔥 YES' : 'No'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Volume Analysis - Using actual nested data */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Volume Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Relative Volume:</span>
                  <span className="font-bold">{result.volume?.relative_volume?.toFixed(2) || 'N/A'}x</span>
                </div>
                <div className="flex justify-between">
                  <span>OBV Divergence:</span>
                  <Badge variant={
                    result.obv_divergence?.divergence?.includes('BULLISH') ? "default" : 
                    result.obv_divergence?.divergence?.includes('BEARISH') ? "destructive" : "secondary"
                  }>
                    {result.obv_divergence?.divergence || 'NONE'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Volume Pattern:</span>
                  <Badge variant={result.volume?.volume_contracting ? "default" : "outline"}>
                    {result.volume?.volume_contracting ? 'Contracting ✅' : result.volume?.volume_pattern || 'Normal'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Institutional:</span>
                  <Badge variant={result.volume?.institutional_activity === 'HIGH' ? "default" : "outline"}>
                    {result.volume?.institutional_activity || 'N/A'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Support/Resistance - Using actual field names */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Support/Resistance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Nearest Resistance:</span>
                  <span className="font-bold text-red-500">${result.nearest_resistance?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Current Price:</span>
                  <span className="font-bold">${result.current_price?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Nearest Support:</span>
                  <span className="font-bold text-green-500">${result.nearest_support?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Testing Level:</span>
                  <Badge variant={
                    result.sr_testing?.testing === 'RESISTANCE' ? "destructive" : 
                    result.sr_testing?.testing === 'SUPPORT' ? "default" : "outline"
                  }>
                    {result.sr_testing?.testing || 'None'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Touches:</span>
                  <span>{result.sr_testing?.touches || 0}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* RSI & ADX */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Momentum & Trend</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>RSI:</span>
                  <span className="font-bold">{result.rsi?.rsi?.toFixed(1) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>RSI Divergence:</span>
                  <Badge variant={
                    result.rsi?.divergence === 'BULLISH' ? "default" : 
                    result.rsi?.divergence === 'BEARISH' ? "destructive" : "outline"
                  }>
                    {result.rsi?.divergence || 'NONE'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>ADX:</span>
                  <span className="font-bold">{result.adx?.adx?.toFixed(1) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Trend Strength:</span>
                  <Badge variant={result.adx?.trend_strength === 'STRONG' ? "default" : "outline"}>
                    {result.adx?.trend_strength || 'N/A'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recommendation */}
          <Card className="md:col-span-2 lg:col-span-3 border-blue-500 border-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">📋 Recommendation</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-lg">{result.recommendation || 'No recommendation available'}</p>
            </CardContent>
          </Card>

          {/* Synergy Bonuses */}
          {result.synergies && result.synergies.length > 0 && (
            <Card className="border-yellow-500 border-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  ⚡ Synergy Bonuses Active
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {result.synergies.map((bonus: string, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 text-sm">
                      <span className="text-yellow-500">★</span>
                      {bonus}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* All Active Signals */}
          {result.active_signals && result.active_signals.length > 0 && (
            <Card className="md:col-span-2 lg:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">All Detected Signals ({result.signal_count})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {result.active_signals.map((signal: string, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 text-sm p-2 rounded bg-muted">
                      {signal}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Education */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">📚 Understanding Breakout Detection</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>NR7/NR4:</strong> Narrowest Range in 7/4 days. Volatility compression often precedes breakouts.
          </p>
          <p>
            <strong>TTM Squeeze + NR7:</strong> "Coiled Spring" - highest probability breakout setup (+15 synergy bonus).
          </p>
          <p>
            <strong>OBV Divergence:</strong> Price vs volume divergence signals accumulation/distribution.
          </p>
          <p>
            <strong>Synergy Bonuses:</strong> Multiple signals aligning increases probability significantly.
          </p>
          <p>
            <strong>Score Interpretation:</strong> 75+ = VERY HIGH probability, 55-74 = HIGH, 35-54 = MODERATE, &lt;35 = LOW
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
