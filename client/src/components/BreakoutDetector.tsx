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
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-green-400';
    if (score >= 40) return 'bg-yellow-500';
    if (score >= 20) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 80) return 'STRONG BREAKOUT';
    if (score >= 60) return 'LIKELY BREAKOUT';
    if (score >= 40) return 'MODERATE';
    if (score >= 20) return 'WEAK';
    return 'NO BREAKOUT';
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
      {result && !result.error && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Overall Score */}
          <Card className="md:col-span-2 lg:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Zap className="h-5 w-5 text-yellow-500" />
                Breakout Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-4">
                  <div className={`text-4xl font-bold rounded-full w-20 h-20 flex items-center justify-center text-white ${getScoreColor(result.breakout_score || 0)}`}>
                    {result.breakout_score?.toFixed(0) || 0}
                  </div>
                  <div>
                    <Badge className={`${getScoreColor(result.breakout_score || 0)} text-white`}>
                      {getScoreLabel(result.breakout_score || 0)}
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      {result.direction === 'bullish' ? '🐂 Bullish Bias' : 
                       result.direction === 'bearish' ? '🐻 Bearish Bias' : 
                       '↔️ Neutral'}
                    </p>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full ${getScoreColor(result.breakout_score || 0)}`}
                    style={{ width: `${result.breakout_score || 0}%` }}
                  />
                </div>
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
                  <span>Bullish Signals:</span>
                  <Badge className="bg-green-500 text-white">{result.bullish_signals || 0}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Bearish Signals:</span>
                  <Badge className="bg-red-500 text-white">{result.bearish_signals || 0}</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Neutral Signals:</span>
                  <Badge variant="secondary">{result.neutral_signals || 0}</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Patterns */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Key Patterns</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span>NR7 Pattern:</span>
                  <Badge variant={result.nr7_detected ? "default" : "outline"}>
                    {result.nr7_detected ? '✅ DETECTED' : 'Not Found'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>NR4 Pattern:</span>
                  <Badge variant={result.nr4_detected ? "default" : "outline"}>
                    {result.nr4_detected ? '✅ DETECTED' : 'Not Found'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>TTM Squeeze:</span>
                  <Badge variant={result.ttm_squeeze_on ? "destructive" : "outline"}>
                    {result.ttm_squeeze_on ? '🔴 ON' : '🟢 OFF'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Volume Analysis */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Volume Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Relative Volume:</span>
                  <span className="font-bold">{result.relative_volume?.toFixed(2) || 1}x</span>
                </div>
                <div className="flex justify-between">
                  <span>OBV Trend:</span>
                  <Badge variant={result.obv_bullish ? "default" : result.obv_bearish ? "destructive" : "secondary"}>
                    {result.obv_bullish ? '📈 Bullish' : result.obv_bearish ? '📉 Bearish' : 'Neutral'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Volume Contraction:</span>
                  <Badge variant={result.volume_contraction ? "default" : "outline"}>
                    {result.volume_contraction ? 'YES' : 'NO'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Support/Resistance */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Support/Resistance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Nearest Resistance:</span>
                  <span className="font-bold text-red-500">${result.resistance?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Current Price:</span>
                  <span className="font-bold">${result.current_price?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Nearest Support:</span>
                  <span className="font-bold text-green-500">${result.support?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Testing Level:</span>
                  <Badge variant={result.testing_resistance ? "destructive" : result.testing_support ? "default" : "outline"}>
                    {result.testing_resistance ? 'Resistance' : result.testing_support ? 'Support' : 'None'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Synergy Bonuses */}
          {result.synergy_bonuses && result.synergy_bonuses.length > 0 && (
            <Card className="border-yellow-500 border-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  ⚡ Synergy Bonuses Active
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {result.synergy_bonuses.map((bonus: string, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 text-sm">
                      <span className="text-yellow-500">★</span>
                      {bonus}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* All Signals */}
          {result.signals && result.signals.length > 0 && (
            <Card className="md:col-span-2 lg:col-span-3">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">All Detected Signals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {result.signals.map((signal: any, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 text-sm p-2 rounded bg-muted">
                      <span className={signal.type === 'bullish' ? 'text-green-500' : signal.type === 'bearish' ? 'text-red-500' : 'text-gray-500'}>
                        {signal.type === 'bullish' ? '🟢' : signal.type === 'bearish' ? '🔴' : '⚪'}
                      </span>
                      {signal.name || signal}
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
            <strong>TTM Squeeze + NR7:</strong> "Coiled Spring" - highest probability breakout setup.
          </p>
          <p>
            <strong>OBV Divergence:</strong> Price vs volume divergence signals accumulation/distribution.
          </p>
          <p>
            <strong>Synergy Bonuses:</strong> Multiple signals aligning increases probability significantly.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
