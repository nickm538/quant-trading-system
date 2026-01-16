import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, AlertCircle, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export function OptionsFlowScanner() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  
  const optionsFlowMutation = trpc.scanners.optionsFlow.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      optionsFlowMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'VERY_BULLISH': return 'bg-green-500';
      case 'BULLISH': return 'bg-green-400';
      case 'NEUTRAL': return 'bg-gray-400';
      case 'BEARISH': return 'bg-red-400';
      case 'VERY_BEARISH': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };

  const formatNumber = (num: number) => {
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num?.toLocaleString() || '0';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <BarChart3 className="h-6 w-6 text-blue-500" />
        <h2 className="text-2xl font-bold">Options Flow Scanner</h2>
        <Badge variant="outline" className="ml-2">Bear ↔ Bull</Badge>
      </div>

      <p className="text-muted-foreground">
        Analyze options flow to detect bullish vs bearish pressure. 
        Tracks put/call ratios, unusual activity, and buy/sell classification.
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
        <Button onClick={handleScan} disabled={optionsFlowMutation.isPending || !symbol.trim()}>
          {optionsFlowMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Search className="h-4 w-4 mr-2" />
          )}
          Scan Options Flow
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
          {/* Pressure Bar - Main Visual */}
          <Card className="md:col-span-2 lg:col-span-3">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                {result.sentiment === 'BULLISH' || result.sentiment === 'VERY_BULLISH' ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : result.sentiment === 'BEARISH' || result.sentiment === 'VERY_BEARISH' ? (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                ) : (
                  <BarChart3 className="h-5 w-5 text-gray-500" />
                )}
                Options Pressure Gauge
              </CardTitle>
              <CardDescription>Bear ← 0 | 50 Neutral | 100 → Bull</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Visual Pressure Bar */}
                <div className="relative">
                  <div className="w-full h-8 rounded-full bg-gradient-to-r from-red-500 via-gray-300 to-green-500" />
                  <div 
                    className="absolute top-0 w-4 h-8 bg-white border-2 border-gray-800 rounded-full transform -translate-x-1/2 shadow-lg"
                    style={{ left: `${result.pressure_bar || 50}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-red-500 font-bold">🐻 BEARISH</span>
                  <span className="text-gray-500">NEUTRAL</span>
                  <span className="text-green-500 font-bold">BULLISH 🐂</span>
                </div>
                <div className="flex items-center justify-center gap-4">
                  <Badge className={`${getSentimentColor(result.sentiment)} text-white text-lg px-4 py-2`}>
                    {result.sentiment}
                  </Badge>
                  <span className="text-2xl font-bold">
                    Net Pressure: {result.net_pressure > 0 ? '+' : ''}{result.net_pressure?.toFixed(1)}%
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Volume Breakdown */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Volume Breakdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-green-500">📈 Call Volume:</span>
                  <span className="font-bold">{formatNumber(result.call_volume || 0)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-red-500">📉 Put Volume:</span>
                  <span className="font-bold">{formatNumber(result.put_volume || 0)}</span>
                </div>
                <div className="flex justify-between items-center border-t pt-2">
                  <span>Total Volume:</span>
                  <span className="font-bold">{formatNumber(result.total_volume || 0)}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Put/Call Ratios */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Put/Call Ratios</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span>P/C (Volume):</span>
                  <Badge variant={result.pcr_volume > 1 ? "destructive" : result.pcr_volume < 0.7 ? "default" : "secondary"}>
                    {result.pcr_volume?.toFixed(2) || 'N/A'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>P/C (Open Interest):</span>
                  <Badge variant={result.pcr_oi > 1 ? "destructive" : result.pcr_oi < 0.7 ? "default" : "secondary"}>
                    {result.pcr_oi?.toFixed(2) || 'N/A'}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  P/C &lt; 0.7 = Bullish | P/C &gt; 1.0 = Bearish
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Buy/Sell Classification */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Buy/Sell Flow</CardTitle>
              <CardDescription>Bid/Ask analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-green-500">Buy Volume:</span>
                  <span className="font-bold">{result.buy_pct?.toFixed(1) || 0}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-red-500">Sell Volume:</span>
                  <span className="font-bold">{result.sell_pct?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className="h-3 rounded-full bg-green-500"
                    style={{ width: `${result.buy_pct || 50}%` }}
                  />
                </div>
                <Badge className={getSentimentColor(result.flow_sentiment) + ' text-white'}>
                  {result.flow_sentiment || 'NEUTRAL'}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Unusual Activity */}
          {result.has_unusual_activity && (
            <Card className="md:col-span-2 lg:col-span-3 border-yellow-500 border-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  ⚠️ Unusual Options Activity Detected
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.unusual_calls?.length > 0 && (
                    <div>
                      <h4 className="font-bold text-green-500 mb-2">Unusual Calls</h4>
                      {result.unusual_calls.map((call: any, idx: number) => (
                        <div key={idx} className="text-sm flex justify-between border-b py-1">
                          <span>${call.strike} strike</span>
                          <span>Vol/OI: {call.vol_oi_ratio}x</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {result.unusual_puts?.length > 0 && (
                    <div>
                      <h4 className="font-bold text-red-500 mb-2">Unusual Puts</h4>
                      {result.unusual_puts.map((put: any, idx: number) => (
                        <div key={idx} className="text-sm flex justify-between border-b py-1">
                          <span>${put.strike} strike</span>
                          <span>Vol/OI: {put.vol_oi_ratio}x</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Education */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">📚 Understanding Options Flow</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>Options flow</strong> tracks the buying and selling of options contracts to gauge market sentiment.
          </p>
          <p>
            <strong>Put/Call Ratio:</strong> Below 0.7 = bullish (more calls). Above 1.0 = bearish (more puts).
          </p>
          <p>
            <strong>Unusual Activity:</strong> When volume exceeds 2x open interest, it signals potential institutional positioning.
          </p>
          <p>
            <strong>Note:</strong> Data is 15-minute delayed from Yahoo Finance.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
