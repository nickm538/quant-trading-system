import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, AlertCircle, Rocket, TrendingUp, TrendingDown, Zap, Globe, Radar } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";

export function BreakoutDetector() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  const [marketResults, setMarketResults] = useState<any>(null);
  const [scanMode, setScanMode] = useState<'single' | 'market'>('single');
  
  const breakoutMutation = trpc.scanners.breakout.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const marketScanMutation = trpc.scanners.marketBreakout.useMutation({
    onSuccess: (data) => {
      setMarketResults(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      breakoutMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const handleMarketScan = () => {
    marketScanMutation.mutate({ maxStocks: 400 });
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

      {/* Scan Mode Tabs */}
      <Tabs value={scanMode} onValueChange={(v) => setScanMode(v as 'single' | 'market')}>
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="single" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Single Stock
          </TabsTrigger>
          <TabsTrigger value="market" className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            Market-Wide Scan
          </TabsTrigger>
        </TabsList>

        {/* Single Stock Tab */}
        <TabsContent value="single" className="space-y-4">
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

          {/* Single Stock Results */}
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

              {/* Key Patterns */}
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

              {/* Volume Analysis */}
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

              {/* Support/Resistance */}
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
                  </div>
                </CardContent>
              </Card>

              {/* Recommendation */}
              <Card className="md:col-span-2 lg:col-span-3">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Recommendation</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{result.recommendation || 'No specific recommendation at this time.'}</p>
                  {result.synergies && result.synergies.length > 0 && (
                    <div className="mt-3">
                      <span className="text-sm font-semibold">Active Synergies:</span>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {result.synergies.map((synergy: string, idx: number) => (
                          <Badge key={idx} variant="outline" className="text-yellow-600 border-yellow-600">
                            ⚡ {synergy}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Market-Wide Scan Tab */}
        <TabsContent value="market" className="space-y-4">
          <Card className="bg-gradient-to-r from-orange-500/10 to-red-500/10 border-orange-500/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Radar className="h-5 w-5 text-orange-500" />
                Market-Wide Breakout Scanner
              </CardTitle>
              <CardDescription>
                Scans 400+ stocks across ALL market caps (mega, large, mid, small, micro) and ETFs 
                to find the best breakout candidates. No bias - every corner of the market is covered.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={handleMarketScan} 
                disabled={marketScanMutation.isPending}
                className="bg-orange-600 hover:bg-orange-700"
                size="lg"
              >
                {marketScanMutation.isPending ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    Scanning Market... (This may take 1-2 minutes)
                  </>
                ) : (
                  <>
                    <Globe className="h-5 w-5 mr-2" />
                    Scan Entire Market for Breakouts
                  </>
                )}
              </Button>
              {marketScanMutation.isPending && (
                <div className="mt-4 space-y-2">
                  <Progress value={33} className="h-2" />
                  <p className="text-sm text-muted-foreground">
                    Analyzing stocks across all market caps and sectors...
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Market Scan Results */}
          {marketResults && marketResults.status === 'success' && (
            <div className="space-y-6">
              {/* Scan Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Radar className="h-5 w-5" />
                    Scan Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-muted rounded-lg">
                      <div className="text-3xl font-bold">{marketResults.stocks_scanned}</div>
                      <div className="text-sm text-muted-foreground">Stocks Scanned</div>
                    </div>
                    <div className="text-center p-4 bg-muted rounded-lg">
                      <div className="text-3xl font-bold text-orange-500">{marketResults.breakout_candidates_found}</div>
                      <div className="text-sm text-muted-foreground">Breakout Candidates</div>
                    </div>
                    <div className="text-center p-4 bg-green-500/10 rounded-lg">
                      <div className="text-3xl font-bold text-green-500">{marketResults.bullish_breakouts?.length || 0}</div>
                      <div className="text-sm text-muted-foreground">Bullish Breakouts</div>
                    </div>
                    <div className="text-center p-4 bg-red-500/10 rounded-lg">
                      <div className="text-3xl font-bold text-red-500">{marketResults.bearish_breakouts?.length || 0}</div>
                      <div className="text-sm text-muted-foreground">Bearish Breakouts</div>
                    </div>
                  </div>
                  <div className="mt-4 text-sm text-muted-foreground">
                    Scan completed in {marketResults.scan_time_seconds}s at {new Date(marketResults.timestamp).toLocaleString()}
                  </div>
                </CardContent>
              </Card>

              {/* Top Bullish Breakouts */}
              {marketResults.bullish_breakouts?.length > 0 && (
                <Card className="border-green-500/30">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-500">
                      <TrendingUp className="h-5 w-5" />
                      Top Bullish Breakout Candidates
                    </CardTitle>
                    <CardDescription>
                      Stocks with high breakout probability and bullish bias - ranked by score
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Rank</th>
                            <th className="text-left p-2">Symbol</th>
                            <th className="text-left p-2">Score</th>
                            <th className="text-left p-2">Probability</th>
                            <th className="text-left p-2">Signals</th>
                            <th className="text-left p-2">Synergies</th>
                            <th className="text-right p-2">Price</th>
                            <th className="text-right p-2">Resistance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResults.bullish_breakouts.slice(0, 15).map((stock: any, idx: number) => (
                            <tr key={stock.symbol} className="border-b hover:bg-muted/50">
                              <td className="p-2 font-bold">{idx + 1}</td>
                              <td className="p-2">
                                <span className="font-mono font-bold">{stock.symbol}</span>
                              </td>
                              <td className="p-2">
                                <Badge className={getScoreColor(stock.breakout_score)}>{stock.breakout_score}</Badge>
                              </td>
                              <td className="p-2">
                                <Badge className={getProbabilityColor(stock.breakout_probability)}>
                                  {stock.breakout_probability}
                                </Badge>
                              </td>
                              <td className="p-2">{stock.signal_count} active</td>
                              <td className="p-2">
                                {stock.synergies?.length > 0 ? (
                                  <Badge variant="outline" className="text-yellow-600">
                                    +{stock.synergy_bonus}
                                  </Badge>
                                ) : '-'}
                              </td>
                              <td className="p-2 text-right">${stock.current_price?.toFixed(2)}</td>
                              <td className="p-2 text-right text-red-500">${stock.nearest_resistance?.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Top Bearish Breakouts */}
              {marketResults.bearish_breakouts?.length > 0 && (
                <Card className="border-red-500/30">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-500">
                      <TrendingDown className="h-5 w-5" />
                      Top Bearish Breakout Candidates
                    </CardTitle>
                    <CardDescription>
                      Stocks with high breakout probability and bearish bias - ranked by score
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Rank</th>
                            <th className="text-left p-2">Symbol</th>
                            <th className="text-left p-2">Score</th>
                            <th className="text-left p-2">Probability</th>
                            <th className="text-left p-2">Signals</th>
                            <th className="text-left p-2">Synergies</th>
                            <th className="text-right p-2">Price</th>
                            <th className="text-right p-2">Support</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResults.bearish_breakouts.slice(0, 15).map((stock: any, idx: number) => (
                            <tr key={stock.symbol} className="border-b hover:bg-muted/50">
                              <td className="p-2 font-bold">{idx + 1}</td>
                              <td className="p-2">
                                <span className="font-mono font-bold">{stock.symbol}</span>
                              </td>
                              <td className="p-2">
                                <Badge className={getScoreColor(stock.breakout_score)}>{stock.breakout_score}</Badge>
                              </td>
                              <td className="p-2">
                                <Badge className={getProbabilityColor(stock.breakout_probability)}>
                                  {stock.breakout_probability}
                                </Badge>
                              </td>
                              <td className="p-2">{stock.signal_count} active</td>
                              <td className="p-2">
                                {stock.synergies?.length > 0 ? (
                                  <Badge variant="outline" className="text-yellow-600">
                                    +{stock.synergy_bonus}
                                  </Badge>
                                ) : '-'}
                              </td>
                              <td className="p-2 text-right">${stock.current_price?.toFixed(2)}</td>
                              <td className="p-2 text-right text-green-500">${stock.nearest_support?.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* No Results Message */}
              {marketResults.breakout_candidates_found === 0 && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    No breakout candidates found in this scan. This can happen during consolidation periods.
                    Try again later or adjust the minimum score threshold.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}

          {/* Market Scan Error */}
          {marketResults?.error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{marketResults.error}</AlertDescription>
            </Alert>
          )}
        </TabsContent>
      </Tabs>

      {/* Education */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">📚 What is a Breakout?</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>Breakout</strong> occurs when price moves beyond a defined support or resistance level 
            with increased volume. This often signals the start of a new trend.
          </p>
          <p>
            <strong>NR7/NR4 Patterns:</strong> Narrowest Range in 7/4 days - indicates volatility compression 
            that often precedes explosive moves.
          </p>
          <p>
            <strong>TTM Squeeze:</strong> When Bollinger Bands move inside Keltner Channels, signaling 
            low volatility before a big move.
          </p>
          <p>
            <strong>OBV Divergence:</strong> When price and On-Balance Volume diverge, it can signal 
            an upcoming reversal or breakout.
          </p>
          <p>
            <strong>Synergy Bonus:</strong> When multiple signals align (e.g., NR7 + Squeeze + Volume), 
            the probability of a successful breakout increases significantly.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
