import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, AlertCircle, TrendingUp, TrendingDown, Circle, Radar, Globe } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";

export function TTMSqueezeScanner() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  const [marketResults, setMarketResults] = useState<any>(null);
  const [scanMode, setScanMode] = useState<'single' | 'market'>('single');
  
  const squeezeMutation = trpc.scanners.ttmSqueeze.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const marketScanMutation = trpc.scanners.marketTTMSqueeze.useMutation({
    onSuccess: (data) => {
      setMarketResults(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      squeezeMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const handleMarketScan = () => {
    marketScanMutation.mutate({ maxStocks: 100 });
  };

  const getMomentumColor = (momentum: number) => {
    if (momentum > 0) return momentum > (result?.prev_momentum || 0) ? 'bg-green-600' : 'bg-green-400';
    return momentum < (result?.prev_momentum || 0) ? 'bg-red-600' : 'bg-red-400';
  };

  const getIntensityColor = (intensity: string) => {
    switch (intensity) {
      case 'EXTREME': return 'bg-purple-500';
      case 'HIGH': return 'bg-red-500';
      case 'MODERATE': return 'bg-orange-500';
      case 'LOW': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <div className="flex gap-1">
          <Circle className="h-4 w-4 fill-red-500 text-red-500" />
          <Circle className="h-4 w-4 fill-green-500 text-green-500" />
        </div>
        <h2 className="text-2xl font-bold">TTM Squeeze Scanner</h2>
        <Badge variant="outline" className="ml-2">John Carter</Badge>
      </div>

      <p className="text-muted-foreground">
        Detect volatility compression (squeeze) that often precedes explosive breakouts. 
        Combines Bollinger Bands and Keltner Channels with momentum analysis.
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
            <Button onClick={handleScan} disabled={squeezeMutation.isPending || !symbol.trim()}>
              {squeezeMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <Search className="h-4 w-4 mr-2" />
              )}
              Scan Squeeze
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
          {result && !result.error && result.status !== 'error' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Squeeze Status */}
              <Card className="md:col-span-2 lg:col-span-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Circle className={`h-6 w-6 ${result.squeeze_on === true || result.squeeze_on === 'True' ? 'fill-red-500 text-red-500' : 'fill-green-500 text-green-500'}`} />
                    Squeeze Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Badge className={`${result.squeeze_on === true || result.squeeze_on === 'True' ? 'bg-red-500' : 'bg-green-500'} text-white text-lg px-4 py-2`}>
                        {result.squeeze_on === true || result.squeeze_on === 'True' ? '🔴 SQUEEZE ON' : '🟢 SQUEEZE OFF'}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {result.squeeze_on === true || result.squeeze_on === 'True'
                        ? 'Volatility is compressed. Price is coiling for a potential breakout!'
                        : 'Volatility is expanding. The squeeze has fired - momentum is in play.'}
                    </p>
                    {result.squeeze_intensity && (
                      <div className="flex items-center gap-2">
                        <span className="text-sm">Intensity:</span>
                        <Badge variant="outline">{result.squeeze_intensity}</Badge>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Momentum */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    {(result.momentum || 0) >= 0 ? (
                      <TrendingUp className="h-5 w-5 text-green-500" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-red-500" />
                    )}
                    Momentum
                  </CardTitle>
                  <CardDescription>Linear regression histogram</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className={`text-3xl font-bold ${(result.momentum || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {(result.momentum || 0) >= 0 ? '+' : ''}{result.momentum?.toFixed(4) || '0.0000'}
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4 relative">
                      <div 
                        className={`h-4 rounded-full ${getMomentumColor(result.momentum || 0)}`}
                        style={{ 
                          width: `${Math.min(Math.abs(result.momentum || 0) * 100, 50)}%`,
                          marginLeft: (result.momentum || 0) >= 0 ? '50%' : `${50 - Math.min(Math.abs(result.momentum || 0) * 100, 50)}%`
                        }}
                      />
                      <div className="absolute top-0 left-1/2 w-0.5 h-4 bg-gray-600" />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Bearish</span>
                      <span>Neutral</span>
                      <span>Bullish</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Signal */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Trading Signal</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Badge 
                      className={`text-lg px-4 py-2 ${
                        result.signal === 'long' ? 'bg-green-500' :
                        result.signal === 'short' ? 'bg-red-500' :
                        result.signal === 'active' ? 'bg-yellow-500' :
                        'bg-gray-500'
                      } text-white`}
                    >
                      {result.signal?.toUpperCase() || 'NONE'}
                    </Badge>
                    <p className="text-sm text-muted-foreground">
                      {result.signal === 'long' && 'Bullish momentum building - consider long positions'}
                      {result.signal === 'short' && 'Bearish momentum building - consider short positions'}
                      {result.signal === 'active' && 'Squeeze is active - wait for direction confirmation'}
                      {(!result.signal || result.signal === 'none') && 'No clear signal - stay patient'}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Technical Details */}
              <Card className="md:col-span-2 lg:col-span-3">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Technical Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">BB Upper:</span>
                      <div className="font-bold">${result.bb_upper?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">BB Lower:</span>
                      <div className="font-bold">${result.bb_lower?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">KC Upper:</span>
                      <div className="font-bold">${result.kc_upper?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">KC Lower:</span>
                      <div className="font-bold">${result.kc_lower?.toFixed(2) || 'N/A'}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Market-Wide Scan Tab */}
        <TabsContent value="market" className="space-y-4">
          <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-500/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Radar className="h-5 w-5 text-purple-500" />
                Market-Wide TTM Squeeze Scanner
              </CardTitle>
              <CardDescription>
                Scans 400+ stocks across ALL market caps (mega, large, mid, small, micro) and ETFs 
                to find the best squeeze setups. No bias - every corner of the market is covered.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={handleMarketScan} 
                disabled={marketScanMutation.isPending}
                className="bg-purple-600 hover:bg-purple-700"
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
                    Scan Entire Market for Squeezes
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
                      <div className="text-3xl font-bold text-purple-500">{marketResults.squeeze_candidates_found}</div>
                      <div className="text-sm text-muted-foreground">Squeeze Setups Found</div>
                    </div>
                    <div className="text-center p-4 bg-green-500/10 rounded-lg">
                      <div className="text-3xl font-bold text-green-500">{marketResults.bullish_setups?.length || 0}</div>
                      <div className="text-sm text-muted-foreground">Bullish Setups</div>
                    </div>
                    <div className="text-center p-4 bg-red-500/10 rounded-lg">
                      <div className="text-3xl font-bold text-red-500">{marketResults.bearish_setups?.length || 0}</div>
                      <div className="text-sm text-muted-foreground">Bearish Setups</div>
                    </div>
                  </div>
                  <div className="mt-4 text-sm text-muted-foreground">
                    Scan completed in {marketResults.scan_time_seconds}s at {new Date(marketResults.timestamp).toLocaleString()}
                  </div>
                </CardContent>
              </Card>

              {/* Top Bullish Setups */}
              {marketResults.bullish_setups?.length > 0 && (
                <Card className="border-green-500/30">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-500">
                      <TrendingUp className="h-5 w-5" />
                      Top Bullish Squeeze Setups
                    </CardTitle>
                    <CardDescription>
                      Stocks with squeeze ON and bullish momentum - ranked by score
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
                            <th className="text-left p-2">Intensity</th>
                            <th className="text-left p-2">Squeeze Bars</th>
                            <th className="text-left p-2">Momentum</th>
                            <th className="text-left p-2">Signal</th>
                            <th className="text-right p-2">Price</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResults.bullish_setups.slice(0, 15).map((stock: any, idx: number) => (
                            <tr key={stock.symbol} className="border-b hover:bg-muted/50">
                              <td className="p-2 font-bold">{idx + 1}</td>
                              <td className="p-2">
                                <span className="font-mono font-bold">{stock.symbol}</span>
                              </td>
                              <td className="p-2">
                                <Badge className="bg-green-500">{stock.score}</Badge>
                              </td>
                              <td className="p-2">
                                <Badge className={getIntensityColor(stock.squeeze_intensity)}>
                                  {stock.squeeze_intensity}
                                </Badge>
                              </td>
                              <td className="p-2">{stock.squeeze_count} bars</td>
                              <td className="p-2 text-green-500">+{stock.momentum?.toFixed(4)}</td>
                              <td className="p-2">
                                <Badge variant="outline" className="text-green-500 border-green-500">
                                  {stock.signal}
                                </Badge>
                              </td>
                              <td className="p-2 text-right">${stock.current_price?.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Top Bearish Setups */}
              {marketResults.bearish_setups?.length > 0 && (
                <Card className="border-red-500/30">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-500">
                      <TrendingDown className="h-5 w-5" />
                      Top Bearish Squeeze Setups
                    </CardTitle>
                    <CardDescription>
                      Stocks with squeeze ON and bearish momentum - ranked by score
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
                            <th className="text-left p-2">Intensity</th>
                            <th className="text-left p-2">Squeeze Bars</th>
                            <th className="text-left p-2">Momentum</th>
                            <th className="text-left p-2">Signal</th>
                            <th className="text-right p-2">Price</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResults.bearish_setups.slice(0, 15).map((stock: any, idx: number) => (
                            <tr key={stock.symbol} className="border-b hover:bg-muted/50">
                              <td className="p-2 font-bold">{idx + 1}</td>
                              <td className="p-2">
                                <span className="font-mono font-bold">{stock.symbol}</span>
                              </td>
                              <td className="p-2">
                                <Badge className="bg-red-500">{stock.score}</Badge>
                              </td>
                              <td className="p-2">
                                <Badge className={getIntensityColor(stock.squeeze_intensity)}>
                                  {stock.squeeze_intensity}
                                </Badge>
                              </td>
                              <td className="p-2">{stock.squeeze_count} bars</td>
                              <td className="p-2 text-red-500">{stock.momentum?.toFixed(4)}</td>
                              <td className="p-2">
                                <Badge variant="outline" className="text-red-500 border-red-500">
                                  {stock.signal}
                                </Badge>
                              </td>
                              <td className="p-2 text-right">${stock.current_price?.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* No Results Message */}
              {marketResults.squeeze_candidates_found === 0 && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    No squeeze setups found in this scan. This is normal during high volatility periods 
                    when squeezes have already fired. Try again later or scan a different set of stocks.
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
          <CardTitle className="text-lg">📚 What is TTM Squeeze?</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>TTM Squeeze</strong> was developed by John Carter. It detects when Bollinger Bands move inside 
            Keltner Channels, indicating low volatility that often precedes big moves.
          </p>
          <p>
            <strong>🔴 Red Dots (Squeeze ON):</strong> Volatility is compressed. Price is coiling like a spring.
          </p>
          <p>
            <strong>🟢 Green Dots (Squeeze OFF):</strong> The squeeze has fired! Volatility is expanding.
          </p>
          <p>
            <strong>Momentum Histogram:</strong> Shows direction. Green = bullish momentum. Red = bearish momentum.
            Darker colors = momentum increasing. Lighter colors = momentum decreasing.
          </p>
          <p>
            <strong>Score Ranking:</strong> Higher scores indicate better setups based on intensity, duration, 
            and momentum alignment. EXTREME intensity with 10+ squeeze bars is ideal.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
