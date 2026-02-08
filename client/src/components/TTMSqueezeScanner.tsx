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
  const [marketIntel, setMarketIntel] = useState<any>(null);
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

  const marketIntelMutation = trpc.scanners.marketIntelligence.useMutation({
    onSuccess: (data) => {
      setMarketIntel(data);
    },
  });

  const handleMarketIntel = () => {
    marketIntelMutation.mutate();
  };

  const handleScan = () => {
    if (symbol.trim()) {
      squeezeMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const handleMarketScan = () => {
    marketScanMutation.mutate({ maxStocks: 500 });
  };

  const getMomentumColor = (color: string) => {
    switch (color) {
      case 'dark_green': return 'bg-green-600';
      case 'light_green': return 'bg-green-400';
      case 'dark_red': return 'bg-red-600';
      case 'light_red': return 'bg-red-400';
      default: return 'bg-gray-400';
    }
  };

  const getMomentumLabel = (color: string) => {
    switch (color) {
      case 'dark_green': return 'Bullish & Accelerating';
      case 'light_green': return 'Bullish but Decelerating';
      case 'dark_red': return 'Bearish & Accelerating';
      case 'light_red': return 'Bearish but Decelerating';
      default: return 'Neutral';
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BULLISH_BREAKOUT': return 'bg-green-500';
      case 'BEARISH_BREAKOUT': return 'bg-red-500';
      case 'SQUEEZE_ON': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getSignalLabel = (signal: string, strength: string) => {
    switch (signal) {
      case 'BULLISH_BREAKOUT': return strength === 'STRONG' ? 'BULLISH BREAKOUT (Strong)' : 'BULLISH BREAKOUT (Weakening)';
      case 'BEARISH_BREAKOUT': return strength === 'STRONG' ? 'BEARISH BREAKOUT (Strong)' : 'BEARISH BREAKOUT (Weakening)';
      case 'SQUEEZE_ON': return strength === 'BULLISH_BUILDING' ? 'SQUEEZE ON (Bullish Building)' : strength === 'BEARISH_BUILDING' ? 'SQUEEZE ON (Bearish Building)' : 'SQUEEZE ON';
      default: return 'NEUTRAL';
    }
  };

  const getSignalDescription = (signal: string, strength: string) => {
    switch (signal) {
      case 'BULLISH_BREAKOUT': return strength === 'STRONG' 
        ? 'Squeeze has fired with strong bullish momentum. Price is breaking out to the upside with increasing momentum — this is the ideal long setup.'
        : 'Squeeze has fired bullish but momentum is starting to decelerate. The move may be maturing — consider tightening stops.';
      case 'BEARISH_BREAKOUT': return strength === 'STRONG'
        ? 'Squeeze has fired with strong bearish momentum. Price is breaking down with increasing momentum — this is the ideal short setup.'
        : 'Squeeze has fired bearish but momentum is starting to decelerate. The move may be maturing — consider tightening stops.';
      case 'SQUEEZE_ON': return strength === 'BULLISH_BUILDING'
        ? 'Volatility is compressed and bullish momentum is building inside the squeeze. Watch for the squeeze to fire — potential long entry coming.'
        : strength === 'BEARISH_BUILDING'
        ? 'Volatility is compressed and bearish momentum is building inside the squeeze. Watch for the squeeze to fire — potential short entry coming.'
        : 'Volatility is compressed. Price is coiling for a potential explosive move. Wait for direction confirmation.';
      default: return 'No clear signal. The squeeze is neither on nor has it recently fired. Wait for a setup to develop.';
    }
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
            <div className="space-y-4">
              {/* Price Header */}
              <div className="flex items-center gap-4 flex-wrap">
                <span className="text-2xl font-bold">{result.symbol}</span>
                <span className="text-2xl">${typeof result.current_price === 'number' ? result.current_price.toFixed(2) : 'N/A'}</span>
                <span className="text-xs text-muted-foreground">{result.data_source || 'Real-time'}</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {/* Squeeze Status */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Circle className={`h-6 w-6 ${result.squeeze_on ? 'fill-red-500 text-red-500' : 'fill-green-500 text-green-500'}`} />
                      Squeeze Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Badge className={`${result.squeeze_on ? 'bg-red-500' : 'bg-green-500'} text-white text-lg px-4 py-2`}>
                        {result.squeeze_on ? 'SQUEEZE ON' : 'SQUEEZE OFF'}
                      </Badge>
                      <div className="text-sm text-muted-foreground">
                        {result.squeeze_count} consecutive bars
                      </div>
                      {result.squeeze_on && result.squeeze_intensity && result.squeeze_intensity !== 'NONE' && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">Intensity:</span>
                          <Badge className={getIntensityColor(result.squeeze_intensity)}>
                            {result.squeeze_intensity}
                          </Badge>
                          {result.squeeze_intensity_ratio && (
                            <span className="text-xs text-muted-foreground">(BB/KC: {result.squeeze_intensity_ratio})</span>
                          )}
                        </div>
                      )}
                      {result.tight_squeeze && (
                        <Badge variant="outline" className="border-purple-500 text-purple-500">Tight Squeeze (1.0x ATR)</Badge>
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
                    <CardDescription>Linear regression of price deviation</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className={`text-3xl font-bold ${(result.momentum || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {(result.momentum || 0) >= 0 ? '+' : ''}{typeof result.momentum === 'number' ? result.momentum.toFixed(4) : '0.0000'}
                      </div>
                      <Badge className={`${getMomentumColor(result.momentum_color)} text-white`}>
                        {getMomentumLabel(result.momentum_color)}
                      </Badge>
                      {/* Momentum History Mini-Chart */}
                      {result.momentum_history && result.momentum_history.length > 0 && (
                        <div className="flex items-end gap-0.5 h-12">
                          {result.momentum_history.map((m: number | null, i: number) => {
                            if (m === null) return <div key={i} className="flex-1 bg-gray-300 rounded-t" style={{ height: '2px' }} />;
                            const maxAbs = Math.max(...result.momentum_history.filter((v: any) => v !== null).map((v: number) => Math.abs(v)), 0.001);
                            const pct = Math.abs(m) / maxAbs * 100;
                            return (
                              <div key={i} className="flex-1 flex flex-col justify-end h-full">
                                {m >= 0 ? (
                                  <div className={`rounded-t ${i === result.momentum_history.length - 1 ? 'bg-green-500' : 'bg-green-500/50'}`} style={{ height: `${Math.max(pct, 4)}%` }} />
                                ) : (
                                  <div className="flex-1" />
                                )}
                                {m < 0 && (
                                  <div className={`rounded-b ${i === result.momentum_history.length - 1 ? 'bg-red-500' : 'bg-red-500/50'}`} style={{ height: `${Math.max(pct, 4)}%` }} />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground">Last 10 bars momentum history</div>
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
                      <Badge className={`text-lg px-4 py-2 ${getSignalColor(result.signal)} text-white`}>
                        {getSignalLabel(result.signal, result.signal_strength)}
                      </Badge>
                      <p className="text-sm text-muted-foreground">
                        {getSignalDescription(result.signal, result.signal_strength)}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Technical Details - Full Width */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Technical Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">BB Upper</span>
                      <div className="font-bold">${typeof result.bb_upper === 'number' ? result.bb_upper.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">BB Middle (SMA 20)</span>
                      <div className="font-bold">${typeof result.bb_middle === 'number' ? result.bb_middle.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">BB Lower</span>
                      <div className="font-bold">${typeof result.bb_lower === 'number' ? result.bb_lower.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">KC Upper</span>
                      <div className="font-bold">${typeof result.kc_upper === 'number' ? result.kc_upper.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">KC Middle (EMA 20)</span>
                      <div className="font-bold">${typeof result.kc_middle === 'number' ? result.kc_middle.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">KC Lower</span>
                      <div className="font-bold">${typeof result.kc_lower === 'number' ? result.kc_lower.toFixed(2) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">ATR (20)</span>
                      <div className="font-bold">{typeof result.atr === 'number' ? result.atr.toFixed(4) : 'N/A'}</div>
                    </div>
                    <div>
                      <span className="text-muted-foreground">BB Width %</span>
                      <div className="font-bold">{typeof result.bb_width === 'number' ? result.bb_width.toFixed(2) + '%' : 'N/A'}</div>
                    </div>
                  </div>

                  {/* Squeeze History Visual */}
                  {result.squeeze_history && result.squeeze_history.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm text-muted-foreground mb-2">Squeeze History (last 10 bars)</div>
                      <div className="flex gap-1">
                        {result.squeeze_history.map((on: boolean, i: number) => (
                          <div key={i} className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                            on ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
                          }`}>
                            {on ? 'S' : 'F'}
                          </div>
                        ))}
                      </div>
                      <div className="flex gap-3 mt-1 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1"><Circle className="h-3 w-3 fill-red-500 text-red-500" /> Squeeze On</span>
                        <span className="flex items-center gap-1"><Circle className="h-3 w-3 fill-green-500 text-green-500" /> Fired</span>
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
                    Scan completed in {marketResults.scan_time_seconds}s at {new Date(marketResults.timestamp).toLocaleString('en-US', { timeZone: 'America/New_York', hour12: true, month: 'numeric', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })} EST
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

      {/* Market Intelligence Section */}
      <Card className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-blue-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Globe className="h-5 w-5 text-blue-500" />
            Market Intelligence
          </CardTitle>
          <CardDescription>
            Real-time market context: VIX, regime detection, sentiment, catalysts, and geopolitics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button 
            onClick={handleMarketIntel} 
            disabled={marketIntelMutation.isPending}
            variant="outline"
            className="border-blue-500/50 hover:bg-blue-500/10"
          >
            {marketIntelMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Fetching Market Intelligence...
              </>
            ) : (
              <>
                <Radar className="h-4 w-4 mr-2" />
                Get Live Market Context
              </>
            )}
          </Button>

          {/* Market Intel Results */}
          {marketIntel && !marketIntel.error && (
            <div className="space-y-4">
              {/* Market Status Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="p-3 bg-muted rounded-lg text-center">
                  <div className="text-xs text-muted-foreground mb-1">Market Status</div>
                  <Badge className={marketIntel.market_status?.is_open ? 'bg-green-500' : 'bg-red-500'}>
                    {marketIntel.market_status?.is_open ? 'OPEN' : 'CLOSED'}
                  </Badge>
                  <div className="text-xs mt-1 text-muted-foreground">
                    {marketIntel.market_status?.day_of_week} {marketIntel.market_status?.current_time_est}
                  </div>
                </div>
                <div className="p-3 bg-muted rounded-lg text-center">
                  <div className="text-xs text-muted-foreground mb-1">VIX</div>
                  <div className="text-2xl font-bold">{marketIntel.vix?.value?.toFixed(2) || 'N/A'}</div>
                  <Badge variant="outline" className="text-xs">
                    {marketIntel.vix?.regime || 'Unknown'}
                  </Badge>
                </div>
                <div className="p-3 bg-muted rounded-lg text-center">
                  <div className="text-xs text-muted-foreground mb-1">Market Regime</div>
                  <Badge className={
                    marketIntel.regime?.current === 'BULL' ? 'bg-green-500' :
                    marketIntel.regime?.current === 'BEAR' ? 'bg-red-500' :
                    marketIntel.regime?.current === 'VOLATILE' ? 'bg-orange-500' : 'bg-gray-500'
                  }>
                    {marketIntel.regime?.current || 'Unknown'}
                  </Badge>
                  <div className="text-xs mt-1 text-muted-foreground">
                    Confidence: {marketIntel.regime?.confidence || 'N/A'}%
                  </div>
                </div>
                <div className="p-3 bg-muted rounded-lg text-center">
                  <div className="text-xs text-muted-foreground mb-1">Sentiment</div>
                  <Badge className={
                    marketIntel.sentiment?.overall === 'BULLISH' ? 'bg-green-500' :
                    marketIntel.sentiment?.overall === 'BEARISH' ? 'bg-red-500' : 'bg-yellow-500'
                  }>
                    {marketIntel.sentiment?.overall || 'Neutral'}
                  </Badge>
                  <div className="text-xs mt-1 text-muted-foreground">
                    Score: {marketIntel.sentiment?.score || 0}/100
                  </div>
                </div>
              </div>

              {/* Historical Pattern Match */}
              {marketIntel.historical_pattern && (
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm font-medium mb-1">Historical Pattern Match</div>
                  <div className="text-sm text-muted-foreground">
                    {marketIntel.historical_pattern.description || 'No significant pattern detected'}
                  </div>
                  {marketIntel.historical_pattern.similarity && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Similarity: {marketIntel.historical_pattern.similarity}% | 
                      Historical Outcome: {marketIntel.historical_pattern.outcome || 'N/A'}
                    </div>
                  )}
                </div>
              )}

              {/* Catalysts */}
              {marketIntel.catalysts?.length > 0 && (
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm font-medium mb-2">Active Catalysts</div>
                  <div className="space-y-1">
                    {marketIntel.catalysts.slice(0, 5).map((catalyst: any, idx: number) => (
                      <div key={idx} className="text-xs flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">{catalyst.type}</Badge>
                        <span className="text-muted-foreground">{catalyst.description}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Geopolitical */}
              {marketIntel.geopolitical && (
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-sm font-medium mb-1">Geopolitical Context</div>
                  <div className="text-xs text-muted-foreground">
                    Risk Level: <Badge variant="outline">{marketIntel.geopolitical.risk_level || 'Normal'}</Badge>
                  </div>
                  {marketIntel.geopolitical.summary && (
                    <div className="text-xs text-muted-foreground mt-1">
                      {marketIntel.geopolitical.summary}
                    </div>
                  )}
                </div>
              )}

              <div className="text-xs text-muted-foreground">
                Last updated: {new Date(marketIntel.timestamp).toLocaleString('en-US', { timeZone: 'America/New_York', hour12: true })} EST
              </div>
            </div>
          )}

          {/* Market Intel Error */}
          {marketIntel?.error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{marketIntel.error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

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
