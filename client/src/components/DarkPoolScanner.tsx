import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, TrendingUp, TrendingDown, AlertCircle, Eye } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export function DarkPoolScanner() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  const [marketResult, setMarketResult] = useState<any>(null);
  
  const darkPoolMutation = trpc.scanners.darkPool.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const marketDarkPoolMutation = trpc.scanners.marketDarkPool.useMutation({
    onSuccess: (data) => {
      setMarketResult(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      darkPoolMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const handleMarketScan = () => {
    marketDarkPoolMutation.mutate({ limit: 400 });
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
    if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num?.toFixed(2) || '0';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Eye className="h-6 w-6 text-purple-500" />
        <h2 className="text-2xl font-bold">Dark Pool / Oracle Scanner</h2>
        <Badge variant="outline" className="ml-2">Insider Movement</Badge>
      </div>

      <p className="text-muted-foreground">
        Detect institutional dark pool activity, short volume analysis, and smart money flow. 
        Data sourced from FINRA and Stockgrid.io.
      </p>

      {/* Search */}
      <div className="flex flex-wrap gap-2">
        <Input
          placeholder="Enter symbol (e.g., AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === 'Enter' && handleScan()}
          className="max-w-xs"
        />
        <Button onClick={handleScan} disabled={darkPoolMutation.isPending || !symbol.trim()}>
          {darkPoolMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Search className="h-4 w-4 mr-2" />
          )}
          Scan Dark Pool
        </Button>
        <Button 
          onClick={handleMarketScan} 
          disabled={marketDarkPoolMutation.isPending}
          variant="outline"
          className="border-purple-500 text-purple-500 hover:bg-purple-500/10"
        >
          {marketDarkPoolMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Eye className="h-4 w-4 mr-2" />
          )}
          Scan Market (400 Stocks)
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
          {/* Overall Sentiment */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                {result.overall_sentiment === 'BULLISH' || result.overall_sentiment === 'VERY_BULLISH' ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : result.overall_sentiment === 'BEARISH' || result.overall_sentiment === 'VERY_BEARISH' ? (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-gray-500" />
                )}
                Overall Sentiment
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <Badge className={`${getSentimentColor(result.overall_sentiment)} text-white`}>
                  {result.overall_sentiment || 'NEUTRAL'}
                </Badge>
                <span className="text-2xl font-bold">{result.overall_score || 50}/100</span>
              </div>
            </CardContent>
          </Card>

          {/* Dark Pool Position */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Net Dark Pool Position</CardTitle>
              <CardDescription>Accumulated institutional position</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="text-2xl font-bold">
                  {result.net_dp_position > 0 ? '+' : ''}{formatNumber(result.net_dp_position || 0)} shares
                </div>
                <div className="text-lg text-muted-foreground">
                  ${formatNumber(result.net_dp_position_dollar || 0)}
                </div>
                <Badge className={`${getSentimentColor(result.dp_sentiment)} text-white`}>
                  {result.dp_sentiment || 'UNKNOWN'}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Short Volume */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Short Volume Analysis</CardTitle>
              <CardDescription>FINRA short sale data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Short Volume:</span>
                  <span className="font-bold">{formatNumber(result.short_volume || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Short Ratio:</span>
                  <span className="font-bold">{result.short_ratio?.toFixed(1) || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${(result.short_ratio || 0) > 50 ? 'bg-red-500' : 'bg-green-500'}`}
                    style={{ width: `${Math.min(result.short_ratio || 0, 100)}%` }}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Price Context */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Price Context</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Current Price:</span>
                  <span className="font-bold">${result.current_price?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Change:</span>
                  <span className={`font-bold ${(result.price_change_pct || 0) >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {(result.price_change_pct || 0) >= 0 ? '+' : ''}{result.price_change_pct?.toFixed(2) || 0}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Volume Ratio:</span>
                  <span className="font-bold">{result.volume_ratio?.toFixed(2) || 1}x</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Signals */}
          {result.signals && result.signals.length > 0 && (
            <Card className="md:col-span-2">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Detected Signals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {result.signals.map((signal: string, idx: number) => (
                    <div key={idx} className="flex items-center gap-2 text-sm">
                      {signal}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Market-Wide Dark Pool Results */}
      {marketResult && (
        <Card className="border-purple-500/50">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Eye className="h-5 w-5 text-purple-500" />
              Market-Wide Dark Pool Scan
              <Badge variant="outline" className="ml-2">
                {marketResult.timestamp || new Date().toLocaleString('en-US', { timeZone: 'America/New_York' })} EST
              </Badge>
            </CardTitle>
            <CardDescription>
              {marketResult.stocks_scanned || 0} stocks scanned - Showing top institutional activity
            </CardDescription>
          </CardHeader>
          <CardContent>
            {marketResult.error ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{marketResult.error}</AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-4">
                {/* Summary Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="text-center p-3 bg-green-500/10 rounded-lg">
                    <div className="text-2xl font-bold text-green-500">{marketResult.bullish_count || 0}</div>
                    <div className="text-sm text-muted-foreground">Bullish Signals</div>
                  </div>
                  <div className="text-center p-3 bg-red-500/10 rounded-lg">
                    <div className="text-2xl font-bold text-red-500">{marketResult.bearish_count || 0}</div>
                    <div className="text-sm text-muted-foreground">Bearish Signals</div>
                  </div>
                  <div className="text-center p-3 bg-purple-500/10 rounded-lg">
                    <div className="text-2xl font-bold text-purple-500">{marketResult.high_activity_count || 0}</div>
                    <div className="text-sm text-muted-foreground">High Activity</div>
                  </div>
                  <div className="text-center p-3 bg-blue-500/10 rounded-lg">
                    <div className="text-2xl font-bold text-blue-500">{marketResult.unusual_volume_count || 0}</div>
                    <div className="text-sm text-muted-foreground">Unusual Volume</div>
                  </div>
                </div>

                {/* Top Bullish */}
                {marketResult.top_bullish && marketResult.top_bullish.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-green-500 mb-2 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Top Bullish Dark Pool Activity
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2">Symbol</th>
                            <th className="text-right py-2 px-2">Net Position</th>
                            <th className="text-right py-2 px-2">Short Ratio</th>
                            <th className="text-right py-2 px-2">Score</th>
                            <th className="text-left py-2 px-2">Signal</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResult.top_bullish.map((stock: any, idx: number) => (
                            <tr key={idx} className="border-b border-border/50 hover:bg-muted/50">
                              <td className="py-2 px-2 font-bold">{stock.symbol}</td>
                              <td className="py-2 px-2 text-right text-green-500">
                                {stock.net_position > 0 ? '+' : ''}{formatNumber(stock.net_position || 0)}
                              </td>
                              <td className="py-2 px-2 text-right">{stock.short_ratio?.toFixed(1) || 0}%</td>
                              <td className="py-2 px-2 text-right">
                                <Badge className="bg-green-500 text-white">{stock.score || 0}</Badge>
                              </td>
                              <td className="py-2 px-2 text-xs text-muted-foreground">{stock.signal || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Top Bearish */}
                {marketResult.top_bearish && marketResult.top_bearish.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-red-500 mb-2 flex items-center gap-2">
                      <TrendingDown className="h-4 w-4" />
                      Top Bearish Dark Pool Activity
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2">Symbol</th>
                            <th className="text-right py-2 px-2">Net Position</th>
                            <th className="text-right py-2 px-2">Short Ratio</th>
                            <th className="text-right py-2 px-2">Score</th>
                            <th className="text-left py-2 px-2">Signal</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResult.top_bearish.map((stock: any, idx: number) => (
                            <tr key={idx} className="border-b border-border/50 hover:bg-muted/50">
                              <td className="py-2 px-2 font-bold">{stock.symbol}</td>
                              <td className="py-2 px-2 text-right text-red-500">
                                {stock.net_position > 0 ? '+' : ''}{formatNumber(stock.net_position || 0)}
                              </td>
                              <td className="py-2 px-2 text-right">{stock.short_ratio?.toFixed(1) || 0}%</td>
                              <td className="py-2 px-2 text-right">
                                <Badge className="bg-red-500 text-white">{stock.score || 0}</Badge>
                              </td>
                              <td className="py-2 px-2 text-xs text-muted-foreground">{stock.signal || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Unusual Activity */}
                {marketResult.unusual_activity && marketResult.unusual_activity.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-purple-500 mb-2 flex items-center gap-2">
                      <AlertCircle className="h-4 w-4" />
                      Unusual Dark Pool Activity
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2">Symbol</th>
                            <th className="text-right py-2 px-2">Volume Ratio</th>
                            <th className="text-right py-2 px-2">DP Volume</th>
                            <th className="text-left py-2 px-2">Reason</th>
                          </tr>
                        </thead>
                        <tbody>
                          {marketResult.unusual_activity.map((stock: any, idx: number) => (
                            <tr key={idx} className="border-b border-border/50 hover:bg-muted/50">
                              <td className="py-2 px-2 font-bold">{stock.symbol}</td>
                              <td className="py-2 px-2 text-right text-purple-500">
                                {stock.volume_ratio?.toFixed(2) || 1}x
                              </td>
                              <td className="py-2 px-2 text-right">{formatNumber(stock.dp_volume || 0)}</td>
                              <td className="py-2 px-2 text-xs text-muted-foreground">{stock.reason || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Education */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">What is Dark Pool Trading?</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>Dark pools</strong> are private exchanges where institutional investors trade large blocks of shares 
            away from public markets. This helps them avoid moving prices against themselves.
          </p>
          <p>
            <strong>Net Position:</strong> Positive = institutions are accumulating (bullish). Negative = institutions are distributing (bearish).
          </p>
          <p>
            <strong>Short Ratio:</strong> Above 50% suggests bearish sentiment. Below 30% suggests bullish sentiment.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
