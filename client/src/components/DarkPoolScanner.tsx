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
  
  const darkPoolMutation = trpc.scanners.darkPool.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      darkPoolMutation.mutate({ symbol: symbol.toUpperCase() });
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
      <div className="flex gap-2">
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

      {/* Education */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">📚 What is Dark Pool Trading?</CardTitle>
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
