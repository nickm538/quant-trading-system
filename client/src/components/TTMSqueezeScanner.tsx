import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Loader2, Search, AlertCircle, TrendingUp, TrendingDown, Circle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

export function TTMSqueezeScanner() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState<any>(null);
  
  const squeezeMutation = trpc.scanners.ttmSqueeze.useMutation({
    onSuccess: (data) => {
      setResult(data);
    },
  });

  const handleScan = () => {
    if (symbol.trim()) {
      squeezeMutation.mutate({ symbol: symbol.toUpperCase() });
    }
  };

  const getMomentumColor = (momentum: number) => {
    if (momentum > 0) return momentum > (result?.prev_momentum || 0) ? 'bg-green-600' : 'bg-green-400';
    return momentum < (result?.prev_momentum || 0) ? 'bg-red-600' : 'bg-red-400';
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

      {/* Search */}
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

      {/* Results */}
      {result && !result.error && result.status !== 'error' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Squeeze Status */}
          <Card className="md:col-span-2 lg:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Circle className={`h-6 w-6 ${result.squeeze_on ? 'fill-red-500 text-red-500' : 'fill-green-500 text-green-500'}`} />
                Squeeze Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <Badge className={`${result.squeeze_on ? 'bg-red-500' : 'bg-green-500'} text-white text-lg px-4 py-2`}>
                    {result.squeeze_on ? '🔴 SQUEEZE ON' : '🟢 SQUEEZE OFF'}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {result.squeeze_on 
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
        </CardContent>
      </Card>
    </div>
  );
}
