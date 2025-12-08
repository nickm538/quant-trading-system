import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { TrendingUp, TrendingDown, AlertCircle, Target, Shield, Zap, Activity } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface OptionRecommendation {
  option_type: string;
  strike: number;
  expiration: string;
  dte: number;
  last_price: number;
  bid: number;
  ask: number;
  mid_price: number;
  final_score: number;
  rating: string;
  scores: {
    volatility: number;
    greeks: number;
    technical: number;
    liquidity: number;
    event_risk: number;
    sentiment: number;
    flow: number;
    expected_value: number;
  };
  key_metrics: {
    delta: number;
    gamma: number;
    vega: number;
    theta: number;
    iv: number;
    spread_pct: number;
    volume: number;
    open_interest: number;
  };
  risk_management: {
    kelly_pct: number;
    conservative_kelly: number;
    max_position_size_pct: number;
  };
  insights: string[];
}

interface InstitutionalOptionsAnalysisProps {
  data: {
    success: boolean;
    symbol: string;
    current_price: number;
    top_calls: OptionRecommendation[];
    top_puts: OptionRecommendation[];
    total_calls_analyzed: number;
    total_puts_analyzed: number;
    calls_passed_filters: number;
    puts_passed_filters: number;
    calls_above_threshold: number;
    puts_above_threshold: number;
    methodology: {
      category_weights: Record<string, number>;
      min_score_threshold: number;
      filters: Record<string, any>;
    };
    market_context: {
      historical_volatility: number;
      earnings_date: string | null;
      days_to_earnings: number;
      sentiment_score: number;
    };
  };
}

const getRatingColor = (rating: string) => {
  switch (rating) {
    case 'EXCEPTIONAL': return 'bg-purple-500';
    case 'EXCELLENT': return 'bg-green-500';
    case 'GOOD': return 'bg-blue-500';
    case 'ACCEPTABLE': return 'bg-yellow-500';
    default: return 'bg-gray-500';
  }
};

const getRatingIcon = (rating: string) => {
  switch (rating) {
    case 'EXCEPTIONAL': return <Zap className="w-4 h-4" />;
    case 'EXCELLENT': return <TrendingUp className="w-4 h-4" />;
    case 'GOOD': return <Target className="w-4 h-4" />;
    case 'ACCEPTABLE': return <Activity className="w-4 h-4" />;
    default: return <AlertCircle className="w-4 h-4" />;
  }
};

const ScoreBar = ({ label, score, color = "bg-blue-500" }: { label: string; score: number; color?: string }) => (
  <div className="space-y-1">
    <div className="flex justify-between text-sm">
      <span className="text-gray-600">{label}</span>
      <span className="font-semibold">{score.toFixed(0)}</span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className={`${color} h-2 rounded-full transition-all duration-300`}
        style={{ width: `${Math.min(score, 100)}%` }}
      />
    </div>
  </div>
);

const OptionCard = ({ option }: { option: OptionRecommendation }) => {
  const isCall = option.option_type === 'CALL';
  
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isCall ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-500" />
            )}
            <CardTitle className="text-lg">
              ${option.strike} {option.option_type}
            </CardTitle>
          </div>
          <Badge className={`${getRatingColor(option.rating)} text-white flex items-center gap-1`}>
            {getRatingIcon(option.rating)}
            {option.rating}
          </Badge>
        </div>
        <CardDescription>
          Expires {option.expiration} ({option.dte} days) • Score: {option.final_score.toFixed(1)}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="space-y-1">
            <div className="text-gray-600">Price</div>
            <div className="font-semibold">${option.mid_price.toFixed(2)}</div>
            <div className="text-xs text-gray-500">
              Bid: ${option.bid.toFixed(2)} / Ask: ${option.ask.toFixed(2)}
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-gray-600">Delta</div>
            <div className="font-semibold">{option.key_metrics.delta.toFixed(3)}</div>
            <div className="text-xs text-gray-500">
              Gamma: {option.key_metrics.gamma.toFixed(3)}
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-gray-600">IV</div>
            <div className="font-semibold">{option.key_metrics.iv.toFixed(1)}%</div>
            <div className="text-xs text-gray-500">
              Vega: {option.key_metrics.vega.toFixed(3)}
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-gray-600">Theta</div>
            <div className="font-semibold">{option.key_metrics.theta.toFixed(3)}</div>
            <div className="text-xs text-gray-500">
              Decay/day
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-gray-600">Volume</div>
            <div className="font-semibold">{option.key_metrics.volume.toLocaleString()}</div>
            <div className="text-xs text-gray-500">
              Spread: {option.key_metrics.spread_pct.toFixed(1)}%
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-gray-600">Open Interest</div>
            <div className="font-semibold">{option.key_metrics.open_interest.toLocaleString()}</div>
            <div className="text-xs text-gray-500">
              Liquidity
            </div>
          </div>
        </div>

        {/* Score Breakdown */}
        <div className="space-y-2 pt-2 border-t">
          <div className="text-sm font-semibold text-gray-700">Score Breakdown</div>
          <div className="grid grid-cols-2 gap-2">
            <ScoreBar label="Volatility" score={option.scores.volatility} color="bg-purple-500" />
            <ScoreBar label="Greeks" score={option.scores.greeks} color="bg-blue-500" />
            <ScoreBar label="Technical" score={option.scores.technical} color="bg-green-500" />
            <ScoreBar label="Liquidity" score={option.scores.liquidity} color="bg-cyan-500" />
            <ScoreBar label="Event Risk" score={option.scores.event_risk} color="bg-orange-500" />
            <ScoreBar label="Sentiment" score={option.scores.sentiment} color="bg-pink-500" />
            <ScoreBar label="Flow" score={option.scores.flow} color="bg-indigo-500" />
            <ScoreBar label="Expected Value" score={option.scores.expected_value} color="bg-teal-500" />
          </div>
        </div>

        {/* Risk Management */}
        <div className="space-y-2 pt-2 border-t">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-700">
            <Shield className="w-4 h-4" />
            Position Sizing (Kelly Criterion)
          </div>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="bg-gray-50 p-2 rounded">
              <div className="text-xs text-gray-600">Full Kelly</div>
              <div className="font-semibold">{(option.risk_management.kelly_pct * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-blue-50 p-2 rounded">
              <div className="text-xs text-gray-600">Conservative</div>
              <div className="font-semibold text-blue-600">
                {(option.risk_management.conservative_kelly * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="text-xs text-gray-600">Max Position</div>
              <div className="font-semibold text-green-600">
                {(option.risk_management.max_position_size_pct * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        {/* Insights */}
        {option.insights && option.insights.length > 0 && (
          <div className="space-y-2 pt-2 border-t">
            <div className="text-sm font-semibold text-gray-700">Key Insights</div>
            <ul className="space-y-1">
              {option.insights.map((insight, idx) => (
                <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                  <span className="text-blue-500 mt-0.5">•</span>
                  <span>{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export const InstitutionalOptionsAnalysis: React.FC<InstitutionalOptionsAnalysisProps> = ({ data }) => {
  if (!data.success) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to analyze options. Please try again.
        </AlertDescription>
      </Alert>
    );
  }

  const hasResults = data.top_calls.length > 0 || data.top_puts.length > 0;

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Current Price</CardDescription>
            <CardTitle className="text-2xl">${data.current_price.toFixed(2)}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Options Analyzed</CardDescription>
            <CardTitle className="text-2xl">
              {data.total_calls_analyzed + data.total_puts_analyzed}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Passed Filters</CardDescription>
            <CardTitle className="text-2xl">
              {data.calls_passed_filters + data.puts_passed_filters}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Top Opportunities</CardDescription>
            <CardTitle className="text-2xl">
              {data.calls_above_threshold + data.puts_above_threshold}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Market Context */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Market Context</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-600">Historical Volatility</div>
              <div className="font-semibold text-lg">
                {(data.market_context.historical_volatility * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-600">Sentiment Score</div>
              <div className="font-semibold text-lg">{data.market_context.sentiment_score}/100</div>
            </div>
            <div>
              <div className="text-gray-600">Days to Earnings</div>
              <div className="font-semibold text-lg">
                {data.market_context.days_to_earnings === 999 ? 'N/A' : data.market_context.days_to_earnings}
              </div>
            </div>
            <div>
              <div className="text-gray-600">Min Score Threshold</div>
              <div className="font-semibold text-lg">{data.methodology.min_score_threshold}/100</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {!hasResults ? (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No options met the institutional-grade quality criteria. This is intentional - the system prioritizes
            precision over quantity. Try analyzing a different symbol or check back when market conditions improve.
          </AlertDescription>
        </Alert>
      ) : (
        <Tabs defaultValue="calls" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="calls" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Calls ({data.top_calls.length})
            </TabsTrigger>
            <TabsTrigger value="puts" className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4" />
              Puts ({data.top_puts.length})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="calls" className="space-y-4 mt-4">
            {data.top_calls.length === 0 ? (
              <Alert>
                <AlertDescription>
                  No call options met the quality threshold of {data.methodology.min_score_threshold}/100.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {data.top_calls.map((option, idx) => (
                  <OptionCard key={idx} option={option} />
                ))}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="puts" className="space-y-4 mt-4">
            {data.top_puts.length === 0 ? (
              <Alert>
                <AlertDescription>
                  No put options met the quality threshold of {data.methodology.min_score_threshold}/100.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {data.top_puts.map((option, idx) => (
                  <OptionCard key={idx} option={option} />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* Methodology Note */}
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          <strong>Institutional-Grade Analysis:</strong> This system uses an 8-factor scoring algorithm with
          Black-Scholes Greeks (including second-order: Vanna, Charm), Kelly Criterion position sizing, and
          AI-powered pattern recognition. Only options scoring ≥{data.methodology.min_score_threshold}/100 are shown.
          All recommendations are for informational purposes only.
        </AlertDescription>
      </Alert>
    </div>
  );
};
