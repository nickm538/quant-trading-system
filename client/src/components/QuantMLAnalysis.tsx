import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { TrendingUp, TrendingDown, Brain, Target, AlertTriangle, Lightbulb, BarChart3, Zap } from "lucide-react";

interface MLPrediction {
  success: boolean;
  symbol: string;
  error?: string;
  current_price?: number;
  predicted_price?: number;
  predicted_change_pct?: number;
  confidence?: number;
  recommendation?: string;
  reasoning?: string;
  horizon_days?: number;
  model_performance?: {
    avg_accuracy: number;
    avg_sharpe_ratio: number;
    avg_win_rate: number;
    num_models_used: number;
  };
  ensemble_details?: {
    individual_predictions: number[];
    model_weights: number[];
  };
}

interface QuantMLAnalysisProps {
  mlPrediction: MLPrediction | null;
  loading?: boolean;
}

export function QuantMLAnalysis({ mlPrediction, loading }: QuantMLAnalysisProps) {
  console.log('📊 QuantMLAnalysis render:', { mlPrediction, loading });
  
  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Quant/ML Analysis
          </CardTitle>
          <CardDescription>
            Advanced machine learning insights
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!mlPrediction || !mlPrediction.success) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Quant/ML Analysis
          </CardTitle>
          <CardDescription>
            Advanced machine learning insights
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {mlPrediction?.error || "No trained ML models available for this stock. Train models first to see AI-powered predictions."}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const { 
    predicted_change_pct, 
    confidence, 
    recommendation, 
    reasoning,
    model_performance,
    horizon_days,
    current_price,
    predicted_price
  } = mlPrediction;

  const isPositive = predicted_change_pct > 0;
  const isHighConfidence = confidence > 70;
  const isStrongSignal = Math.abs(predicted_change_pct) > 3;

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-500" />
          Quant/ML Analysis
          <Badge variant={isHighConfidence ? "default" : "secondary"}>
            {confidence.toFixed(1)}% Confidence
          </Badge>
        </CardTitle>
        <CardDescription>
          Advanced machine learning insights powered by {model_performance.num_models_used} ensemble models
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Main Prediction */}
        <div className="p-4 rounded-lg bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950/20 dark:to-blue-950/20 border">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {isPositive ? (
                  <TrendingUp className="h-6 w-6 text-green-500" />
                ) : (
                  <TrendingDown className="h-6 w-6 text-red-500" />
                )}
                <h3 className="text-2xl font-bold">
                  {isPositive ? '+' : ''}{predicted_change_pct.toFixed(2)}%
                </h3>
                <span className="text-sm text-muted-foreground">
                  in {horizon_days} days
                </span>
              </div>
              <p className="text-sm text-muted-foreground">
                ${current_price.toFixed(2)} → ${predicted_price.toFixed(2)}
              </p>
            </div>
            <Badge 
              variant={
                recommendation.includes('STRONG BUY') ? 'default' :
                recommendation.includes('BUY') ? 'default' :
                recommendation.includes('STRONG SELL') ? 'destructive' :
                recommendation.includes('SELL') ? 'destructive' :
                'secondary'
              }
              className="text-lg px-4 py-2"
            >
              {recommendation}
            </Badge>
          </div>
        </div>

        {/* AI Insights - What Humans Might Miss */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            <h4 className="font-semibold">AI-Detected Insights</h4>
          </div>
          
          <div className="space-y-2">
            {isStrongSignal && (
              <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20">
                <Zap className="h-4 w-4 text-yellow-600" />
                <AlertDescription className="text-sm">
                  <strong>Strong Signal Detected:</strong> Our models predict a {Math.abs(predicted_change_pct).toFixed(1)}% 
                  {isPositive ? ' upward' : ' downward'} movement - significantly above normal market noise. 
                  This suggests {isPositive ? 'accumulation by smart money' : 'distribution before a decline'}.
                </AlertDescription>
              </Alert>
            )}
            
            {isHighConfidence && (
              <Alert className="border-blue-500 bg-blue-50 dark:bg-blue-950/20">
                <Target className="h-4 w-4 text-blue-600" />
                <AlertDescription className="text-sm">
                  <strong>High Model Agreement:</strong> {model_performance.num_models_used} independent ML models 
                  agree with {confidence.toFixed(1)}% consensus. This level of agreement typically precedes 
                  actual price movements within {horizon_days} days.
                </AlertDescription>
              </Alert>
            )}

            {model_performance.avg_sharpe_ratio > 1.5 && (
              <Alert className="border-green-500 bg-green-50 dark:bg-green-950/20">
                <BarChart3 className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-sm">
                  <strong>Superior Risk-Adjusted Returns:</strong> Models show Sharpe Ratio of {model_performance.avg_sharpe_ratio.toFixed(2)}, 
                  indicating excellent risk-adjusted performance. This suggests the predicted move has favorable risk/reward.
                </AlertDescription>
              </Alert>
            )}
          </div>
        </div>

        {/* How It Works - Educational */}
        <div className="space-y-3 pt-4 border-t">
          <h4 className="font-semibold text-sm text-muted-foreground">How This System Works</h4>
          <div className="text-sm space-y-2 text-muted-foreground">
            <p>
              <strong className="text-foreground">🤖 Ensemble Learning:</strong> We use {model_performance.num_models_used} different 
              AI models (XGBoost, LightGBM, and Ensemble) that each analyze 50+ technical indicators. 
              By combining their predictions, we reduce individual model bias and increase accuracy.
            </p>
            <p>
              <strong className="text-foreground">📊 Continuous Learning:</strong> Every time you train models, 
              they analyze 5 years of historical data across 15 random stocks from 90+ in our pool. 
              Over time, the system learns patterns that repeat before major price movements.
            </p>
            <p>
              <strong className="text-foreground">🎯 Pattern Recognition:</strong> The models detect subtle patterns 
              in price action, volume, momentum, and volatility that human traders often miss. These patterns 
              include hidden divergences, institutional accumulation/distribution, and regime changes.
            </p>
            <p>
              <strong className="text-foreground">⚡ Early Warning System:</strong> When models detect high-confidence 
              predictions (&gt;{confidence.toFixed(0)}%), it means they've identified a pattern that historically 
              precedes price movements. This gives you an edge to act before the market moves.
            </p>
          </div>
        </div>

        {/* Model Performance Metrics */}
        <div className="space-y-3 pt-4 border-t">
          <h4 className="font-semibold text-sm text-muted-foreground">Model Performance</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold text-green-600">
                {model_performance.avg_accuracy.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Direction Accuracy
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {model_performance.avg_accuracy > 55 ? 'Excellent' : 'Good'} (&gt;50% beats random)
              </div>
            </div>
            
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold text-blue-600">
                {model_performance.avg_sharpe_ratio.toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Sharpe Ratio
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {model_performance.avg_sharpe_ratio > 1 ? 'Strong' : 'Moderate'} risk-adjusted returns
              </div>
            </div>
            
            <div className="text-center p-3 rounded-lg bg-muted/50">
              <div className="text-2xl font-bold text-purple-600">
                {model_performance.avg_win_rate.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Win Rate
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Historical success rate
              </div>
            </div>
          </div>
        </div>

        {/* What The Data Shows */}
        <div className="space-y-3 pt-4 border-t">
          <h4 className="font-semibold text-sm text-muted-foreground">What The Data Is Telling Us</h4>
          <div className="text-sm space-y-2">
            <p className="leading-relaxed">
              {reasoning}
            </p>
            <p className="leading-relaxed text-muted-foreground">
              {isPositive ? (
                <>
                  The models are detecting <strong className="text-foreground">bullish momentum patterns</strong> similar 
                  to historical instances that preceded upward moves. Key indicators show increasing buying pressure, 
                  positive momentum divergence, and technical setup favoring continuation to the upside.
                </>
              ) : (
                <>
                  The models are detecting <strong className="text-foreground">bearish momentum patterns</strong> similar 
                  to historical instances that preceded downward moves. Key indicators show increasing selling pressure, 
                  negative momentum divergence, and technical setup favoring continuation to the downside.
                </>
              )}
            </p>
            {Math.abs(predicted_change_pct) < 1 && (
              <p className="leading-relaxed text-muted-foreground">
                The predicted movement is <strong className="text-foreground">minimal</strong>, suggesting the stock 
                is in a consolidation phase. This is often a period where smart money accumulates before the next 
                significant move. Consider waiting for a clearer signal or using this as an opportunity to build 
                positions gradually.
              </p>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="pt-4 border-t">
          <p className="text-xs text-muted-foreground italic">
            ⚠️ ML predictions are probabilistic, not guarantees. Past performance doesn't ensure future results. 
            Always combine AI insights with fundamental analysis, risk management, and your own judgment.
          </p>
        </div>

      </CardContent>
    </Card>
  );
}
