import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Target,
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Clock,
  DollarSign,
  BarChart3,
  Brain,
  Zap,
  ChevronDown,
  ChevronUp,
  Minus,
} from "lucide-react";
import { useState } from "react";

interface PersonalRecommendationProps {
  recommendation: any;
  symbol: string;
  currentPrice: number;
}

export function PersonalRecommendation({
  recommendation,
  symbol,
  currentPrice,
}: PersonalRecommendationProps) {
  const [showBreakdown, setShowBreakdown] = useState(false);

  if (!recommendation || recommendation.error) {
    return null;
  }

  const action = recommendation.action || "HOLD";
  const conviction = recommendation.conviction_score || 50;
  const convictionLevel = recommendation.conviction_level || "MODERATE";
  const narrative = recommendation.narrative || "";
  const riskWarnings = recommendation.risk_warnings || [];
  const timeHorizon = recommendation.time_horizon || {};
  const signalBreakdown = recommendation.signal_breakdown || {};
  const convictionBreakdown = recommendation.conviction_breakdown || {};

  // Determine action color and icon
  const getActionStyle = (act: string) => {
    const upper = act.toUpperCase();
    if (upper.includes("BUY") || upper.includes("LONG"))
      return {
        bg: "bg-emerald-500/10 border-emerald-500/30",
        text: "text-emerald-400",
        badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
        icon: TrendingUp,
      };
    if (upper.includes("SELL") || upper.includes("SHORT"))
      return {
        bg: "bg-red-500/10 border-red-500/30",
        text: "text-red-400",
        badge: "bg-red-500/20 text-red-300 border-red-500/40",
        icon: TrendingDown,
      };
    return {
      bg: "bg-amber-500/10 border-amber-500/30",
      text: "text-amber-400",
      badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
      icon: Minus,
    };
  };

  const actionStyle = getActionStyle(action);
  const ActionIcon = actionStyle.icon;

  // Conviction bar color
  const getConvictionColor = (score: number) => {
    if (score >= 75) return "bg-emerald-500";
    if (score >= 60) return "bg-emerald-400";
    if (score >= 50) return "bg-amber-400";
    if (score >= 35) return "bg-orange-400";
    return "bg-red-500";
  };

  // Format signal breakdown for display
  const formatSignalValue = (value: number) => {
    if (value >= 70) return { label: "Bullish", color: "text-emerald-400" };
    if (value >= 55) return { label: "Lean Bullish", color: "text-emerald-300" };
    if (value >= 45) return { label: "Neutral", color: "text-amber-400" };
    if (value >= 30) return { label: "Lean Bearish", color: "text-orange-400" };
    return { label: "Bearish", color: "text-red-400" };
  };

  // Parse narrative into sections for rich rendering
  const renderNarrative = (text: string) => {
    if (!text) return null;

    // Split by double newlines to get paragraphs
    const paragraphs = text.split("\n\n").filter((p) => p.trim());

    return paragraphs.map((para, i) => {
      // Convert **bold** markers to JSX
      const parts = para.split(/(\*\*[^*]+\*\*)/g);
      return (
        <p key={i} className="text-sm text-muted-foreground leading-relaxed mb-3">
          {parts.map((part, j) => {
            if (part.startsWith("**") && part.endsWith("**")) {
              return (
                <span key={j} className="font-semibold text-foreground">
                  {part.slice(2, -2)}
                </span>
              );
            }
            return <span key={j}>{part}</span>;
          })}
        </p>
      );
    });
  };

  return (
    <Card className={`border-2 ${actionStyle.bg}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`p-2 rounded-lg ${actionStyle.bg}`}
            >
              <Brain className={`h-6 w-6 ${actionStyle.text}`} />
            </div>
            <div>
              <CardTitle className="text-xl">
                If I Were Trading {symbol}
              </CardTitle>
              <CardDescription>
                Personal recommendation engine — synthesized from all data
                sources
              </CardDescription>
            </div>
          </div>
          <Badge
            variant="outline"
            className={`text-lg px-4 py-1 ${actionStyle.badge}`}
          >
            <ActionIcon className="h-5 w-5 mr-2" />
            {action}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        {/* Conviction Score Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Conviction Score</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">{conviction.toFixed(0)}</span>
              <span className="text-sm text-muted-foreground">/ 100</span>
              <Badge variant="outline" className="ml-2">
                {convictionLevel}
              </Badge>
            </div>
          </div>
          <div className="h-3 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${getConvictionColor(conviction)}`}
              style={{ width: `${Math.min(100, Math.max(0, conviction))}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Strong Sell</span>
            <span>Neutral</span>
            <span>Strong Buy</span>
          </div>
        </div>

        {/* Position & Time Horizon Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <DollarSign className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Position</span>
            </div>
            <div className="text-sm font-semibold">
              {recommendation.position_size_shares > 0
                ? `${recommendation.position_size_shares} shares`
                : "No position"}
            </div>
            {recommendation.position_size_dollars > 0 && (
              <div className="text-xs text-muted-foreground">
                ${recommendation.position_size_dollars?.toLocaleString()}
              </div>
            )}
          </div>

          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <Shield className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Risk</span>
            </div>
            <div className="text-sm font-semibold text-red-400">
              ${recommendation.risk_per_trade_dollars?.toFixed(2)}
            </div>
            <div className="text-xs text-muted-foreground">
              {recommendation.position_pct_of_bankroll?.toFixed(1)}% of bankroll
            </div>
          </div>

          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <Clock className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Time Horizon</span>
            </div>
            <div className="text-sm font-semibold">
              {timeHorizon.horizon?.replace(/_/g, " ") || "N/A"}
            </div>
            <div className="text-xs text-muted-foreground">
              {timeHorizon.suggested_days || ""}
            </div>
          </div>

          <div className="p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Confluence</span>
            </div>
            <div className="text-sm font-semibold">
              {convictionBreakdown.confluence || "N/A"}
            </div>
            <div className="text-xs text-muted-foreground">
              {convictionBreakdown.bullish_signals || 0}B /{" "}
              {convictionBreakdown.bearish_signals || 0}S /{" "}
              {convictionBreakdown.neutral_signals || 0}N
            </div>
          </div>
        </div>

        {/* Narrative — The Heart of the Recommendation */}
        <div className="p-4 bg-muted/20 rounded-lg border border-border/50">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="h-4 w-4 text-amber-400" />
            <span className="text-sm font-semibold text-foreground">
              The Analysis
            </span>
          </div>
          <div className="prose prose-sm prose-invert max-w-none">
            {renderNarrative(narrative)}
          </div>
        </div>

        {/* Risk Warnings */}
        {riskWarnings.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <span className="text-sm font-semibold">Risk Warnings</span>
            </div>
            <div className="space-y-1.5">
              {riskWarnings.map((warning: string, i: number) => (
                <div
                  key={i}
                  className="flex items-start gap-2 p-2.5 bg-amber-500/5 border border-amber-500/20 rounded-lg"
                >
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />
                  <span className="text-xs text-amber-200/80">{warning}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Entry/Exit Strategy */}
        {(recommendation.recommended_entry || recommendation.recommended_exit) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {recommendation.recommended_entry && (
              <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
                <div className="text-xs font-semibold text-emerald-400 mb-1.5">
                  Entry Strategy
                </div>
                <div className="text-xs text-muted-foreground leading-relaxed">
                  {recommendation.recommended_entry}
                </div>
              </div>
            )}
            {recommendation.recommended_exit && (
              <div className="p-3 bg-red-500/5 border border-red-500/20 rounded-lg">
                <div className="text-xs font-semibold text-red-400 mb-1.5">
                  Exit Strategy
                </div>
                <div className="text-xs text-muted-foreground leading-relaxed">
                  {recommendation.recommended_exit}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Signal Breakdown (Collapsible) */}
        <div>
          <button
            onClick={() => setShowBreakdown(!showBreakdown)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full"
          >
            {showBreakdown ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            <span className="font-medium">Signal Breakdown by Source</span>
          </button>

          {showBreakdown && (
            <div className="mt-3 space-y-2">
              {Object.entries(signalBreakdown).map(
                ([key, signal]: [string, any]) => {
                  const formatted = formatSignalValue(signal.value || 50);
                  const weight = ((signal.weight || 0) * 100).toFixed(0);
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between p-2.5 bg-muted/20 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-medium w-40 capitalize">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          Weight: {weight}%
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              (signal.value || 50) >= 55
                                ? "bg-emerald-500"
                                : (signal.value || 50) >= 45
                                  ? "bg-amber-400"
                                  : "bg-red-500"
                            }`}
                            style={{
                              width: `${Math.min(100, Math.max(0, signal.value || 50))}%`,
                            }}
                          />
                        </div>
                        <span
                          className={`text-xs font-medium w-24 text-right ${formatted.color}`}
                        >
                          {(signal.value || 50).toFixed(0)} — {formatted.label}
                        </span>
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          )}
        </div>

        {/* Time Horizon Reasoning */}
        {timeHorizon.reasoning && (
          <div className="text-xs text-muted-foreground italic p-3 bg-muted/10 rounded-lg border border-border/30">
            <span className="font-semibold text-foreground not-italic">
              Why {timeHorizon.horizon?.replace(/_/g, " ")}?
            </span>{" "}
            {timeHorizon.reasoning}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
