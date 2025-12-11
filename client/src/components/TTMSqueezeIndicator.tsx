import React from 'react';
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Activity, TrendingUp, TrendingDown, Circle } from "lucide-react";

interface TTMSqueezeIndicatorProps {
  squeeze_active: boolean;
  squeeze_bars: number;
  squeeze_momentum: number;
  squeeze_signal: 'long' | 'short' | 'active' | 'none';
  expected_move_pct?: number;
  compact?: boolean;
  realtime?: boolean; // If true, shows live indicator
  lastUpdate?: string; // Timestamp of last update
}

export function TTMSqueezeIndicator({
  squeeze_active,
  squeeze_bars,
  squeeze_momentum,
  squeeze_signal,
  expected_move_pct,
  compact = false,
  realtime = false,
  lastUpdate
}: TTMSqueezeIndicatorProps) {
  
  // Determine squeeze dot color (matches TradingView LazyBear)
  const getDotColor = () => {
    if (squeeze_active) {
      return "bg-red-500"; // Red = Squeeze ON (compression)
    } else {
      return "bg-green-500"; // Green = Squeeze OFF (expansion)
    }
  };

  // Determine momentum color
  const getMomentumColor = () => {
    if (squeeze_momentum > 0) {
      return squeeze_momentum > 5 ? "text-green-600" : "text-green-400";
    } else {
      return squeeze_momentum < -5 ? "text-red-600" : "text-red-400";
    }
  };

  // Get signal badge
  const getSignalBadge = () => {
    switch (squeeze_signal) {
      case 'long':
        return (
          <Badge className="bg-green-500 text-white border-0">
            <TrendingUp className="w-3 h-3 mr-1" />
            LONG FIRE
          </Badge>
        );
      case 'short':
        return (
          <Badge className="bg-red-500 text-white border-0">
            <TrendingDown className="w-3 h-3 mr-1" />
            SHORT FIRE
          </Badge>
        );
      case 'active':
        return (
          <Badge className="bg-yellow-500 text-white border-0">
            <Activity className="w-3 h-3 mr-1" />
            ACTIVE
          </Badge>
        );
      default:
        return null;
    }
  };

  // Compact view (for table rows)
  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-2">
              {/* Squeeze dot */}
              <div className={`w-3 h-3 rounded-full ${getDotColor()} animate-pulse`} />
              
              {/* Momentum value */}
              <span className={`text-sm font-semibold ${getMomentumColor()}`}>
                {squeeze_momentum > 0 ? '+' : ''}{squeeze_momentum.toFixed(1)}
              </span>
              
              {/* Signal badge (if active) */}
              {squeeze_signal !== 'none' && getSignalBadge()}
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <div className="space-y-1">
              <p className="font-semibold">
                {squeeze_active ? '🔴 Squeeze ON' : '🟢 Squeeze OFF'}
              </p>
              {squeeze_active && (
                <p className="text-xs">Compressed for {squeeze_bars} bars</p>
              )}
              <p className="text-xs">Momentum: {squeeze_momentum.toFixed(2)}</p>
              {expected_move_pct && (
                <p className="text-xs">Expected Move: {expected_move_pct.toFixed(2)}%</p>
              )}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // Full view (for detailed displays)
  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          <span className="font-semibold text-gray-900 dark:text-gray-100">TTM Squeeze</span>
          {realtime && (
            <Badge className="bg-green-500 text-white border-0 text-xs animate-pulse">
              <Circle className="w-2 h-2 mr-1 fill-current" />
              LIVE
            </Badge>
          )}
        </div>
        {getSignalBadge()}
      </div>

      {/* Squeeze State */}
      <div className="flex items-center gap-3">
        <div className={`w-4 h-4 rounded-full ${getDotColor()} animate-pulse`} />
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {squeeze_active ? 'Squeeze ON' : 'Squeeze OFF'}
          </p>
          {squeeze_active && squeeze_bars > 0 && (
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Compressed for {squeeze_bars} bar{squeeze_bars !== 1 ? 's' : ''}
              {squeeze_bars >= 3 && (
                <span className="ml-1 text-yellow-600 dark:text-yellow-400 font-semibold">
                  (High-probability setup!)
                </span>
              )}
            </p>
          )}
        </div>
      </div>

      {/* Momentum */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-600 dark:text-gray-400">Momentum</span>
        <span className={`text-lg font-bold ${getMomentumColor()}`}>
          {squeeze_momentum > 0 ? '+' : ''}{squeeze_momentum.toFixed(2)}
        </span>
      </div>

      {/* Expected Move */}
      {expected_move_pct !== undefined && (
        <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-800">
          <span className="text-sm text-gray-600 dark:text-gray-400">Expected Move</span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            ±{expected_move_pct.toFixed(2)}%
            {squeeze_active && squeeze_bars >= 3 && (
              <span className="ml-1 text-xs text-yellow-600 dark:text-yellow-400">
                (Amplified 1.5x)
              </span>
            )}
          </span>
        </div>
      )}

      {/* Interpretation */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-800">
        <p className="text-xs text-gray-600 dark:text-gray-400">
          {squeeze_active ? (
            <>
              <span className="font-semibold text-red-600 dark:text-red-400">Volatility compression detected.</span>
              {squeeze_bars >= 3 ? (
                <> Expect breakout soon. Consider ATM options.</>
              ) : (
                <> Monitor for 3+ bars for high-probability setup.</>
              )}
            </>
          ) : (
            <>
              <span className="font-semibold text-green-600 dark:text-green-400">Volatility expansion phase.</span>
              {squeeze_signal === 'long' ? (
                <> Bullish breakout confirmed. Strong upside momentum.</>
              ) : squeeze_signal === 'short' ? (
                <> Bearish breakout confirmed. Strong downside momentum.</>
              ) : (
                <> Normal volatility conditions. No squeeze setup.</>
              )}
            </>
          )}
        </p>
      </div>
    </div>
  );
}
