import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface RawDataDisplayProps {
  analysis: any;
}

// Safe number formatting helper - handles null, undefined, strings, and non-numbers
const safeFixed = (value: any, decimals: number = 2): string => {
  if (value === null || value === undefined) return 'N/A';
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (typeof num !== 'number' || isNaN(num)) return 'N/A';
  return num.toFixed(decimals);
};

export function RawDataDisplay({ analysis }: RawDataDisplayProps) {
  const technical = analysis.technical_analysis || {};
  const stochastic = analysis.stochastic_analysis || {};
  const garch = stochastic.garch_analysis || {};
  const monteCarlo = stochastic.monte_carlo || {};
  const positionSizing = analysis.position_sizing || {};
  const recommendation = analysis.recommendation || {};

  return (
    <Card className="bg-card/50 border-border/50">
      <CardHeader>
        <CardTitle className="text-primary">📊 Detailed Raw Data & Calculations</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Noise Filter & Data Quality Banner */}
        {analysis.noise_filter && !analysis.noise_filter.error && (
          <div className="mb-4 space-y-2">
            <div className="flex flex-wrap items-center gap-3 p-3 rounded-lg bg-muted/30 border border-border/50">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">Data Quality:</span>
                <span className={`text-sm font-bold px-2 py-0.5 rounded ${(analysis.noise_filter.data_quality?.score ?? 0) >= 80 ? 'bg-green-500/20 text-green-400' : (analysis.noise_filter.data_quality?.score ?? 0) >= 60 ? 'bg-yellow-500/20 text-yellow-400' : 'bg-red-500/20 text-red-400'}`}>
                  {analysis.noise_filter.data_quality?.score ?? 'N/A'}/100
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">Signal Strength:</span>
                <span className="text-sm text-muted-foreground">{analysis.noise_filter.signal_strength?.interpretation || 'N/A'}</span>
              </div>
              {analysis.confidence_adjustment && (
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold">Confidence Adj:</span>
                  <span className={`text-sm font-bold ${analysis.confidence_adjustment.noise_filter_delta > 0 ? 'text-green-400' : analysis.confidence_adjustment.noise_filter_delta < 0 ? 'text-red-400' : 'text-muted-foreground'}`}>
                    {analysis.confidence_adjustment.noise_filter_delta > 0 ? '+' : ''}{safeFixed(analysis.confidence_adjustment.noise_filter_delta, 1)} pts
                  </span>
                </div>
              )}
            </div>
            {/* Bias Warnings */}
            {analysis.noise_filter.bias_warnings?.filter((b: any) => b.severity !== 'INFO').length > 0 && (
              <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                <span className="text-sm font-semibold text-yellow-400">Bias Warnings:</span>
                <div className="mt-1 space-y-1">
                  {analysis.noise_filter.bias_warnings.filter((b: any) => b.severity !== 'INFO').map((b: any, i: number) => (
                    <div key={i} className="text-xs text-yellow-300/80">
                      <span className={`font-bold ${b.severity === 'HIGH' ? 'text-red-400' : 'text-yellow-400'}`}>[{b.severity}]</span> {b.note}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <Tabs defaultValue="technical" className="w-full">
          {/* Mobile: horizontally scrollable tab bar */}
          <div className="md:hidden overflow-x-auto -mx-2 px-2 pb-2 scrollbar-thin">
            <TabsList className="inline-flex w-max gap-1 h-auto p-1">
              <TabsTrigger value="technical" className="px-3 py-2 text-xs">Technical</TabsTrigger>
              <TabsTrigger value="advanced" className="px-3 py-2 text-xs">Pivot/Fib</TabsTrigger>
              <TabsTrigger value="candlestick" className="px-3 py-2 text-xs">Candles</TabsTrigger>
              <TabsTrigger value="fundamentals" className="px-3 py-2 text-xs">Cash Flow</TabsTrigger>
              <TabsTrigger value="garch" className="px-3 py-2 text-xs">GARCH</TabsTrigger>
              <TabsTrigger value="montecarlo" className="px-3 py-2 text-xs">Monte Carlo</TabsTrigger>
              <TabsTrigger value="position" className="px-3 py-2 text-xs">Position</TabsTrigger>
              <TabsTrigger value="market" className="px-3 py-2 text-xs">Market</TabsTrigger>
              <TabsTrigger value="darkpools" className="px-3 py-2 text-xs">Dark Pools</TabsTrigger>
              <TabsTrigger value="arima" className="px-3 py-2 text-xs">ARIMA</TabsTrigger>
              <TabsTrigger value="all" className="px-3 py-2 text-xs">All Data</TabsTrigger>
            </TabsList>
          </div>
          {/* Desktop: full grid */}
          <TabsList className="hidden md:grid w-full grid-cols-11">
            <TabsTrigger value="technical">Technical</TabsTrigger>
            <TabsTrigger value="advanced">Pivot/Fib</TabsTrigger>
            <TabsTrigger value="candlestick">Candlesticks</TabsTrigger>
            <TabsTrigger value="fundamentals">Cash Flow</TabsTrigger>
            <TabsTrigger value="garch">GARCH</TabsTrigger>
            <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
            <TabsTrigger value="position">Position</TabsTrigger>
            <TabsTrigger value="market">Market</TabsTrigger>
            <TabsTrigger value="darkpools">Dark Pools</TabsTrigger>
            <TabsTrigger value="arima">ARIMA</TabsTrigger>
            <TabsTrigger value="all">All Data</TabsTrigger>
          </TabsList>

          <TabsContent value="technical" className="space-y-4">
            {/* Composite Scores */}
            <div>
              <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Composite Scores</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <DataItem label="Technical Score" value={`${safeFixed(technical.technical_score)}/100`} />
                <DataItem label="Momentum Score" value={`${safeFixed(technical.momentum_score)}/100`} />
                <DataItem label="Trend Score" value={`${safeFixed(technical.trend_score)}/100`} />
                <DataItem label="Volatility Score" value={`${safeFixed(technical.volatility_score)}/100`} />
              </div>
            </div>

            {/* Comprehensive Technical Score */}
            {analysis.comprehensive_technicals?.intelligent_score && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Comprehensive Technical Rating</h4>
                <div className="p-3 rounded-lg border" style={{borderColor: analysis.comprehensive_technicals.intelligent_score.color === 'red' ? '#ef4444' : analysis.comprehensive_technicals.intelligent_score.color === 'green' ? '#22c55e' : '#eab308'}}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-lg font-bold" style={{color: analysis.comprehensive_technicals.intelligent_score.color === 'red' ? '#ef4444' : analysis.comprehensive_technicals.intelligent_score.color === 'green' ? '#22c55e' : '#eab308'}}>
                      {analysis.comprehensive_technicals.intelligent_score.score}/100 — {analysis.comprehensive_technicals.intelligent_score.rating?.replace(/_/g, ' ')}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {analysis.comprehensive_technicals.intelligent_score.bullish_count} Bullish · {analysis.comprehensive_technicals.intelligent_score.bearish_count} Bearish · {analysis.comprehensive_technicals.intelligent_score.neutral_count} Neutral
                    </span>
                  </div>
                  {analysis.comprehensive_technicals.intelligent_score.bullish_signals?.length > 0 && (
                    <div className="text-xs text-green-600 dark:text-green-400 mb-1">
                      <span className="font-medium">Bullish: </span>{analysis.comprehensive_technicals.intelligent_score.bullish_signals.join(' · ')}
                    </div>
                  )}
                  {analysis.comprehensive_technicals.intelligent_score.bearish_signals?.length > 0 && (
                    <div className="text-xs text-red-600 dark:text-red-400">
                      <span className="font-medium">Bearish: </span>{analysis.comprehensive_technicals.intelligent_score.bearish_signals.join(' · ')}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Core Indicators */}
            <div className="mt-4">
              <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Core Indicators</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <DataItem label="RSI (14)" value={safeFixed(technical.rsi)} />
                <DataItem label="MACD" value={safeFixed(technical.macd, 4)} />
                <DataItem label="MACD Signal" value={safeFixed(analysis.technical_indicators?.macd_signal, 4)} />
                <DataItem label="ADX" value={safeFixed(technical.adx)} />
                <DataItem label="ATR" value={`$${safeFixed(analysis.technical_indicators?.atr)}`} />
                <DataItem label="ATR %" value={analysis.technical_indicators?.atr_pct ? `${safeFixed(analysis.technical_indicators.atr_pct * 100)}%` : 'N/A'} />
                <DataItem label="Current Volatility" value={`${safeFixed(technical.current_volatility * 100)}%`} />
                <DataItem label="ROC (10)" value={safeFixed(analysis.technical_indicators?.roc)} />
              </div>
            </div>

            {/* Moving Averages */}
            <div className="mt-4">
              <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Moving Averages</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <DataItem label="SMA 20" value={`$${safeFixed(analysis.technical_indicators?.sma_20)}`} />
                <DataItem label="SMA 50" value={`$${safeFixed(analysis.technical_indicators?.sma_50)}`} />
                <DataItem label="SMA 200" value={`$${safeFixed(analysis.technical_indicators?.sma_200)}`} />
                <DataItem label="VWAP" value={`$${safeFixed(analysis.technical_indicators?.vwap)}`} />
              </div>
              {analysis.comprehensive_technicals?.golden_death_cross && (
                <div className="mt-2 p-2 bg-muted/30 rounded text-sm">
                  <span className={`font-semibold ${analysis.comprehensive_technicals.golden_death_cross.golden_cross ? 'text-green-600' : analysis.comprehensive_technicals.golden_death_cross.death_cross ? 'text-red-600' : 'text-muted-foreground'}`}>
                    {analysis.comprehensive_technicals.golden_death_cross.signal?.replace(/_/g, ' ') || 'No Cross'}
                  </span>
                  {analysis.comprehensive_technicals.golden_death_cross.days_since_cross && (
                    <span className="text-xs text-muted-foreground ml-2">({analysis.comprehensive_technicals.golden_death_cross.days_since_cross} days ago)</span>
                  )}
                </div>
              )}
            </div>

            {/* Bollinger Bands */}
            <div className="mt-4">
              <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Bollinger Bands</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                <DataItem label="Upper Band" value={`$${safeFixed(analysis.technical_indicators?.bb_upper)}`} />
                <DataItem label="Middle Band" value={`$${safeFixed(analysis.technical_indicators?.bb_middle)}`} />
                <DataItem label="Lower Band" value={`$${safeFixed(analysis.technical_indicators?.bb_lower)}`} />
                <DataItem label="VWAP Distance" value={analysis.technical_indicators?.vwap_distance ? `${safeFixed(analysis.technical_indicators.vwap_distance * 100)}%` : 'N/A'} />
              </div>
            </div>

            {/* Momentum Oscillators from comprehensive_technicals */}
            {analysis.comprehensive_technicals && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Momentum Oscillators</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {analysis.comprehensive_technicals.stochastic_rsi && (
                    <>
                      <DataItem label="Stoch RSI %K" value={safeFixed(analysis.comprehensive_technicals.stochastic_rsi.stoch_rsi_k)} />
                      <DataItem label="Stoch RSI %D" value={safeFixed(analysis.comprehensive_technicals.stochastic_rsi.stoch_rsi_d)} />
                    </>
                  )}
                  {analysis.comprehensive_technicals.williams_r && (
                    <DataItem label="Williams %R" value={safeFixed(analysis.comprehensive_technicals.williams_r.williams_r)} />
                  )}
                  {analysis.comprehensive_technicals.cci && (
                    <DataItem label="CCI (20)" value={safeFixed(analysis.comprehensive_technicals.cci.cci)} />
                  )}
                  {analysis.comprehensive_technicals.trix && (
                    <>
                      <DataItem label="TRIX" value={safeFixed(analysis.comprehensive_technicals.trix.trix, 4)} />
                      <DataItem label="TRIX Signal" value={safeFixed(analysis.comprehensive_technicals.trix.trix_signal, 4)} />
                    </>
                  )}
                  <DataItem label="MFI (14)" value={safeFixed(analysis.technical_indicators?.mfi || analysis.comprehensive_technicals.mfi?.mfi)} />
                </div>
                {/* Signal badges */}
                <div className="mt-2 flex flex-wrap gap-1">
                  {analysis.comprehensive_technicals.stochastic_rsi?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.stochastic_rsi.signal === 'OVERSOLD' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.stochastic_rsi.signal === 'OVERBOUGHT' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      StochRSI: {analysis.comprehensive_technicals.stochastic_rsi.signal}
                    </span>
                  )}
                  {analysis.comprehensive_technicals.williams_r?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.williams_r.signal === 'OVERSOLD' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.williams_r.signal === 'OVERBOUGHT' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      Williams: {analysis.comprehensive_technicals.williams_r.signal}
                    </span>
                  )}
                  {analysis.comprehensive_technicals.cci?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.cci.signal.includes('OVERSOLD') ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.cci.signal.includes('OVERBOUGHT') ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      CCI: {analysis.comprehensive_technicals.cci.signal}
                    </span>
                  )}
                  {analysis.comprehensive_technicals.mfi?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.mfi.signal === 'BULLISH' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.mfi.signal === 'BEARISH' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      MFI: {analysis.comprehensive_technicals.mfi.signal}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Trend Indicators from comprehensive_technicals */}
            {analysis.comprehensive_technicals && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Trend Indicators</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {analysis.comprehensive_technicals.aroon && (
                    <>
                      <DataItem label="Aroon Up" value={safeFixed(analysis.comprehensive_technicals.aroon.aroon_up)} />
                      <DataItem label="Aroon Down" value={safeFixed(analysis.comprehensive_technicals.aroon.aroon_down)} />
                      <DataItem label="Aroon Osc" value={safeFixed(analysis.comprehensive_technicals.aroon.aroon_oscillator)} />
                    </>
                  )}
                  {analysis.comprehensive_technicals.dmi && (
                    <>
                      <DataItem label="+DI" value={safeFixed(analysis.comprehensive_technicals.dmi.plus_di)} />
                      <DataItem label="-DI" value={safeFixed(analysis.comprehensive_technicals.dmi.minus_di)} />
                      <DataItem label="DMI ADX" value={safeFixed(analysis.comprehensive_technicals.dmi.adx)} />
                    </>
                  )}
                </div>
                {/* Signal badges */}
                <div className="mt-2 flex flex-wrap gap-1">
                  {analysis.comprehensive_technicals.aroon?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.aroon.signal === 'BULLISH' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.aroon.signal === 'BEARISH' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      Aroon: {analysis.comprehensive_technicals.aroon.signal}
                    </span>
                  )}
                  {analysis.comprehensive_technicals.dmi?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.dmi.signal.includes('BULLISH') ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.dmi.signal.includes('BEARISH') ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                      DMI: {analysis.comprehensive_technicals.dmi.signal?.replace(/_/g, ' ')}
                    </span>
                  )}
                  {analysis.comprehensive_technicals.dmi?.trend_strength && (
                    <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300">
                      Trend: {analysis.comprehensive_technicals.dmi.trend_strength}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Ichimoku Cloud */}
            {analysis.comprehensive_technicals?.ichimoku && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Ichimoku Cloud</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <DataItem label="Tenkan-sen" value={`$${safeFixed(analysis.comprehensive_technicals.ichimoku.tenkan_sen)}`} />
                  <DataItem label="Kijun-sen" value={`$${safeFixed(analysis.comprehensive_technicals.ichimoku.kijun_sen)}`} />
                  <DataItem label="Cloud Top" value={`$${safeFixed(analysis.comprehensive_technicals.ichimoku.cloud_top)}`} />
                  <DataItem label="Cloud Bottom" value={`$${safeFixed(analysis.comprehensive_technicals.ichimoku.cloud_bottom)}`} />
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.ichimoku.trend === 'BULLISH' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}>
                    {analysis.comprehensive_technicals.ichimoku.price_position?.replace(/_/g, ' ')}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.ichimoku.cloud_bullish ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}>
                    Cloud: {analysis.comprehensive_technicals.ichimoku.cloud_bullish ? 'Bullish' : 'Bearish'}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.ichimoku.tk_cross === 'BULLISH' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}>
                    TK Cross: {analysis.comprehensive_technicals.ichimoku.tk_cross}
                  </span>
                </div>
              </div>
            )}

            {/* Volume Indicators */}
            {analysis.comprehensive_technicals?.obv && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Volume Indicators</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <DataItem label="OBV" value={analysis.comprehensive_technicals.obv.obv ? (analysis.comprehensive_technicals.obv.obv > 1e6 ? `${safeFixed(analysis.comprehensive_technicals.obv.obv / 1e6)}M` : safeFixed(analysis.comprehensive_technicals.obv.obv, 0)) : 'N/A'} />
                  <DataItem label="OBV Trend" value={analysis.comprehensive_technicals.obv.obv_trend || 'N/A'} />
                  <DataItem label="OBV Divergence" value={analysis.comprehensive_technicals.obv.divergence || 'NONE'} />
                  {analysis.comprehensive_technicals.vwap && (
                    <DataItem label="VWAP Dev %" value={`${safeFixed(analysis.comprehensive_technicals.vwap.deviation_pct)}%`} />
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.obv.signal === 'RISING' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : analysis.comprehensive_technicals.obv.signal === 'FALLING' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
                    OBV: {analysis.comprehensive_technicals.obv.signal}
                  </span>
                  {analysis.comprehensive_technicals.vwap?.signal && (
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${analysis.comprehensive_technicals.vwap.signal.includes('ABOVE') ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}>
                      VWAP: {analysis.comprehensive_technicals.vwap.signal?.replace(/_/g, ' ')}
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* TTM Squeeze */}
            {analysis.technical_indicators?.ttm_squeeze_state !== undefined && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">TTM Squeeze</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  <DataItem label="Squeeze Active" value={analysis.technical_indicators.ttm_squeeze_state ? '🔴 YES' : '🟢 NO'} />
                  <DataItem label="Momentum" value={safeFixed(analysis.technical_indicators.ttm_squeeze_momentum, 4)} />
                  <DataItem label="Signal" value={analysis.technical_indicators.ttm_squeeze_signal?.toUpperCase() || 'N/A'} />
                  <DataItem label="Score Impact" value={safeFixed(analysis.technical_indicators.ttm_squeeze_score)} />
                </div>
              </div>
            )}

            {/* Candlestick Patterns from comprehensive_technicals */}
            {analysis.comprehensive_technicals?.candlestick?.patterns?.length > 0 && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2 text-cyan-900 dark:text-cyan-100">Candlestick Patterns Detected</h4>
                <div className="space-y-1">
                  {analysis.comprehensive_technicals.candlestick.patterns.map((p: any, i: number) => (
                    <div key={i} className={`text-sm p-2 rounded ${p.type?.includes('BULLISH') ? 'bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300' : 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300'}`}>
                      <span className="font-medium">{p.name?.replace(/_/g, ' ')}</span>
                      <span className="text-xs ml-2">({p.type?.replace(/_/g, ' ')}) — Strength: {p.strength}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Pivot Points & Fibonacci - R2 moved to ML/Quant page */}
          <TabsContent value="advanced" className="space-y-4">
            <div className="mb-4 p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
              <h4 className="font-semibold mb-2 text-indigo-900 dark:text-indigo-100">Pivot Points & Fibonacci Levels</h4>
              <p className="text-sm text-indigo-800 dark:text-indigo-200">
                Pivot points and Fibonacci levels identify key support/resistance zones. R2 Score is available on the ML/Quant page.
              </p>
            </div>
            
            {analysis.advanced_technicals ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="Trend Direction" value={analysis.advanced_technicals.r2_analysis?.trend_direction || analysis.advanced_technicals.trend_direction || 'N/A'} />
                  <DataItem label="Trend Strength" value={analysis.advanced_technicals.r2_analysis?.trend_strength || 'N/A'} />
                  <DataItem label="Current Price" value={`$${safeFixed(analysis.advanced_technicals.current_price)}`} />
                  <DataItem label="Nearest Support" value={`$${safeFixed(analysis.advanced_technicals.key_levels?.nearest_support?.price)}`} />
                </div>
                
                {/* Dynamic Guidance Based on Results */}
                {(() => {
                  const price = analysis.advanced_technicals.current_price;
                  const pivotStd = analysis.advanced_technicals.pivot_points?.standard;
                  const nearestSupport = analysis.advanced_technicals.key_levels?.nearest_support?.price;
                  const nearestResistance = analysis.advanced_technicals.key_levels?.nearest_resistance?.price;
                  
                  if (!price || !pivotStd) return null;
                  
                  const abovePivot = price > (pivotStd.pivot || 0);
                  const distToPivot = pivotStd.pivot ? ((price - pivotStd.pivot) / pivotStd.pivot * 100).toFixed(1) : 'N/A';
                  
                  return (
                    <div className={`p-3 rounded-lg border-l-4 ${
                      abovePivot ? 'bg-green-50 dark:bg-green-950 border-green-500' : 'bg-red-50 dark:bg-red-950 border-red-500'
                    }`}>
                      <p className="text-sm font-medium mb-1">
                        {abovePivot ? '✅ Price is ABOVE the daily pivot' : '⚠️ Price is BELOW the daily pivot'} ({distToPivot}% {abovePivot ? 'above' : 'below'})
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {abovePivot 
                          ? `Bullish bias. Watch R1 ($${safeFixed(pivotStd.r1)}) as next resistance. If rejected, pivot ($${safeFixed(pivotStd.pivot)}) becomes support.`
                          : `Bearish bias. Watch S1 ($${safeFixed(pivotStd.s1)}) as next support. If broken, S2 ($${safeFixed(pivotStd.s2)}) is the next target.`
                        }
                      </p>
                    </div>
                  );
                })()}
                
                {analysis.advanced_technicals.pivot_points && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Standard Pivot Points</h5>
                    <div className="grid grid-cols-3 md:grid-cols-7 gap-2">
                      <DataItem label="R3" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r3)}`} />
                      <DataItem label="R2" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r2)}`} />
                      <DataItem label="R1" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.r1)}`} />
                      <DataItem label="Pivot" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.pivot)}`} />
                      <DataItem label="S1" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s1)}`} />
                      <DataItem label="S2" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s2)}`} />
                      <DataItem label="S3" value={`$${safeFixed(analysis.advanced_technicals.pivot_points.standard?.s3)}`} />
                    </div>
                  </div>
                )}
                
                {analysis.advanced_technicals.fibonacci && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Fibonacci Levels</h5>
                    <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
                      {Object.entries(analysis.advanced_technicals.fibonacci.retracement || {}).map(([level, price]: [string, any]) => (
                        <DataItem key={level} label={`${level}`} value={`$${safeFixed(price)}`} />
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Education: Pivot Points & Fibonacci */}
                <div className="mt-6 p-4 bg-indigo-50 dark:bg-indigo-950 rounded-lg border border-indigo-200 dark:border-indigo-800">
                  <h5 className="font-semibold mb-3 text-indigo-900 dark:text-indigo-100">🎓 How to Use Pivot Points & Fibonacci</h5>
                  <div className="text-xs text-indigo-800 dark:text-indigo-200 space-y-3">
                    <div>
                      <p className="font-medium mb-1">Pivot Points (Calculated from Previous Day's High, Low, Close):</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>Pivot (P)</strong> = (High + Low + Close) / 3. This is the day's equilibrium. Price above = bullish bias, below = bearish bias.</li>
                        <li><strong>R1, R2, R3</strong> = Resistance levels. Each acts as a potential ceiling. R1 is the first target for longs; R3 is extreme overbought territory.</li>
                        <li><strong>S1, S2, S3</strong> = Support levels. Each acts as a potential floor. S1 is the first target for shorts; S3 is extreme oversold territory.</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-medium mb-1">Fibonacci Retracement (Based on Recent Swing High/Low):</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>23.6%</strong> = Shallow pullback. Strong trends often bounce here.</li>
                        <li><strong>38.2%</strong> = Moderate pullback. Most common retracement in healthy trends.</li>
                        <li><strong>50.0%</strong> = Not a true Fibonacci number, but widely watched. "Half-back" level.</li>
                        <li><strong>61.8%</strong> = The "Golden Ratio." Deep pullback but still within trend. Key decision level.</li>
                        <li><strong>78.6%</strong> = Very deep pullback. If this breaks, the trend may be reversing.</li>
                      </ul>
                    </div>
                    <div className="p-2 bg-indigo-100 dark:bg-indigo-900 rounded">
                      <p className="font-medium">💡 Pro Tips:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li>When pivot support/resistance <strong>aligns</strong> with a Fibonacci level, that zone is significantly stronger ("confluence").</li>
                        <li>Day traders use pivots for intraday entries/exits. Swing traders use Fibonacci for multi-day setups.</li>
                        <li>Volume confirmation at these levels increases reliability. A bounce on heavy volume is more trustworthy than one on thin volume.</li>
                        <li>If price gaps above R1 at open, R2 becomes the next target. If it gaps below S1, S2 is in play.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Advanced technicals data not available. Run a full analysis to see R2, Pivot, and Fibonacci levels.
              </div>
            )}
          </TabsContent>

          {/* Candlestick Patterns */}
          <TabsContent value="candlestick" className="space-y-4">
            <div className="mb-4 p-4 bg-amber-50 dark:bg-amber-950 rounded-lg border border-amber-200 dark:border-amber-800">
              <h4 className="font-semibold mb-2 text-amber-900 dark:text-amber-100">Candlestick Pattern Detection</h4>
              <p className="text-sm text-amber-800 dark:text-amber-200">
                Expert-level pattern recognition including Doji, Hammer, Engulfing, Morning/Evening Star, Ichimoku Cloud, and more.
              </p>
              <p className="text-xs text-amber-700 dark:text-amber-300 mt-2">
                ⚠️ <strong>Note:</strong> This page shows two types of analysis: <strong>Algorithmic Detection</strong> (mathematical formulas on OHLC data) and <strong>Vision AI Analysis</strong> (AI visually analyzing chart images). They may show different patterns because they analyze different timeframes and use different methods. When signals conflict, consider both perspectives and use additional confirmation.
              </p>
            </div>
            
            {analysis.candlestick_patterns ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="Overall Bias" value={analysis.candlestick_patterns.overall_bias || 'N/A'} />
                  <DataItem label="Confidence" value={analysis.candlestick_patterns.recommendation?.confidence || 'N/A'} />
                  <DataItem label="Patterns Found" value={analysis.candlestick_patterns.patterns_found || analysis.candlestick_patterns.patterns?.length || 0} />
                  <DataItem label="Pattern Strength" value={analysis.candlestick_patterns.recommendation?.action || 'N/A'} />
                </div>
                
                {(analysis.candlestick_patterns.patterns?.length > 0 || analysis.candlestick_patterns.patterns_detected?.length > 0) && (
                  <div className="mt-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-blue-600 dark:text-blue-400">📊</span>
                      <h5 className="font-semibold">Algorithmic Pattern Detection</h5>
                      <span className="text-xs bg-blue-100 dark:bg-blue-900 px-2 py-0.5 rounded text-blue-700 dark:text-blue-300">OHLC Data Analysis</span>
                    </div>
                    <p className="text-xs text-muted-foreground mb-3">Patterns detected using mathematical formulas on recent price data (Open, High, Low, Close). Analyzes the last 100 trading days.</p>
                    <div className="space-y-2">
                      {(analysis.candlestick_patterns.patterns || analysis.candlestick_patterns.patterns_detected || []).map((pattern: any, idx: number) => (
                        <div key={idx} className={`p-3 rounded-lg border ${pattern.type?.includes('BULLISH') ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800' : pattern.type?.includes('BEARISH') ? 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800' : 'bg-muted/20 border-border'}`}>
                          <div className="flex justify-between items-center">
                            <span className="font-medium">{pattern.pattern || pattern.name}</span>
                            <span className={`text-sm ${pattern.type?.includes('BULLISH') ? 'text-green-600' : pattern.type?.includes('BEARISH') ? 'text-red-600' : 'text-muted-foreground'}`}>
                              {pattern.type || pattern.signal?.toUpperCase()} ({pattern.reliability || 'N/A'})
                            </span>
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {pattern.date && <span className="mr-2">Date: {pattern.date}</span>}
                            {pattern.price && <span className="mr-2">Price: ${pattern.price.toFixed(2)}</span>}
                          </div>
                          {pattern.description && <p className="text-xs text-muted-foreground mt-1">{pattern.description}</p>}
                          {pattern.action && <p className="text-xs font-medium mt-1">{pattern.action}</p>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {analysis.candlestick_patterns.ichimoku && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">☁️ Ichimoku Cloud Analysis</h5>
                    
                    {/* Overall Signal Badge */}
                    <div className="flex items-center gap-3 mb-3">
                      <span className={`px-3 py-1.5 rounded-full text-sm font-bold ${
                        analysis.candlestick_patterns.ichimoku.overall_signal?.includes('BULLISH') ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                        analysis.candlestick_patterns.ichimoku.overall_signal?.includes('BEARISH') ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' :
                        'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300'
                      }`}>
                        {analysis.candlestick_patterns.ichimoku.overall_signal || 'N/A'}
                      </span>
                      {analysis.candlestick_patterns.ichimoku.interpretation && (
                        <span className="text-sm text-muted-foreground italic">{analysis.candlestick_patterns.ichimoku.interpretation}</span>
                      )}
                    </div>
                    
                    {/* 5 Ichimoku Components */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-3">
                      <DataItem label="Tenkan-sen (9)" value={analysis.candlestick_patterns.ichimoku.tenkan_sen ? `$${safeFixed(analysis.candlestick_patterns.ichimoku.tenkan_sen)}` : 'N/A'} />
                      <DataItem label="Kijun-sen (26)" value={analysis.candlestick_patterns.ichimoku.kijun_sen ? `$${safeFixed(analysis.candlestick_patterns.ichimoku.kijun_sen)}` : 'N/A'} />
                      <DataItem label="Senkou A" value={analysis.candlestick_patterns.ichimoku.senkou_span_a ? `$${safeFixed(analysis.candlestick_patterns.ichimoku.senkou_span_a)}` : 'N/A'} />
                      <DataItem label="Senkou B" value={analysis.candlestick_patterns.ichimoku.senkou_span_b ? `$${safeFixed(analysis.candlestick_patterns.ichimoku.senkou_span_b)}` : 'N/A'} />
                      <DataItem label="Chikou Span" value={analysis.candlestick_patterns.ichimoku.chikou_signal || 'N/A'} />
                    </div>
                    
                    {/* Signal Components */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                      <DataItem label="TK Cross" value={analysis.candlestick_patterns.ichimoku.tk_cross || 'N/A'} />
                      <DataItem label="Cloud Color" value={analysis.candlestick_patterns.ichimoku.cloud_color || 'N/A'} />
                      <DataItem label="Price vs Cloud" value={analysis.candlestick_patterns.ichimoku.cloud_position || analysis.candlestick_patterns.ichimoku.price_vs_cloud || 'N/A'} />
                      <DataItem label="Cloud Range" value={analysis.candlestick_patterns.ichimoku.cloud_top && analysis.candlestick_patterns.ichimoku.cloud_bottom ? `$${safeFixed(analysis.candlestick_patterns.ichimoku.cloud_bottom)} - $${safeFixed(analysis.candlestick_patterns.ichimoku.cloud_top)}` : 'N/A'} />
                    </div>
                    
                    {/* Ichimoku Education */}
                    <div className="mt-3 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
                      <p className="font-medium mb-1">📖 Understanding Ichimoku Cloud (5 Components):</p>
                      <ul className="space-y-1 ml-2">
                        <li><span className="font-medium">Tenkan-sen (Conversion):</span> (9-period High + Low) / 2. Acts as fast signal line. When above Kijun = bullish momentum.</li>
                        <li><span className="font-medium">Kijun-sen (Base):</span> (26-period High + Low) / 2. Acts as support/resistance and trend confirmation.</li>
                        <li><span className="font-medium">Senkou A (Leading A):</span> (Tenkan + Kijun) / 2, plotted 26 periods ahead. Forms one edge of the cloud.</li>
                        <li><span className="font-medium">Senkou B (Leading B):</span> (52-period High + Low) / 2, plotted 26 periods ahead. Forms the other edge.</li>
                        <li><span className="font-medium">Chikou Span (Lagging):</span> Current close plotted 26 periods back. Confirms trend when above/below past prices.</li>
                      </ul>
                      <p className="mt-2 font-medium">Signal Scoring (4 signals):</p>
                      <ul className="space-y-1 ml-2">
                        <li>3-4 bullish signals = <span className="text-green-600 dark:text-green-400 font-medium">STRONG BULLISH</span></li>
                        <li>2 bullish signals = <span className="text-green-500 dark:text-green-400">BULLISH</span></li>
                        <li>0-1 bullish signals = <span className="text-red-500 dark:text-red-400">BEARISH</span></li>
                      </ul>
                      <p className="mt-2 italic">Tip: The cloud itself acts as dynamic support/resistance. Thicker clouds = stronger S/R zones. Price entering the cloud often signals consolidation before the next move.</p>
                    </div>
                  </div>
                )}
                
                {/* Golden Cross / Death Cross Analysis - Only show when a cross is actually detected */}
                {analysis.candlestick_patterns.golden_death_cross && (() => {
                  const gdc = analysis.candlestick_patterns.golden_death_cross;
                  const hasGoldenCross = gdc.recent_golden_cross === true;
                  const hasDeathCross = gdc.recent_death_cross === true;
                  const hasCrossEvent = hasGoldenCross || hasDeathCross;
                  
                  // Only render this section if a cross was recently detected
                  if (!hasCrossEvent) return null;
                  
                  return (
                    <div className="mt-4">
                      <h5 className="font-semibold mb-2">⚔️ {hasGoldenCross ? 'Golden Cross' : 'Death Cross'} Detected!</h5>
                      <div className={`p-4 rounded-lg border-2 ${hasGoldenCross ? 'bg-green-50 dark:bg-green-950 border-green-400 dark:border-green-600' : 'bg-red-50 dark:bg-red-950 border-red-400 dark:border-red-600'}`}>
                        
                        {/* Alert Banner */}
                        <div className={`flex items-center gap-2 mb-3 p-2 rounded-lg ${hasGoldenCross ? 'bg-green-100 dark:bg-green-900' : 'bg-red-100 dark:bg-red-900'}`}>
                          <span className="text-2xl">{hasGoldenCross ? '🔔' : '⚠️'}</span>
                          <div>
                            <p className={`font-bold ${hasGoldenCross ? 'text-green-800 dark:text-green-200' : 'text-red-800 dark:text-red-200'}`}>
                              {hasGoldenCross ? 'GOLDEN CROSS' : 'DEATH CROSS'} — {gdc.days_since_cross ? `${gdc.days_since_cross} day(s) ago` : 'Recent'}
                            </p>
                            <p className={`text-xs ${hasGoldenCross ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'}`}>
                              {hasGoldenCross ? 'The 50-day SMA crossed ABOVE the 200-day SMA — a major bullish trend signal' : 'The 50-day SMA crossed BELOW the 200-day SMA — a major bearish trend signal'}
                            </p>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
                          <DataItem label="50-Day SMA" value={`$${safeFixed(gdc.sma_50)}`} />
                          <DataItem label="200-Day SMA" value={`$${safeFixed(gdc.sma_200)}`} />
                          <DataItem label="Signal" value={gdc.signal || 'N/A'} />
                          <DataItem label="Days Since Cross" value={gdc.days_since_cross || 'N/A'} />
                        </div>
                        
                        {gdc.explanation && (
                          <div className="text-sm text-muted-foreground whitespace-pre-line mt-2">{gdc.explanation}</div>
                        )}
                      </div>
                      
                      {/* Education */}
                      <div className="mt-3 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
                        <p className="font-medium mb-1">📖 Understanding Golden/Death Cross:</p>
                        <ul className="space-y-1 ml-2">
                          <li><span className="font-medium">Golden Cross:</span> 50-day SMA crosses ABOVE 200-day SMA. Historically preceded average gains of 15-25% over 12 months.</li>
                          <li><span className="font-medium">Death Cross:</span> 50-day SMA crosses BELOW 200-day SMA. Historically preceded average declines of 10-20%.</li>
                          <li><span className="font-medium">Confirmation:</span> Volume should increase on the cross day. Low-volume crosses are less reliable.</li>
                          <li><span className="font-medium">Caveat:</span> These are lagging indicators. By the time the cross occurs, a significant portion of the move may have already happened. Always confirm with other indicators.</li>
                        </ul>
                      </div>
                    </div>
                  );
                })()}
                
                {/* ====== COMPREHENSIVE SIGNAL CONFLICT ANALYSIS ====== */}
                {analysis.candlestick_conflict_analysis && !analysis.candlestick_conflict_analysis.error && (() => {
                  const cca = analysis.candlestick_conflict_analysis;
                  const summary = cca.summary;
                  if (!summary) return null;

                  const agreementColors: Record<string, string> = {
                    green: 'bg-green-100 dark:bg-green-900/50 border-green-400 dark:border-green-600 text-green-800 dark:text-green-200',
                    red: 'bg-red-100 dark:bg-red-900/50 border-red-400 dark:border-red-600 text-red-800 dark:text-red-200',
                    yellow: 'bg-yellow-100 dark:bg-yellow-900/50 border-yellow-400 dark:border-yellow-600 text-yellow-800 dark:text-yellow-200',
                    gray: 'bg-muted/30 border-border text-muted-foreground',
                  };

                  const dirColors: Record<string, string> = {
                    BULLISH: 'text-green-600 dark:text-green-400',
                    BEARISH: 'text-red-600 dark:text-red-400',
                    NEUTRAL: 'text-yellow-600 dark:text-yellow-400',
                    UNAVAILABLE: 'text-muted-foreground',
                  };

                  const dirBg: Record<string, string> = {
                    BULLISH: 'bg-green-100 dark:bg-green-900/40',
                    BEARISH: 'bg-red-100 dark:bg-red-900/40',
                    NEUTRAL: 'bg-yellow-100 dark:bg-yellow-900/40',
                  };

                  const sevColors: Record<string, string> = {
                    HIGH: 'bg-red-500/20 text-red-400 border-red-500/40',
                    MODERATE: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
                    LOW: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
                  };

                  return (
                    <div className="mt-4 space-y-3">
                      {/* Agreement Banner */}
                      <div className={`p-4 rounded-lg border-2 ${agreementColors[summary.agreement_color] || agreementColors.gray}`}>
                        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                          <div className="flex items-center gap-2">
                            <span className="text-xl">
                              {summary.agreement_color === 'green' ? '✅' : summary.agreement_color === 'red' ? '🔴' : '⚠️'}
                            </span>
                            <h5 className="font-bold text-base">{summary.agreement_label}</h5>
                          </div>
                          {summary.conflict_count > 0 && (
                            <span className={`text-xs font-bold px-2 py-1 rounded border ${sevColors[summary.max_severity] || sevColors.LOW}`}>
                              {summary.max_severity} SEVERITY
                            </span>
                          )}
                        </div>

                        {/* Signal Breakdown Table */}
                        {summary.signal_breakdown?.length > 0 && (
                          <div className="mb-3">
                            <h6 className="text-xs font-semibold mb-2 uppercase tracking-wider opacity-70">Signal Breakdown ({summary.total_methods} Methods)</h6>
                            <div className="grid gap-1.5">
                              {summary.signal_breakdown.map((sig: any, idx: number) => (
                                <div key={idx} className={`flex items-center justify-between p-2 rounded ${dirBg[sig.direction] || 'bg-muted/20'}`}>
                                  <div className="flex items-center gap-2 min-w-0">
                                    <span className={`text-xs font-bold w-16 text-center px-1.5 py-0.5 rounded ${dirColors[sig.direction]}`}>
                                      {sig.direction}
                                    </span>
                                    <span className="text-sm font-medium truncate">{sig.label}</span>
                                  </div>
                                  <div className="flex items-center gap-2 flex-shrink-0">
                                    <span className="text-xs text-muted-foreground">{sig.raw_signal}</span>
                                    <span className="text-xs opacity-50">({(sig.weight * 100).toFixed(0)}% wt)</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Weighted Consensus Bar */}
                        {(summary.weighted_bullish_pct > 0 || summary.weighted_bearish_pct > 0) && (
                          <div className="mb-2">
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-green-600 dark:text-green-400 font-medium">Bullish {summary.weighted_bullish_pct}%</span>
                              <span className="text-red-600 dark:text-red-400 font-medium">Bearish {summary.weighted_bearish_pct}%</span>
                            </div>
                            <div className="h-2.5 rounded-full bg-muted/40 overflow-hidden flex">
                              <div className="bg-green-500 h-full transition-all" style={{ width: `${summary.weighted_bullish_pct}%` }} />
                              <div className="bg-gray-400 h-full transition-all flex-1" />
                              <div className="bg-red-500 h-full transition-all" style={{ width: `${summary.weighted_bearish_pct}%` }} />
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Individual Conflict Explanations */}
                      {cca.has_conflicts && summary.conflicts?.length > 0 && (
                        <div className="space-y-2">
                          <h6 className="text-sm font-semibold flex items-center gap-2">
                            <span>🔍</span>
                            Why Do These Methods Disagree? ({summary.conflict_count} conflict{summary.conflict_count > 1 ? 's' : ''})
                          </h6>
                          {summary.conflicts.map((c: any, idx: number) => (
                            <div key={idx} className={`p-3 rounded-lg border ${c.is_ok ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-700' : 'bg-orange-50 dark:bg-orange-950/30 border-orange-300 dark:border-orange-700'}`}>
                              {/* Conflict Header */}
                              <div className="flex flex-wrap items-center gap-2 mb-2">
                                <span className={`text-xs font-bold px-2 py-0.5 rounded ${dirColors[c.signal_a]} ${dirBg[c.signal_a]}`}>
                                  {c.method_a_label}: {c.signal_a}
                                </span>
                                <span className="text-xs text-muted-foreground">vs</span>
                                <span className={`text-xs font-bold px-2 py-0.5 rounded ${dirColors[c.signal_b]} ${dirBg[c.signal_b]}`}>
                                  {c.method_b_label}: {c.signal_b}
                                </span>
                                <span className={`text-xs font-bold px-1.5 py-0.5 rounded border ml-auto ${sevColors[c.severity] || sevColors.LOW}`}>
                                  {c.severity}
                                </span>
                              </div>

                              {/* WHY explanation */}
                              <p className="text-sm leading-relaxed mb-2">{c.why}</p>

                              {/* Is it OK? Assessment */}
                              <div className={`p-2 rounded text-xs ${c.is_ok ? 'bg-blue-100 dark:bg-blue-900/30' : 'bg-orange-100 dark:bg-orange-900/30'}`}>
                                <span className="font-bold">{c.is_ok ? '✅ Assessment:' : '⚠️ Concern:'}</span>{' '}
                                {c.assessment}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* No Conflicts - All Clear */}
                      {!cca.has_conflicts && summary.total_methods >= 2 && (
                        <div className="p-3 rounded-lg bg-green-50 dark:bg-green-950/30 border border-green-300 dark:border-green-700">
                          <p className="text-sm text-green-700 dark:text-green-300">
                            <span className="font-bold">✅ No Conflicts Detected.</span>{' '}
                            All {summary.total_methods} chart reading methods are in agreement.
                            {summary.bullish_count === summary.total_methods && ' All methods read BULLISH — strong signal alignment.'}
                            {summary.bearish_count === summary.total_methods && ' All methods read BEARISH — strong signal alignment.'}
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })()}
                
                {/* Vision AI Chart Analysis */}
                {analysis.candlestick_patterns.vision_ai_analysis && (
                  <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-800">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-purple-600 dark:text-purple-400">🤖</span>
                      <h5 className="font-semibold text-purple-900 dark:text-purple-100">Vision AI Chart Analysis</h5>
                      <span className="text-xs bg-purple-100 dark:bg-purple-900 px-2 py-0.5 rounded text-purple-700 dark:text-purple-300">
                        {analysis.candlestick_patterns.vision_ai_analysis.chart_source} + {analysis.candlestick_patterns.vision_ai_analysis.ai_model}
                      </span>
                    </div>
                    <p className="text-xs text-purple-700 dark:text-purple-300 mb-3">AI visually analyzes chart images like a human trader would. Now analyzes MULTIPLE timeframes: Daily (Finviz), 5-Minute Intraday (Polygon.io), and Weekly (Finviz).</p>
                    
                    {/* Multi-Timeframe Alignment Banner */}
                    {analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment && (
                      <div className={`mb-4 p-3 rounded-lg border ${
                        analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'ALL_BULLISH' ? 'bg-green-50 dark:bg-green-950 border-green-300 dark:border-green-700' :
                        analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'ALL_BEARISH' ? 'bg-red-50 dark:bg-red-950 border-red-300 dark:border-red-700' :
                        analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'CONFLICTING' ? 'bg-amber-50 dark:bg-amber-950 border-amber-300 dark:border-amber-700' :
                        'bg-gray-50 dark:bg-gray-900 border-gray-300'
                      }`}>
                        <div className="flex items-center gap-2 mb-1">
                          <span>{analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'ALL_BULLISH' ? '✅' : analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'ALL_BEARISH' ? '🔴' : analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment === 'CONFLICTING' ? '⚠️' : 'ℹ️'}</span>
                          <span className="font-semibold text-sm">Multi-Timeframe Alignment: {analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment?.replace(/_/g, ' ')}</span>
                          <span className="text-xs text-muted-foreground ml-2">({(analysis.candlestick_patterns.vision_ai_analysis.timeframes_analyzed || []).join(', ')})</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{analysis.candlestick_patterns.vision_ai_analysis.timeframe_alignment_note}</p>
                      </div>
                    )}
                    
                    {/* Intraday Summary (5-min Polygon data) */}
                    {analysis.candlestick_patterns.vision_ai_analysis.intraday_summary && (
                      <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                          <span>📊</span>
                          <h6 className="font-semibold text-blue-900 dark:text-blue-100 text-sm">5-Minute Intraday Snapshot (Polygon.io)</h6>
                          {analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.data_date && (
                            <span className={`text-xs px-2 py-0.5 rounded-full ${
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.is_live 
                                ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' 
                                : 'bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-300'
                            }`}>
                              {analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.is_live 
                                ? '🟢 Live' 
                                : `📅 Data from ${analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.data_day} ${analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.data_date} (market closed)`}
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                          <div>
                            <span className="text-muted-foreground text-xs">Current: </span>
                            <span className="font-semibold">${analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.current_price}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">VWAP: </span>
                            <span className={`font-semibold ${
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.vwap_signal === 'ABOVE_VWAP' ? 'text-green-600' : 'text-red-600'
                            }`}>${analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.vwap}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">EMA 9/21: </span>
                            <span className={`font-semibold ${
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.ema_signal === 'BULLISH' ? 'text-green-600' : 
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.ema_signal === 'BEARISH' ? 'text-red-600' : 'text-yellow-600'
                            }`}>{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.ema_signal}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">Intraday Trend: </span>
                            <span className={`font-semibold ${
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.intraday_trend === 'UPTREND' ? 'text-green-600' : 
                              analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.intraday_trend === 'DOWNTREND' ? 'text-red-600' : 'text-yellow-600'
                            }`}>{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.intraday_trend}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">Session Change: </span>
                            <span className={`font-semibold ${(analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.change_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {(analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.change_pct || 0) >= 0 ? '+' : ''}{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.change_pct}%
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">Range: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.range_pct}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">Volume Trend: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.volume_trend}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground text-xs">Bars: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.intraday_summary.bars}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Per-Timeframe Breakdowns */}
                    {analysis.candlestick_patterns.vision_ai_analysis.multi_timeframe && Object.keys(analysis.candlestick_patterns.vision_ai_analysis.multi_timeframe).length > 1 && (
                      <div className="mb-4">
                        <h6 className="font-medium text-sm mb-2">Per-Timeframe Vision AI Readings</h6>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {Object.entries(analysis.candlestick_patterns.vision_ai_analysis.multi_timeframe).map(([tfKey, tfData]: [string, any]) => (
                            <div key={tfKey} className={`p-3 rounded-lg border ${
                              (tfData.overall_bias || '').toUpperCase().includes('BULL') ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800' :
                              (tfData.overall_bias || '').toUpperCase().includes('BEAR') ? 'bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800' :
                              'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-800'
                            }`}>
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-semibold text-sm">{tfData.label || tfKey}</span>
                                <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                                  (tfData.overall_bias || '').toUpperCase().includes('BULL') ? 'bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200' :
                                  (tfData.overall_bias || '').toUpperCase().includes('BEAR') ? 'bg-red-200 dark:bg-red-800 text-red-800 dark:text-red-200' :
                                  'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
                                }`}>{tfData.overall_bias || 'N/A'}</span>
                              </div>
                              <div className="text-xs space-y-1 text-muted-foreground">
                                <div>Source: {tfData.source}</div>
                                <div>Trend: {tfData.trend?.direction || 'N/A'} ({tfData.trend?.strength || 'N/A'})</div>
                                {tfData.recommendation?.signal && (
                                  <div>Signal: <span className={`font-semibold ${
                                    tfData.recommendation.signal === 'BUY' ? 'text-green-600' :
                                    tfData.recommendation.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                                  }`}>{tfData.recommendation.signal}</span> ({tfData.recommendation.confidence}% conf)</div>
                                )}
                                {tfData.candlestick_patterns?.length > 0 && (
                                  <div>Patterns: {tfData.candlestick_patterns.slice(0, 3).map((p: any) => p.name).join(', ')}{tfData.candlestick_patterns.length > 3 ? ` +${tfData.candlestick_patterns.length - 3} more` : ''}</div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <DataItem label="AI Bias" value={analysis.candlestick_patterns.vision_ai_analysis.overall_bias || 'N/A'} />
                      <DataItem label="Trend Direction" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.direction || 'N/A'} />
                      <DataItem label="Trend Strength" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.strength || 'N/A'} />
                      <DataItem label="Momentum" value={analysis.candlestick_patterns.vision_ai_analysis.trend?.momentum || 'N/A'} />
                    </div>
                    
                    {/* AI Recommendation */}
                    {analysis.candlestick_patterns.vision_ai_analysis.recommendation && (
                      <div className="mb-4 p-3 bg-white dark:bg-gray-800 rounded border">
                        <h6 className="font-medium mb-2">AI Trading Recommendation</h6>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">Signal: </span>
                            <span className={`font-semibold ${
                              analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal === 'BUY' ? 'text-green-600' :
                              analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal === 'SELL' ? 'text-red-600' : 'text-yellow-600'
                            }`}>
                              {analysis.candlestick_patterns.vision_ai_analysis.recommendation.signal}
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Confidence: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.recommendation.confidence}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">R/R: </span>
                            <span className="font-semibold">{analysis.candlestick_patterns.vision_ai_analysis.recommendation.risk_reward}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Stop Loss: </span>
                            <span className="font-semibold">${analysis.candlestick_patterns.vision_ai_analysis.recommendation.stop_loss}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Support/Resistance from AI */}
                    {(analysis.candlestick_patterns.vision_ai_analysis.support_levels?.length > 0 || analysis.candlestick_patterns.vision_ai_analysis.resistance_levels?.length > 0) && (
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <h6 className="font-medium text-green-700 dark:text-green-400 mb-1">AI Support Levels</h6>
                          <div className="text-sm space-y-1">
                            {analysis.candlestick_patterns.vision_ai_analysis.support_levels?.map((level: number, idx: number) => (
                              <div key={idx} className="text-green-600">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h6 className="font-medium text-red-700 dark:text-red-400 mb-1">AI Resistance Levels</h6>
                          <div className="text-sm space-y-1">
                            {analysis.candlestick_patterns.vision_ai_analysis.resistance_levels?.map((level: number, idx: number) => (
                              <div key={idx} className="text-red-600">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* AI Key Observations */}
                    {analysis.candlestick_patterns.vision_ai_analysis.key_observations?.length > 0 && (
                      <div>
                        <h6 className="font-medium mb-2">AI Key Observations</h6>
                        <ul className="text-sm space-y-1">
                          {analysis.candlestick_patterns.vision_ai_analysis.key_observations.map((obs: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-2">
                              <span className="text-purple-500">•</span>
                              <span>{obs}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* AI Detected Patterns */}
                    {analysis.candlestick_patterns.vision_ai_analysis.candlestick_patterns?.length > 0 && (
                      <div className="mt-4">
                        <h6 className="font-medium mb-2">AI Detected Candlestick Patterns</h6>
                        <div className="space-y-2">
                          {analysis.candlestick_patterns.vision_ai_analysis.candlestick_patterns.map((pattern: any, idx: number) => (
                            <div key={idx} className={`p-2 rounded border ${
                              pattern.signal === 'bullish' ? 'bg-green-50 dark:bg-green-950 border-green-300' :
                              pattern.signal === 'bearish' ? 'bg-red-50 dark:bg-red-950 border-red-300' : 'bg-gray-50 dark:bg-gray-900 border-gray-300'
                            }`}>
                              <div className="flex justify-between">
                                <span className="font-medium">{pattern.name}</span>
                                <span className={`text-xs ${
                                  pattern.signal === 'bullish' ? 'text-green-600' : pattern.signal === 'bearish' ? 'text-red-600' : 'text-gray-600'
                                }`}>
                                  {pattern.signal?.toUpperCase()} • {pattern.reliability}
                                </span>
                              </div>
                              {pattern.description && <p className="text-xs text-muted-foreground mt-1">{pattern.description}</p>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* EXA AI Real-Time Web Intelligence */}
                {analysis.exa_intelligence && !analysis.exa_intelligence.error && (
                  <div className="mt-6 p-4 bg-cyan-50 dark:bg-cyan-950 rounded-lg border border-cyan-200 dark:border-cyan-800">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-cyan-600 dark:text-cyan-400">🌐</span>
                      <h5 className="font-semibold text-cyan-900 dark:text-cyan-100">EXA AI Real-Time Web Intelligence</h5>
                      <span className="text-xs bg-cyan-100 dark:bg-cyan-900 px-2 py-0.5 rounded text-cyan-700 dark:text-cyan-300">
                        Neural Search • {analysis.exa_intelligence.timestamp ? new Date(analysis.exa_intelligence.timestamp).toLocaleDateString() : 'Live'}
                      </span>
                    </div>
                    <p className="text-xs text-cyan-700 dark:text-cyan-300 mb-4">
                      EXA AI performs neural web search across financial sites (TradingView, StockCharts, Finviz, SeekingAlpha, etc.) to find real-time candlestick chart analysis, expert opinions, and pattern mentions from professional analysts.
                    </p>
                    
                    {/* Synthesis / Key Findings */}
                    {analysis.exa_intelligence.candlestick_analysis?.synthesis && (() => {
                      const syn = analysis.exa_intelligence.candlestick_analysis.synthesis;
                      return (
                        <div className="space-y-3">
                          {/* Consensus & Confidence */}
                          <div className="flex items-center gap-3 flex-wrap">
                            <span className={`px-3 py-1.5 rounded-full text-sm font-bold ${
                              syn.consensus_trend === 'bullish' ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                              syn.consensus_trend === 'bearish' ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' :
                              'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300'
                            }`}>
                              Web Consensus: {syn.consensus_trend?.toUpperCase() || 'N/A'}
                            </span>
                            <span className={`px-3 py-1.5 rounded-full text-sm font-bold ${
                              syn.expert_consensus === 'bullish' ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                              syn.expert_consensus === 'bearish' ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' :
                              'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300'
                            }`}>
                              Expert Consensus: {syn.expert_consensus?.toUpperCase() || 'N/A'}
                            </span>
                            <span className="text-sm text-muted-foreground">
                              Confidence: {syn.confidence || 0}% • {syn.total_sources_analyzed || 0} sources
                            </span>
                          </div>
                          
                          {/* Key Findings */}
                          {syn.key_findings?.length > 0 && (
                            <div className="p-3 bg-white dark:bg-gray-800 rounded border">
                              <h6 className="font-medium mb-2 text-sm">Key Findings from Web Analysis</h6>
                              <ul className="text-sm space-y-1">
                                {syn.key_findings.map((finding: string, idx: number) => (
                                  <li key={idx} className="flex items-start gap-2">
                                    <span className="text-cyan-500 mt-0.5">▸</span>
                                    <span>{finding}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {/* Candlestick Patterns from Web */}
                          {syn.candlestick_patterns_detected?.length > 0 && (
                            <div>
                              <h6 className="font-medium mb-2 text-sm">Candlestick Patterns Mentioned by Analysts</h6>
                              <div className="flex flex-wrap gap-2">
                                {syn.candlestick_patterns_detected.map((p: any, idx: number) => (
                                  <span key={idx} className="px-2 py-1 bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 rounded text-xs font-medium">
                                    {p.pattern} ({p.mentions} mention{p.mentions > 1 ? 's' : ''})
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Chart Patterns from Web */}
                          {syn.chart_patterns_detected?.length > 0 && (
                            <div>
                              <h6 className="font-medium mb-2 text-sm">Chart Patterns Identified by Analysts</h6>
                              <div className="flex flex-wrap gap-2">
                                {syn.chart_patterns_detected.map((p: any, idx: number) => (
                                  <span key={idx} className={`px-2 py-1 rounded text-xs font-medium ${
                                    p.pattern?.toLowerCase().includes('bullish') ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' :
                                    p.pattern?.toLowerCase().includes('bearish') ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200' :
                                    'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
                                  }`}>
                                    {p.pattern} ({p.mentions})
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Support / Resistance from Web */}
                          {(syn.support_levels?.length > 0 || syn.resistance_levels?.length > 0) && (
                            <div className="grid grid-cols-2 gap-4">
                              {syn.support_levels?.length > 0 && (
                                <div>
                                  <h6 className="font-medium text-green-700 dark:text-green-400 mb-1 text-sm">Web-Sourced Support</h6>
                                  <div className="text-sm space-y-1">
                                    {syn.support_levels.slice(0, 5).map((level: number, idx: number) => (
                                      <div key={idx} className="text-green-600 dark:text-green-400">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {syn.resistance_levels?.length > 0 && (
                                <div>
                                  <h6 className="font-medium text-red-700 dark:text-red-400 mb-1 text-sm">Web-Sourced Resistance</h6>
                                  <div className="text-sm space-y-1">
                                    {syn.resistance_levels.slice(0, 5).map((level: number, idx: number) => (
                                      <div key={idx} className="text-red-600 dark:text-red-400">${typeof level === 'number' ? level.toFixed(2) : level}</div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })()}
                    
                    {/* Market Sentiment from EXA */}
                    {analysis.exa_intelligence.market_sentiment?.overall_sentiment && (
                      <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border">
                        <h6 className="font-medium mb-2 text-sm">Market Sentiment (from News Analysis)</h6>
                        <div className="flex items-center gap-4 mb-2">
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                            analysis.exa_intelligence.market_sentiment.overall_sentiment.bias === 'bullish' ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300' :
                            analysis.exa_intelligence.market_sentiment.overall_sentiment.bias === 'bearish' ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300' :
                            'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
                          }`}>
                            {analysis.exa_intelligence.market_sentiment.overall_sentiment.bias?.toUpperCase()}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            Bullish: {analysis.exa_intelligence.market_sentiment.overall_sentiment.bullish_pct}% • 
                            Bearish: {analysis.exa_intelligence.market_sentiment.overall_sentiment.bearish_pct}% • 
                            Neutral: {analysis.exa_intelligence.market_sentiment.overall_sentiment.neutral_pct}%
                          </span>
                        </div>
                        {analysis.exa_intelligence.market_sentiment.key_headlines?.length > 0 && (
                          <div className="text-xs text-muted-foreground space-y-1">
                            {analysis.exa_intelligence.market_sentiment.key_headlines.slice(0, 3).map((h: string, idx: number) => (
                              <p key={idx}>• {h}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Insider / Dark Pool from EXA */}
                    {analysis.exa_intelligence.insider_dark_pool && (
                      <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border">
                        <h6 className="font-medium mb-2 text-sm">Insider & Dark Pool Intelligence (Web Sources)</h6>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                          <div className="p-2 bg-muted/30 rounded">
                            <div className="text-muted-foreground">Insider Reports</div>
                            <div className="font-semibold">{analysis.exa_intelligence.insider_dark_pool.insider_activity?.length || 0}</div>
                          </div>
                          <div className="p-2 bg-muted/30 rounded">
                            <div className="text-muted-foreground">Dark Pool Mentions</div>
                            <div className="font-semibold">{analysis.exa_intelligence.insider_dark_pool.dark_pool_mentions?.length || 0}</div>
                          </div>
                          <div className="p-2 bg-muted/30 rounded">
                            <div className="text-muted-foreground">Unusual Options</div>
                            <div className="font-semibold">{analysis.exa_intelligence.insider_dark_pool.unusual_options?.length || 0}</div>
                          </div>
                          <div className="p-2 bg-muted/30 rounded">
                            <div className="text-muted-foreground">Congress Trades</div>
                            <div className="font-semibold">{analysis.exa_intelligence.insider_dark_pool.congress_trades?.length || 0}</div>
                          </div>
                        </div>
                        {analysis.exa_intelligence.insider_dark_pool.insider_activity?.length > 0 && (
                          <div className="mt-2 text-xs text-muted-foreground">
                            <p className="font-medium">Recent Insider Activity:</p>
                            {analysis.exa_intelligence.insider_dark_pool.insider_activity.slice(0, 3).map((item: any, idx: number) => (
                              <p key={idx} className="ml-2">• {item.title}</p>
                            ))}
                          </div>
                        )}
                        {analysis.exa_intelligence.insider_dark_pool.congress_trades?.length > 0 && (
                          <div className="mt-2 text-xs text-muted-foreground">
                            <p className="font-medium">Congress Trades:</p>
                            {analysis.exa_intelligence.insider_dark_pool.congress_trades.slice(0, 3).map((item: any, idx: number) => (
                              <p key={idx} className="ml-2">• {item.title}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Source URLs */}
                    {analysis.exa_intelligence.candlestick_analysis?.source_urls?.length > 0 && (
                      <div className="mt-3 text-xs text-muted-foreground">
                        <details>
                          <summary className="cursor-pointer font-medium">View Source URLs ({analysis.exa_intelligence.candlestick_analysis.source_urls.length})</summary>
                          <div className="mt-1 space-y-1 ml-2">
                            {analysis.exa_intelligence.candlestick_analysis.source_urls.slice(0, 10).map((url: string, idx: number) => (
                              <a key={idx} href={url} target="_blank" rel="noopener noreferrer" className="block text-cyan-600 dark:text-cyan-400 hover:underline truncate">{url}</a>
                            ))}
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Candlestick pattern data not available. Run a full analysis to see pattern detection.
              </div>
            )}
          </TabsContent>

          {/* Enhanced Fundamentals - Cash Flow */}
          <TabsContent value="fundamentals" className="space-y-4">
            <div className="mb-4 p-4 bg-emerald-50 dark:bg-emerald-950 rounded-lg border border-emerald-200 dark:border-emerald-800">
              <h4 className="font-semibold mb-2 text-emerald-900 dark:text-emerald-100">Enhanced Cash Flow & Valuation Metrics</h4>
              <p className="text-sm text-emerald-800 dark:text-emerald-200">
                Deep fundamental analysis including PE, PEG, GARP scoring, FCF, EBITDA/EV, Free Float, and Liquidity metrics.
              </p>
            </div>
            
            {analysis.enhanced_fundamentals ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DataItem label="P/E Ratio" value={safeFixed(analysis.enhanced_fundamentals.valuation?.pe_ratio)} />
                  <DataItem label="Forward P/E" value={safeFixed(analysis.enhanced_fundamentals.valuation?.forward_pe)} />
                  <DataItem label="PEG Ratio" value={safeFixed(analysis.enhanced_fundamentals.valuation?.peg_ratio)} />
                  <DataItem label="EV/EBITDA" value={safeFixed(analysis.enhanced_fundamentals.valuation?.ev_to_ebitda)} />
                </div>
                
                {analysis.enhanced_fundamentals.cash_flow && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Cash Flow Metrics</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Free Cash Flow" value={analysis.enhanced_fundamentals.cash_flow.fcf_formatted || 'N/A'} />
                      <DataItem label="Op Cash Flow" value={analysis.enhanced_fundamentals.cash_flow.ocf_formatted || 'N/A'} />
                      <DataItem label="EBITDA" value={analysis.enhanced_fundamentals.cash_flow.ebitda_formatted || 'N/A'} />
                      <DataItem label="FCF/OCF Ratio" value={analysis.enhanced_fundamentals.cash_flow.fcf_to_ocf_ratio != null ? safeFixed(analysis.enhanced_fundamentals.cash_flow.fcf_to_ocf_ratio) : 'N/A'} />
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2">
                      <DataItem label="FCF Yield" value={analysis.enhanced_fundamentals.cash_flow.fcf_yield_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.cash_flow.fcf_yield_pct)}%` : 'N/A'} />
                      <DataItem label="FCF Margin" value={analysis.enhanced_fundamentals.cash_flow.fcf_margin_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.cash_flow.fcf_margin_pct)}%` : 'N/A'} />
                      <DataItem label="FCF Positive" value={analysis.enhanced_fundamentals.cash_flow.fcf_positive ? '✅ Yes' : '❌ No'} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.profitability && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Profitability Metrics</h5>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      <DataItem label="Gross Margin" value={analysis.enhanced_fundamentals.profitability.gross_margin_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.profitability.gross_margin_pct)}%` : 'N/A'} />
                      <DataItem label="Operating Margin" value={analysis.enhanced_fundamentals.profitability.operating_margin_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.profitability.operating_margin_pct)}%` : 'N/A'} />
                      <DataItem label="Profit Margin" value={analysis.enhanced_fundamentals.profitability.profit_margin_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.profitability.profit_margin_pct)}%` : 'N/A'} />
                      <DataItem label="ROE" value={analysis.enhanced_fundamentals.profitability.roe_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.profitability.roe_pct)}%` : 'N/A'} />
                      <DataItem label="ROA" value={analysis.enhanced_fundamentals.profitability.roa_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.profitability.roa_pct)}%` : 'N/A'} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.growth && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Growth Metrics</h5>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      <DataItem label="Earnings Growth" value={analysis.enhanced_fundamentals.growth.earnings_growth_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.growth.earnings_growth_pct)}%` : 'N/A'} />
                      <DataItem label="Revenue Growth" value={analysis.enhanced_fundamentals.growth.revenue_growth_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.growth.revenue_growth_pct)}%` : 'N/A'} />
                      <DataItem label="Quarterly Earnings Growth" value={analysis.enhanced_fundamentals.growth.earnings_quarterly_growth_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.growth.earnings_quarterly_growth_pct)}%` : 'N/A'} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.garp_analysis && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">GARP Analysis (Growth at Reasonable Price)</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="GARP Score" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.score, 1)}/100`} />
                      <DataItem label="GARP Signal" value={analysis.enhanced_fundamentals.garp_analysis.verdict || 'N/A'} />
                      <DataItem label="Growth Rate" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.growth_rate_used, 1)}%`} />
                      <DataItem label="Value Score" value={`${safeFixed(analysis.enhanced_fundamentals.garp_analysis.value_score, 1)}/100`} />
                    </div>
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.liquidity_analysis && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Liquidity Analysis</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Risk Level" value={analysis.enhanced_fundamentals.liquidity_analysis.risk_level || 'N/A'} />
                      <DataItem label="Current Ratio" value={safeFixed(analysis.enhanced_fundamentals.liquidity_analysis.current_ratio)} />
                      <DataItem label="Quick Ratio" value={safeFixed(analysis.enhanced_fundamentals.liquidity_analysis.quick_ratio)} />
                      <DataItem label="Free Float %" value={analysis.enhanced_fundamentals.share_structure?.free_float_pct ? `${safeFixed(analysis.enhanced_fundamentals.share_structure.free_float_pct)}%` : 'N/A'} />
                    </div>
                    {analysis.enhanced_fundamentals.liquidity_analysis.signals?.length > 0 && (
                      <div className="mt-2 text-sm text-muted-foreground">
                        {analysis.enhanced_fundamentals.liquidity_analysis.signals.map((s: string, i: number) => (
                          <div key={i}>• {s}</div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                
                {analysis.enhanced_fundamentals.debt_liquidity && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">Financial Health & Debt</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Total Debt" value={analysis.enhanced_fundamentals.debt_liquidity.total_debt_formatted || 'N/A'} />
                      <DataItem label="Total Cash" value={analysis.enhanced_fundamentals.debt_liquidity.total_cash_formatted || 'N/A'} />
                      <DataItem label="Net Debt" value={analysis.enhanced_fundamentals.debt_liquidity.net_debt_formatted || 'N/A'} />
                      <DataItem label="Debt/Equity" value={safeFixed(analysis.enhanced_fundamentals.debt_liquidity.debt_to_equity)} />
                      <DataItem label="Current Ratio" value={safeFixed(analysis.enhanced_fundamentals.debt_liquidity.current_ratio)} />
                      <DataItem label="Quick Ratio" value={safeFixed(analysis.enhanced_fundamentals.debt_liquidity.quick_ratio)} />
                      <DataItem label="Interest Coverage" value={safeFixed(analysis.enhanced_fundamentals.debt_liquidity.interest_coverage)} />
                      {analysis.enhanced_fundamentals.altman_z_score?.score && (
                        <DataItem label="Altman Z-Score" value={`${safeFixed(analysis.enhanced_fundamentals.altman_z_score.score)} (${analysis.enhanced_fundamentals.altman_z_score.rating})`} />
                      )}
                    </div>
                  </div>
                )}
                
                {/* NEW: Analyst Ratings */}
                {analysis.enhanced_fundamentals.analyst_ratings && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">📊 Analyst Ratings</h5>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      <DataItem label="Consensus" value={analysis.enhanced_fundamentals.analyst_ratings.consensus_rating || 'N/A'} />
                      <DataItem label="Strong Buy" value={analysis.enhanced_fundamentals.analyst_ratings.strong_buy || 0} />
                      <DataItem label="Buy" value={analysis.enhanced_fundamentals.analyst_ratings.buy || 0} />
                      <DataItem label="Hold" value={analysis.enhanced_fundamentals.analyst_ratings.hold || 0} />
                      <DataItem label="Sell" value={analysis.enhanced_fundamentals.analyst_ratings.sell || 0} />
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                      <DataItem label="Target Price" value={analysis.enhanced_fundamentals.analyst_ratings.target_price ? `$${safeFixed(analysis.enhanced_fundamentals.analyst_ratings.target_price)}` : 'N/A'} />
                      <DataItem label="Target High" value={analysis.enhanced_fundamentals.analyst_ratings.target_high ? `$${safeFixed(analysis.enhanced_fundamentals.analyst_ratings.target_high)}` : 'N/A'} />
                      <DataItem label="Target Low" value={analysis.enhanced_fundamentals.analyst_ratings.target_low ? `$${safeFixed(analysis.enhanced_fundamentals.analyst_ratings.target_low)}` : 'N/A'} />
                      <DataItem label="# Analysts" value={analysis.enhanced_fundamentals.analyst_ratings.number_of_analysts || 0} />
                    </div>
                  </div>
                )}
                
                {/* NEW: Insider Activity */}
                {analysis.enhanced_fundamentals.insider_activity && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">🏢 Insider Activity (90 Days)</h5>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      <DataItem label="Net Activity" value={analysis.enhanced_fundamentals.insider_activity.net_insider_activity || 'NEUTRAL'} />
                      <DataItem label="Total Buys" value={analysis.enhanced_fundamentals.insider_activity.total_buys_90d || 0} />
                      <DataItem label="Total Sells" value={analysis.enhanced_fundamentals.insider_activity.total_sells_90d || 0} />
                      <DataItem label="Buy Value" value={analysis.enhanced_fundamentals.insider_activity.buy_value_90d ? `$${safeFixed(analysis.enhanced_fundamentals.insider_activity.buy_value_90d / 1e6)}M` : '$0'} />
                      <DataItem label="Sell Value" value={analysis.enhanced_fundamentals.insider_activity.sell_value_90d ? `$${safeFixed(analysis.enhanced_fundamentals.insider_activity.sell_value_90d / 1e6)}M` : '$0'} />
                    </div>
                    {analysis.enhanced_fundamentals.insider_activity.recent_transactions?.length > 0 && (
                      <div className="mt-2 text-xs">
                        <p className="text-muted-foreground">Recent: {analysis.enhanced_fundamentals.insider_activity.recent_transactions.slice(0, 3).map((t: any) => `${t.name?.split(' ')[0] || 'Insider'} ${t.type} ${t.shares?.toLocaleString() || 0} shares`).join('; ')}</p>
                      </div>
                    )}
                  </div>
                )}
                
                {/* NEW: Dividends */}
                {analysis.enhanced_fundamentals.dividends && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">💰 Dividend Information</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Dividend Yield (TTM)" value={analysis.enhanced_fundamentals.dividends.dividend_yield_pct ? `${safeFixed(analysis.enhanced_fundamentals.dividends.dividend_yield_pct)}%` : 'N/A'} />
                      <DataItem label="Annual Dividend/Share" value={analysis.enhanced_fundamentals.dividends.annual_dividend != null ? `$${safeFixed(analysis.enhanced_fundamentals.dividends.annual_dividend)}` : 'N/A'} />
                      <DataItem label="Payout Ratio" value={analysis.enhanced_fundamentals.dividends.payout_ratio_pct ? `${safeFixed(analysis.enhanced_fundamentals.dividends.payout_ratio_pct)}%` : 'N/A'} />
                      <DataItem label="Ex-Dividend Date" value={analysis.enhanced_fundamentals.dividends.ex_dividend_date || 'N/A'} />
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                      <DataItem label="3Y Div Growth" value={analysis.enhanced_fundamentals.dividends.dividend_growth_3yr != null ? `${safeFixed(analysis.enhanced_fundamentals.dividends.dividend_growth_3yr)}%` : 'N/A'} />
                      <DataItem label="5Y Div Growth" value={analysis.enhanced_fundamentals.dividends.dividend_growth_5yr != null ? `${safeFixed(analysis.enhanced_fundamentals.dividends.dividend_growth_5yr)}%` : 'N/A'} />
                      <DataItem label="Est. Dividend" value={analysis.enhanced_fundamentals.dividends.dividend_est_per_share != null ? `$${safeFixed(analysis.enhanced_fundamentals.dividends.dividend_est_per_share)}` : 'N/A'} />
                      <DataItem label="Est. Yield" value={analysis.enhanced_fundamentals.dividends.dividend_est_yield_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.dividends.dividend_est_yield_pct)}%` : 'N/A'} />
                    </div>
                    {analysis.enhanced_fundamentals.dividends.dividend_history?.length > 0 && (
                      <div className="mt-2 text-xs">
                        <p className="text-muted-foreground">Recent Payments: {analysis.enhanced_fundamentals.dividends.dividend_history.slice(0, 4).map((d: any) => `${d.date}: $${d.dividend?.toFixed(2) || 'N/A'}`).join(' • ')}</p>
                      </div>
                    )}
                  </div>
                )}
                
                {/* NEW: Earnings */}
                {analysis.enhanced_fundamentals.earnings && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">📈 Earnings Data</h5>
                    {analysis.enhanced_fundamentals.earnings.eps_estimates && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        <DataItem label="Est. EPS Avg" value={analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_avg ? `$${safeFixed(analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_avg)}` : 'N/A'} />
                        <DataItem label="Est. EPS High" value={analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_high ? `$${safeFixed(analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_high)}` : 'N/A'} />
                        <DataItem label="Est. EPS Low" value={analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_low ? `$${safeFixed(analysis.enhanced_fundamentals.earnings.eps_estimates.estimated_eps_low)}` : 'N/A'} />
                        <DataItem label="# Analysts" value={analysis.enhanced_fundamentals.earnings.eps_estimates.number_of_analysts || 'N/A'} />
                      </div>
                    )}
                    {analysis.enhanced_fundamentals.earnings.earnings_surprises?.length > 0 && (
                      <div className="mt-2 text-xs">
                        <p className="text-muted-foreground">Recent Surprises: {analysis.enhanced_fundamentals.earnings.earnings_surprises.slice(0, 4).map((s: any) => `${s.date}: ${s.surprise > 0 ? '+' : ''}${safeFixed(s.surprise, 3)}`).join(', ')}</p>
                      </div>
                    )}
                  </div>
                )}
                
                {/* NEW: Share Structure & Free Float */}
                {analysis.enhanced_fundamentals.share_structure && (
                  <div className="mt-4">
                    <h5 className="font-semibold mb-2">📊 Share Structure & Free Float</h5>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      <DataItem label="Shares Outstanding" value={analysis.enhanced_fundamentals.share_structure.shares_outstanding_formatted?.replace('$', '') || 'N/A'} />
                      <DataItem label="Float Shares" value={analysis.enhanced_fundamentals.share_structure.float_shares_formatted?.replace('$', '') || 'N/A'} />
                      <DataItem label="Free Float %" value={analysis.enhanced_fundamentals.share_structure.free_float_pct ? `${safeFixed(analysis.enhanced_fundamentals.share_structure.free_float_pct)}%` : 'N/A'} />
                      <DataItem label="Insider Ownership" value={analysis.enhanced_fundamentals.share_structure.insider_ownership_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.share_structure.insider_ownership_pct)}%` : 'N/A'} />
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                      <DataItem label="Institutional %" value={analysis.enhanced_fundamentals.share_structure.institutional_ownership_pct != null ? `${safeFixed(analysis.enhanced_fundamentals.share_structure.institutional_ownership_pct)}%` : 'N/A'} />
                      <DataItem label="Shares Short" value={analysis.enhanced_fundamentals.share_structure.shares_short_formatted?.replace('$', '') || 'N/A'} />
                      <DataItem label="Short % of Float" value={analysis.enhanced_fundamentals.share_structure.short_pct_of_float != null ? `${safeFixed(analysis.enhanced_fundamentals.share_structure.short_pct_of_float)}%` : 'N/A'} />
                      <DataItem label="Short Ratio (Days)" value={analysis.enhanced_fundamentals.share_structure.short_ratio_days != null ? safeFixed(analysis.enhanced_fundamentals.share_structure.short_ratio_days) : 'N/A'} />
                    </div>
                    {/* Insider & Institutional Activity */}
                    {(analysis.enhanced_fundamentals.share_structure.insider_transactions_pct != null || analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct != null) && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                        <DataItem 
                          label="Insider Transactions" 
                          value={analysis.enhanced_fundamentals.share_structure.insider_transactions_pct != null 
                            ? <span className={analysis.enhanced_fundamentals.share_structure.insider_transactions_pct > 0 ? 'text-green-500' : analysis.enhanced_fundamentals.share_structure.insider_transactions_pct < 0 ? 'text-red-500' : ''}>
                                {analysis.enhanced_fundamentals.share_structure.insider_transactions_pct > 0 ? '+' : ''}{safeFixed(analysis.enhanced_fundamentals.share_structure.insider_transactions_pct)}%
                              </span>
                            : 'N/A'} 
                        />
                        <DataItem 
                          label="Institutional Transactions" 
                          value={analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct != null 
                            ? <span className={analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct > 0 ? 'text-green-500' : analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct < 0 ? 'text-red-500' : ''}>
                                {analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct > 0 ? '+' : ''}{safeFixed(analysis.enhanced_fundamentals.share_structure.institutional_transactions_pct)}%
                              </span>
                            : 'N/A'} 
                        />
                      </div>
                    )}
                    {analysis.enhanced_fundamentals.share_structure.data_source && (
                      <p className="text-xs text-muted-foreground mt-2">Source: {analysis.enhanced_fundamentals.share_structure.data_source === 'finviz' ? 'Finviz (scraped)' : analysis.enhanced_fundamentals.share_structure.data_source}</p>
                    )}
                  </div>
                )}
                
                {/* CAGR Analysis Section */}
                {analysis.enhanced_fundamentals.financial_trends && (
                  <div className="mt-4 space-y-4">
                    {/* CAGR Summary */}
                    {analysis.enhanced_fundamentals.financial_trends.cagr_summary && (
                      <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-semibold text-blue-900 dark:text-blue-100">📈 CAGR Summary</h5>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                            analysis.enhanced_fundamentals.financial_trends.cagr_summary.assessment === 'Exceptional Growth' ? 'bg-green-500 text-white' :
                            analysis.enhanced_fundamentals.financial_trends.cagr_summary.assessment === 'Strong Growth' ? 'bg-emerald-500 text-white' :
                            analysis.enhanced_fundamentals.financial_trends.cagr_summary.assessment === 'Moderate Growth' ? 'bg-yellow-500 text-black' :
                            analysis.enhanced_fundamentals.financial_trends.cagr_summary.assessment === 'Slow Growth' ? 'bg-orange-500 text-white' :
                            'bg-red-500 text-white'
                          }`}>
                            {analysis.enhanced_fundamentals.financial_trends.cagr_summary.assessment}
                          </span>
                        </div>
                        <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                          {safeFixed(analysis.enhanced_fundamentals.financial_trends.cagr_summary.average_5yr_cagr)}% <span className="text-sm font-normal">Avg 5-Year CAGR</span>
                        </p>
                      </div>
                    )}
                    
                    {/* Revenue CAGR */}
                    <div>
                      <h5 className="font-semibold mb-2">💰 Revenue CAGR</h5>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">1-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.revenue_1yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.revenue_1yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.revenue_1yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">3-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.revenue_3yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.revenue_3yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.revenue_3yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">5-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.revenue_5yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.revenue_5yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.revenue_5yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Earnings CAGR */}
                    <div>
                      <h5 className="font-semibold mb-2">📊 Earnings CAGR</h5>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">1-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.earnings_1yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.earnings_1yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.earnings_1yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">3-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.earnings_3yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.earnings_3yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.earnings_3yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">5-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.earnings_5yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.earnings_5yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.earnings_5yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Stock Price CAGR */}
                    <div>
                      <h5 className="font-semibold mb-2">📈 Stock Price CAGR</h5>
                      <div className="grid grid-cols-4 gap-2">
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">1-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.stock_price_1yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.stock_price_1yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.stock_price_1yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">3-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.stock_price_3yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.stock_price_3yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.stock_price_3yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">5-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.stock_price_5yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.stock_price_5yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.stock_price_5yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                        <div className="p-2 bg-muted rounded text-center">
                          <p className="text-xs text-muted-foreground">10-Year</p>
                          <p className={`font-bold ${(analysis.enhanced_fundamentals.financial_trends.stock_price_10yr_cagr || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {analysis.enhanced_fundamentals.financial_trends.stock_price_10yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.stock_price_10yr_cagr)}%` : 'N/A'}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    {/* Other CAGR Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      <DataItem 
                        label="FCF 3yr CAGR" 
                        value={analysis.enhanced_fundamentals.financial_trends.fcf_3yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.fcf_3yr_cagr)}%` : 'N/A'} 
                      />
                      <DataItem 
                        label="FCF 5yr CAGR" 
                        value={analysis.enhanced_fundamentals.financial_trends.fcf_5yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.fcf_5yr_cagr)}%` : 'N/A'} 
                      />
                      <DataItem 
                        label="Dividend 5yr CAGR" 
                        value={analysis.enhanced_fundamentals.financial_trends.dividend_5yr_cagr !== null ? `${safeFixed(analysis.enhanced_fundamentals.financial_trends.dividend_5yr_cagr)}%` : 'N/A'} 
                      />
                    </div>
                    
                    {/* Revenue Trend Table */}
                    {analysis.enhanced_fundamentals.financial_trends.revenue_trend?.length > 0 && (
                      <div>
                        <h5 className="font-semibold mb-2">📋 Historical Revenue & Margins</h5>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b">
                                <th className="text-left p-2">Year</th>
                                <th className="text-right p-2">Revenue</th>
                                <th className="text-right p-2">Net Income</th>
                                <th className="text-right p-2">Gross Margin</th>
                                <th className="text-right p-2">Op Margin</th>
                                <th className="text-right p-2">Net Margin</th>
                              </tr>
                            </thead>
                            <tbody>
                              {analysis.enhanced_fundamentals.financial_trends.revenue_trend.slice(0, 5).map((item: any, idx: number) => (
                                <tr key={idx} className="border-b border-muted">
                                  <td className="p-2 font-medium">{item.year}</td>
                                  <td className="p-2 text-right">{item.revenue ? `$${(item.revenue / 1e9).toFixed(1)}B` : 'N/A'}</td>
                                  <td className="p-2 text-right">{item.net_income ? `$${(item.net_income / 1e9).toFixed(1)}B` : 'N/A'}</td>
                                  <td className="p-2 text-right">{item.gross_margin ? `${safeFixed(item.gross_margin)}%` : 'N/A'}</td>
                                  <td className="p-2 text-right">{item.operating_margin ? `${safeFixed(item.operating_margin)}%` : 'N/A'}</td>
                                  <td className="p-2 text-right">{item.net_margin ? `${safeFixed(item.net_margin)}%` : 'N/A'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Dynamic Cash Flow Guidance Based on Results */}
                {(() => {
                  const val = analysis.enhanced_fundamentals.valuation;
                  const cf = analysis.enhanced_fundamentals.cash_flow;
                  const garp = analysis.enhanced_fundamentals.garp_analysis;
                  const health = analysis.enhanced_fundamentals.financial_health;
                  
                  if (!val && !cf) return null;
                  
                  const pe = val?.pe_ratio;
                  const fcfYield = cf?.fcf_yield_pct;
                  const zScore = health?.altman_z_score;
                  const garpScore = garp?.score;
                  
                  // Determine overall fundamental health
                  let healthSignal = 'neutral';
                  let healthColor = 'yellow';
                  if (zScore > 3 && fcfYield > 3 && pe < 25) { healthSignal = 'strong'; healthColor = 'green'; }
                  else if (zScore > 2.5 && fcfYield > 1) { healthSignal = 'healthy'; healthColor = 'green'; }
                  else if (zScore < 1.8 || fcfYield < 0) { healthSignal = 'concerning'; healthColor = 'red'; }
                  
                  return (
                    <div className={`mt-4 p-4 rounded-lg border-l-4 ${
                      healthColor === 'green' ? 'bg-green-50 dark:bg-green-950 border-green-500' :
                      healthColor === 'red' ? 'bg-red-50 dark:bg-red-950 border-red-500' :
                      'bg-yellow-50 dark:bg-yellow-950 border-yellow-500'
                    }`}>
                      <h5 className="font-semibold mb-2 text-sm">🎯 What These Fundamentals Tell You</h5>
                      <div className="text-xs space-y-2">
                        {pe && (
                          <p>
                            <strong>Valuation:</strong> P/E of {safeFixed(pe)} means you're paying ${safeFixed(pe)} for every $1 of earnings.
                            {pe < 15 ? ' This is relatively cheap — either a value opportunity or the market sees problems ahead.' :
                             pe < 25 ? ' This is fairly valued for a growth company.' :
                             pe < 40 ? ' This is premium-priced. The market expects strong future growth to justify this multiple.' :
                             ' This is very expensive. Only justified if the company is growing earnings 25%+ annually.'}
                          </p>
                        )}
                        {fcfYield && (
                          <p>
                            <strong>Cash Flow:</strong> FCF Yield of {safeFixed(fcfYield)}%.
                            {fcfYield > 5 ? ' Excellent. The company generates significant free cash relative to its price. This is a cash machine.' :
                             fcfYield > 2 ? ' Healthy cash generation. The company can fund growth, buybacks, or dividends.' :
                             fcfYield > 0 ? ' Modest cash flow. The company is reinvesting heavily, which is fine for growth but watch for sustainability.' :
                             ' Negative FCF. The company is burning cash. This is acceptable for high-growth companies but risky for mature ones.'}
                          </p>
                        )}
                        {zScore && (
                          <p>
                            <strong>Bankruptcy Risk:</strong> Altman Z-Score of {safeFixed(zScore)}.
                            {zScore > 3 ? ' Safe zone. Very low bankruptcy risk.' :
                             zScore > 1.8 ? ' Gray zone. Not immediately concerning but worth monitoring.' :
                             ' Distress zone. Elevated bankruptcy risk. Proceed with extreme caution.'}
                          </p>
                        )}
                        {garpScore && (
                          <p>
                            <strong>GARP (Growth at Reasonable Price):</strong> Score of {safeFixed(garpScore, 0)}/100.
                            {garpScore > 70 ? ' Strong GARP candidate — good growth at a reasonable valuation.' :
                             garpScore > 40 ? ' Moderate GARP score. Either growth is slowing or valuation is stretched.' :
                             ' Weak GARP score. The growth-to-price ratio is unfavorable.'}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })()}
                
                {/* Cash Flow Education */}
                <div className="mt-6 p-4 bg-emerald-50 dark:bg-emerald-950 rounded-lg border border-emerald-200 dark:border-emerald-800">
                  <h5 className="font-semibold mb-3 text-emerald-900 dark:text-emerald-100">🎓 Understanding Cash Flow & Valuation Metrics</h5>
                  <div className="text-xs text-emerald-800 dark:text-emerald-200 space-y-3">
                    <div>
                      <p className="font-medium mb-1">Why Cash Flow Matters More Than Earnings:</p>
                      <p>Earnings can be manipulated through accounting tricks (depreciation schedules, revenue recognition, one-time charges). Cash flow is harder to fake — either the cash is in the bank or it isn't. Think of earnings as what the company <em>says</em> it made, and cash flow as what it <em>actually</em> made.</p>
                    </div>
                    <div>
                      <p className="font-medium mb-1">Key Metrics Explained:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>Free Cash Flow (FCF)</strong> = Operating cash flow minus capital expenditures. This is the money left over after running the business and maintaining equipment. It's what's available for dividends, buybacks, debt paydown, or acquisitions.</li>
                        <li><strong>FCF Yield</strong> = FCF / Market Cap. Think of it as the "interest rate" you earn on your investment. A 5% FCF yield means the company generates 5 cents of free cash for every dollar of market value.</li>
                        <li><strong>FCF Margin</strong> = FCF / Revenue. Shows how efficiently the company converts sales into cash. Software companies often have 25-40% FCF margins; retailers might have 3-5%.</li>
                        <li><strong>P/E Ratio</strong> = Price / Earnings per share. The most common valuation metric. Compare to industry peers, not across sectors (tech P/E of 30 is different from utility P/E of 30).</li>
                        <li><strong>PEG Ratio</strong> = P/E / Growth Rate. Adjusts valuation for growth. PEG {'<'} 1 = potentially undervalued. PEG {'>'} 2 = potentially overvalued.</li>
                        <li><strong>EV/EBITDA</strong> = Enterprise Value / EBITDA. Better than P/E for comparing companies with different debt levels. Lower = cheaper.</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-medium mb-1">GARP Analysis (Growth at Reasonable Price):</p>
                      <p>Popularized by Peter Lynch, GARP combines growth investing with value discipline. A high GARP score means the company offers strong earnings growth relative to its valuation — the sweet spot between overpaying for growth and settling for cheap-but-stagnant companies.</p>
                    </div>
                    <div>
                      <p className="font-medium mb-1">Altman Z-Score (Bankruptcy Predictor):</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>&gt; 3.0</strong> = Safe zone. Very low probability of bankruptcy.</li>
                        <li><strong>1.8 - 3.0</strong> = Gray zone. Some financial stress. Monitor quarterly.</li>
                        <li><strong>&lt; 1.8</strong> = Distress zone. Elevated bankruptcy risk. Not necessarily imminent, but a red flag.</li>
                      </ul>
                    </div>
                    <div className="p-2 bg-emerald-100 dark:bg-emerald-900 rounded">
                      <p className="font-medium">💡 Pro Tips:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li>Always compare valuation metrics to <strong>industry peers</strong>, not the broad market. A P/E of 40 is cheap for a high-growth SaaS company but expensive for a bank.</li>
                        <li>Watch for <strong>declining FCF margins</strong> over multiple quarters — this often precedes earnings misses.</li>
                        <li>High insider ownership ({'>'}10%) often aligns management incentives with shareholders.</li>
                        <li>Short interest above 10% of float can create squeeze potential but also signals bearish institutional sentiment.</li>
                        <li>Revenue CAGR vs. FCF CAGR divergence: If revenue grows faster than FCF, the company may be sacrificing profitability for growth.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Enhanced fundamentals data not available. Run a full analysis to see cash flow and valuation metrics.
              </div>
            )}
          </TabsContent>

          <TabsContent value="garch" className="space-y-4">
            <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-100">📊 What is GARCH?</h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models time-varying volatility. 
                Unlike simple historical volatility, GARCH adapts to market conditions - volatility clusters during turbulent periods and calms during stable periods.
              </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <DataItem label="Model" value="GARCH(1,1)" />
                <p className="text-xs text-muted-foreground">Standard model: 1 lag for shocks, 1 lag for volatility</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Distribution" value="Student-t" />
                <p className="text-xs text-muted-foreground">Captures extreme moves better than normal distribution</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Fat-Tail DF" value={safeFixed(garch.fat_tail_df)} />
                <p className="text-xs text-muted-foreground">Lower = fatter tails (more extreme events). Typical: 3-10</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="AIC" value={safeFixed(garch.aic)} />
                <p className="text-xs text-muted-foreground">Model quality: Lower is better. Compare across stocks</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="BIC" value={safeFixed(garch.bic)} />
                <p className="text-xs text-muted-foreground">Similar to AIC but penalizes complexity more</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Current Vol" value={`${safeFixed(garch.current_volatility * 100)}%`} />
                <p className="text-xs text-muted-foreground">Annualized volatility forecast for next period</p>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">GARCH(1,1) Model:</h4>
              <code className="text-xs block">
                σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁<br/>
                Where:<br/>
                - σ²ₜ = Conditional variance at time t<br/>
                - ω = Long-run variance<br/>
                - α = ARCH parameter (shock impact)<br/>
                - β = GARCH parameter (persistence)<br/>
                - ε²ₜ₋₁ = Previous squared residual<br/>
                <br/>
                Fitted using Maximum Likelihood Estimation (MLE)<br/>
                Student-t distribution captures fat tails (df={safeFixed(garch.fat_tail_df)})
              </code>
            </div>
            
            <div className="mt-4 p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-800">
              <h4 className="font-semibold mb-2 text-green-900 dark:text-green-100">💡 How to Use GARCH</h4>
              <ul className="text-sm text-green-800 dark:text-green-200 space-y-2">
                <li><strong>Current Vol:</strong> Use this for options pricing and risk assessment. Higher vol = higher option premiums.</li>
                <li><strong>Fat-Tail DF:</strong> Lower values (3-5) mean more extreme moves are likely. Be cautious with leverage.</li>
                <li><strong>AIC/BIC:</strong> Compare these across different stocks. Lower scores = model fits better = more reliable forecasts.</li>
                <li><strong>Volatility Clustering:</strong> If recent volatility is high, GARCH predicts it will stay high short-term (and vice versa).</li>
              </ul>
            </div>
          </TabsContent>

          <TabsContent value="montecarlo" className="space-y-4">
            <div className="mb-4 p-4 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-800">
              <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-100">🎲 What is Monte Carlo?</h4>
              <p className="text-sm text-purple-800 dark:text-purple-200">
                Monte Carlo simulation runs 20,000 possible future price paths using GARCH volatility and fat-tail distributions. 
                This shows the range of outcomes and helps quantify risk better than simple predictions.
              </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <DataItem label="Simulations" value="20,000" />
                <p className="text-xs text-muted-foreground">Number of price paths simulated for statistical confidence</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Time Steps" value="30 days" />
                <p className="text-xs text-muted-foreground">Forecast horizon (1 month ahead)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Expected Price" value={`$${safeFixed(stochastic.expected_price)}`} />
                <p className="text-xs text-muted-foreground">Average price across all 20,000 simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Expected Return" value={`${safeFixed(stochastic.expected_return * 100)}%`} />
                <p className="text-xs text-muted-foreground">Average return over 30 days (annualized)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="VaR (95%)" value={`${safeFixed(stochastic.var_95 * 100)}%`} />
                <p className="text-xs text-muted-foreground">Worst-case loss in 95% of scenarios (5% chance worse)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="CVaR (95%)" value={`${safeFixed(stochastic.cvar_95 * 100)}%`} />
                <p className="text-xs text-muted-foreground">Average loss when VaR is exceeded (tail risk)</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="Max Drawdown" value={`${safeFixed(stochastic.max_drawdown * 100)}%`} />
                <p className="text-xs text-muted-foreground">Largest peak-to-trough decline across simulations</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Lower" value={`$${safeFixed(stochastic.confidence_interval_lower)}`} />
                <p className="text-xs text-muted-foreground">Lower bound: 95% chance price stays above this</p>
              </div>
              
              <div className="space-y-1">
                <DataItem label="95% CI Upper" value={`$${safeFixed(stochastic.confidence_interval_upper)}`} />
                <p className="text-xs text-muted-foreground">Upper bound: 95% chance price stays below this</p>
              </div>
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Monte Carlo GBM with Fat Tails:</h4>
              <code className="text-xs block">
                dS/S = μ·dt + σ·dW<br/>
                Log-step: log(Sₜ₊₁/Sₜ) = (μ - 0.5σ²)·dt + σ·√dt·Zₜ<br/>
                <br/>
                Where:<br/>
                - Zₜ ~ Student-t(df=5) for fat tails<br/>
                - μ = Annualized drift (historical mean return)<br/>
                - σ = GARCH-forecasted volatility<br/>
                - Antithetic variates for variance reduction<br/>
                <br/>
                VaR(95%) = 5th percentile of return distribution<br/>
                CVaR(95%) = Mean of returns below VaR<br/>
                Max Drawdown = max((Running Max - Price) / Running Max)
              </code>
            </div>
            
            <div className="mt-4 p-4 bg-orange-50 dark:bg-orange-950 rounded-lg border border-orange-200 dark:border-orange-800">
              <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-100">🎯 How to Use Monte Carlo</h4>
              <ul className="text-sm text-orange-800 dark:text-orange-200 space-y-2">
                <li><strong>Expected Price:</strong> Your best estimate for 30 days out. Don't treat as guaranteed - it's an average.</li>
                <li><strong>VaR (95%):</strong> Your downside risk. If VaR is -15%, you have 5% chance of losing more than 15%.</li>
                <li><strong>CVaR (95%):</strong> Tail risk - how bad it gets when VaR is breached. Critical for position sizing.</li>
                <li><strong>95% CI Range:</strong> Price likely stays within this range. Wider range = more uncertainty = higher risk.</li>
                <li><strong>Max Drawdown:</strong> Worst decline you might see. Use this to set stop losses and manage emotions.</li>
              </ul>
            </div>
          </TabsContent>

          <TabsContent value="position" className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <DataItem label="Bankroll" value={`$${safeFixed(analysis.bankroll || 1000)}`} />
              <DataItem label="Position Size" value={`${safeFixed((positionSizing.position_size_pct || 0) * 100)}%`} />
              <DataItem label="Shares" value={safeFixed(positionSizing.shares, 0)} />
              <DataItem label="Position Value" value={`$${safeFixed(positionSizing.position_value)}`} />
              <DataItem label="Dollar Risk" value={`$${safeFixed(positionSizing.dollar_risk)}`} />
              <DataItem label="Dollar Reward" value={`$${safeFixed(positionSizing.dollar_reward)}`} />
              <DataItem label="Risk/Reward" value={`1:${safeFixed(positionSizing.risk_reward_ratio)}`} />
              <DataItem label="Risk % of Bankroll" value={`${safeFixed((positionSizing.risk_pct_of_bankroll || 0) * 100)}%`} />
            </div>
            
            <div className="mt-4 p-4 bg-muted/20 rounded-lg">
              <h4 className="font-semibold mb-2">Kelly Criterion Position Sizing:</h4>
              <code className="text-xs block">
                Kelly Fraction = (p·b - q) / b<br/>
                Where:<br/>
                - p = Win probability (confidence / 100)<br/>
                - q = Loss probability (1 - p)<br/>
                - b = Win/loss ratio (target gain / stop loss)<br/>
                <br/>
                Position Size = min(Half-Kelly, Risk-Based Max)<br/>
                Risk-Based Max = Max Risk % / Stop Distance %<br/>
                <br/>
                Constraints:<br/>
                - Max 2% risk per trade (moderate risk)<br/>
                - Half-Kelly for safety (0.5 × Kelly)<br/>
                - Position size between 1% and 20% of bankroll<br/>
                <br/>
                Shares = floor(Position Value / Current Price)<br/>
                Dollar Risk = Shares × |Current Price - Stop Loss|
              </code>
            </div>
          </TabsContent>

          <TabsContent value="market" className="space-y-4">
            {/* Market Context - Live market status, VIX, regime, sentiment */}
            {analysis.market_context ? (
              <div className="space-y-6">
                {/* Market Status */}
                <div className="p-4 bg-muted/20 rounded-lg">
                  <h4 className="font-semibold text-primary mb-3">🕐 Market Status</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <DataItem label="Status" value={analysis.market_context.market_status?.status || 'N/A'} />
                    <DataItem label="Session" value={analysis.market_context.market_status?.session || 'N/A'} />
                    <DataItem label="Day" value={analysis.market_context.market_status?.day_of_week || 'N/A'} />
                    <DataItem label="Time (EST)" value={analysis.market_context.market_status?.current_time_est?.split(' ')[1] || 'N/A'} />
                  </div>
                  {analysis.market_context.market_status?.reason && (
                    <p className="text-sm text-muted-foreground mt-2">{analysis.market_context.market_status.reason}</p>
                  )}
                </div>

                {/* VIX */}
                {analysis.market_context.vix?.success && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📈 VIX (Volatility Index)</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <DataItem label="VIX" value={safeFixed(analysis.market_context.vix.vix)} />
                      <DataItem label="Level" value={analysis.market_context.vix.level || 'N/A'} />
                      <DataItem label="Change" value={`${analysis.market_context.vix.change >= 0 ? '+' : ''}${safeFixed(analysis.market_context.vix.change)}`} />
                      <DataItem label="Change %" value={`${analysis.market_context.vix.change_percent >= 0 ? '+' : ''}${safeFixed(analysis.market_context.vix.change_percent)}%`} />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">{analysis.market_context.vix.interpretation}</p>
                  </div>
                )}

                {/* Market Regime */}
                {analysis.market_context.regime?.success && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">🎯 Market Regime</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <DataItem label="Regime" value={analysis.market_context.regime.regime || 'N/A'} />
                      <DataItem label="SPY Momentum" value={`${safeFixed(analysis.market_context.regime.momentum_20d)}%`} />
                      <DataItem label="Volatility" value={`${safeFixed(analysis.market_context.regime.volatility_annualized)}%`} />
                      <DataItem label="Confidence" value={`${safeFixed(analysis.market_context.regime.confidence)}%`} />
                    </div>
                    {analysis.market_context.regime.characteristics?.length > 0 && (
                      <div className="text-sm text-muted-foreground mt-2">
                        {analysis.market_context.regime.characteristics.map((c: string, i: number) => (
                          <span key={i}>{i > 0 ? ' | ' : ''}{c}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Overall Assessment */}
                {analysis.market_context.overall && (
                  <div className="p-4 bg-muted/20 rounded-lg border-l-4 border-primary">
                    <h4 className="font-semibold text-primary mb-2">📋 Overall Assessment</h4>
                    <div className="flex items-center gap-4 mb-2">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        analysis.market_context.overall.conditions === 'FAVORABLE' ? 'bg-green-500/20 text-green-400' :
                        analysis.market_context.overall.conditions === 'CAUTIOUS' ? 'bg-red-500/20 text-red-400' :
                        analysis.market_context.overall.conditions === 'VOLATILE' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {analysis.market_context.overall.conditions}
                      </span>
                    </div>
                    <p className="text-sm">{analysis.market_context.overall.recommendation}</p>
                  </div>
                )}

                {/* Historical Patterns */}
                {analysis.market_context.patterns?.success && analysis.market_context.patterns.patterns_found > 0 && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📊 Historical Pattern Matches</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-3">
                      <DataItem label="Avg 20D Return" value={`${safeFixed(analysis.market_context.patterns.average_subsequent_return)}%`} />
                      <DataItem label="Bullish Probability" value={`${safeFixed(analysis.market_context.patterns.bullish_probability)}%`} />
                      <DataItem label="Patterns Found" value={analysis.market_context.patterns.patterns_found || 0} />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {analysis.market_context.patterns.top_matches?.slice(0, 3).map((m: any, i: number) => (
                        <div key={i}>{m.date}: {m.subsequent_20d_return > 0 ? '+' : ''}{safeFixed(m.subsequent_20d_return)}% (corr: {safeFixed(m.correlation, 3)})</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 text-center text-muted-foreground">
                Market context data not available
              </div>
            )}
          </TabsContent>

          {/* Dark Pools - Institutional flow from StockGrid.io */}
          <TabsContent value="darkpools" className="space-y-4">
            {analysis.stockgrid_analysis?.dark_pools ? (
              <div className="space-y-6">
                {/* Dark Pool Summary */}
                <div className="p-4 bg-muted/20 rounded-lg border-l-4 border-purple-500">
                  <h4 className="font-semibold text-purple-400 mb-3">🏦 Dark Pool Activity (FINRA TRF)</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <DataItem 
                      label="Net Short Volume" 
                      value={analysis.stockgrid_analysis.dark_pools.net_short_volume ? 
                        `${analysis.stockgrid_analysis.dark_pools.net_short_volume > 0 ? '+' : ''}${(analysis.stockgrid_analysis.dark_pools.net_short_volume / 1000000).toFixed(2)}M` : 'N/A'} 
                    />
                    <DataItem 
                      label="Net Short $" 
                      value={analysis.stockgrid_analysis.dark_pools.net_short_volume_dollar ? 
                        `$${(analysis.stockgrid_analysis.dark_pools.net_short_volume_dollar / 1000000).toFixed(1)}M` : 'N/A'} 
                    />
                    <DataItem 
                      label="20-Day Position" 
                      value={analysis.stockgrid_analysis.dark_pools.position ? 
                        `${(analysis.stockgrid_analysis.dark_pools.position / 1000000).toFixed(1)}M shares` : 'N/A'} 
                    />
                    <DataItem 
                      label="Position $" 
                      value={analysis.stockgrid_analysis.dark_pools.position_dollar ? 
                        `$${(analysis.stockgrid_analysis.dark_pools.position_dollar / 1000000000).toFixed(2)}B` : 'N/A'} 
                    />
                  </div>
                </div>

                {/* Sentiment Interpretation */}
                {analysis.stockgrid_analysis.dark_pools.sentiment && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📊 Institutional Flow Signal</h4>
                    <div className="flex items-center gap-4 mb-3">
                      <span className={`px-4 py-2 rounded-full text-sm font-bold ${
                        analysis.stockgrid_analysis.dark_pools.sentiment.sentiment?.includes('BULLISH') ? 'bg-green-500/20 text-green-400' :
                        analysis.stockgrid_analysis.dark_pools.sentiment.sentiment?.includes('BEARISH') ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {analysis.stockgrid_analysis.dark_pools.sentiment.sentiment || 'N/A'}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Score: {analysis.stockgrid_analysis.dark_pools.sentiment.score || 'N/A'}/100
                      </span>
                    </div>
                    <p className="text-sm text-primary">{analysis.stockgrid_analysis.dark_pools.sentiment.signal}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Daily Trend: {analysis.stockgrid_analysis.dark_pools.sentiment.daily_trend || 'N/A'}
                    </p>
                  </div>
                )}

                {/* Additional Metrics */}
                <div className="p-4 bg-muted/20 rounded-lg">
                  <h4 className="font-semibold text-primary mb-3">📋 Daily Metrics</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <DataItem label="Short Volume" value={analysis.stockgrid_analysis.dark_pools.short_volume ? `${(analysis.stockgrid_analysis.dark_pools.short_volume / 1000000).toFixed(2)}M` : 'N/A'} />
                    <DataItem label="Total Volume" value={analysis.stockgrid_analysis.dark_pools.total_volume ? `${(analysis.stockgrid_analysis.dark_pools.total_volume / 1000000).toFixed(2)}M` : 'N/A'} />
                    <DataItem label="Short Ratio" value={analysis.stockgrid_analysis.dark_pools.short_ratio ? `${(analysis.stockgrid_analysis.dark_pools.short_ratio * 100).toFixed(1)}%` : 'N/A'} />
                    <DataItem label="Data Date" value={analysis.stockgrid_analysis.dark_pools.date || 'N/A'} />
                  </div>
                </div>

                {/* Short Volume History */}
                {analysis.stockgrid_analysis.dark_pools.short_volume_history?.length > 0 && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📈 Short Volume History (Last {analysis.stockgrid_analysis.dark_pools.short_volume_history.length} Days)</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-muted">
                            <th className="text-left py-1 px-2">Date</th>
                            <th className="text-right py-1 px-2">Short Vol</th>
                            <th className="text-right py-1 px-2">Total Vol</th>
                            <th className="text-right py-1 px-2">Short %</th>
                            <th className="text-right py-1 px-2">Net Short</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysis.stockgrid_analysis.dark_pools.short_volume_history.slice(0, 10).map((day: any, i: number) => {
                            const netShort = day.short_volume && day.total_volume ? (2 * day.short_volume) - day.total_volume : 0;
                            return (
                              <tr key={i} className="border-b border-muted/30">
                                <td className="py-1 px-2">{day.date}</td>
                                <td className="text-right py-1 px-2">{day.short_volume ? `${(day.short_volume / 1000000).toFixed(2)}M` : 'N/A'}</td>
                                <td className="text-right py-1 px-2">{day.total_volume ? `${(day.total_volume / 1000000).toFixed(2)}M` : 'N/A'}</td>
                                <td className="text-right py-1 px-2">{day.short_ratio ? `${(day.short_ratio * 100).toFixed(1)}%` : 'N/A'}</td>
                                <td className={`text-right py-1 px-2 font-medium ${netShort < 0 ? 'text-green-400' : netShort > 0 ? 'text-red-400' : ''}`}>
                                  {netShort ? `${(netShort / 1000000).toFixed(2)}M` : '0'}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Explanation */}
                {analysis.stockgrid_analysis.dark_pools.explanation && (
                  <div className="p-4 bg-muted/20 rounded-lg border-l-4 border-blue-500">
                    <h4 className="font-semibold text-blue-400 mb-2">🔍 Analysis</h4>
                    <p className="text-sm">{analysis.stockgrid_analysis.dark_pools.explanation}</p>
                  </div>
                )}

                {/* How to Interpret */}
                <div className="p-4 bg-muted/10 rounded-lg border border-muted">
                  <h4 className="font-semibold text-muted-foreground mb-2">ℹ️ How to Interpret Dark Pool Data</h4>
                  <div className="text-xs text-muted-foreground space-y-2">
                    <p><strong>Key Insight (SqueezeMetrics Research):</strong> Dark pool "short volume" mostly represents market makers selling shares to meet buyer demand. Counterintuitively:</p>
                    <ul className="list-disc list-inside ml-2">
                      <li><strong>Negative Position</strong> = More buying than shorting = <span className="text-green-400">BULLISH</span></li>
                      <li><strong>Positive Position</strong> = More shorting than buying = <span className="text-red-400">BEARISH</span></li>
                    </ul>
                    <p className="mt-2"><strong>Thresholds:</strong></p>
                    <ul className="list-disc list-inside ml-2">
                      <li>Position $ &lt; -$1B: Strong institutional accumulation</li>
                      <li>Position $ &gt; $1B: Strong institutional distribution</li>
                    </ul>
                    <p className="text-xs mt-2 italic">Data source: StockGrid.io (FINRA TRF). Updated daily at 6pm EST.</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 text-center text-muted-foreground">
                Dark pool data not available. {analysis.stockgrid_analysis?.dark_pools?.error || ''}
              </div>
            )}
          </TabsContent>

          {/* ARIMA Forecasting */}
          <TabsContent value="arima" className="space-y-4">
            {analysis.stockgrid_analysis?.arima?.status === 'success' ? (
              <div className="space-y-6">
                {/* ARIMA Summary */}
                <div className="p-4 bg-muted/20 rounded-lg border-l-4 border-blue-500">
                  <h4 className="font-semibold text-blue-400 mb-3">📈 ARIMA Time Series Forecast</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <DataItem label="Model" value={analysis.stockgrid_analysis.arima.model || 'N/A'} />
                    <DataItem label="Current Price" value={`$${safeFixed(analysis.stockgrid_analysis.arima.current_price)}`} />
                    <DataItem 
                      label="5-Day Forecast" 
                      value={analysis.stockgrid_analysis.arima.forecast?.[4] ? 
                        `$${safeFixed(analysis.stockgrid_analysis.arima.forecast[4].predicted)}` : 'N/A'} 
                    />
                    <DataItem 
                      label="Expected Change" 
                      value={analysis.stockgrid_analysis.arima.interpretation?.expected_change_pct ? 
                        `${analysis.stockgrid_analysis.arima.interpretation.expected_change_pct > 0 ? '+' : ''}${safeFixed(analysis.stockgrid_analysis.arima.interpretation.expected_change_pct)}%` : 'N/A'} 
                    />
                  </div>
                </div>

                {/* Signal */}
                {analysis.stockgrid_analysis.arima.interpretation && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">🎯 ARIMA Signal</h4>
                    <div className="flex items-center gap-4 mb-3">
                      <span className={`px-4 py-2 rounded-full text-sm font-bold ${
                        analysis.stockgrid_analysis.arima.interpretation.signal?.includes('BUY') ? 'bg-green-500/20 text-green-400' :
                        analysis.stockgrid_analysis.arima.interpretation.signal?.includes('SELL') ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {analysis.stockgrid_analysis.arima.interpretation.signal || 'N/A'}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Confidence: {analysis.stockgrid_analysis.arima.interpretation.confidence || 'N/A'}
                      </span>
                    </div>
                    <p className="text-sm">{analysis.stockgrid_analysis.arima.interpretation.summary}</p>
                  </div>
                )}

                {/* Forecast Table */}
                {analysis.stockgrid_analysis.arima.forecast?.length > 0 && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📅 5-Day Price Forecast</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-muted">
                            <th className="text-left py-2">Date</th>
                            <th className="text-right py-2">Predicted</th>
                            <th className="text-right py-2">95% CI Low</th>
                            <th className="text-right py-2">95% CI High</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysis.stockgrid_analysis.arima.forecast.map((f: any, i: number) => (
                            <tr key={i} className="border-b border-muted/50">
                              <td className="py-2">{f.date}</td>
                              <td className="text-right">${safeFixed(f.predicted)}</td>
                              <td className="text-right text-red-400">${safeFixed(f.lower_95)}</td>
                              <td className="text-right text-green-400">${safeFixed(f.upper_95)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Model Statistics */}
                {analysis.stockgrid_analysis.arima.model_stats && (
                  <div className="p-4 bg-muted/20 rounded-lg">
                    <h4 className="font-semibold text-primary mb-3">📊 Model Statistics</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <DataItem label="AIC" value={safeFixed(analysis.stockgrid_analysis.arima.model_stats.aic)} />
                      <DataItem label="BIC" value={safeFixed(analysis.stockgrid_analysis.arima.model_stats.bic)} />
                      <DataItem label="Observations" value={analysis.stockgrid_analysis.arima.model_stats.observations || 'N/A'} />
                      <DataItem label="Stationary" value={analysis.stockgrid_analysis.arima.model_stats.is_stationary ? 'Yes' : 'No'} />
                    </div>
                  </div>
                )}

                {/* Dynamic ARIMA Guidance Based on Results */}
                {analysis.stockgrid_analysis.arima.interpretation && (() => {
                  const interp = analysis.stockgrid_analysis.arima.interpretation;
                  const forecast = analysis.stockgrid_analysis.arima.forecast;
                  const stats = analysis.stockgrid_analysis.arima.model_stats;
                  const changePct = interp.expected_change_pct || 0;
                  const isStationary = stats?.is_stationary;
                  
                  // Calculate CI width as % of price for reliability assessment
                  const lastForecast = forecast?.[forecast.length - 1];
                  const ciWidth = lastForecast ? ((lastForecast.upper_95 - lastForecast.lower_95) / lastForecast.predicted * 100) : null;
                  const isReliable = ciWidth !== null && ciWidth < 10;
                  
                  return (
                    <div className={`p-4 rounded-lg border-l-4 ${
                      changePct > 1 ? 'bg-green-50 dark:bg-green-950 border-green-500' :
                      changePct < -1 ? 'bg-red-50 dark:bg-red-950 border-red-500' :
                      'bg-yellow-50 dark:bg-yellow-950 border-yellow-500'
                    }`}>
                      <h5 className="font-semibold mb-2 text-sm">🎯 What This ARIMA Forecast Means For You</h5>
                      <div className="text-xs space-y-2">
                        <p>
                          <strong>Direction:</strong> ARIMA projects a <strong>{changePct > 0 ? `+${safeFixed(changePct)}%` : `${safeFixed(changePct)}%`}</strong> move over 5 days.
                          {changePct > 2 ? ' This is a meaningful bullish projection.' :
                           changePct < -2 ? ' This is a meaningful bearish projection.' :
                           ' This is a relatively flat projection, suggesting consolidation.'}
                        </p>
                        <p>
                          <strong>Reliability:</strong> {ciWidth !== null ? (
                            ciWidth < 5 ? `The confidence interval is tight (${safeFixed(ciWidth)}% width), indicating high model confidence. This forecast is more reliable than average.` :
                            ciWidth < 10 ? `The confidence interval is moderate (${safeFixed(ciWidth)}% width). Reasonable confidence, but use other indicators for confirmation.` :
                            `The confidence interval is wide (${safeFixed(ciWidth)}% width), indicating significant uncertainty. Treat this as directional guidance only, not a precise target.`
                          ) : 'Unable to assess CI width.'}
                        </p>
                        {!isStationary && (
                          <p className="text-yellow-700 dark:text-yellow-300">
                            ⚠️ <strong>Caution:</strong> The series required differencing to achieve stationarity. This means the stock has a strong trend component that ARIMA may struggle to capture during trend reversals.
                          </p>
                        )}
                        <p>
                          <strong>How to use:</strong> {changePct > 1 && isReliable ? 
                            'The model supports a bullish short-term thesis. Consider this as one confirmation signal alongside your technical and fundamental analysis.' :
                            changePct < -1 && isReliable ?
                            'The model supports a bearish short-term thesis. Consider tightening stops or reducing position size.' :
                            'The model shows no strong directional conviction. This is a wait-and-see signal. Do not force a trade based on this alone.'}
                        </p>
                      </div>
                    </div>
                  );
                })()}
                
                {/* Comprehensive ARIMA Education */}
                <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                  <h5 className="font-semibold mb-3 text-blue-900 dark:text-blue-100">🎓 Understanding ARIMA Forecasting</h5>
                  <div className="text-xs text-blue-800 dark:text-blue-200 space-y-3">
                    <div>
                      <p className="font-medium mb-1">What ARIMA Does (Think of it like weather forecasting for stocks):</p>
                      <p>ARIMA looks at the stock's recent price history and finds mathematical patterns in how prices move from day to day. It's like noticing that after 3 rainy days, there's usually a sunny day — except with prices.</p>
                    </div>
                    <div>
                      <p className="font-medium mb-1">The Three Components:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>AR (AutoRegressive)</strong> = "Yesterday's price influences today's price." Uses past values. The (p) number tells you how many days back it looks.</li>
                        <li><strong>I (Integrated)</strong> = "Remove the trend first." Differencing makes the data stationary (flat). The (d) number is how many times it differences.</li>
                        <li><strong>MA (Moving Average)</strong> = "Learn from past mistakes." Uses previous forecast errors to self-correct. The (q) number is how many error terms it uses.</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-medium mb-1">Reading the Forecast Table:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>Predicted</strong> = The model's best guess for that day's closing price.</li>
                        <li><strong>95% CI Low/High</strong> = There's a 95% chance the actual price falls within this range. Wider = more uncertainty.</li>
                        <li>Notice how the CI <strong>widens</strong> each day — this is normal. Predictions get less reliable further out.</li>
                      </ul>
                    </div>
                    <div>
                      <p className="font-medium mb-1">Model Quality Metrics:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li><strong>AIC/BIC</strong> = Lower is better. These measure how well the model fits without overfitting. Compare across different stocks to gauge relative quality.</li>
                        <li><strong>Stationary = Yes</strong> means the model's assumptions are met. "No" means the data needed extra processing (less ideal).</li>
                      </ul>
                    </div>
                    <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded">
                      <p className="font-medium">💡 Pro Tips:</p>
                      <ul className="space-y-1 ml-3 list-disc">
                        <li>ARIMA is <strong>best for 1-3 day forecasts</strong>. Day 4-5 predictions are significantly less reliable.</li>
                        <li>ARIMA does NOT account for earnings, news, or market sentiment. It's purely mathematical.</li>
                        <li>If the CI range is wider than 5% of the price, the forecast has low practical value for trading decisions.</li>
                        <li>ARIMA works best on <strong>liquid, large-cap stocks</strong> with consistent trading patterns. It struggles with penny stocks, IPOs, and meme stocks.</li>
                        <li>Use ARIMA as <strong>one input among many</strong>. When ARIMA agrees with your technical analysis (RSI, MACD, etc.), the combined signal is stronger.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 text-center text-muted-foreground">
                ARIMA forecast not available. {analysis.stockgrid_analysis?.arima?.error || ''}
              </div>
            )}
          </TabsContent>

          <TabsContent value="all" className="space-y-4">
            <div className="p-4 bg-muted/20 rounded-lg max-h-[600px] overflow-auto">
              <pre className="text-xs">
                {JSON.stringify(analysis, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function DataItem({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="p-3 bg-muted/30 rounded-lg">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className={`text-sm font-semibold ${value === null || value === undefined ? 'text-muted-foreground/50 italic' : ''}`}>
        {value !== null && value !== undefined ? value : '—'}
      </div>
    </div>
  );
}
