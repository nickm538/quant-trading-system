import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface GreeksHeatmapProps {
  data: {
    symbol: string;
    current_price: number;
    risk_free_rate: number;
    strikes: number[];
    expirations: Array<{ date: string; days: number }>;
    calls: {
      delta: number[][];
      gamma: number[][];
      theta: number[][];
      vega: number[][];
      rho: number[][];
    };
    puts: {
      delta: number[][];
      gamma: number[][];
      theta: number[][];
      vega: number[][];
      rho: number[][];
    };
  };
}

type GreekType = 'delta' | 'gamma' | 'theta' | 'vega' | 'rho';
type OptionType = 'calls' | 'puts';

export function GreeksHeatmap({ data }: GreeksHeatmapProps) {
  const [selectedGreek, setSelectedGreek] = useState<GreekType>('delta');
  const [selectedType, setSelectedType] = useState<OptionType>('calls');

  // Get color for heatmap cell based on value
  const getColor = (value: number, greek: GreekType): string => {
    if (greek === 'delta') {
      // Delta: -1 to 1 (red to green)
      const normalized = (value + 1) / 2; // 0 to 1
      if (normalized < 0.33) return `rgb(239, 68, 68)`; // red
      if (normalized < 0.67) return `rgb(234, 179, 8)`; // yellow
      return `rgb(34, 197, 94)`; // green
    } else if (greek === 'gamma') {
      // Gamma: 0 to max (white to blue)
      const max = Math.max(...data[selectedType][greek].flat());
      const intensity = Math.min(value / max, 1);
      return `rgba(59, 130, 246, ${intensity})`;
    } else if (greek === 'theta') {
      // Theta: negative (red gradient)
      const max = Math.max(...data[selectedType][greek].flat().map(Math.abs));
      const intensity = Math.min(Math.abs(value) / max, 1);
      return `rgba(239, 68, 68, ${intensity})`;
    } else if (greek === 'vega') {
      // Vega: 0 to max (white to purple)
      const max = Math.max(...data[selectedType][greek].flat());
      const intensity = Math.min(value / max, 1);
      return `rgba(168, 85, 247, ${intensity})`;
    } else {
      // Rho: can be positive or negative (red/green)
      const max = Math.max(...data[selectedType][greek].flat().map(Math.abs));
      const normalized = value / max;
      if (normalized < 0) {
        return `rgba(239, 68, 68, ${Math.abs(normalized)})`;
      } else {
        return `rgba(34, 197, 94, ${normalized})`;
      }
    }
  };

  // Format value for display
  const formatValue = (value: number, greek: GreekType): string => {
    if (greek === 'delta') return value.toFixed(3);
    if (greek === 'gamma') return value.toFixed(4);
    if (greek === 'theta') return value.toFixed(2);
    if (greek === 'vega') return value.toFixed(2);
    if (greek === 'rho') return value.toFixed(2);
    return value.toFixed(2);
  };

  const greekData = data[selectedType][selectedGreek];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Options Greeks Heatmap - {data.symbol}</CardTitle>
        <CardDescription>
          Current Price: ${data.current_price.toFixed(2)} | Risk-Free Rate: {(data.risk_free_rate * 100).toFixed(2)}%
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Greek Selector */}
          <Tabs value={selectedGreek} onValueChange={(v) => setSelectedGreek(v as GreekType)}>
            {/* Mobile: scrollable */}
            <div className="md:hidden overflow-x-auto -mx-1 px-1 pb-1">
              <TabsList className="inline-flex w-max gap-1 h-auto p-1">
                <TabsTrigger value="delta" className="px-3 py-2 text-xs">Delta</TabsTrigger>
                <TabsTrigger value="gamma" className="px-3 py-2 text-xs">Gamma</TabsTrigger>
                <TabsTrigger value="theta" className="px-3 py-2 text-xs">Theta</TabsTrigger>
                <TabsTrigger value="vega" className="px-3 py-2 text-xs">Vega</TabsTrigger>
                <TabsTrigger value="rho" className="px-3 py-2 text-xs">Rho</TabsTrigger>
              </TabsList>
            </div>
            {/* Desktop: grid */}
            <TabsList className="hidden md:grid w-full grid-cols-5">
              <TabsTrigger value="delta">Delta</TabsTrigger>
              <TabsTrigger value="gamma">Gamma</TabsTrigger>
              <TabsTrigger value="theta">Theta</TabsTrigger>
              <TabsTrigger value="vega">Vega</TabsTrigger>
              <TabsTrigger value="rho">Rho</TabsTrigger>
            </TabsList>
          </Tabs>

          {/* Call/Put Selector */}
          <Tabs value={selectedType} onValueChange={(v) => setSelectedType(v as OptionType)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="calls">Calls</TabsTrigger>
              <TabsTrigger value="puts">Puts</TabsTrigger>
            </TabsList>
          </Tabs>

          {/* Heatmap */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border border-border p-2 bg-muted font-semibold">Strike</th>
                  {data.expirations.map((exp, i) => (
                    <th key={i} className="border border-border p-2 bg-muted font-semibold">
                      {exp.date}
                      <div className="text-xs text-muted-foreground">{exp.days}d</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.strikes.map((strike, strikeIdx) => (
                  <tr key={strikeIdx}>
                    <td className="border border-border p-2 font-medium bg-muted">
                      ${strike.toFixed(2)}
                      {Math.abs(strike - data.current_price) < 5 && (
                        <span className="ml-1 text-xs text-primary">ATM</span>
                      )}
                    </td>
                    {data.expirations.map((_, expIdx) => {
                      const value = greekData[expIdx][strikeIdx];
                      return (
                        <td
                          key={expIdx}
                          className="border border-border p-2 text-center font-mono cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                          style={{ backgroundColor: getColor(value, selectedGreek) }}
                          title={`${selectedGreek.toUpperCase()}: ${formatValue(value, selectedGreek)}`}
                        >
                          <span className={value < 0 && selectedGreek !== 'delta' ? 'text-white' : ''}>
                            {formatValue(value, selectedGreek)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="mt-4 p-4 bg-muted rounded-lg">
            <h4 className="font-semibold mb-2">Greek Definitions:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
              <div>
                <span className="font-medium">Delta:</span> Rate of change in option price per $1 move in stock (-1 to 1)
              </div>
              <div>
                <span className="font-medium">Gamma:</span> Rate of change in Delta per $1 move in stock
              </div>
              <div>
                <span className="font-medium">Theta:</span> Time decay - option value loss per day (negative)
              </div>
              <div>
                <span className="font-medium">Vega:</span> Sensitivity to 1% change in implied volatility
              </div>
              <div>
                <span className="font-medium">Rho:</span> Sensitivity to 1% change in interest rates
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
