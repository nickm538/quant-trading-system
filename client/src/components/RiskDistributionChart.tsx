import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

interface RiskDistributionChartProps {
  returns: number[];
  var95: number;
  cvar95: number;
}

export function RiskDistributionChart({ returns, var95, cvar95 }: RiskDistributionChartProps) {
  // Create histogram bins
  const numBins = 50;
  const minReturn = Math.min(...returns);
  const maxReturn = Math.max(...returns);
  const binWidth = (maxReturn - minReturn) / numBins;
  
  const bins = Array(numBins).fill(0).map((_, i) => ({
    return: (minReturn + i * binWidth + binWidth / 2) * 100,
    count: 0,
    label: `${((minReturn + i * binWidth) * 100).toFixed(1)}%`
  }));
  
  // Fill bins
  returns.forEach(ret => {
    const binIndex = Math.min(
      Math.floor((ret - minReturn) / binWidth),
      numBins - 1
    );
    if (binIndex >= 0 && binIndex < numBins) {
      bins[binIndex].count++;
    }
  });

  return (
    <div className="w-full h-[350px]">
      <h3 className="text-lg font-semibold mb-2">Return Distribution & Risk Metrics</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bins} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis 
            dataKey="return"
            stroke="#888"
            label={{ value: 'Return (%)', position: 'insideBottom', offset: -5, fill: '#888' }}
            tick={{ fontSize: 10 }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
          />
          <YAxis 
            stroke="#888"
            label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#888' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
            labelStyle={{ color: '#888' }}
            formatter={(value: number) => [`${value} simulations`, 'Frequency']}
            labelFormatter={(value) => `Return: ${Number(value).toFixed(1)}%`}
          />
          <Legend />
          
          {/* VaR 95% line */}
          <ReferenceLine 
            x={var95 * 100} 
            stroke="#ef4444" 
            strokeWidth={2}
            label={{ value: `VaR 95%: ${(var95 * 100).toFixed(1)}%`, position: 'top', fill: '#ef4444' }}
          />
          
          {/* CVaR 95% line */}
          <ReferenceLine 
            x={cvar95 * 100} 
            stroke="#dc2626" 
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{ value: `CVaR 95%: ${(cvar95 * 100).toFixed(1)}%`, position: 'bottom', fill: '#dc2626' }}
          />
          
          <Bar dataKey="count" fill="#10b981" name="Simulations" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-muted-foreground">
        <div>VaR (95%): Maximum loss at 95% confidence = {(var95 * 100).toFixed(2)}%</div>
        <div>CVaR (95%): Average loss beyond VaR = {(cvar95 * 100).toFixed(2)}%</div>
      </div>
    </div>
  );
}
