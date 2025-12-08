import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface MonteCarloChartProps {
  meanPath: number[];
  ci95Lower: number[];
  ci95Upper: number[];
  ci68Lower: number[];
  ci68Upper: number[];
  currentPrice: number;
  forecastDays: number;
}

export function MonteCarloChart({
  meanPath,
  ci95Lower,
  ci95Upper,
  ci68Lower,
  ci68Upper,
  currentPrice,
  forecastDays
}: MonteCarloChartProps) {
  // Prepare data for chart
  const data = meanPath.map((price, index) => ({
    day: index,
    mean: price,
    ci95Lower: ci95Lower[index],
    ci95Upper: ci95Upper[index],
    ci68Lower: ci68Lower[index],
    ci68Upper: ci68Upper[index],
  }));

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="color95" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.05}/>
            </linearGradient>
            <linearGradient id="color68" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.15}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis 
            dataKey="day" 
            stroke="#888"
            label={{ value: 'Days', position: 'insideBottom', offset: -5, fill: '#888' }}
          />
          <YAxis 
            stroke="#888"
            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#888' }}
            domain={['dataMin - 10', 'dataMax + 10']}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
            labelStyle={{ color: '#888' }}
            formatter={(value: number) => `$${value.toFixed(2)}`}
          />
          <Legend />
          
          {/* 95% Confidence Interval */}
          <Area
            type="monotone"
            dataKey="ci95Upper"
            stroke="none"
            fill="url(#color95)"
            name="95% CI Upper"
          />
          <Area
            type="monotone"
            dataKey="ci95Lower"
            stroke="none"
            fill="url(#color95)"
            name="95% CI Lower"
          />
          
          {/* 68% Confidence Interval */}
          <Area
            type="monotone"
            dataKey="ci68Upper"
            stroke="none"
            fill="url(#color68)"
            name="68% CI Upper"
          />
          <Area
            type="monotone"
            dataKey="ci68Lower"
            stroke="none"
            fill="url(#color68)"
            name="68% CI Lower"
          />
          
          {/* Mean Path */}
          <Line
            type="monotone"
            dataKey="mean"
            stroke="#10b981"
            strokeWidth={3}
            dot={false}
            name="Expected Price"
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-muted-foreground text-center">
        Monte Carlo Simulation: 20,000 paths with fat-tail distributions
      </div>
    </div>
  );
}
