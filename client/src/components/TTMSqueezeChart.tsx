import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

interface TTMSqueezeDataPoint {
  date: string;
  momentum: number;
  squeeze_active: boolean;
}

interface TTMSqueezeChartProps {
  data: TTMSqueezeDataPoint[];
  title?: string;
  height?: number;
}

export function TTMSqueezeChart({ 
  data, 
  title = "TTM Squeeze Momentum Histogram",
  height = 300
}: TTMSqueezeChartProps) {
  
  // Take last 60 bars for visibility
  const displayData = data.slice(-60);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 shadow-lg">
          <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {data.date}
          </p>
          <p className={`text-sm font-bold ${data.momentum >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            Momentum: {data.momentum.toFixed(2)}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <div className={`w-2 h-2 rounded-full ${data.squeeze_active ? 'bg-red-500' : 'bg-green-500'}`} />
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {data.squeeze_active ? 'Squeeze ON' : 'Squeeze OFF'}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  // Get bar color based on momentum and squeeze state
  const getBarColor = (momentum: number, squeeze_active: boolean) => {
    if (momentum >= 0) {
      // Positive momentum (bullish)
      return squeeze_active ? '#22c55e' : '#86efac'; // Darker green when squeeze active
    } else {
      // Negative momentum (bearish)
      return squeeze_active ? '#ef4444' : '#fca5a5'; // Darker red when squeeze active
    }
  };

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-gray-100">{title}</h3>
      )}
      
      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-gray-600 dark:text-gray-400">Squeeze ON</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-gray-600 dark:text-gray-400">Squeeze OFF</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-3 bg-green-600" />
          <span className="text-gray-600 dark:text-gray-400">Bullish</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-3 bg-red-600" />
          <span className="text-gray-600 dark:text-gray-400">Bearish</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart 
          data={displayData} 
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.3} />
          
          <XAxis 
            dataKey="date" 
            stroke="#888"
            tick={{ fontSize: 10 }}
            interval={Math.floor(displayData.length / 8)}
          />
          
          <YAxis 
            stroke="#888"
            label={{ 
              value: 'Momentum', 
              angle: -90, 
              position: 'insideLeft', 
              fill: '#888',
              style: { fontSize: 12 }
            }}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          {/* Zero line */}
          <ReferenceLine y={0} stroke="#666" strokeWidth={2} />
          
          {/* Momentum bars with dynamic colors */}
          <Bar 
            dataKey="momentum" 
            radius={[4, 4, 0, 0]}
          >
            {displayData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getBarColor(entry.momentum, entry.squeeze_active)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Interpretation guide */}
      <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800">
        <p className="text-xs text-gray-600 dark:text-gray-400">
          <span className="font-semibold">Interpretation:</span> Darker bars indicate squeeze is active (volatility compression). 
          Green bars = bullish momentum, Red bars = bearish momentum. 
          Watch for squeeze fires (transition from red dots to green dots) with strong momentum for high-probability trades.
        </p>
      </div>
    </div>
  );
}

// Mini version for compact displays
interface TTMSqueezeMiniChartProps {
  data: TTMSqueezeDataPoint[];
  width?: number;
  height?: number;
}

export function TTMSqueezeMiniChart({ 
  data, 
  width = 120, 
  height = 40 
}: TTMSqueezeMiniChartProps) {
  
  // Take last 20 bars for mini chart
  const displayData = data.slice(-20);

  return (
    <ResponsiveContainer width={width} height={height}>
      <BarChart data={displayData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
        <Bar dataKey="momentum" radius={[2, 2, 0, 0]}>
          {displayData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.momentum >= 0 ? '#22c55e' : '#ef4444'}
              opacity={entry.squeeze_active ? 1.0 : 0.5}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
