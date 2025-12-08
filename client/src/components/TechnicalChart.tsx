import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PriceData {
  date: string;
  close: number;
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  bb_upper?: number;
  bb_lower?: number;
}

interface TechnicalChartProps {
  priceData: PriceData[];
  title?: string;
}

export function TechnicalChart({ priceData, title = "Price & Technical Indicators" }: TechnicalChartProps) {
  // Take last 60 days for better visibility
  const displayData = priceData.slice(-60);

  return (
    <div className="w-full h-[400px]">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={displayData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis 
            dataKey="date" 
            stroke="#888"
            tick={{ fontSize: 10 }}
            interval={Math.floor(displayData.length / 6)}
          />
          <YAxis 
            stroke="#888"
            domain={['dataMin - 5', 'dataMax + 5']}
            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft', fill: '#888' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
            labelStyle={{ color: '#888' }}
            formatter={(value: number) => `$${value.toFixed(2)}`}
          />
          <Legend />
          
          {/* Bollinger Bands */}
          {displayData[0]?.bb_upper && (
            <>
              <Line
                type="monotone"
                dataKey="bb_upper"
                stroke="#ef4444"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                name="BB Upper"
              />
              <Line
                type="monotone"
                dataKey="bb_lower"
                stroke="#ef4444"
                strokeWidth={1}
                strokeDasharray="3 3"
                dot={false}
                name="BB Lower"
              />
            </>
          )}
          
          {/* Moving Averages */}
          {displayData[0]?.sma_20 && (
            <Line
              type="monotone"
              dataKey="sma_20"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
              name="SMA 20"
            />
          )}
          {displayData[0]?.sma_50 && (
            <Line
              type="monotone"
              dataKey="sma_50"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              name="SMA 50"
            />
          )}
          {displayData[0]?.sma_200 && (
            <Line
              type="monotone"
              dataKey="sma_200"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              name="SMA 200"
            />
          )}
          
          {/* Price */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#10b981"
            strokeWidth={2.5}
            dot={false}
            name="Close Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
