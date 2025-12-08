import { Server as SocketIOServer } from 'socket.io';
import type { Server as HTTPServer } from 'http';

interface PriceUpdate {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

let io: SocketIOServer | null = null;
const activeSymbols = new Set<string>();
const priceIntervals = new Map<string, NodeJS.Timeout>();

/**
 * Initialize WebSocket server for real-time price streaming
 */
export function initializePriceStream(httpServer: HTTPServer) {
  io = new SocketIOServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    },
    path: "/api/socket.io"
  });

  io.on('connection', (socket) => {
    console.log(`[WebSocket] Client connected: ${socket.id}`);

    // Handle stock subscription
    socket.on('subscribe', (symbol: string) => {
      console.log(`[WebSocket] ${socket.id} subscribed to ${symbol}`);
      socket.join(symbol);
      activeSymbols.add(symbol);
      
      // Start price updates for this symbol if not already running
      if (!priceIntervals.has(symbol)) {
        startPriceUpdates(symbol);
      }
    });

    // Handle stock unsubscription
    socket.on('unsubscribe', (symbol: string) => {
      console.log(`[WebSocket] ${socket.id} unsubscribed from ${symbol}`);
      socket.leave(symbol);
      
      // Check if anyone else is subscribed
      const room = io?.sockets.adapter.rooms.get(symbol);
      if (!room || room.size === 0) {
        stopPriceUpdates(symbol);
        activeSymbols.delete(symbol);
      }
    });

    socket.on('disconnect', () => {
      console.log(`[WebSocket] Client disconnected: ${socket.id}`);
    });
  });

  console.log('[WebSocket] Price streaming initialized');
  return io;
}

/**
 * Start broadcasting price updates for a symbol
 */
function startPriceUpdates(symbol: string) {
  const interval = setInterval(async () => {
    try {
      const priceData = await fetchLatestPrice(symbol);
      if (priceData && io) {
        io.to(symbol).emit('price-update', priceData);
      }
    } catch (error) {
      console.error(`[WebSocket] Error fetching price for ${symbol}:`, error);
    }
  }, 5000); // Update every 5 seconds

  priceIntervals.set(symbol, interval);
  console.log(`[WebSocket] Started price updates for ${symbol}`);
}

/**
 * Stop broadcasting price updates for a symbol
 */
function stopPriceUpdates(symbol: string) {
  const interval = priceIntervals.get(symbol);
  if (interval) {
    clearInterval(interval);
    priceIntervals.delete(symbol);
    console.log(`[WebSocket] Stopped price updates for ${symbol}`);
  }
}

/**
 * Fetch latest price from Yahoo Finance
 */
async function fetchLatestPrice(symbol: string): Promise<PriceUpdate | null> {
  try {
    // Use the data API client
    const { callDataApi } = await import('./_core/dataApi');
    
    const response = await callDataApi('YahooFinance/get_stock_chart', {
      query: {
        symbol,
        region: 'US',
        interval: '1m',
        range: '1d',
        includeAdjustedClose: false
      }
    });

    if (!response || !(response as any).chart || !(response as any).chart.result) {
      return null;
    }

    const result = (response as any).chart.result[0];
    const meta = result.meta;
    const quotes = result.indicators.quote[0];
    const timestamps = result.timestamp;

    // Get the latest data point
    const lastIndex = timestamps.length - 1;
    const currentPrice = quotes.close[lastIndex];
    const previousClose = meta.chartPreviousClose;

    if (!currentPrice) {
      return null;
    }

    const change = currentPrice - previousClose;
    const changePercent = (change / previousClose) * 100;

    return {
      symbol,
      price: currentPrice,
      change,
      changePercent,
      volume: quotes.volume[lastIndex] || 0,
      timestamp: Date.now()
    };
  } catch (error) {
    console.error(`[WebSocket] Error in fetchLatestPrice for ${symbol}:`, error);
    return null;
  }
}

/**
 * Manually broadcast a price update (for testing or manual triggers)
 */
export function broadcastPriceUpdate(symbol: string, priceData: PriceUpdate) {
  if (io) {
    io.to(symbol).emit('price-update', priceData);
  }
}

/**
 * Get list of actively streamed symbols
 */
export function getActiveSymbols(): string[] {
  return Array.from(activeSymbols);
}

/**
 * Cleanup all price streams
 */
export function cleanupPriceStreams() {
  priceIntervals.forEach((interval, symbol) => {
    clearInterval(interval);
  });
  priceIntervals.clear();
  activeSymbols.clear();
  console.log('[WebSocket] All price streams cleaned up');
}
