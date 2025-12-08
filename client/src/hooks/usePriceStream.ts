import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

interface PriceUpdate {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

/**
 * React hook for real-time price streaming via WebSocket
 * 
 * @param symbol - Stock symbol to stream prices for
 * @param enabled - Whether to enable streaming (default: true)
 * @returns Latest price data and connection status
 */
export function usePriceStream(symbol: string | null, enabled: boolean = true) {
  const [priceData, setPriceData] = useState<PriceUpdate | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!symbol || !enabled) {
      return;
    }

    // Initialize socket connection
    const socket = io({
      path: '/api/socket.io',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    socketRef.current = socket;

    // Connection handlers
    socket.on('connect', () => {
      console.log(`[WebSocket] Connected, subscribing to ${symbol}`);
      setIsConnected(true);
      setError(null);
      socket.emit('subscribe', symbol);
    });

    socket.on('disconnect', () => {
      console.log('[WebSocket] Disconnected');
      setIsConnected(false);
    });

    socket.on('connect_error', (err) => {
      console.error('[WebSocket] Connection error:', err);
      setError(err.message);
      setIsConnected(false);
    });

    // Price update handler
    socket.on('price-update', (data: PriceUpdate) => {
      if (data.symbol === symbol) {
        setPriceData(data);
      }
    });

    // Cleanup
    return () => {
      if (socket.connected) {
        socket.emit('unsubscribe', symbol);
      }
      socket.disconnect();
    };
  }, [symbol, enabled]);

  return {
    priceData,
    isConnected,
    error,
    subscribe: (newSymbol: string) => {
      if (socketRef.current?.connected) {
        socketRef.current.emit('subscribe', newSymbol);
      }
    },
    unsubscribe: (oldSymbol: string) => {
      if (socketRef.current?.connected) {
        socketRef.current.emit('unsubscribe', oldSymbol);
      }
    }
  };
}
