/**
 * React Hook for TTM Squeeze Real-Time WebSocket Stream
 * Connects to backend squeeze monitoring service
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

export interface SqueezeState {
  symbol: string;
  timestamp: string;
  squeeze_active: boolean;
  squeeze_bars: number;
  momentum: number;
  squeeze_signal: 'long' | 'short' | 'active' | 'none';
  current_price: number;
  bars_in_history: number;
}

export interface Alert {
  id: string;
  timestamp: string;
  type: string;
  symbol: string;
  message: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  data?: any;
}

export interface Position {
  position_id: string;
  symbol: string;
  side: 'long_call' | 'long_put';
  entry_price: number;
  entry_option_price: number;
  strike: number;
  expiration: string;
  entry_date: string;
}

export interface SystemStatus {
  running: boolean;
  watchlist: string[];
  positions: number;
  active_alerts: number;
}

interface UseSqueezeStreamReturn {
  // Connection state
  connected: boolean;
  error: string | null;
  
  // Data
  squeezeStates: Record<string, SqueezeState>;
  alerts: Alert[];
  positions: Position[];
  status: SystemStatus | null;
  
  // Actions
  addToWatchlist: (symbol: string) => void;
  removeFromWatchlist: (symbol: string) => void;
  addPosition: (position: Position) => void;
  removePosition: (positionId: string) => void;
  getStatus: () => void;
  
  // Connection control
  connect: () => void;
  disconnect: () => void;
}

export function useSqueezeStream(autoConnect: boolean = true): UseSqueezeStreamReturn {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [squeezeStates, setSqueezeStates] = useState<Record<string, SqueezeState>>({});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  
  const socketRef = useRef<Socket | null>(null);
  
  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      console.log('[SqueezeStream] Already connected');
      return;
    }
    
    console.log('[SqueezeStream] Connecting to squeeze stream...');
    
    // Connect to Socket.IO server
    const socket = io('/', {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity,
    });
    
    // Connection events
    socket.on('connect', () => {
      console.log('[SqueezeStream] Connected');
      setConnected(true);
      setError(null);
    });
    
    socket.on('disconnect', (reason) => {
      console.log('[SqueezeStream] Disconnected:', reason);
      setConnected(false);
    });
    
    socket.on('connect_error', (err) => {
      console.error('[SqueezeStream] Connection error:', err);
      setError(err.message);
      setConnected(false);
    });
    
    // Squeeze events
    socket.on('squeeze:update', (state: SqueezeState) => {
      console.log('[SqueezeStream] Squeeze update:', state.symbol, state);
      setSqueezeStates(prev => ({
        ...prev,
        [state.symbol]: state
      }));
    });
    
    socket.on('squeeze:fire', (state: SqueezeState) => {
      console.log('[SqueezeStream] 🔥 Squeeze fired:', state.symbol, state.squeeze_signal);
      setSqueezeStates(prev => ({
        ...prev,
        [state.symbol]: state
      }));
    });
    
    // Alert events
    socket.on('alert:new', (alert: Alert) => {
      console.log('[SqueezeStream] 🔔 New alert:', alert.type, alert.message);
      setAlerts(prev => [alert, ...prev].slice(0, 100)); // Keep last 100 alerts
    });
    
    // Position events
    socket.on('position:exit', (data: any) => {
      console.log('[SqueezeStream] 📤 Position exit:', data.position_id, data.reason);
      setPositions(prev => prev.filter(p => p.position_id !== data.position_id));
    });
    
    // Status events
    socket.on('status', (data: any) => {
      console.log('[SqueezeStream] Status update:', data);
      if (data.data?.status) {
        setStatus(data.data.status);
      }
    });
    
    socketRef.current = socket;
  }, []);
  
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      console.log('[SqueezeStream] Disconnecting...');
      socketRef.current.disconnect();
      socketRef.current = null;
      setConnected(false);
    }
  }, []);
  
  const addToWatchlist = useCallback((symbol: string) => {
    if (!socketRef.current?.connected) {
      console.warn('[SqueezeStream] Not connected, cannot add to watchlist');
      return;
    }
    
    console.log('[SqueezeStream] Adding to watchlist:', symbol);
    socketRef.current.emit('squeeze:subscribe', symbol);
  }, []);
  
  const removeFromWatchlist = useCallback((symbol: string) => {
    if (!socketRef.current?.connected) {
      console.warn('[SqueezeStream] Not connected, cannot remove from watchlist');
      return;
    }
    
    console.log('[SqueezeStream] Removing from watchlist:', symbol);
    socketRef.current.emit('squeeze:unsubscribe', symbol);
  }, []);
  
  const addPosition = useCallback((position: Position) => {
    if (!socketRef.current?.connected) {
      console.warn('[SqueezeStream] Not connected, cannot add position');
      return;
    }
    
    console.log('[SqueezeStream] Adding position:', position.position_id);
    socketRef.current.emit('position:add', position);
    setPositions(prev => [...prev, position]);
  }, []);
  
  const removePosition = useCallback((positionId: string) => {
    if (!socketRef.current?.connected) {
      console.warn('[SqueezeStream] Not connected, cannot remove position');
      return;
    }
    
    console.log('[SqueezeStream] Removing position:', positionId);
    socketRef.current.emit('position:remove', positionId);
    setPositions(prev => prev.filter(p => p.position_id !== positionId));
  }, []);
  
  const getStatus = useCallback(() => {
    if (!socketRef.current?.connected) {
      console.warn('[SqueezeStream] Not connected, cannot get status');
      return;
    }
    
    console.log('[SqueezeStream] Requesting status...');
    socketRef.current.emit('status:get');
  }, []);
  
  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);
  
  return {
    connected,
    error,
    squeezeStates,
    alerts,
    positions,
    status,
    addToWatchlist,
    removeFromWatchlist,
    addPosition,
    removePosition,
    getStatus,
    connect,
    disconnect,
  };
}
