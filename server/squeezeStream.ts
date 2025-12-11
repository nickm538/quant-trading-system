/**
 * TTM Squeeze Real-Time WebSocket Stream
 * Provides live squeeze state updates, alerts, and position monitoring
 */

import { Server as SocketIOServer } from 'socket.io';
import type { Server as HTTPServer } from 'http';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface SqueezeState {
  symbol: string;
  timestamp: string;
  squeeze_active: boolean;
  squeeze_bars: number;
  momentum: number;
  squeeze_signal: 'long' | 'short' | 'active' | 'none';
  current_price: number;
  bars_in_history: number;
}

interface Alert {
  id: string;
  type: string;
  priority: string;
  symbol: string;
  message: string;
  data: any;
  timestamp: string;
  acknowledged: boolean;
}

interface Position {
  position_id: string;
  symbol: string;
  side: 'long_call' | 'long_put';
  entry_price: number;
  entry_option_price: number;
  strike: number;
  expiration: string;
  entry_date: string;
}

let io: SocketIOServer | null = null;
let pythonProcess: ChildProcess | null = null;
const activeSqueezeSubscriptions = new Set<string>();

/**
 * Initialize TTM Squeeze WebSocket server
 */
export function initializeSqueezeStream(httpServer: HTTPServer) {
  io = new SocketIOServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    },
    path: "/api/squeeze-socket.io"
  });

  io.on('connection', (socket) => {
    console.log(`[Squeeze WS] Client connected: ${socket.id}`);

    // Send current status on connect
    socket.emit('squeeze:status', {
      connected: true,
      timestamp: new Date().toISOString()
    });

    // Handle squeeze monitoring subscription
    socket.on('squeeze:subscribe', (symbol: string) => {
      console.log(`[Squeeze WS] ${socket.id} subscribed to ${symbol}`);
      socket.join(`squeeze:${symbol}`);
      activeSqueezeSubscriptions.add(symbol);
      
      // Add symbol to Python trading service watchlist
      sendToPythonService({
        action: 'add_watchlist',
        symbol: symbol
      });
    });

    // Handle squeeze monitoring unsubscription
    socket.on('squeeze:unsubscribe', (symbol: string) => {
      console.log(`[Squeeze WS] ${socket.id} unsubscribed from ${symbol}`);
      socket.leave(`squeeze:${symbol}`);
      
      // Check if anyone else is subscribed
      const room = io?.sockets.adapter.rooms.get(`squeeze:${symbol}`);
      if (!room || room.size === 0) {
        activeSqueezeSubscriptions.delete(symbol);
        
        // Remove from Python service watchlist
        sendToPythonService({
          action: 'remove_watchlist',
          symbol: symbol
        });
      }
    });

    // Handle position tracking
    socket.on('position:add', (position: Position) => {
      console.log(`[Squeeze WS] Adding position: ${position.position_id}`);
      sendToPythonService({
        action: 'add_position',
        position_id: position.position_id,
        position_data: position
      });
    });

    socket.on('position:remove', (position_id: string) => {
      console.log(`[Squeeze WS] Removing position: ${position_id}`);
      sendToPythonService({
        action: 'remove_position',
        position_id: position_id
      });
    });

    // Handle alert acknowledgment
    socket.on('alert:acknowledge', (alert_id: string) => {
      console.log(`[Squeeze WS] Acknowledging alert: ${alert_id}`);
      sendToPythonService({
        action: 'acknowledge_alert',
        alert_id: alert_id
      });
    });

    // Handle status request
    socket.on('squeeze:get_status', () => {
      sendToPythonService({
        action: 'get_status',
        reply_to: socket.id
      });
    });

    // Handle disconnect
    socket.on('disconnect', () => {
      console.log(`[Squeeze WS] Client disconnected: ${socket.id}`);
    });
  });

  // Start Python trading service bridge
  startPythonServiceBridge();

  console.log('[Squeeze WS] TTM Squeeze WebSocket server initialized');
}

/**
 * Start Python trading service bridge
 * Spawns Python process and handles IPC
 */
function startPythonServiceBridge() {
  if (pythonProcess) {
    console.log('[Squeeze WS] Python service already running');
    return;
  }

  const pythonScript = path.join(__dirname, '../python_system/realtime/websocket_bridge.py');
  
  pythonProcess = spawn('python3.11', [pythonScript], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  // Handle stdout (messages from Python)
  pythonProcess.stdout?.on('data', (data) => {
    const lines = data.toString().split('\n');
    
    for (const line of lines) {
      if (!line.trim()) continue;
      
      try {
        const message = JSON.parse(line);
        handlePythonMessage(message);
      } catch (e) {
        // Not JSON, probably a log message
        console.log(`[Python] ${line}`);
      }
    }
  });

  // Handle stderr (errors from Python)
  pythonProcess.stderr?.on('data', (data) => {
    console.error(`[Python Error] ${data.toString()}`);
  });

  // Handle process exit
  pythonProcess.on('exit', (code) => {
    console.log(`[Python] Process exited with code ${code}`);
    pythonProcess = null;
    
    // Restart after 5 seconds
    setTimeout(() => {
      console.log('[Python] Restarting service...');
      startPythonServiceBridge();
    }, 5000);
  });

  console.log('[Squeeze WS] Python trading service bridge started');
}

/**
 * Send message to Python trading service
 */
function sendToPythonService(message: any) {
  if (!pythonProcess || !pythonProcess.stdin) {
    console.error('[Squeeze WS] Python service not available');
    return;
  }

  try {
    pythonProcess.stdin.write(JSON.stringify(message) + '\n');
  } catch (e) {
    console.error('[Squeeze WS] Error sending to Python service:', e);
  }
}

/**
 * Handle messages from Python trading service
 */
function handlePythonMessage(message: any) {
  const { type, data } = message;

  switch (type) {
    case 'squeeze_update':
      // Broadcast squeeze state update
      const squeezeState: SqueezeState = data;
      io?.to(`squeeze:${squeezeState.symbol}`).emit('squeeze:update', squeezeState);
      break;

    case 'squeeze_fire':
      // Broadcast squeeze fire alert
      const fireState: SqueezeState = data;
      io?.to(`squeeze:${fireState.symbol}`).emit('squeeze:fire', fireState);
      console.log(`[Squeeze WS] 🔥 Squeeze fire: ${fireState.symbol} - ${fireState.squeeze_signal}`);
      break;

    case 'alert':
      // Broadcast alert
      const alert: Alert = data;
      io?.to(`squeeze:${alert.symbol}`).emit('alert:new', alert);
      io?.emit('alert:global', alert); // Also broadcast globally
      break;

    case 'exit_signal':
      // Broadcast exit signal
      const exitData = data;
      io?.to(`squeeze:${exitData.position.symbol}`).emit('position:exit', exitData);
      console.log(`[Squeeze WS] 📤 Exit signal: ${exitData.position_id} - ${exitData.signal}`);
      break;

    case 'status':
      // Send status to requesting client
      if (data.reply_to) {
        io?.to(data.reply_to).emit('squeeze:status', data.status);
      }
      break;

    default:
      console.log(`[Squeeze WS] Unknown message type: ${type}`);
  }
}

/**
 * Broadcast squeeze state update (called from external code)
 */
export function broadcastSqueezeUpdate(squeezeState: SqueezeState) {
  io?.to(`squeeze:${squeezeState.symbol}`).emit('squeeze:update', squeezeState);
}

/**
 * Broadcast alert (called from external code)
 */
export function broadcastAlert(alert: Alert) {
  io?.to(`squeeze:${alert.symbol}`).emit('alert:new', alert);
  io?.emit('alert:global', alert);
}

/**
 * Get active squeeze subscriptions
 */
export function getActiveSqueezeSubscriptions(): string[] {
  return Array.from(activeSqueezeSubscriptions);
}

/**
 * Stop Python service bridge
 */
export function stopPythonServiceBridge() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
    console.log('[Squeeze WS] Python service bridge stopped');
  }
}
