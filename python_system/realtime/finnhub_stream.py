"""
Real-Time Market Data Service using Finnhub WebSocket
Provides live quote updates for TTM Squeeze monitoring
"""

import websocket
import json
import threading
import time
import os
from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinnhubRealtimeStream:
    """
    Real-time market data stream using Finnhub WebSocket API
    Maintains live quotes for TTM Squeeze monitoring
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.ws = None
        self.thread = None
        self.running = False
        
        # Subscribed symbols
        self.subscribed_symbols = set()
        
        # Latest quotes (symbol -> quote data)
        self.latest_quotes = {}
        
        # Quote history for bar construction (symbol -> deque of quotes)
        self.quote_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Callbacks for quote updates
        self.quote_callbacks = []
        
        # Connection state
        self.connected = False
        self.last_heartbeat = None
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle trade messages
            if data.get('type') == 'trade':
                for trade in data.get('data', []):
                    symbol = trade.get('s')
                    price = trade.get('p')
                    volume = trade.get('v')
                    timestamp = trade.get('t')  # Unix timestamp in milliseconds
                    
                    if symbol and price:
                        # Update latest quote
                        quote = {
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'timestamp': timestamp,
                            'datetime': datetime.fromtimestamp(timestamp / 1000)
                        }
                        
                        self.latest_quotes[symbol] = quote
                        self.quote_history[symbol].append(quote)
                        
                        # Trigger callbacks
                        for callback in self.quote_callbacks:
                            try:
                                callback(quote)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
            
            # Handle ping/pong
            elif data.get('type') == 'ping':
                self.last_heartbeat = datetime.now()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Auto-reconnect if running
        if self.running:
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self._connect()
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connection established")
        self.connected = True
        self.last_heartbeat = datetime.now()
        
        # Resubscribe to all symbols
        for symbol in self.subscribed_symbols:
            self._send_subscribe(symbol)
    
    def _send_subscribe(self, symbol: str):
        """Send subscribe message"""
        if self.ws and self.connected:
            msg = json.dumps({'type': 'subscribe', 'symbol': symbol})
            self.ws.send(msg)
            logger.info(f"Subscribed to {symbol}")
    
    def _send_unsubscribe(self, symbol: str):
        """Send unsubscribe message"""
        if self.ws and self.connected:
            msg = json.dumps({'type': 'unsubscribe', 'symbol': symbol})
            self.ws.send(msg)
            logger.info(f"Unsubscribed from {symbol}")
    
    def _connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run in separate thread
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.thread.start()
            
            # Wait for connection
            timeout = 10
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                raise Exception("Connection timeout")
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise
    
    def start(self):
        """Start the real-time stream"""
        if self.running:
            logger.warning("Stream already running")
            return
        
        self.running = True
        self._connect()
        logger.info("Real-time stream started")
    
    def stop(self):
        """Stop the real-time stream"""
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Real-time stream stopped")
    
    def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        for symbol in symbols:
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.add(symbol)
                self._send_subscribe(symbol)
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
                self._send_unsubscribe(symbol)
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_quote_history(self, symbol: str, count: int = 100) -> List[Dict]:
        """Get recent quote history for symbol"""
        history = self.quote_history.get(symbol, deque())
        return list(history)[-count:]
    
    def add_quote_callback(self, callback: Callable[[Dict], None]):
        """Add callback for quote updates"""
        self.quote_callbacks.append(callback)
    
    def is_connected(self) -> bool:
        """Check if stream is connected"""
        return self.connected
    
    def get_status(self) -> Dict:
        """Get stream status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'subscribed_symbols': list(self.subscribed_symbols),
            'symbols_with_data': list(self.latest_quotes.keys()),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }


# Singleton instance
_stream_instance = None
_stream_lock = threading.Lock()


def get_realtime_stream(api_key: str = None) -> FinnhubRealtimeStream:
    """Get singleton instance of realtime stream"""
    global _stream_instance
    
    with _stream_lock:
        if _stream_instance is None:
            if api_key is None:
                # Try KEY first (user's env var), then FINNHUB_API_KEY
                api_key = os.getenv('KEY', os.getenv('FINNHUB_API_KEY'))
                if not api_key:
                    raise ValueError("Finnhub API key required (set KEY or FINNHUB_API_KEY environment variable)")
            
            _stream_instance = FinnhubRealtimeStream(api_key)
        
        return _stream_instance


if __name__ == '__main__':
    # Test the stream
    stream = get_realtime_stream()
    
    def on_quote(quote):
        print(f"{quote['symbol']}: ${quote['price']:.2f} @ {quote['datetime']}")
    
    stream.add_quote_callback(on_quote)
    stream.start()
    stream.subscribe(['AAPL', 'MSFT', 'GOOGL'])
    
    try:
        while True:
            time.sleep(1)
            status = stream.get_status()
            print(f"\nStatus: {status['connected']} | Symbols: {len(status['symbols_with_data'])}")
    except KeyboardInterrupt:
        print("\nStopping stream...")
        stream.stop()
