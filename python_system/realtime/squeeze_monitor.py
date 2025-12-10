"""
TTM Squeeze Real-Time Monitoring Service
Continuously calculates TTM Squeeze on live market data
Detects squeeze fires and generates trading signals
"""

import threading
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import logging

from python_system.indicators.ttm_squeeze import TTMSqueeze
from python_system.realtime.finnhub_stream import get_realtime_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SqueezeMonitor:
    """
    Real-time TTM Squeeze monitoring service
    Maintains historical bars and calculates squeeze state continuously
    """
    
    def __init__(self, api_key: str = None, bar_interval_seconds: int = 60):
        """
        Initialize squeeze monitor
        
        Args:
            api_key: Finnhub API key (optional, uses env var if not provided)
            bar_interval_seconds: Interval for bar construction (default 60s = 1min bars)
        """
        self.bar_interval = bar_interval_seconds
        self.stream = get_realtime_stream(api_key)
        self.squeeze_calculator = TTMSqueeze()
        
        # Monitored symbols
        self.monitored_symbols = set()
        
        # Historical bars (symbol -> DataFrame)
        self.historical_bars = {}
        
        # Current bar being constructed (symbol -> bar data)
        self.current_bars = {}
        
        # Latest squeeze state (symbol -> squeeze data)
        self.squeeze_states = {}
        
        # Previous squeeze state for fire detection (symbol -> bool)
        self.previous_squeeze_active = {}
        
        # Callbacks for squeeze events
        self.squeeze_fire_callbacks = []
        self.squeeze_update_callbacks = []
        
        # Monitoring thread
        self.monitor_thread = None
        self.running = False
        
        # Register quote callback
        self.stream.add_quote_callback(self._on_quote)
    
    def _on_quote(self, quote: Dict):
        """Handle incoming quote from real-time stream"""
        symbol = quote['symbol']
        
        if symbol not in self.monitored_symbols:
            return
        
        # Update current bar
        self._update_current_bar(symbol, quote)
    
    def _update_current_bar(self, symbol: str, quote: Dict):
        """Update current bar with new quote"""
        timestamp = quote['datetime']
        price = quote['price']
        volume = quote.get('volume', 0)
        
        # Get or create current bar
        if symbol not in self.current_bars:
            self.current_bars[symbol] = {
                'start_time': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume,
                'trade_count': 1
            }
        else:
            bar = self.current_bars[symbol]
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += volume
            bar['trade_count'] += 1
    
    def _finalize_bar(self, symbol: str) -> Optional[Dict]:
        """Finalize current bar and start new one"""
        if symbol not in self.current_bars:
            return None
        
        bar = self.current_bars[symbol]
        
        # Only finalize if we have trades
        if bar['trade_count'] == 0:
            return None
        
        # Create finalized bar
        finalized = {
            'timestamp': bar['start_time'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume']
        }
        
        # Reset current bar
        self.current_bars[symbol] = {
            'start_time': datetime.now(),
            'open': bar['close'],
            'high': bar['close'],
            'low': bar['close'],
            'close': bar['close'],
            'volume': 0,
            'trade_count': 0
        }
        
        return finalized
    
    def _add_bar_to_history(self, symbol: str, bar: Dict):
        """Add finalized bar to historical data"""
        if symbol not in self.historical_bars:
            self.historical_bars[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Append bar
        new_row = pd.DataFrame([bar])
        self.historical_bars[symbol] = pd.concat([self.historical_bars[symbol], new_row], ignore_index=True)
        
        # Keep last 200 bars (enough for TTM Squeeze calculation)
        if len(self.historical_bars[symbol]) > 200:
            self.historical_bars[symbol] = self.historical_bars[symbol].iloc[-200:]
    
    def _calculate_squeeze(self, symbol: str):
        """Calculate TTM Squeeze for symbol"""
        if symbol not in self.historical_bars:
            return
        
        df = self.historical_bars[symbol]
        
        # Need at least 50 bars for accurate calculation
        if len(df) < 50:
            logger.warning(f"{symbol}: Not enough bars ({len(df)}) for squeeze calculation")
            return
        
        try:
            # Calculate squeeze
            result = self.squeeze_calculator.calculate(df)
            
            if not result['success']:
                logger.error(f"{symbol}: Squeeze calculation failed: {result.get('error')}")
                return
            
            # Get latest squeeze state
            latest_idx = len(result['squeeze_active']) - 1
            squeeze_active = result['squeeze_active'][latest_idx]
            momentum = result['momentum'][latest_idx]
            squeeze_signal = result['squeeze_signal'][latest_idx]
            
            # Count consecutive squeeze bars
            squeeze_bars = 0
            for i in range(latest_idx, -1, -1):
                if result['squeeze_active'][i]:
                    squeeze_bars += 1
                else:
                    break
            
            # Store squeeze state
            squeeze_state = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'squeeze_active': squeeze_active,
                'squeeze_bars': squeeze_bars,
                'momentum': momentum,
                'squeeze_signal': squeeze_signal,
                'current_price': df['close'].iloc[-1],
                'bars_in_history': len(df)
            }
            
            self.squeeze_states[symbol] = squeeze_state
            
            # Detect squeeze fire
            prev_active = self.previous_squeeze_active.get(symbol, False)
            if prev_active and not squeeze_active:
                # Squeeze just fired!
                logger.info(f"🔥 SQUEEZE FIRE: {symbol} - Signal: {squeeze_signal}, Momentum: {momentum:.2f}")
                self._trigger_squeeze_fire(squeeze_state)
            
            # Update previous state
            self.previous_squeeze_active[symbol] = squeeze_active
            
            # Trigger update callbacks
            self._trigger_squeeze_update(squeeze_state)
            
        except Exception as e:
            logger.error(f"{symbol}: Error calculating squeeze: {e}")
    
    def _trigger_squeeze_fire(self, squeeze_state: Dict):
        """Trigger squeeze fire callbacks"""
        for callback in self.squeeze_fire_callbacks:
            try:
                callback(squeeze_state)
            except Exception as e:
                logger.error(f"Squeeze fire callback error: {e}")
    
    def _trigger_squeeze_update(self, squeeze_state: Dict):
        """Trigger squeeze update callbacks"""
        for callback in self.squeeze_update_callbacks:
            try:
                callback(squeeze_state)
            except Exception as e:
                logger.error(f"Squeeze update callback error: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Squeeze monitor loop started")
        
        last_bar_time = {}
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Process each monitored symbol
                for symbol in list(self.monitored_symbols):
                    # Check if it's time to finalize bar
                    last_time = last_bar_time.get(symbol, current_time - timedelta(seconds=self.bar_interval))
                    elapsed = (current_time - last_time).total_seconds()
                    
                    if elapsed >= self.bar_interval:
                        # Finalize bar
                        bar = self._finalize_bar(symbol)
                        if bar:
                            self._add_bar_to_history(symbol, bar)
                            self._calculate_squeeze(symbol)
                        
                        last_bar_time[symbol] = current_time
                
                # Sleep for 1 second
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)
        
        logger.info("Squeeze monitor loop stopped")
    
    def start(self):
        """Start the squeeze monitor"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        # Start real-time stream if not already running
        if not self.stream.is_connected():
            self.stream.start()
        
        # Start monitor thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Squeeze monitor started")
    
    def stop(self):
        """Stop the squeeze monitor"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Squeeze monitor stopped")
    
    def add_symbol(self, symbol: str, historical_df: Optional[pd.DataFrame] = None):
        """
        Add symbol to monitoring
        
        Args:
            symbol: Stock symbol to monitor
            historical_df: Optional historical data to initialize (must have OHLCV columns)
        """
        if symbol in self.monitored_symbols:
            logger.warning(f"{symbol} already being monitored")
            return
        
        # Add to monitored symbols
        self.monitored_symbols.add(symbol)
        
        # Subscribe to real-time stream
        self.stream.subscribe([symbol])
        
        # Initialize with historical data if provided
        if historical_df is not None and len(historical_df) > 0:
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in historical_df.columns for col in required_cols):
                # Add timestamp if not present
                if 'timestamp' not in historical_df.columns:
                    historical_df = historical_df.copy()
                    historical_df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(historical_df), freq='1min')
                
                self.historical_bars[symbol] = historical_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                
                # Calculate initial squeeze
                self._calculate_squeeze(symbol)
                
                logger.info(f"Added {symbol} with {len(historical_df)} historical bars")
            else:
                logger.warning(f"{symbol}: Historical data missing required columns")
        else:
            logger.info(f"Added {symbol} (no historical data)")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from monitoring"""
        if symbol not in self.monitored_symbols:
            return
        
        self.monitored_symbols.remove(symbol)
        self.stream.unsubscribe([symbol])
        
        # Clean up data
        self.historical_bars.pop(symbol, None)
        self.current_bars.pop(symbol, None)
        self.squeeze_states.pop(symbol, None)
        self.previous_squeeze_active.pop(symbol, None)
        
        logger.info(f"Removed {symbol} from monitoring")
    
    def get_squeeze_state(self, symbol: str) -> Optional[Dict]:
        """Get latest squeeze state for symbol"""
        return self.squeeze_states.get(symbol)
    
    def get_all_squeeze_states(self) -> Dict[str, Dict]:
        """Get all squeeze states"""
        return self.squeeze_states.copy()
    
    def add_squeeze_fire_callback(self, callback: Callable[[Dict], None]):
        """Add callback for squeeze fire events"""
        self.squeeze_fire_callbacks.append(callback)
    
    def add_squeeze_update_callback(self, callback: Callable[[Dict], None]):
        """Add callback for squeeze update events"""
        self.squeeze_update_callbacks.append(callback)
    
    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            'running': self.running,
            'monitored_symbols': list(self.monitored_symbols),
            'symbols_with_data': list(self.squeeze_states.keys()),
            'stream_status': self.stream.get_status()
        }


# Singleton instance
_monitor_instance = None
_monitor_lock = threading.Lock()


def get_squeeze_monitor(api_key: str = None, bar_interval_seconds: int = 60) -> SqueezeMonitor:
    """Get singleton instance of squeeze monitor"""
    global _monitor_instance
    
    with _monitor_lock:
        if _monitor_instance is None:
            _monitor_instance = SqueezeMonitor(api_key, bar_interval_seconds)
        
        return _monitor_instance


if __name__ == '__main__':
    # Test the monitor
    import yfinance as yf
    
    monitor = get_squeeze_monitor()
    
    def on_fire(state):
        print(f"\n🔥 SQUEEZE FIRE: {state['symbol']}")
        print(f"   Signal: {state['squeeze_signal']}")
        print(f"   Momentum: {state['momentum']:.2f}")
        print(f"   Price: ${state['current_price']:.2f}")
    
    def on_update(state):
        status = "🔴 ON" if state['squeeze_active'] else "🟢 OFF"
        print(f"{state['symbol']}: {status} | Momentum: {state['momentum']:+.2f} | Bars: {state['squeeze_bars']}")
    
    monitor.add_squeeze_fire_callback(on_fire)
    monitor.add_squeeze_update_callback(on_update)
    
    # Get historical data
    print("Fetching historical data...")
    hist = yf.download('AAPL', period='1mo', interval='1h')
    hist = hist.reset_index()
    hist.columns = [col.lower() if isinstance(col, str) else col for col in hist.columns]
    
    monitor.start()
    monitor.add_symbol('AAPL', hist)
    
    try:
        print("\nMonitoring AAPL... Press Ctrl+C to stop")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()
