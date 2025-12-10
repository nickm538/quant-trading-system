"""
Integrated Real-Time Trading Service
Orchestrates squeeze monitoring, position tracking, exit signals, and alerts
"""

import threading
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import yfinance as yf
import pandas as pd

from python_system.realtime.squeeze_monitor import get_squeeze_monitor
from python_system.realtime.exit_strategy import PositionTracker, ExitStrategy, PositionSide
from python_system.realtime.alert_system import AlertSystem, SqueezeAlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingService:
    """
    Integrated real-time trading service
    Manages squeeze monitoring, positions, exits, and alerts
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize trading service
        
        Args:
            api_key: Finnhub API key (optional)
        """
        # Initialize components
        self.squeeze_monitor = get_squeeze_monitor(api_key)
        self.exit_strategy = ExitStrategy(
            profit_target_pct=0.50,
            stop_loss_pct=0.30,
            min_dte_exit=3
        )
        self.position_tracker = PositionTracker(self.exit_strategy)
        self.alert_system = AlertSystem()
        self.alert_manager = SqueezeAlertManager(self.alert_system)
        
        # Register callbacks
        self.squeeze_monitor.add_squeeze_fire_callback(self._on_squeeze_fire)
        self.squeeze_monitor.add_squeeze_update_callback(self._on_squeeze_update)
        self.position_tracker.add_exit_callback(self._on_exit_signal)
        
        # Monitoring thread
        self.monitor_thread = None
        self.running = False
        
        # Watchlist
        self.watchlist = set()
    
    def _on_squeeze_fire(self, squeeze_state: Dict):
        """Handle squeeze fire event"""
        self.alert_manager.on_squeeze_fire(squeeze_state)
    
    def _on_squeeze_update(self, squeeze_state: Dict):
        """Handle squeeze update event"""
        # Alert if squeeze active for 3+ bars
        if squeeze_state['squeeze_active'] and squeeze_state['squeeze_bars'] >= 3:
            self.alert_manager.on_squeeze_active(squeeze_state)
    
    def _on_exit_signal(self, position_id: str, position: Dict, signal, reason: str):
        """Handle exit signal event"""
        self.alert_manager.on_exit_signal(position_id, position, signal, reason)
    
    def _position_monitor_loop(self):
        """Monitor positions for exit signals"""
        logger.info("Position monitor loop started")
        
        while self.running:
            try:
                # Get all positions
                positions = self.position_tracker.get_all_positions()
                
                if not positions:
                    time.sleep(5)
                    continue
                
                # Get current prices and squeeze states
                price_data = {}
                squeeze_states = {}
                
                for position_id, position in positions.items():
                    symbol = position['symbol']
                    
                    # Get squeeze state
                    squeeze_state = self.squeeze_monitor.get_squeeze_state(symbol)
                    if squeeze_state:
                        squeeze_states[symbol] = squeeze_state
                        price_data[symbol] = squeeze_state['current_price']
                
                # Check all positions for exit signals
                if price_data and squeeze_states:
                    exit_signals = self.position_tracker.check_all_positions(
                        price_data, squeeze_states
                    )
                    
                    if exit_signals:
                        logger.info(f"Found {len(exit_signals)} exit signals")
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                time.sleep(30)
        
        logger.info("Position monitor loop stopped")
    
    def start(self):
        """Start the trading service"""
        if self.running:
            logger.warning("Trading service already running")
            return
        
        # Start squeeze monitor
        self.squeeze_monitor.start()
        
        # Start position monitor
        self.running = True
        self.monitor_thread = threading.Thread(target=self._position_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Trading service started")
    
    def stop(self):
        """Stop the trading service"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.squeeze_monitor.stop()
        
        logger.info("Trading service stopped")
    
    def add_to_watchlist(self, symbol: str, fetch_history: bool = True):
        """
        Add symbol to watchlist
        
        Args:
            symbol: Stock symbol
            fetch_history: Whether to fetch historical data
        """
        if symbol in self.watchlist:
            logger.warning(f"{symbol} already in watchlist")
            return
        
        self.watchlist.add(symbol)
        
        # Fetch historical data if requested
        historical_df = None
        if fetch_history:
            try:
                logger.info(f"Fetching historical data for {symbol}...")
                hist = yf.download(symbol, period='2mo', interval='1h', progress=False)
                if not hist.empty:
                    hist = hist.reset_index()
                    hist.columns = [col.lower() if isinstance(col, str) else col for col in hist.columns]
                    historical_df = hist
                    logger.info(f"Fetched {len(hist)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching history for {symbol}: {e}")
        
        # Add to squeeze monitor
        self.squeeze_monitor.add_symbol(symbol, historical_df)
        
        logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist"""
        if symbol not in self.watchlist:
            return
        
        self.watchlist.remove(symbol)
        self.squeeze_monitor.remove_symbol(symbol)
        
        logger.info(f"Removed {symbol} from watchlist")
    
    def add_position(self, position_id: str, position_data: Dict):
        """
        Add position to tracking
        
        Args:
            position_id: Unique position identifier
            position_data: Position data
        """
        # Ensure symbol is in watchlist
        symbol = position_data['symbol']
        if symbol not in self.watchlist:
            self.add_to_watchlist(symbol)
        
        # Add to position tracker
        self.position_tracker.add_position(position_id, position_data)
        
        logger.info(f"Added position {position_id}")
    
    def remove_position(self, position_id: str):
        """Remove position from tracking"""
        self.position_tracker.remove_position(position_id)
        logger.info(f"Removed position {position_id}")
    
    def get_squeeze_state(self, symbol: str) -> Optional[Dict]:
        """Get current squeeze state for symbol"""
        return self.squeeze_monitor.get_squeeze_state(symbol)
    
    def get_all_squeeze_states(self) -> Dict[str, Dict]:
        """Get all squeeze states"""
        return self.squeeze_monitor.get_all_squeeze_states()
    
    def get_position(self, position_id: str) -> Optional[Dict]:
        """Get position data"""
        return self.position_tracker.get_position(position_id)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions"""
        return self.position_tracker.get_all_positions()
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List:
        """Get active alerts"""
        return self.alert_system.get_active_alerts(symbol)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert"""
        self.alert_system.acknowledge_alert(alert_id)
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'running': self.running,
            'watchlist': list(self.watchlist),
            'squeeze_monitor': self.squeeze_monitor.get_status(),
            'positions': len(self.position_tracker.get_all_positions()),
            'active_alerts': len(self.alert_system.get_active_alerts()),
            'alert_stats': self.alert_system.get_stats()
        }


# Singleton instance
_service_instance = None
_service_lock = threading.Lock()


def get_trading_service(api_key: str = None) -> TradingService:
    """Get singleton instance of trading service"""
    global _service_instance
    
    with _service_lock:
        if _service_instance is None:
            _service_instance = TradingService(api_key)
        
        return _service_instance


if __name__ == '__main__':
    # Test the trading service
    service = get_trading_service()
    
    # Start service
    service.start()
    
    # Add symbols to watchlist
    print("Adding symbols to watchlist...")
    service.add_to_watchlist('AAPL')
    service.add_to_watchlist('MSFT')
    
    # Add test position
    print("\nAdding test position...")
    from datetime import timedelta
    service.add_position('test_pos_1', {
        'symbol': 'AAPL',
        'side': 'long_call',
        'entry_price': 180.0,
        'entry_option_price': 5.0,
        'strike': 185.0,
        'expiration': datetime.now() + timedelta(days=30),
        'entry_date': datetime.now()
    })
    
    # Monitor for 60 seconds
    try:
        print("\nMonitoring... Press Ctrl+C to stop\n")
        for i in range(12):
            time.sleep(5)
            
            # Get status
            status = service.get_status()
            print(f"\n[{i*5}s] Status:")
            print(f"  Watchlist: {status['watchlist']}")
            print(f"  Positions: {status['positions']}")
            print(f"  Active Alerts: {status['active_alerts']}")
            
            # Get squeeze states
            states = service.get_all_squeeze_states()
            for symbol, state in states.items():
                status_icon = "🔴" if state['squeeze_active'] else "🟢"
                print(f"  {symbol}: {status_icon} Momentum {state['momentum']:+.2f}")
            
            # Get active alerts
            alerts = service.get_active_alerts()
            if alerts:
                print(f"\n  Active Alerts:")
                for alert in alerts[:3]:
                    print(f"    - {alert.message}")
    
    except KeyboardInterrupt:
        print("\n\nStopping service...")
    
    service.stop()
    print("Service stopped")
