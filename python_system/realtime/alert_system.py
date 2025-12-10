"""
Real-Time Alert System for TTM Squeeze Trading
Sends alerts for squeeze fires, exit signals, and position updates
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import logging
import json
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Alert types"""
    SQUEEZE_FIRE = "squeeze_fire"
    SQUEEZE_ACTIVE = "squeeze_active"
    EXIT_SIGNAL = "exit_signal"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    MOMENTUM_REVERSAL = "momentum_reversal"
    TIME_DECAY_WARNING = "time_decay_warning"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Alert:
    """Alert data structure"""
    
    def __init__(self,
                 alert_type: AlertType,
                 priority: AlertPriority,
                 symbol: str,
                 message: str,
                 data: Dict = None):
        self.id = f"{symbol}_{alert_type.value}_{int(datetime.now().timestamp())}"
        self.alert_type = alert_type
        self.priority = priority
        self.symbol = symbol
        self.message = message
        self.data = data or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'type': self.alert_type.value,
            'priority': self.priority.value,
            'symbol': self.symbol,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }
    
    def __repr__(self) -> str:
        return f"Alert({self.priority.value.upper()}: {self.symbol} - {self.message})"


class AlertSystem:
    """
    Real-time alert system for trading signals
    Manages alert generation, delivery, and history
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize alert system
        
        Args:
            max_history: Maximum number of alerts to keep in history
        """
        self.max_history = max_history
        
        # Active alerts (not acknowledged)
        self.active_alerts = {}  # alert_id -> Alert
        
        # Alert history
        self.alert_history = deque(maxlen=max_history)
        
        # Alert callbacks (for real-time delivery)
        self.alert_callbacks = []
        
        # Alert filters (symbol -> enabled alert types)
        self.alert_filters = {}
    
    def create_alert(self,
                    alert_type: AlertType,
                    priority: AlertPriority,
                    symbol: str,
                    message: str,
                    data: Dict = None) -> Alert:
        """
        Create and send alert
        
        Args:
            alert_type: Type of alert
            priority: Priority level
            symbol: Stock symbol
            message: Alert message
            data: Additional data
        
        Returns:
            Created alert
        """
        # Check if alert is filtered
        if symbol in self.alert_filters:
            if alert_type not in self.alert_filters[symbol]:
                logger.debug(f"Alert filtered: {symbol} - {alert_type.value}")
                return None
        
        # Create alert
        alert = Alert(alert_type, priority, symbol, message, data)
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Log alert
        logger.info(f"🔔 {alert}")
        
        # Trigger callbacks
        self._trigger_callbacks(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert (remove from active)"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            del self.active_alerts[alert_id]
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def acknowledge_all(self, symbol: Optional[str] = None):
        """Acknowledge all alerts (optionally for specific symbol)"""
        if symbol:
            to_ack = [aid for aid, alert in self.active_alerts.items() if alert.symbol == symbol]
        else:
            to_ack = list(self.active_alerts.keys())
        
        for alert_id in to_ack:
            self.acknowledge_alert(alert_id)
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List[Alert]:
        """Get active alerts (optionally filtered by symbol)"""
        alerts = list(self.active_alerts.values())
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        # Sort by priority and timestamp
        priority_order = {
            AlertPriority.CRITICAL: 0,
            AlertPriority.HIGH: 1,
            AlertPriority.MEDIUM: 2,
            AlertPriority.LOW: 3
        }
        
        alerts.sort(key=lambda a: (priority_order[a.priority], a.timestamp), reverse=True)
        
        return alerts
    
    def get_alert_history(self, 
                         symbol: Optional[str] = None,
                         alert_type: Optional[AlertType] = None,
                         limit: int = 100) -> List[Alert]:
        """Get alert history"""
        alerts = list(self.alert_history)
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts[-limit:]
    
    def set_alert_filter(self, symbol: str, enabled_types: List[AlertType]):
        """Set alert filter for symbol"""
        self.alert_filters[symbol] = set(enabled_types)
    
    def clear_alert_filter(self, symbol: str):
        """Clear alert filter for symbol"""
        if symbol in self.alert_filters:
            del self.alert_filters[symbol]
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add callback for real-time alert delivery"""
        self.alert_callbacks.append(callback)
    
    def _trigger_callbacks(self, alert: Alert):
        """Trigger alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_stats(self) -> Dict:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            type_name = alert.alert_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by priority
        priority_counts = {}
        for alert in self.active_alerts.values():
            priority_name = alert.priority.value
            priority_counts[priority_name] = priority_counts.get(priority_name, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'alerts_by_type': type_counts,
            'active_by_priority': priority_counts
        }


class SqueezeAlertManager:
    """
    High-level alert manager for TTM Squeeze trading
    Integrates with squeeze monitor and exit strategy
    """
    
    def __init__(self, alert_system: AlertSystem = None):
        """
        Initialize squeeze alert manager
        
        Args:
            alert_system: Alert system instance (creates new if None)
        """
        self.alert_system = alert_system or AlertSystem()
        
        # Track last alert time to avoid spam
        self.last_alert_time = {}
        self.alert_cooldown_seconds = 60  # 1 minute cooldown
    
    def on_squeeze_fire(self, squeeze_state: Dict):
        """Handle squeeze fire event"""
        symbol = squeeze_state['symbol']
        signal = squeeze_state['squeeze_signal']
        momentum = squeeze_state['momentum']
        
        # Determine priority based on momentum strength
        if abs(momentum) > 20:
            priority = AlertPriority.CRITICAL
        elif abs(momentum) > 10:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM
        
        # Create message
        direction = "LONG" if signal == 'long' else "SHORT"
        message = f"🔥 TTM Squeeze FIRE - {direction} signal with momentum {momentum:+.2f}"
        
        # Create alert
        self.alert_system.create_alert(
            AlertType.SQUEEZE_FIRE,
            priority,
            symbol,
            message,
            data={
                'signal': signal,
                'momentum': momentum,
                'squeeze_bars': squeeze_state.get('squeeze_bars', 0),
                'price': squeeze_state.get('current_price')
            }
        )
    
    def on_squeeze_active(self, squeeze_state: Dict):
        """Handle squeeze active event (compression >3 bars)"""
        symbol = squeeze_state['symbol']
        squeeze_bars = squeeze_state['squeeze_bars']
        
        # Only alert if squeeze active for 3+ bars
        if squeeze_bars < 3:
            return
        
        # Check cooldown
        if not self._check_cooldown(symbol, AlertType.SQUEEZE_ACTIVE):
            return
        
        message = f"⚠️ TTM Squeeze ACTIVE for {squeeze_bars} bars - High-probability setup forming"
        
        self.alert_system.create_alert(
            AlertType.SQUEEZE_ACTIVE,
            AlertPriority.HIGH,
            symbol,
            message,
            data={
                'squeeze_bars': squeeze_bars,
                'momentum': squeeze_state.get('momentum'),
                'price': squeeze_state.get('current_price')
            }
        )
    
    def on_exit_signal(self, position_id: str, position: Dict, signal, reason: str):
        """Handle exit signal event"""
        symbol = position['symbol']
        
        # Determine priority based on signal type
        if 'stop_loss' in signal.value:
            priority = AlertPriority.CRITICAL
        elif 'profit_target' in signal.value:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM
        
        message = f"📤 EXIT SIGNAL: {signal.value.replace('_', ' ').title()} - {reason}"
        
        self.alert_system.create_alert(
            AlertType.EXIT_SIGNAL,
            priority,
            symbol,
            message,
            data={
                'position_id': position_id,
                'signal_type': signal.value,
                'reason': reason,
                'position_side': position['side'],
                'strike': position.get('strike'),
                'expiration': position.get('expiration').isoformat() if hasattr(position.get('expiration'), 'isoformat') else str(position.get('expiration'))
            }
        )
    
    def on_time_decay_warning(self, position_id: str, position: Dict, dte: int):
        """Handle time decay warning (DTE approaching threshold)"""
        symbol = position['symbol']
        
        message = f"⏰ Time Decay Warning: {dte} days to expiration"
        
        self.alert_system.create_alert(
            AlertType.TIME_DECAY_WARNING,
            AlertPriority.MEDIUM,
            symbol,
            message,
            data={
                'position_id': position_id,
                'dte': dte,
                'strike': position.get('strike'),
                'expiration': position.get('expiration').isoformat() if hasattr(position.get('expiration'), 'isoformat') else str(position.get('expiration'))
            }
        )
    
    def _check_cooldown(self, symbol: str, alert_type: AlertType) -> bool:
        """Check if alert is in cooldown period"""
        key = f"{symbol}_{alert_type.value}"
        
        if key in self.last_alert_time:
            elapsed = (datetime.now() - self.last_alert_time[key]).total_seconds()
            if elapsed < self.alert_cooldown_seconds:
                return False
        
        self.last_alert_time[key] = datetime.now()
        return True


if __name__ == '__main__':
    # Test alert system
    alert_system = AlertSystem()
    alert_manager = SqueezeAlertManager(alert_system)
    
    def on_alert(alert):
        print(f"\n{alert}")
        print(f"  Data: {alert.data}")
    
    alert_system.add_callback(on_alert)
    
    # Test squeeze fire
    print("=== Testing Squeeze Fire Alert ===")
    alert_manager.on_squeeze_fire({
        'symbol': 'AAPL',
        'squeeze_signal': 'long',
        'momentum': 15.5,
        'squeeze_bars': 4,
        'current_price': 180.50
    })
    
    # Test squeeze active
    print("\n=== Testing Squeeze Active Alert ===")
    alert_manager.on_squeeze_active({
        'symbol': 'MSFT',
        'squeeze_bars': 5,
        'momentum': 3.2,
        'current_price': 420.00
    })
    
    # Test exit signal
    print("\n=== Testing Exit Signal Alert ===")
    from python_system.realtime.exit_strategy import ExitSignal
    alert_manager.on_exit_signal(
        'pos_001',
        {'symbol': 'GOOGL', 'side': 'long_call', 'strike': 150, 'expiration': datetime.now()},
        ExitSignal.PROFIT_TARGET,
        "Profit target reached: 50.0%"
    )
    
    # Get stats
    print("\n=== Alert Stats ===")
    stats = alert_system.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Get active alerts
    print("\n=== Active Alerts ===")
    active = alert_system.get_active_alerts()
    for alert in active:
        print(f"  {alert.priority.value.upper()}: {alert.symbol} - {alert.message}")
