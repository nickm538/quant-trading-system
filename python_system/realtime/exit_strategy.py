"""
Exit Strategy Engine for TTM Squeeze Options Trading
Determines optimal exit points based on momentum reversal, profit targets, and risk management
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExitSignal(Enum):
    """Exit signal types"""
    NONE = "none"
    MOMENTUM_REVERSAL = "momentum_reversal"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    SQUEEZE_REVERSAL = "squeeze_reversal"


class PositionSide(Enum):
    """Position side"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"


class ExitStrategy:
    """
    Exit strategy engine for options positions based on TTM Squeeze
    
    Rules:
    1. Momentum Reversal: Exit when momentum crosses zero against position
    2. Profit Target: Exit when position reaches target profit (default 50%)
    3. Stop Loss: Exit when position reaches max loss (default 30%)
    4. Time Decay: Exit when DTE < 3 days (theta burn acceleration)
    5. Squeeze Reversal: Exit if squeeze fires opposite direction
    """
    
    def __init__(self,
                 profit_target_pct: float = 0.50,
                 stop_loss_pct: float = 0.30,
                 min_dte_exit: int = 3,
                 momentum_reversal_threshold: float = 0.0):
        """
        Initialize exit strategy
        
        Args:
            profit_target_pct: Profit target as percentage of premium (default 50%)
            stop_loss_pct: Stop loss as percentage of premium (default 30%)
            min_dte_exit: Minimum DTE before forced exit (default 3 days)
            momentum_reversal_threshold: Momentum level for reversal signal (default 0.0)
        """
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_dte_exit = min_dte_exit
        self.momentum_reversal_threshold = momentum_reversal_threshold
    
    def check_exit(self,
                   position: Dict,
                   current_price: float,
                   squeeze_state: Dict,
                   current_option_price: Optional[float] = None) -> Tuple[bool, ExitSignal, str]:
        """
        Check if position should be exited
        
        Args:
            position: Position data (side, entry_price, entry_date, strike, expiration, etc.)
            current_price: Current underlying price
            squeeze_state: Current TTM Squeeze state
            current_option_price: Current option price (if available)
        
        Returns:
            (should_exit, exit_signal, reason)
        """
        
        # Extract position data
        side = PositionSide(position['side'])
        entry_price = position['entry_price']
        entry_date = position['entry_date']
        expiration = position['expiration']
        strike = position['strike']
        entry_option_price = position.get('entry_option_price', entry_price)
        
        # Calculate DTE
        if isinstance(expiration, str):
            expiration = datetime.fromisoformat(expiration)
        dte = (expiration - datetime.now()).days
        
        # 1. Time Decay Check
        if dte <= self.min_dte_exit:
            return (True, ExitSignal.TIME_DECAY, 
                   f"DTE {dte} <= {self.min_dte_exit} days - theta burn acceleration")
        
        # 2. Profit Target / Stop Loss Check (if option price available)
        if current_option_price is not None:
            pnl_pct = (current_option_price - entry_option_price) / entry_option_price
            
            if pnl_pct >= self.profit_target_pct:
                return (True, ExitSignal.PROFIT_TARGET,
                       f"Profit target reached: {pnl_pct*100:.1f}% >= {self.profit_target_pct*100:.1f}%")
            
            if pnl_pct <= -self.stop_loss_pct:
                return (True, ExitSignal.STOP_LOSS,
                       f"Stop loss hit: {pnl_pct*100:.1f}% <= -{self.stop_loss_pct*100:.1f}%")
        
        # 3. Momentum Reversal Check
        momentum = squeeze_state.get('momentum', 0)
        
        if side == PositionSide.LONG_CALL:
            # Exit call if momentum turns negative
            if momentum < self.momentum_reversal_threshold:
                return (True, ExitSignal.MOMENTUM_REVERSAL,
                       f"Momentum reversal: {momentum:.2f} < {self.momentum_reversal_threshold:.2f} (bearish)")
        
        elif side == PositionSide.LONG_PUT:
            # Exit put if momentum turns positive
            if momentum > self.momentum_reversal_threshold:
                return (True, ExitSignal.MOMENTUM_REVERSAL,
                       f"Momentum reversal: {momentum:.2f} > {self.momentum_reversal_threshold:.2f} (bullish)")
        
        # 4. Squeeze Reversal Check
        squeeze_signal = squeeze_state.get('squeeze_signal', 'none')
        
        if side == PositionSide.LONG_CALL and squeeze_signal == 'short':
            return (True, ExitSignal.SQUEEZE_REVERSAL,
                   f"Squeeze fired short - exit long call")
        
        if side == PositionSide.LONG_PUT and squeeze_signal == 'long':
            return (True, ExitSignal.SQUEEZE_REVERSAL,
                   f"Squeeze fired long - exit long put")
        
        # No exit signal
        return (False, ExitSignal.NONE, "No exit conditions met")
    
    def calculate_profit_target_price(self, entry_option_price: float) -> float:
        """Calculate option price for profit target"""
        return entry_option_price * (1 + self.profit_target_pct)
    
    def calculate_stop_loss_price(self, entry_option_price: float) -> float:
        """Calculate option price for stop loss"""
        return entry_option_price * (1 - self.stop_loss_pct)
    
    def get_exit_levels(self, position: Dict) -> Dict:
        """
        Get all exit levels for a position
        
        Returns:
            {
                'profit_target_price': float,
                'stop_loss_price': float,
                'min_dte': int,
                'momentum_threshold': float
            }
        """
        entry_option_price = position.get('entry_option_price', position['entry_price'])
        
        return {
            'profit_target_price': self.calculate_profit_target_price(entry_option_price),
            'stop_loss_price': self.calculate_stop_loss_price(entry_option_price),
            'profit_target_pct': self.profit_target_pct * 100,
            'stop_loss_pct': self.stop_loss_pct * 100,
            'min_dte': self.min_dte_exit,
            'momentum_threshold': self.momentum_reversal_threshold
        }
    
    def update_trailing_stop(self, 
                            position: Dict, 
                            current_option_price: float,
                            trailing_stop_pct: float = 0.20) -> Optional[float]:
        """
        Calculate trailing stop loss
        
        Args:
            position: Position data
            current_option_price: Current option price
            trailing_stop_pct: Trailing stop percentage (default 20%)
        
        Returns:
            New stop loss price or None if not applicable
        """
        entry_option_price = position.get('entry_option_price', position['entry_price'])
        current_stop = position.get('stop_loss_price', self.calculate_stop_loss_price(entry_option_price))
        
        # Only trail if in profit
        if current_option_price <= entry_option_price:
            return None
        
        # Calculate trailing stop
        trailing_stop = current_option_price * (1 - trailing_stop_pct)
        
        # Only update if trailing stop is higher than current stop
        if trailing_stop > current_stop:
            return trailing_stop
        
        return None


class PositionTracker:
    """
    Track open positions and monitor for exit signals
    """
    
    def __init__(self, exit_strategy: ExitStrategy = None):
        """
        Initialize position tracker
        
        Args:
            exit_strategy: Exit strategy instance (creates default if None)
        """
        self.exit_strategy = exit_strategy or ExitStrategy()
        self.positions = {}  # position_id -> position data
        self.exit_callbacks = []
    
    def add_position(self, position_id: str, position_data: Dict):
        """
        Add position to tracking
        
        Args:
            position_id: Unique position identifier
            position_data: Position data (side, entry_price, strike, expiration, etc.)
        """
        # Add entry timestamp if not present
        if 'entry_date' not in position_data:
            position_data['entry_date'] = datetime.now()
        
        # Calculate exit levels
        exit_levels = self.exit_strategy.get_exit_levels(position_data)
        position_data['exit_levels'] = exit_levels
        
        self.positions[position_id] = position_data
        logger.info(f"Added position {position_id}: {position_data['symbol']} {position_data['side']}")
    
    def remove_position(self, position_id: str):
        """Remove position from tracking"""
        if position_id in self.positions:
            del self.positions[position_id]
            logger.info(f"Removed position {position_id}")
    
    def check_position(self, 
                      position_id: str, 
                      current_price: float,
                      squeeze_state: Dict,
                      current_option_price: Optional[float] = None) -> Tuple[bool, ExitSignal, str]:
        """
        Check if position should be exited
        
        Returns:
            (should_exit, exit_signal, reason)
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        should_exit, signal, reason = self.exit_strategy.check_exit(
            position, current_price, squeeze_state, current_option_price
        )
        
        if should_exit:
            logger.info(f"Exit signal for {position_id}: {signal.value} - {reason}")
            self._trigger_exit_callbacks(position_id, position, signal, reason)
        
        return should_exit, signal, reason
    
    def check_all_positions(self, 
                           price_data: Dict[str, float],
                           squeeze_states: Dict[str, Dict],
                           option_prices: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Check all positions for exit signals
        
        Args:
            price_data: {symbol: current_price}
            squeeze_states: {symbol: squeeze_state}
            option_prices: {position_id: current_option_price} (optional)
        
        Returns:
            List of positions with exit signals
        """
        exit_signals = []
        
        for position_id, position in list(self.positions.items()):
            symbol = position['symbol']
            
            if symbol not in price_data or symbol not in squeeze_states:
                continue
            
            current_price = price_data[symbol]
            squeeze_state = squeeze_states[symbol]
            current_option_price = option_prices.get(position_id) if option_prices else None
            
            should_exit, signal, reason = self.check_position(
                position_id, current_price, squeeze_state, current_option_price
            )
            
            if should_exit:
                exit_signals.append({
                    'position_id': position_id,
                    'position': position,
                    'signal': signal,
                    'reason': reason,
                    'timestamp': datetime.now()
                })
        
        return exit_signals
    
    def get_position(self, position_id: str) -> Optional[Dict]:
        """Get position data"""
        return self.positions.get(position_id)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions"""
        return self.positions.copy()
    
    def add_exit_callback(self, callback):
        """Add callback for exit signals"""
        self.exit_callbacks.append(callback)
    
    def _trigger_exit_callbacks(self, position_id: str, position: Dict, signal: ExitSignal, reason: str):
        """Trigger exit callbacks"""
        for callback in self.exit_callbacks:
            try:
                callback(position_id, position, signal, reason)
            except Exception as e:
                logger.error(f"Exit callback error: {e}")


if __name__ == '__main__':
    # Test exit strategy
    strategy = ExitStrategy(
        profit_target_pct=0.50,
        stop_loss_pct=0.30,
        min_dte_exit=3
    )
    
    # Test position
    position = {
        'side': 'long_call',
        'symbol': 'AAPL',
        'entry_price': 100.0,
        'entry_option_price': 5.0,
        'strike': 105.0,
        'expiration': datetime.now() + timedelta(days=10),
        'entry_date': datetime.now() - timedelta(days=5)
    }
    
    # Test scenarios
    print("=== Exit Strategy Tests ===\n")
    
    # Scenario 1: Profit target
    squeeze_state = {'momentum': 10.0, 'squeeze_signal': 'none'}
    should_exit, signal, reason = strategy.check_exit(position, 110.0, squeeze_state, 7.5)
    print(f"1. Profit Target: {should_exit} - {signal.value} - {reason}\n")
    
    # Scenario 2: Stop loss
    should_exit, signal, reason = strategy.check_exit(position, 95.0, squeeze_state, 3.5)
    print(f"2. Stop Loss: {should_exit} - {signal.value} - {reason}\n")
    
    # Scenario 3: Momentum reversal
    squeeze_state = {'momentum': -5.0, 'squeeze_signal': 'none'}
    should_exit, signal, reason = strategy.check_exit(position, 102.0, squeeze_state, 5.5)
    print(f"3. Momentum Reversal: {should_exit} - {signal.value} - {reason}\n")
    
    # Scenario 4: Time decay
    position['expiration'] = datetime.now() + timedelta(days=2)
    should_exit, signal, reason = strategy.check_exit(position, 102.0, squeeze_state, 5.5)
    print(f"4. Time Decay: {should_exit} - {signal.value} - {reason}\n")
    
    # Scenario 5: Squeeze reversal
    squeeze_state = {'momentum': -10.0, 'squeeze_signal': 'short'}
    should_exit, signal, reason = strategy.check_exit(position, 102.0, squeeze_state, 5.5)
    print(f"5. Squeeze Reversal: {should_exit} - {signal.value} - {reason}\n")
    
    # Get exit levels
    exit_levels = strategy.get_exit_levels(position)
    print(f"Exit Levels: {exit_levels}")
