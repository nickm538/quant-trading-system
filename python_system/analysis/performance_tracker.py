"""
Performance Tracking System
===========================

Track paper trading performance to validate strategy before live trading:
- Trade logging
- P&L tracking
- Win rate calculation
- Sharpe ratio
- Maximum drawdown
- Performance attribution

Author: Institutional Trading System
Date: 2024-12-11
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    signal_type: str  # BUY, SELL
    confidence: float
    target_price: float
    stop_loss: float
    
    # Exit info (filled when trade closes)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TARGET, STOP, MANUAL, TIME
    
    # Performance
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None  # After costs
    return_pct: Optional[float] = None
    hold_days: Optional[int] = None
    
    # Costs
    execution_costs: Optional[float] = None
    
    # Status
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Gross profit / Gross loss
    
    max_drawdown: float
    max_drawdown_pct: float
    
    sharpe_ratio: float
    sortino_ratio: float
    
    avg_hold_days: float
    total_return_pct: float
    
    # By signal type
    buy_win_rate: float
    sell_win_rate: float


class PerformanceTracker:
    """
    Track paper trading performance
    
    Saves trades to JSON file for persistence
    Calculates comprehensive performance metrics
    """
    
    def __init__(self, data_file: str = "paper_trades.json"):
        """
        Initialize performance tracker
        
        Args:
            data_file: Path to JSON file for storing trades
        """
        self.data_file = data_file
        self.trades: List[Trade] = []
        self.initial_capital = 1000.0  # Default
        self.current_capital = self.initial_capital
        
        # Load existing trades
        self._load_trades()
    
    def _load_trades(self):
        """Load trades from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.initial_capital = data.get('initial_capital', 1000.0)
                    self.current_capital = data.get('current_capital', self.initial_capital)
                    
                    trades_data = data.get('trades', [])
                    self.trades = [Trade(**t) for t in trades_data]
                    
                logger.info(f"Loaded {len(self.trades)} trades from {self.data_file}")
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
                self.trades = []
        else:
            logger.info(f"No existing trades file found. Starting fresh.")
    
    def _save_trades(self):
        """Save trades to JSON file"""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'trades': [asdict(t) for t in self.trades],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.trades)} trades to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def log_trade_entry(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        signal_type: str,
        confidence: float,
        target_price: float,
        stop_loss: float
    ) -> str:
        """
        Log a new trade entry
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            shares: Number of shares
            signal_type: BUY or SELL
            confidence: Signal confidence
            target_price: Target price
            stop_loss: Stop loss price
            
        Returns:
            Trade ID
        """
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            entry_date=datetime.now().isoformat(),
            entry_price=entry_price,
            shares=shares,
            signal_type=signal_type,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            status="OPEN"
        )
        
        self.trades.append(trade)
        self._save_trades()
        
        logger.info(f"Logged trade entry: {trade_id} - {signal_type} {shares} {symbol} @ ${entry_price:.2f}")
        
        return trade_id
    
    def log_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        execution_costs: float = 0.0
    ):
        """
        Log trade exit
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit (TARGET, STOP, MANUAL, TIME)
            execution_costs: Total execution costs
        """
        trade = next((t for t in self.trades if t.trade_id == trade_id), None)
        
        if not trade:
            logger.error(f"Trade {trade_id} not found")
            return
        
        if trade.status != "OPEN":
            logger.warning(f"Trade {trade_id} is not open (status: {trade.status})")
            return
        
        # Calculate performance
        entry_date = datetime.fromisoformat(trade.entry_date)
        exit_date = datetime.now()
        hold_days = (exit_date - entry_date).days
        
        if trade.signal_type == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * trade.shares
        else:  # SELL
            gross_pnl = (trade.entry_price - exit_price) * trade.shares
        
        net_pnl = gross_pnl - execution_costs
        return_pct = (net_pnl / (trade.entry_price * trade.shares)) * 100
        
        # Update trade
        trade.exit_date = exit_date.isoformat()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.return_pct = return_pct
        trade.hold_days = hold_days
        trade.execution_costs = execution_costs
        trade.status = "CLOSED"
        
        # Update capital
        self.current_capital += net_pnl
        
        self._save_trades()
        
        logger.info(
            f"Logged trade exit: {trade_id} - "
            f"P&L: ${net_pnl:.2f} ({return_pct:.2f}%) - "
            f"Reason: {exit_reason}"
        )
    
    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return [t for t in self.trades if t.status == "OPEN"]
    
    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades"""
        return [t for t in self.trades if t.status == "CLOSED"]
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            PerformanceMetrics object
        """
        closed_trades = self.get_closed_trades()
        
        if not closed_trades:
            return PerformanceMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                avg_hold_days=0,
                total_return_pct=0,
                buy_win_rate=0,
                sell_win_rate=0
            )
        
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.net_pnl > 0]
        losing_trades = [t for t in closed_trades if t.net_pnl <= 0]
        
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.net_pnl for t in closed_trades)
        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
        
        gross_profit = sum(t.net_pnl for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Drawdown
        capital_curve = [self.initial_capital]
        for trade in sorted(closed_trades, key=lambda t: t.exit_date):
            capital_curve.append(capital_curve[-1] + trade.net_pnl)
        
        peak = capital_curve[0]
        max_dd = 0
        max_dd_pct = 0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            dd = peak - capital
            dd_pct = (dd / peak) * 100 if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        # Risk-adjusted returns
        returns = [t.return_pct for t in closed_trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate, daily returns)
        sharpe = (avg_return / std_return) if std_return > 0 else 0
        
        # Sortino ratio (downside deviation only)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else std_return
        sortino = (avg_return / downside_std) if downside_std > 0 else 0
        
        # Hold time
        avg_hold_days = np.mean([t.hold_days for t in closed_trades if t.hold_days is not None])
        
        # Total return
        total_return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # By signal type
        buy_trades = [t for t in closed_trades if t.signal_type == "BUY"]
        sell_trades = [t for t in closed_trades if t.signal_type == "SELL"]
        
        buy_wins = len([t for t in buy_trades if t.net_pnl > 0])
        sell_wins = len([t for t in sell_trades if t.net_pnl > 0])
        
        buy_win_rate = (buy_wins / len(buy_trades) * 100) if buy_trades else 0
        sell_win_rate = (sell_wins / len(sell_trades) * 100) if sell_trades else 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_hold_days=avg_hold_days,
            total_return_pct=total_return_pct,
            buy_win_rate=buy_win_rate,
            sell_win_rate=sell_win_rate
        )
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 80)
        print("PAPER TRADING PERFORMANCE REPORT")
        print("=" * 80)
        
        print(f"\n📊 CAPITAL:")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Current: ${self.current_capital:,.2f}")
        print(f"  Total Return: {metrics.total_return_pct:.2f}%")
        print(f"  Total P&L: ${metrics.total_pnl:,.2f}")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Winning Trades: {metrics.winning_trades}")
        print(f"  Losing Trades: {metrics.losing_trades}")
        print(f"  Win Rate: {metrics.win_rate:.2f}%")
        print(f"  Average Hold: {metrics.avg_hold_days:.1f} days")
        
        print(f"\n💰 P&L METRICS:")
        print(f"  Average Win: ${metrics.avg_win:.2f}")
        print(f"  Average Loss: ${metrics.avg_loss:.2f}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        print(f"\n🎯 BY SIGNAL TYPE:")
        print(f"  BUY Win Rate: {metrics.buy_win_rate:.2f}%")
        print(f"  SELL Win Rate: {metrics.sell_win_rate:.2f}%")
        
        print("\n" + "=" * 80)
        
        # Open positions
        open_trades = self.get_open_trades()
        if open_trades:
            print(f"\n📂 OPEN POSITIONS ({len(open_trades)}):")
            for trade in open_trades:
                print(f"  {trade.symbol}: {trade.signal_type} {trade.shares} @ ${trade.entry_price:.2f}")
                print(f"    Entry: {trade.entry_date[:10]}")
                print(f"    Target: ${trade.target_price:.2f} | Stop: ${trade.stop_loss:.2f}")
        
        print("\n" + "=" * 80)
