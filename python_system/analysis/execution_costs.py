"""
Execution Costs Module
======================

Models realistic trading costs:
- Slippage (market impact)
- Commission fees
- Bid-ask spread
- Market order vs limit order costs

Author: Institutional Trading System
Date: 2024-12-11
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionCost:
    """Execution cost breakdown"""
    entry_slippage: float  # Entry slippage cost
    exit_slippage: float  # Exit slippage cost
    commission: float  # Total commission (entry + exit)
    spread_cost: float  # Bid-ask spread cost
    total_cost: float  # Total execution cost
    total_cost_pct: float  # Total cost as % of position value
    
    # Adjusted prices
    effective_entry_price: float
    effective_exit_price: float
    
    # Impact on returns
    gross_return: float  # Return before costs
    net_return: float  # Return after costs
    cost_drag: float  # How much costs reduced returns


class ExecutionCostModel:
    """
    Models realistic execution costs for trading
    
    Based on institutional trading research:
    - Slippage increases with position size and volatility
    - Commission varies by broker (using zero-commission model)
    - Spread widens during volatile periods
    """
    
    def __init__(
        self,
        commission_per_share: float = 0.0,  # Zero-commission brokers (Robinhood, Webull)
        min_commission: float = 0.0,
        max_commission: float = 0.0,
        spread_bps: float = 5.0,  # 5 basis points typical spread
        slippage_factor: float = 0.10  # 10% of volatility as slippage
    ):
        """
        Initialize execution cost model
        
        Args:
            commission_per_share: Commission per share (0 for zero-commission brokers)
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
            spread_bps: Bid-ask spread in basis points (1 bp = 0.01%)
            slippage_factor: Slippage as fraction of volatility
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.spread_bps = spread_bps
        self.slippage_factor = slippage_factor
    
    def calculate_slippage(
        self,
        price: float,
        shares: int,
        volatility: float,
        avg_volume: float,
        is_entry: bool = True
    ) -> float:
        """
        Calculate slippage based on market impact
        
        Slippage increases with:
        - Position size relative to volume
        - Volatility
        - Market order urgency
        
        Args:
            price: Stock price
            shares: Number of shares
            volatility: Annualized volatility (e.g., 0.25 = 25%)
            avg_volume: Average daily volume
            is_entry: True for entry, False for exit
            
        Returns:
            Slippage amount in dollars
        """
        # Base slippage from volatility
        daily_vol = volatility / np.sqrt(252)  # Convert annual to daily
        base_slippage_pct = daily_vol * self.slippage_factor
        
        # Market impact from position size
        position_value = price * shares
        if avg_volume > 0:
            # Assume average trade value is price * 10% of daily volume
            typical_trade_value = price * (avg_volume * 0.10)
            size_impact = (position_value / typical_trade_value) ** 0.5
            size_impact = min(size_impact, 3.0)  # Cap at 3x
        else:
            size_impact = 1.0
        
        # Entry vs exit (exits typically have less slippage)
        direction_multiplier = 1.0 if is_entry else 0.7
        
        # Total slippage
        total_slippage_pct = base_slippage_pct * size_impact * direction_multiplier
        slippage_amount = position_value * total_slippage_pct
        
        return slippage_amount
    
    def calculate_commission(self, shares: int) -> float:
        """
        Calculate commission cost
        
        Args:
            shares: Number of shares
            
        Returns:
            Commission amount in dollars
        """
        commission = shares * self.commission_per_share
        
        # Apply min/max
        if self.min_commission > 0:
            commission = max(commission, self.min_commission)
        if self.max_commission > 0:
            commission = min(commission, self.max_commission)
        
        return commission
    
    def calculate_spread_cost(
        self,
        price: float,
        shares: int
    ) -> float:
        """
        Calculate bid-ask spread cost
        
        Args:
            price: Stock price
            shares: Number of shares
            
        Returns:
            Spread cost in dollars
        """
        spread_pct = self.spread_bps / 10000  # Convert bps to decimal
        position_value = price * shares
        spread_cost = position_value * spread_pct
        
        return spread_cost
    
    def calculate_total_costs(
        self,
        entry_price: float,
        exit_price: float,
        shares: int,
        volatility: float,
        avg_volume: float
    ) -> ExecutionCost:
        """
        Calculate total execution costs for a round-trip trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            shares: Number of shares
            volatility: Annualized volatility
            avg_volume: Average daily volume
            
        Returns:
            ExecutionCost object with full breakdown
        """
        # Entry costs
        entry_slippage = self.calculate_slippage(
            entry_price, shares, volatility, avg_volume, is_entry=True
        )
        entry_commission = self.calculate_commission(shares)
        entry_spread = self.calculate_spread_cost(entry_price, shares)
        
        # Exit costs
        exit_slippage = self.calculate_slippage(
            exit_price, shares, volatility, avg_volume, is_entry=False
        )
        exit_commission = self.calculate_commission(shares)
        exit_spread = self.calculate_spread_cost(exit_price, shares)
        
        # Total costs
        total_slippage = entry_slippage + exit_slippage
        total_commission = entry_commission + exit_commission
        total_spread = entry_spread + exit_spread
        total_cost = total_slippage + total_commission + total_spread
        
        # Position value
        position_value = entry_price * shares
        total_cost_pct = (total_cost / position_value) * 100 if position_value > 0 else 0
        
        # Effective prices (worse prices after costs)
        effective_entry_price = entry_price + (entry_slippage + entry_commission + entry_spread) / shares
        effective_exit_price = exit_price - (exit_slippage + exit_commission + exit_spread) / shares
        
        # Returns
        gross_return = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        net_return = ((effective_exit_price - effective_entry_price) / effective_entry_price) * 100 if effective_entry_price > 0 else 0
        cost_drag = gross_return - net_return
        
        return ExecutionCost(
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            commission=total_commission,
            spread_cost=total_spread,
            total_cost=total_cost,
            total_cost_pct=total_cost_pct,
            effective_entry_price=effective_entry_price,
            effective_exit_price=effective_exit_price,
            gross_return=gross_return,
            net_return=net_return,
            cost_drag=cost_drag
        )
    
    def adjust_target_for_costs(
        self,
        entry_price: float,
        target_price: float,
        shares: int,
        volatility: float,
        avg_volume: float
    ) -> float:
        """
        Adjust target price to account for execution costs
        
        Args:
            entry_price: Entry price
            target_price: Desired target price
            shares: Number of shares
            volatility: Annualized volatility
            avg_volume: Average daily volume
            
        Returns:
            Adjusted target price that achieves desired net return
        """
        costs = self.calculate_total_costs(
            entry_price, target_price, shares, volatility, avg_volume
        )
        
        # Adjust target to compensate for costs
        cost_per_share = costs.total_cost / shares if shares > 0 else 0
        adjusted_target = target_price + cost_per_share
        
        return adjusted_target


# Default models for different account types
def get_retail_cost_model() -> ExecutionCostModel:
    """Zero-commission retail broker (Robinhood, Webull, etc.)"""
    return ExecutionCostModel(
        commission_per_share=0.0,
        min_commission=0.0,
        max_commission=0.0,
        spread_bps=5.0,  # 5 bps spread
        slippage_factor=0.10  # 10% of volatility
    )


def get_traditional_broker_model() -> ExecutionCostModel:
    """Traditional broker with commissions"""
    return ExecutionCostModel(
        commission_per_share=0.005,  # $0.005 per share
        min_commission=1.0,  # $1 minimum
        max_commission=10.0,  # $10 maximum
        spread_bps=5.0,
        slippage_factor=0.10
    )


def get_institutional_model() -> ExecutionCostModel:
    """Institutional broker with better execution"""
    return ExecutionCostModel(
        commission_per_share=0.001,  # $0.001 per share
        min_commission=0.0,
        max_commission=0.0,
        spread_bps=2.0,  # Tighter spreads
        slippage_factor=0.05  # Better execution
    )
