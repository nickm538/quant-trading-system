"""
Risk Management Controls
========================

Institutional-grade risk controls to prevent catastrophic losses:
- Position size limits
- Portfolio heat monitoring
- Daily loss limits
- Concentration limits
- Correlation-based diversification

Author: Institutional Trading System
Date: 2024-12-11
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size_pct: float = 0.20  # Max 20% in single position
    max_sector_exposure_pct: float = 0.40  # Max 40% in single sector
    max_portfolio_heat: float = 0.06  # Max 6% total portfolio at risk
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_correlation: float = 0.70  # Max correlation between positions
    min_positions: int = 3  # Minimum number of positions for diversification
    max_positions: int = 10  # Maximum number of positions


@dataclass
class RiskCheckResult:
    """Result of risk check"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    adjusted_position_size: float  # Adjusted position size if needed
    risk_metrics: Dict


class RiskManager:
    """
    Institutional-grade risk management
    
    Prevents:
    - Over-concentration in single positions
    - Excessive portfolio heat
    - Correlation risk
    - Daily loss spirals
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.daily_pnl = {}  # Track daily P&L
        self.positions = {}  # Current positions
        self.logger = logging.getLogger(__name__)
    
    def check_position_limits(
        self,
        symbol: str,
        proposed_size_pct: float,
        current_positions: Dict[str, Dict]
    ) -> RiskCheckResult:
        """
        Check if proposed position violates risk limits
        
        Args:
            symbol: Stock symbol
            proposed_size_pct: Proposed position size as % of portfolio
            current_positions: Dictionary of current positions
            
        Returns:
            RiskCheckResult with pass/fail and adjustments
        """
        warnings = []
        errors = []
        adjusted_size = proposed_size_pct
        
        # Check 1: Single position size limit
        if proposed_size_pct > self.limits.max_position_size_pct:
            warnings.append(
                f"Position size {proposed_size_pct*100:.1f}% exceeds limit "
                f"{self.limits.max_position_size_pct*100:.1f}%"
            )
            adjusted_size = self.limits.max_position_size_pct
        
        # Check 2: Portfolio heat (total risk)
        current_heat = sum(
            pos.get('position_size_pct', 0) 
            for pos in current_positions.values()
        )
        
        total_heat = current_heat + adjusted_size
        
        if total_heat > self.limits.max_portfolio_heat:
            warnings.append(
                f"Total portfolio heat {total_heat*100:.1f}% exceeds limit "
                f"{self.limits.max_portfolio_heat*100:.1f}%"
            )
            # Reduce position to fit within heat limit
            available_heat = self.limits.max_portfolio_heat - current_heat
            adjusted_size = max(0, available_heat)
        
        # Check 3: Position count limits
        num_positions = len(current_positions)
        
        if num_positions >= self.limits.max_positions:
            errors.append(
                f"Already at max positions ({self.limits.max_positions}). "
                f"Close a position before opening new one."
            )
            adjusted_size = 0
        
        # Check 4: Minimum diversification
        if num_positions < self.limits.min_positions and adjusted_size > self.limits.max_position_size_pct * 0.5:
            warnings.append(
                f"Less than {self.limits.min_positions} positions. "
                f"Consider smaller position sizes for diversification."
            )
            adjusted_size = min(adjusted_size, self.limits.max_position_size_pct * 0.5)
        
        # Risk metrics
        risk_metrics = {
            'current_heat': current_heat,
            'proposed_heat': total_heat,
            'num_positions': num_positions,
            'heat_utilization_pct': (total_heat / self.limits.max_portfolio_heat) * 100,
            'position_slots_used': num_positions,
            'position_slots_available': self.limits.max_positions - num_positions
        }
        
        passed = len(errors) == 0
        
        return RiskCheckResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            adjusted_position_size=adjusted_size,
            risk_metrics=risk_metrics
        )
    
    def check_correlation_risk(
        self,
        symbol: str,
        current_positions: Dict[str, Dict],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if new position creates excessive correlation risk
        
        Args:
            symbol: New symbol to add
            current_positions: Current positions
            correlation_matrix: Correlation matrix between symbols
            
        Returns:
            (passed, warnings)
        """
        warnings = []
        
        if not correlation_matrix or symbol not in correlation_matrix:
            warnings.append("No correlation data available - cannot check correlation risk")
            return True, warnings
        
        # Check correlation with existing positions
        high_correlation_positions = []
        
        for pos_symbol in current_positions.keys():
            if pos_symbol in correlation_matrix[symbol]:
                corr = correlation_matrix[symbol][pos_symbol]
                
                if abs(corr) > self.limits.max_correlation:
                    high_correlation_positions.append((pos_symbol, corr))
        
        if high_correlation_positions:
            warnings.append(
                f"High correlation detected with existing positions:"
            )
            for pos_symbol, corr in high_correlation_positions:
                warnings.append(f"  - {pos_symbol}: {corr:.2f}")
            warnings.append(
                f"Consider reducing position size or diversifying"
            )
        
        return True, warnings
    
    def check_daily_loss_limit(
        self,
        current_pnl: float,
        bankroll: float,
        date: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if daily loss limit has been hit
        
        Args:
            current_pnl: Current P&L for the day
            bankroll: Total bankroll
            date: Date to check (default: today)
            
        Returns:
            (can_trade, message)
        """
        if date is None:
            date = datetime.now().date()
        
        loss_pct = abs(current_pnl) / bankroll if bankroll > 0 else 0
        
        if current_pnl < 0 and loss_pct > self.limits.max_daily_loss_pct:
            return False, (
                f"Daily loss limit hit: {loss_pct*100:.2f}% "
                f"(limit: {self.limits.max_daily_loss_pct*100:.1f}%). "
                f"No new trades allowed today."
            )
        
        return True, "Daily loss limit OK"
    
    def calculate_portfolio_var(
        self,
        positions: Dict[str, Dict],
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate portfolio Value at Risk
        
        Args:
            positions: Dictionary of positions with var_95 values
            confidence: Confidence level (default 95%)
            
        Returns:
            Dictionary with VaR metrics
        """
        if not positions:
            return {
                'portfolio_var': 0,
                'portfolio_cvar': 0,
                'max_expected_loss': 0
            }
        
        # Simple VaR aggregation (assumes independence - conservative)
        total_var = 0
        total_cvar = 0
        total_position_value = 0
        
        for pos in positions.values():
            position_value = pos.get('position_value', 0)
            var_95 = pos.get('var_95', 0)
            cvar_95 = pos.get('cvar_95', 0)
            
            total_var += position_value * abs(var_95)
            total_cvar += position_value * abs(cvar_95)
            total_position_value += position_value
        
        return {
            'portfolio_var': total_var,
            'portfolio_cvar': total_cvar,
            'portfolio_var_pct': (total_var / total_position_value * 100) if total_position_value > 0 else 0,
            'portfolio_cvar_pct': (total_cvar / total_position_value * 100) if total_position_value > 0 else 0,
            'max_expected_loss': total_cvar
        }
    
    def get_position_sizing_recommendation(
        self,
        symbol: str,
        base_position_size: float,
        confidence: float,
        current_positions: Dict[str, Dict],
        bankroll: float
    ) -> Dict:
        """
        Get recommended position size with risk adjustments
        
        Args:
            symbol: Stock symbol
            base_position_size: Base position size from Kelly/risk calc
            confidence: Signal confidence (0-100)
            current_positions: Current positions
            bankroll: Total bankroll
            
        Returns:
            Dictionary with recommended size and reasoning
        """
        # Start with base size
        recommended_size = base_position_size
        adjustments = []
        
        # Adjustment 1: Confidence-based scaling
        if confidence < 60:
            confidence_scale = 0.5
            recommended_size *= confidence_scale
            adjustments.append(f"Low confidence ({confidence:.1f}%): scaled by {confidence_scale}x")
        elif confidence < 70:
            confidence_scale = 0.75
            recommended_size *= confidence_scale
            adjustments.append(f"Medium confidence ({confidence:.1f}%): scaled by {confidence_scale}x")
        
        # Adjustment 2: Portfolio heat
        current_heat = sum(pos.get('position_size_pct', 0) for pos in current_positions.values())
        heat_utilization = current_heat / self.limits.max_portfolio_heat
        
        if heat_utilization > 0.75:
            heat_scale = 0.5
            recommended_size *= heat_scale
            adjustments.append(f"High portfolio heat ({heat_utilization*100:.1f}%): scaled by {heat_scale}x")
        
        # Adjustment 3: Number of positions (encourage diversification)
        num_positions = len(current_positions)
        if num_positions < self.limits.min_positions:
            diversification_scale = 0.7
            recommended_size *= diversification_scale
            adjustments.append(f"Low diversification ({num_positions} positions): scaled by {diversification_scale}x")
        
        # Apply hard limits
        recommended_size = min(recommended_size, self.limits.max_position_size_pct)
        
        # Check if we have room
        available_heat = self.limits.max_portfolio_heat - current_heat
        recommended_size = min(recommended_size, available_heat)
        
        return {
            'recommended_size_pct': recommended_size,
            'base_size_pct': base_position_size,
            'adjustments': adjustments,
            'current_heat': current_heat,
            'available_heat': available_heat,
            'heat_utilization_pct': heat_utilization * 100
        }


def get_conservative_risk_limits() -> RiskLimits:
    """Conservative risk limits for small accounts"""
    return RiskLimits(
        max_position_size_pct=0.15,  # Max 15% per position
        max_sector_exposure_pct=0.30,  # Max 30% per sector
        max_portfolio_heat=0.04,  # Max 4% total risk
        max_daily_loss_pct=0.03,  # Max 3% daily loss
        max_correlation=0.60,
        min_positions=3,
        max_positions=8
    )


def get_moderate_risk_limits() -> RiskLimits:
    """Moderate risk limits (default)"""
    return RiskLimits(
        max_position_size_pct=0.20,  # Max 20% per position
        max_sector_exposure_pct=0.40,  # Max 40% per sector
        max_portfolio_heat=0.06,  # Max 6% total risk
        max_daily_loss_pct=0.05,  # Max 5% daily loss
        max_correlation=0.70,
        min_positions=3,
        max_positions=10
    )


def get_aggressive_risk_limits() -> RiskLimits:
    """Aggressive risk limits for experienced traders"""
    return RiskLimits(
        max_position_size_pct=0.30,  # Max 30% per position
        max_sector_exposure_pct=0.50,  # Max 50% per sector
        max_portfolio_heat=0.10,  # Max 10% total risk
        max_daily_loss_pct=0.08,  # Max 8% daily loss
        max_correlation=0.80,
        min_positions=2,
        max_positions=15
    )
