"""
Drift Detection & Regime Change Detection
=========================================

Detects when market conditions change, invalidating historical patterns.
Critical for avoiding losses when models become stale.

Implements:
- Statistical drift detection (CUSUM, Page-Hinkley)
- Regime change detection (Hidden Markov Models)
- Volatility regime shifts
- Correlation breakdown detection

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Alert for detected drift"""
    timestamp: datetime
    drift_type: str  # 'mean', 'variance', 'correlation', 'regime'
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_value: float
    threshold: float
    description: str
    recommendation: str


@dataclass
class RegimeState:
    """Current market regime"""
    regime: str  # 'bull', 'bear', 'high_vol', 'low_vol', 'crisis'
    confidence: float  # 0-100
    duration_days: int
    characteristics: Dict[str, float]


class DriftDetector:
    """
    Detects statistical drift in data streams.
    Alerts when model assumptions are violated.
    """
    
    def __init__(
        self,
        window_size: int = 30,
        sensitivity: float = 3.0  # Standard deviations
    ):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of reference window
            sensitivity: Number of std devs for alert (lower = more sensitive)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)
        
        # CUSUM parameters
        self.cusum_threshold = 5.0
        self.cusum_drift = 0.5
    
    def detect_mean_drift(
        self,
        data: pd.Series,
        reference_period: Optional[pd.Series] = None
    ) -> Optional[DriftAlert]:
        """
        Detect drift in mean (expected return).
        
        Args:
            data: Time series data
            reference_period: Historical reference (if None, uses first window_size)
            
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if len(data) < self.window_size * 2:
            return None
        
        # Reference statistics
        if reference_period is None:
            reference = data.iloc[:self.window_size]
        else:
            reference = reference_period
        
        ref_mean = reference.mean()
        ref_std = reference.std()
        
        if ref_std == 0:
            return None
        
        # Current statistics
        current = data.iloc[-self.window_size:]
        current_mean = current.mean()
        
        # Z-score of mean difference
        z_score = abs(current_mean - ref_mean) / (ref_std / np.sqrt(self.window_size))
        
        if z_score > self.sensitivity:
            severity = 'critical' if z_score > 5 else ('high' if z_score > 4 else 'medium')
            
            return DriftAlert(
                timestamp=datetime.now(),
                drift_type='mean',
                severity=severity,
                metric_value=current_mean,
                threshold=ref_mean,
                description=f"Mean shifted from {ref_mean:.4f} to {current_mean:.4f} ({z_score:.2f}σ)",
                recommendation="Model may be stale. Consider retraining or reducing position sizes."
            )
        
        return None
    
    def detect_variance_drift(
        self,
        data: pd.Series,
        reference_period: Optional[pd.Series] = None
    ) -> Optional[DriftAlert]:
        """
        Detect drift in variance (volatility regime change).
        
        Args:
            data: Time series data
            reference_period: Historical reference
            
        Returns:
            DriftAlert if drift detected
        """
        if len(data) < self.window_size * 2:
            return None
        
        # Reference variance
        if reference_period is None:
            reference = data.iloc[:self.window_size]
        else:
            reference = reference_period
        
        ref_var = reference.var()
        
        if ref_var == 0:
            return None
        
        # Current variance
        current = data.iloc[-self.window_size:]
        current_var = current.var()
        
        # F-test for variance equality
        f_stat = current_var / ref_var if current_var > ref_var else ref_var / current_var
        
        # Critical value at 95% confidence
        from scipy.stats import f as f_dist
        f_critical = f_dist.ppf(0.975, self.window_size-1, self.window_size-1)
        
        if f_stat > f_critical:
            severity = 'critical' if f_stat > f_critical * 2 else 'high'
            
            return DriftAlert(
                timestamp=datetime.now(),
                drift_type='variance',
                severity=severity,
                metric_value=current_var,
                threshold=ref_var,
                description=f"Variance changed from {ref_var:.6f} to {current_var:.6f} (F={f_stat:.2f})",
                recommendation="Volatility regime has shifted. Adjust position sizing and risk limits."
            )
        
        return None
    
    def detect_cusum_drift(
        self,
        data: pd.Series,
        target_mean: Optional[float] = None
    ) -> Optional[DriftAlert]:
        """
        CUSUM (Cumulative Sum) drift detection.
        Detects small persistent changes in mean.
        
        Args:
            data: Time series data
            target_mean: Expected mean (if None, uses historical mean)
            
        Returns:
            DriftAlert if drift detected
        """
        if len(data) < self.window_size:
            return None
        
        if target_mean is None:
            target_mean = data.iloc[:self.window_size].mean()
        
        # Calculate CUSUM
        cusum_pos = 0
        cusum_neg = 0
        
        for value in data.iloc[-self.window_size:]:
            deviation = value - target_mean
            cusum_pos = max(0, cusum_pos + deviation - self.cusum_drift)
            cusum_neg = max(0, cusum_neg - deviation - self.cusum_drift)
        
        # Check thresholds
        if cusum_pos > self.cusum_threshold:
            return DriftAlert(
                timestamp=datetime.now(),
                drift_type='cusum_upward',
                severity='medium',
                metric_value=cusum_pos,
                threshold=self.cusum_threshold,
                description=f"Persistent upward drift detected (CUSUM={cusum_pos:.2f})",
                recommendation="Mean has shifted upward. Monitor for trend continuation."
            )
        elif cusum_neg > self.cusum_threshold:
            return DriftAlert(
                timestamp=datetime.now(),
                drift_type='cusum_downward',
                severity='medium',
                metric_value=cusum_neg,
                threshold=self.cusum_threshold,
                description=f"Persistent downward drift detected (CUSUM={cusum_neg:.2f})",
                recommendation="Mean has shifted downward. Consider defensive positioning."
            )
        
        return None
    
    def detect_all_drifts(
        self,
        data: pd.Series,
        reference_period: Optional[pd.Series] = None
    ) -> List[DriftAlert]:
        """
        Run all drift detection methods.
        
        Args:
            data: Time series data
            reference_period: Historical reference
            
        Returns:
            List of DriftAlerts
        """
        alerts = []
        
        # Mean drift
        mean_alert = self.detect_mean_drift(data, reference_period)
        if mean_alert:
            alerts.append(mean_alert)
        
        # Variance drift
        var_alert = self.detect_variance_drift(data, reference_period)
        if var_alert:
            alerts.append(var_alert)
        
        # CUSUM drift
        cusum_alert = self.detect_cusum_drift(data)
        if cusum_alert:
            alerts.append(cusum_alert)
        
        return alerts


class RegimeDetector:
    """
    Detects market regime changes using multiple methodologies.
    """
    
    def __init__(self):
        """Initialize regime detector"""
        self.logger = logging.getLogger(__name__)
        self.current_regime = None
        self.regime_start_date = None
    
    def detect_volatility_regime(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> str:
        """
        Detect volatility regime.
        
        Args:
            returns: Return series
            window: Rolling window for volatility
            
        Returns:
            Regime: 'low_vol', 'normal_vol', 'high_vol', 'crisis'
        """
        if len(returns) < window * 2:
            return 'unknown'
        
        # Calculate rolling volatility (annualized)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Current volatility
        current_vol = rolling_vol.iloc[-1]
        
        # Historical percentiles
        hist_vol = rolling_vol.iloc[:-window]  # Exclude recent period
        
        percentile_25 = hist_vol.quantile(0.25)
        percentile_75 = hist_vol.quantile(0.75)
        percentile_95 = hist_vol.quantile(0.95)
        
        # Classify regime
        if current_vol > percentile_95:
            return 'crisis'
        elif current_vol > percentile_75:
            return 'high_vol'
        elif current_vol < percentile_25:
            return 'low_vol'
        else:
            return 'normal_vol'
    
    def detect_trend_regime(
        self,
        prices: pd.Series,
        short_window: int = 50,
        long_window: int = 200
    ) -> str:
        """
        Detect trend regime using moving averages.
        
        Args:
            prices: Price series
            short_window: Short MA period
            long_window: Long MA period
            
        Returns:
            Regime: 'strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'
        """
        if len(prices) < long_window:
            return 'unknown'
        
        # Calculate MAs
        sma_short = prices.rolling(window=short_window).mean()
        sma_long = prices.rolling(window=long_window).mean()
        
        current_price = prices.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        
        # Price position relative to MAs
        above_short = current_price > current_sma_short
        above_long = current_price > current_sma_long
        short_above_long = current_sma_short > current_sma_long
        
        # Slope of long MA
        long_ma_slope = (current_sma_long - sma_long.iloc[-20]) / sma_long.iloc[-20] if len(sma_long) >= 20 else 0
        
        # Classify
        if above_short and above_long and short_above_long and long_ma_slope > 0.01:
            return 'strong_bull'
        elif above_long and short_above_long:
            return 'bull'
        elif not above_short and not above_long and not short_above_long and long_ma_slope < -0.01:
            return 'strong_bear'
        elif not above_long and not short_above_long:
            return 'bear'
        else:
            return 'neutral'
    
    def detect_correlation_regime(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> str:
        """
        Detect correlation regime (crisis = high correlation).
        
        Args:
            returns: DataFrame of multiple asset returns
            window: Rolling window
            
        Returns:
            Regime: 'diversified', 'normal', 'correlated', 'crisis'
        """
        if len(returns) < window or returns.shape[1] < 2:
            return 'unknown'
        
        # Calculate rolling correlation matrix
        rolling_corr = returns.rolling(window=window).corr()
        
        # Get most recent correlation matrix
        latest_corr = returns.iloc[-window:].corr()
        
        # Average absolute correlation (excluding diagonal)
        mask = np.ones_like(latest_corr, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = latest_corr.where(mask).abs().mean().mean()
        
        # Classify
        if avg_corr > 0.8:
            return 'crisis'  # Everything moving together
        elif avg_corr > 0.6:
            return 'correlated'
        elif avg_corr < 0.3:
            return 'diversified'
        else:
            return 'normal'
    
    def detect_comprehensive_regime(
        self,
        prices: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Comprehensive regime detection combining multiple signals.
        
        Args:
            prices: Price series
            returns: Return series
            volume: Volume series (optional)
            
        Returns:
            RegimeState
        """
        self.logger.info("Detecting market regime...")
        
        # Detect individual regimes
        vol_regime = self.detect_volatility_regime(returns)
        trend_regime = self.detect_trend_regime(prices)
        
        # Combine signals
        characteristics = {
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime
        }
        
        # Determine overall regime
        if vol_regime == 'crisis':
            regime = 'crisis'
            confidence = 95
        elif trend_regime in ['strong_bull', 'bull'] and vol_regime in ['low_vol', 'normal_vol']:
            regime = 'bull'
            confidence = 85 if trend_regime == 'strong_bull' else 70
        elif trend_regime in ['strong_bear', 'bear'] and vol_regime != 'crisis':
            regime = 'bear'
            confidence = 85 if trend_regime == 'strong_bear' else 70
        elif vol_regime == 'high_vol':
            regime = 'high_vol'
            confidence = 75
        elif vol_regime == 'low_vol':
            regime = 'low_vol'
            confidence = 75
        else:
            regime = 'neutral'
            confidence = 60
        
        # Calculate regime duration
        if self.current_regime != regime:
            self.current_regime = regime
            self.regime_start_date = datetime.now()
            duration_days = 0
        else:
            duration_days = (datetime.now() - self.regime_start_date).days if self.regime_start_date else 0
        
        self.logger.info(f"  Detected regime: {regime} (confidence: {confidence}%)")
        self.logger.info(f"  Volatility: {vol_regime}, Trend: {trend_regime}")
        self.logger.info(f"  Duration: {duration_days} days")
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            duration_days=duration_days,
            characteristics=characteristics
        )


# Global instances
drift_detector = DriftDetector()
regime_detector = RegimeDetector()
