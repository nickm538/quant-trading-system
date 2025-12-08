"""
Walk-Forward Cross-Validation & Advanced Momentum Analysis
===========================================================

Implements robust validation methodology used by institutional hedge funds.
Prevents overfitting through time-series aware validation.

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    train_scores: List[float]
    test_scores: List[float]
    predictions: pd.Series
    actuals: pd.Series
    feature_importance: Dict[str, float]
    best_parameters: Dict[str, Any]
    overfitting_score: float  # 0 = no overfitting, 1 = severe overfitting
    stability_score: float  # 0-100, higher = more stable
    confidence: float  # 0-100


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for time-series data.
    
    This is the GOLD STANDARD for validating trading strategies.
    Used by top hedge funds to ensure models work in live trading.
    """
    
    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 21,    # 1 month
        step_days: int = 21             # Move forward 1 month each iteration
    ):
        """
        Initialize walk-forward cross-validation.
        
        Args:
            train_period_days: Days in training window
            test_period_days: Days in test window
            step_days: Days to step forward each iteration
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.logger = logging.getLogger(__name__)
    
    def split_data(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward train/test splits.
        
        Args:
            data: Time-indexed DataFrame
            
        Returns:
            List of (train, test) DataFrame tuples
        """
        splits = []
        
        # Ensure data is sorted
        data = data.sort_index()
        
        # Calculate number of splits
        total_days = (data.index[-1] - data.index[0]).days
        n_splits = max(1, (total_days - self.train_period_days) // self.step_days)
        
        self.logger.info(f"Creating {n_splits} walk-forward splits")
        self.logger.info(f"  Train period: {self.train_period_days} days")
        self.logger.info(f"  Test period: {self.test_period_days} days")
        self.logger.info(f"  Step size: {self.step_days} days")
        
        for i in range(n_splits):
            # Calculate date ranges
            train_start_offset = i * self.step_days
            train_end_offset = train_start_offset + self.train_period_days
            test_end_offset = train_end_offset + self.test_period_days
            
            train_start = data.index[0] + timedelta(days=train_start_offset)
            train_end = data.index[0] + timedelta(days=train_end_offset)
            test_end = data.index[0] + timedelta(days=test_end_offset)
            
            # Get data slices
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= train_end) & (data.index < test_end)]
            
            if len(train_data) > 0 and len(test_data) > 0:
                splits.append((train_data, test_data))
                self.logger.debug(
                    f"Split {i+1}: Train {len(train_data)} samples "
                    f"({train_start.date()} to {train_end.date()}), "
                    f"Test {len(test_data)} samples "
                    f"({train_end.date()} to {test_end.date()})"
                )
        
        return splits
    
    def validate_model(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        model_func: Callable,
        metric: str = 'r2'  # 'r2', 'mse', 'mae'
    ) -> WalkForwardResult:
        """
        Perform walk-forward validation on a model.
        
        Args:
            data: Full dataset
            feature_columns: Feature column names
            target_column: Target variable column
            model_func: Function that takes (X_train, y_train) and returns fitted model
            metric: Evaluation metric
            
        Returns:
            WalkForwardResult with comprehensive validation metrics
        """
        self.logger.info("="*80)
        self.logger.info("Starting Walk-Forward Cross-Validation")
        self.logger.info("="*80)
        
        splits = self.split_data(data)
        
        train_scores = []
        test_scores = []
        all_predictions = []
        all_actuals = []
        feature_importances = {col: [] for col in feature_columns}
        
        for i, (train_data, test_data) in enumerate(splits):
            self.logger.info(f"\nFold {i+1}/{len(splits)}")
            
            # Prepare data
            X_train = train_data[feature_columns].values
            y_train = train_data[target_column].values
            X_test = test_data[feature_columns].values
            y_test = test_data[target_column].values
            
            # Train model
            model = model_func(X_train, y_train)
            
            # Predict
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate scores
            if metric == 'r2':
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            elif metric == 'mse':
                train_score = -mean_squared_error(y_train, train_pred)  # Negative for consistency
                test_score = -mean_squared_error(y_test, test_pred)
            elif metric == 'mae':
                train_score = -mean_absolute_error(y_train, train_pred)
                test_score = -mean_absolute_error(y_test, test_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
            # Store predictions
            for pred, actual, idx in zip(test_pred, y_test, test_data.index):
                all_predictions.append({'date': idx, 'prediction': pred})
                all_actuals.append({'date': idx, 'actual': actual})
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                for col, importance in zip(feature_columns, model.feature_importances_):
                    feature_importances[col].append(importance)
            
            self.logger.info(f"  Train {metric}: {train_score:.4f}")
            self.logger.info(f"  Test {metric}: {test_score:.4f}")
            self.logger.info(f"  Gap: {train_score - test_score:.4f}")
        
        # Aggregate results
        predictions_df = pd.DataFrame(all_predictions).set_index('date')['prediction']
        actuals_df = pd.DataFrame(all_actuals).set_index('date')['actual']
        
        # Calculate feature importance (average across folds)
        avg_feature_importance = {}
        for col, importances in feature_importances.items():
            if importances:
                avg_feature_importance[col] = np.mean(importances)
        
        # Calculate overfitting score
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        score_gap = avg_train - avg_test
        
        # Overfitting score: 0 = no overfitting, 1 = severe
        overfitting_score = min(1.0, max(0.0, score_gap / 0.2))  # 0.2 = severe threshold
        
        # Stability score: consistency of test scores across folds
        test_score_std = np.std(test_scores)
        stability_score = max(0, 100 * (1 - test_score_std))  # Higher = more stable
        
        # Confidence: based on test performance and stability
        if avg_test > 0.7:
            base_confidence = 90
        elif avg_test > 0.5:
            base_confidence = 75
        elif avg_test > 0.3:
            base_confidence = 60
        else:
            base_confidence = 40
        
        # Adjust for overfitting and stability
        confidence = base_confidence * (1 - overfitting_score * 0.3) * (stability_score / 100)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Walk-Forward Validation Results")
        self.logger.info("="*80)
        self.logger.info(f"Average Train Score: {avg_train:.4f}")
        self.logger.info(f"Average Test Score: {avg_test:.4f}")
        self.logger.info(f"Score Gap: {score_gap:.4f}")
        self.logger.info(f"Overfitting Score: {overfitting_score:.2f} (0=none, 1=severe)")
        self.logger.info(f"Stability Score: {stability_score:.1f}/100")
        self.logger.info(f"Final Confidence: {confidence:.1f}%")
        self.logger.info("="*80 + "\n")
        
        return WalkForwardResult(
            train_scores=train_scores,
            test_scores=test_scores,
            predictions=predictions_df,
            actuals=actuals_df,
            feature_importance=avg_feature_importance,
            best_parameters={},  # Can be extended for hyperparameter tuning
            overfitting_score=overfitting_score,
            stability_score=stability_score,
            confidence=confidence
        )


class AdvancedMomentumAnalysis:
    """
    Advanced momentum analysis using multiple timeframes and methodologies.
    Combines technical, statistical, and machine learning approaches.
    """
    
    def __init__(self):
        """Initialize momentum analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_multi_timeframe_momentum(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, float]:
        """
        Calculate momentum across multiple timeframes.
        
        Args:
            data: OHLCV DataFrame
            price_column: Column to use for momentum
            
        Returns:
            Dict of momentum metrics
        """
        momentum = {}
        
        # Short-term momentum (1-5 days)
        if len(data) >= 5:
            momentum['momentum_1d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-2] - 1) * 100
            momentum['momentum_5d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-6] - 1) * 100
        
        # Medium-term momentum (10-20 days)
        if len(data) >= 20:
            momentum['momentum_10d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-11] - 1) * 100
            momentum['momentum_20d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-21] - 1) * 100
        
        # Long-term momentum (60-120 days)
        if len(data) >= 120:
            momentum['momentum_60d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-61] - 1) * 100
            momentum['momentum_120d'] = (data[price_column].iloc[-1] / data[price_column].iloc[-121] - 1) * 100
        
        return momentum
    
    def calculate_momentum_quality(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, float]:
        """
        Assess quality of momentum (not just magnitude).
        
        High-quality momentum:
        - Consistent direction
        - Increasing strength
        - Supported by volume
        
        Args:
            data: OHLCV DataFrame
            price_column: Price column
            
        Returns:
            Dict of quality metrics
        """
        quality = {}
        
        if len(data) < 20:
            return quality
        
        # Calculate returns
        returns = data[price_column].pct_change()
        
        # 1. Consistency: % of days moving in same direction
        recent_returns = returns.tail(20)
        positive_days = (recent_returns > 0).sum()
        negative_days = (recent_returns < 0).sum()
        quality['consistency'] = max(positive_days, negative_days) / 20 * 100
        
        # 2. Acceleration: is momentum increasing?
        if len(data) >= 40:
            recent_momentum = (data[price_column].iloc[-1] / data[price_column].iloc[-21] - 1)
            past_momentum = (data[price_column].iloc[-21] / data[price_column].iloc[-41] - 1)
            quality['acceleration'] = (recent_momentum - past_momentum) * 100
        
        # 3. Volume confirmation
        if 'volume' in data.columns:
            recent_volume = data['volume'].tail(20).mean()
            past_volume = data['volume'].iloc[-40:-20].mean() if len(data) >= 40 else recent_volume
            quality['volume_confirmation'] = (recent_volume / past_volume - 1) * 100 if past_volume > 0 else 0
        
        # 4. Volatility-adjusted momentum (Sharpe-like)
        if len(recent_returns) > 1:
            mean_return = recent_returns.mean()
            std_return = recent_returns.std()
            quality['risk_adjusted_momentum'] = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        return quality
    
    def detect_momentum_regime(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> str:
        """
        Detect current momentum regime.
        
        Regimes:
        - 'strong_uptrend': Consistent strong positive momentum
        - 'weak_uptrend': Inconsistent positive momentum
        - 'consolidation': No clear direction
        - 'weak_downtrend': Inconsistent negative momentum
        - 'strong_downtrend': Consistent strong negative momentum
        
        Args:
            data: OHLCV DataFrame
            price_column: Price column
            
        Returns:
            Regime string
        """
        momentum = self.calculate_multi_timeframe_momentum(data, price_column)
        quality = self.calculate_momentum_quality(data, price_column)
        
        if not momentum or not quality:
            return 'unknown'
        
        # Average momentum across timeframes
        avg_momentum = np.mean([v for k, v in momentum.items() if 'momentum_' in k])
        consistency = quality.get('consistency', 50)
        
        # Classify regime
        if avg_momentum > 2 and consistency > 65:
            return 'strong_uptrend'
        elif avg_momentum > 0.5 and consistency > 55:
            return 'weak_uptrend'
        elif avg_momentum < -2 and consistency > 65:
            return 'strong_downtrend'
        elif avg_momentum < -0.5 and consistency > 55:
            return 'weak_downtrend'
        else:
            return 'consolidation'
    
    def calculate_momentum_score(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> float:
        """
        Calculate comprehensive momentum score (0-100).
        
        Args:
            data: OHLCV DataFrame
            price_column: Price column
            
        Returns:
            Momentum score 0-100 (higher = stronger bullish momentum)
        """
        momentum = self.calculate_multi_timeframe_momentum(data, price_column)
        quality = self.calculate_momentum_quality(data, price_column)
        
        if not momentum:
            return 50  # Neutral
        
        # Weight different timeframes
        score = 50  # Start neutral
        
        # Short-term (30% weight)
        if 'momentum_5d' in momentum:
            score += momentum['momentum_5d'] * 1.5  # 30% weight
        
        # Medium-term (40% weight)
        if 'momentum_20d' in momentum:
            score += momentum['momentum_20d'] * 2.0  # 40% weight
        
        # Long-term (30% weight)
        if 'momentum_60d' in momentum:
            score += momentum['momentum_60d'] * 1.5  # 30% weight
        
        # Adjust for quality
        if quality:
            consistency_factor = quality.get('consistency', 50) / 100
            score = score * (0.7 + 0.3 * consistency_factor)  # Up to 30% boost for consistency
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return score


# Global instances
walk_forward_cv = WalkForwardCV()
momentum_analyzer = AdvancedMomentumAnalysis()
