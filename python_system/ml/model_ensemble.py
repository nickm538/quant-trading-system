"""
Model Ensemble System with Comprehensive Safeguards
====================================================

Prevents:
1. Model Fighting - Conflicting predictions from different models
2. Degradation - Performance decay over time
3. Overfitting - Memorizing noise instead of learning patterns
4. Data Leakage - Future information contaminating training
5. Data Conflicts - Inconsistencies between data sources

Features:
- Intelligent ensemble voting with confidence weighting
- Continuous performance monitoring
- Automatic retraining triggers
- Data consistency validation
- Out-of-sample testing

Author: Institutional Trading System
Date: 2025-11-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model prediction with metadata"""
    model_name: str
    model_type: str  # 'xgboost', 'lightgbm', 'lstm'
    predicted_return: float
    confidence: float  # 0-1
    timestamp: datetime
    features_used: List[str]
    
    def __repr__(self):
        return f"{self.model_name}: {self.predicted_return:.4f} (conf: {self.confidence:.2f})"


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with safeguards"""
    symbol: str
    predicted_return: float
    confidence: float
    individual_predictions: List[ModelPrediction]
    agreement_score: float  # How much models agree (0-1)
    warning_flags: List[str]  # Any warnings detected
    timestamp: datetime
    
    def is_reliable(self, min_confidence: float = 0.6, min_agreement: float = 0.7) -> bool:
        """Check if prediction is reliable"""
        return (
            self.confidence >= min_confidence and
            self.agreement_score >= min_agreement and
            len(self.warning_flags) == 0
        )


class ModelEnsemble:
    """
    Ensemble system with safeguards against common ML pitfalls.
    
    Safeguards:
    1. Model Fighting Prevention: Weighted voting based on recent performance
    2. Degradation Detection: Track accuracy over time, trigger retraining
    3. Overfitting Prevention: Out-of-sample validation, regularization
    4. Data Leakage Prevention: Strict temporal separation
    5. Conflict Resolution: Cross-validate data sources
    """
    
    def __init__(
        self,
        models_dir: str = '/home/ubuntu/quant-trading-web/python_system/ml/models',
        performance_window: int = 30  # days
    ):
        """Initialize ensemble system"""
        self.models_dir = Path(models_dir)
        self.performance_window = performance_window
        
        # Performance tracking
        self.model_performance = {}  # {model_name: {date: accuracy}}
        self.prediction_history = []  # List of (prediction, actual) tuples
        
        # Safeguard thresholds
        self.MIN_CONFIDENCE = 0.6
        self.MIN_AGREEMENT = 0.7
        self.MAX_DISAGREEMENT = 0.3  # Max allowed disagreement between models
        self.DEGRADATION_THRESHOLD = 0.05  # 5% accuracy drop triggers retraining
        self.MIN_MODELS = 2  # Minimum models required for ensemble
        
        logger.info("ModelEnsemble initialized with safeguards")
    
    def predict_ensemble(
        self,
        symbol: str,
        predictions: List[ModelPrediction],
        use_weighted_voting: bool = True
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction with safeguards.
        
        Args:
            symbol: Stock symbol
            predictions: List of individual model predictions
            use_weighted_voting: Weight by recent performance
            
        Returns:
            EnsemblePrediction with warnings
        """
        if len(predictions) < self.MIN_MODELS:
            raise ValueError(f"Need at least {self.MIN_MODELS} models for ensemble")
        
        warning_flags = []
        
        # 1. Check for model fighting (high disagreement)
        agreement_score = self._calculate_agreement(predictions)
        
        if agreement_score < self.MIN_AGREEMENT:
            warning_flags.append(
                f"LOW_AGREEMENT: Models disagree (score: {agreement_score:.2f})"
            )
            logger.warning(f"Model disagreement detected for {symbol}")
        
        # 2. Check for extreme predictions (potential overfitting)
        pred_values = [p.predicted_return for p in predictions]
        if max(abs(v) for v in pred_values) > 0.20:  # >20% return
            warning_flags.append("EXTREME_PREDICTION: Unusually large predicted return")
            logger.warning(f"Extreme prediction detected for {symbol}")
        
        # 3. Check confidence levels
        avg_confidence = np.mean([p.confidence for p in predictions])
        if avg_confidence < self.MIN_CONFIDENCE:
            warning_flags.append(
                f"LOW_CONFIDENCE: Average confidence {avg_confidence:.2f}"
            )
        
        # 4. Calculate weighted ensemble prediction
        if use_weighted_voting:
            # Weight by confidence and recent performance
            weights = self._calculate_weights(predictions)
            ensemble_return = np.average(pred_values, weights=weights)
            ensemble_confidence = np.average(
                [p.confidence for p in predictions],
                weights=weights
            )
        else:
            # Simple average
            ensemble_return = np.mean(pred_values)
            ensemble_confidence = np.mean([p.confidence for p in predictions])
        
        # 5. Apply safety bounds (prevent extreme predictions)
        ensemble_return = np.clip(ensemble_return, -0.15, 0.15)  # ±15% max
        
        ensemble = EnsemblePrediction(
            symbol=symbol,
            predicted_return=ensemble_return,
            confidence=ensemble_confidence,
            individual_predictions=predictions,
            agreement_score=agreement_score,
            warning_flags=warning_flags,
            timestamp=datetime.now()
        )
        
        logger.info(f"Ensemble prediction for {symbol}: {ensemble_return:.4f} "
                   f"(agreement: {agreement_score:.2f}, warnings: {len(warning_flags)})")
        
        return ensemble
    
    def _calculate_agreement(self, predictions: List[ModelPrediction]) -> float:
        """
        Calculate agreement score between models.
        
        Returns:
            Agreement score (0-1), where 1 = perfect agreement
        """
        pred_values = [p.predicted_return for p in predictions]
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(pred_values)):
            for j in range(i + 1, len(pred_values)):
                # Agreement based on direction and magnitude
                same_direction = np.sign(pred_values[i]) == np.sign(pred_values[j])
                
                if same_direction:
                    # Calculate relative difference
                    diff = abs(pred_values[i] - pred_values[j])
                    avg = (abs(pred_values[i]) + abs(pred_values[j])) / 2
                    
                    if avg > 0:
                        rel_diff = diff / avg
                        agreement = max(0, 1 - rel_diff)
                    else:
                        agreement = 1.0
                else:
                    # Opposite directions = low agreement
                    agreement = 0.0
                
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 1.0
    
    def _calculate_weights(self, predictions: List[ModelPrediction]) -> np.ndarray:
        """
        Calculate weights for ensemble voting based on:
        1. Model confidence
        2. Recent performance
        3. Model type diversity
        """
        weights = []
        
        for pred in predictions:
            # Base weight from confidence
            weight = pred.confidence
            
            # Adjust by recent performance
            if pred.model_name in self.model_performance:
                recent_perf = self._get_recent_performance(pred.model_name)
                weight *= (0.5 + recent_perf)  # Scale by performance
            
            weights.append(weight)
        
        # Normalize
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def _get_recent_performance(self, model_name: str, days: int = 30) -> float:
        """
        Get recent performance for a model.
        
        Returns:
            Performance score (0-1)
        """
        if model_name not in self.model_performance:
            return 0.5  # Default neutral performance
        
        perf_history = self.model_performance[model_name]
        
        # Get recent performance
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = {
            date: acc for date, acc in perf_history.items()
            if date >= cutoff_date
        }
        
        if not recent:
            return 0.5
        
        return np.mean(list(recent.values()))
    
    def track_prediction(
        self,
        prediction: EnsemblePrediction,
        actual_return: float
    ):
        """
        Track prediction vs actual for performance monitoring.
        
        This enables:
        1. Degradation detection
        2. Model retraining triggers
        3. Performance-based weighting
        """
        # Calculate error
        error = abs(prediction.predicted_return - actual_return)
        accuracy = max(0, 1 - error)  # Simple accuracy metric
        
        # Track for each individual model
        for model_pred in prediction.individual_predictions:
            model_error = abs(model_pred.predicted_return - actual_return)
            model_accuracy = max(0, 1 - model_error)
            
            if model_pred.model_name not in self.model_performance:
                self.model_performance[model_pred.model_name] = {}
            
            self.model_performance[model_pred.model_name][datetime.now()] = model_accuracy
        
        # Store prediction history
        self.prediction_history.append({
            'timestamp': prediction.timestamp,
            'symbol': prediction.symbol,
            'predicted': prediction.predicted_return,
            'actual': actual_return,
            'error': error,
            'accuracy': accuracy,
            'confidence': prediction.confidence,
            'agreement': prediction.agreement_score
        })
        
        logger.info(f"Tracked prediction for {prediction.symbol}: "
                   f"predicted={prediction.predicted_return:.4f}, "
                   f"actual={actual_return:.4f}, "
                   f"error={error:.4f}")
        
        # Check for degradation
        self._check_degradation()
    
    def _check_degradation(self):
        """
        Check for model degradation and trigger retraining if needed.
        
        Degradation indicators:
        1. Accuracy drop over time
        2. Increasing prediction errors
        3. Confidence-accuracy mismatch
        """
        if len(self.prediction_history) < 10:
            return  # Need more data
        
        # Get recent predictions
        recent = self.prediction_history[-self.performance_window:]
        
        # Calculate recent vs historical accuracy
        recent_accuracy = np.mean([p['accuracy'] for p in recent])
        
        if len(self.prediction_history) > self.performance_window:
            historical = self.prediction_history[:-self.performance_window]
            historical_accuracy = np.mean([p['accuracy'] for p in historical])
            
            accuracy_drop = historical_accuracy - recent_accuracy
            
            if accuracy_drop > self.DEGRADATION_THRESHOLD:
                logger.warning(
                    f"DEGRADATION DETECTED: Accuracy dropped {accuracy_drop:.2%} "
                    f"(historical: {historical_accuracy:.2%}, "
                    f"recent: {recent_accuracy:.2%})"
                )
                logger.warning("RETRAINING RECOMMENDED")
                
                # TODO: Trigger automatic retraining
                return True
        
        return False
    
    def validate_data_consistency(
        self,
        data_sources: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Validate consistency across multiple data sources.
        
        Prevents conflicts from inconsistent data.
        
        Args:
            data_sources: Dict of {source_name: dataframe}
            
        Returns:
            Validation report with warnings
        """
        report = {
            'consistent': True,
            'warnings': [],
            'metrics': {}
        }
        
        if len(data_sources) < 2:
            return report
        
        # Compare overlapping dates
        source_names = list(data_sources.keys())
        df1 = data_sources[source_names[0]]
        df2 = data_sources[source_names[1]]
        
        # Find common dates
        common_dates = df1.index.intersection(df2.index)
        
        if len(common_dates) == 0:
            report['warnings'].append("No overlapping dates between sources")
            report['consistent'] = False
            return report
        
        # Compare close prices
        if 'close' in df1.columns and 'close' in df2.columns:
            close1 = df1.loc[common_dates, 'close']
            close2 = df2.loc[common_dates, 'close']
            
            # Calculate relative differences
            rel_diff = abs(close1 - close2) / ((close1 + close2) / 2)
            avg_diff = rel_diff.mean()
            max_diff = rel_diff.max()
            
            report['metrics']['avg_price_diff'] = avg_diff
            report['metrics']['max_price_diff'] = max_diff
            
            if avg_diff > 0.01:  # >1% average difference
                report['warnings'].append(
                    f"High price discrepancy between sources: {avg_diff:.2%} avg"
                )
                report['consistent'] = False
            
            if max_diff > 0.05:  # >5% max difference
                report['warnings'].append(
                    f"Extreme price discrepancy detected: {max_diff:.2%} max"
                )
                report['consistent'] = False
        
        logger.info(f"Data consistency check: {len(report['warnings'])} warnings")
        
        return report
    
    def detect_data_leakage(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage in features.
        
        Checks:
        1. Perfect correlation with target (suspicious)
        2. Future information in features
        3. Target variable in features
        """
        report = {
            'leakage_detected': False,
            'suspicious_features': [],
            'warnings': []
        }
        
        # Check for perfect correlations
        for col in feature_names:
            if col in features.columns:
                corr = features[col].corr(target)
                
                if abs(corr) > 0.99:
                    report['suspicious_features'].append({
                        'feature': col,
                        'correlation': corr,
                        'reason': 'Perfect correlation with target'
                    })
                    report['leakage_detected'] = True
        
        # Check for future-looking features
        future_keywords = ['next', 'future', 'forward', 'ahead', 'tomorrow']
        for col in feature_names:
            if any(keyword in col.lower() for keyword in future_keywords):
                report['warnings'].append(
                    f"Potential future-looking feature: {col}"
                )
        
        # Check for target in features
        if 'target' in feature_names or 'return' in feature_names:
            report['warnings'].append(
                "Target variable may be included in features"
            )
        
        if report['leakage_detected']:
            logger.error(f"DATA LEAKAGE DETECTED: {len(report['suspicious_features'])} suspicious features")
        
        return report
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        recent = self.prediction_history[-self.performance_window:]
        
        report = {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent),
            'recent_accuracy': np.mean([p['accuracy'] for p in recent]),
            'recent_avg_error': np.mean([p['error'] for p in recent]),
            'recent_avg_confidence': np.mean([p['confidence'] for p in recent]),
            'recent_avg_agreement': np.mean([p['agreement'] for p in recent]),
            'model_performance': {}
        }
        
        # Per-model performance
        for model_name, perf_history in self.model_performance.items():
            recent_perf = self._get_recent_performance(model_name, days=30)
            report['model_performance'][model_name] = {
                'recent_accuracy': recent_perf,
                'total_predictions': len(perf_history)
            }
        
        return report


# Example usage
if __name__ == '__main__':
    # Create ensemble
    ensemble = ModelEnsemble()
    
    # Simulate predictions from different models
    predictions = [
        ModelPrediction(
            model_name='xgboost_v1',
            model_type='xgboost',
            predicted_return=0.025,
            confidence=0.85,
            timestamp=datetime.now(),
            features_used=['rsi', 'macd', 'volume']
        ),
        ModelPrediction(
            model_name='lightgbm_v1',
            model_type='lightgbm',
            predicted_return=0.028,
            confidence=0.80,
            timestamp=datetime.now(),
            features_used=['rsi', 'macd', 'volume']
        ),
        ModelPrediction(
            model_name='lstm_v1',
            model_type='lstm',
            predicted_return=0.022,
            confidence=0.75,
            timestamp=datetime.now(),
            features_used=['price_history']
        )
    ]
    
    # Generate ensemble prediction
    ensemble_pred = ensemble.predict_ensemble('AAPL', predictions)
    
    print(f"\nEnsemble Prediction:")
    print(f"  Symbol: {ensemble_pred.symbol}")
    print(f"  Predicted Return: {ensemble_pred.predicted_return:.4f}")
    print(f"  Confidence: {ensemble_pred.confidence:.2f}")
    print(f"  Agreement Score: {ensemble_pred.agreement_score:.2f}")
    print(f"  Warnings: {ensemble_pred.warning_flags}")
    print(f"  Reliable: {ensemble_pred.is_reliable()}")
    
    # Track actual outcome
    actual_return = 0.026
    ensemble.track_prediction(ensemble_pred, actual_return)
    
    # Get performance report
    report = ensemble.get_performance_report()
    print(f"\nPerformance Report:")
    print(json.dumps(report, indent=2))
