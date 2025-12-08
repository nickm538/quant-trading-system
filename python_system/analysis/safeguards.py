"""
Advanced Safeguards for Production Trading
==========================================

Implements critical safeguards to prevent:
- Overfitting
- Look-ahead bias
- Data leakage
- Noise-induced false signals

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class SafeguardReport:
    """Report on safeguard checks"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    confidence_adjustment: float  # Multiplier for final confidence (0-1)
    details: Dict[str, Any]


class TradingSafeguards:
    """
    Comprehensive safeguards for production trading system.
    Ensures model integrity and prevents common pitfalls.
    """
    
    def __init__(self):
        """Initialize safeguards"""
        self.logger = logging.getLogger(__name__)
    
    # ==================== LOOK-AHEAD BIAS PREVENTION ====================
    
    def check_temporal_integrity(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> SafeguardReport:
        """
        Verify no future data is used in calculations.
        
        Args:
            data: DataFrame with time-indexed data
            feature_columns: Columns to check for look-ahead bias
            
        Returns:
            SafeguardReport
        """
        warnings = []
        errors = []
        
        # Check 1: Ensure data is sorted by time
        if not data.index.is_monotonic_increasing:
            errors.append("Data is not sorted chronologically - CRITICAL look-ahead bias risk")
        
        # Check 2: Verify no NaN at beginning (indicates forward-filling)
        for col in feature_columns:
            if col in data.columns:
                first_valid_idx = data[col].first_valid_index()
                if first_valid_idx != data.index[0]:
                    warnings.append(
                        f"Column '{col}' has NaN values at start - "
                        f"ensure no forward-filling occurred"
                    )
        
        # Check 3: Verify rolling calculations don't use future data
        # (This is a heuristic check - looks for suspicious patterns)
        for col in feature_columns:
            if col in data.columns and len(data) > 100:
                # Check if values at end are suspiciously similar to middle
                # (could indicate future data leakage)
                mid_section = data[col].iloc[40:60]
                end_section = data[col].iloc[-20:]
                
                if not mid_section.empty and not end_section.empty:
                    correlation = mid_section.corr(end_section.iloc[:len(mid_section)])
                    if correlation > 0.99:
                        warnings.append(
                            f"Column '{col}' shows suspiciously high correlation "
                            f"between past and recent values ({correlation:.4f}) - "
                            f"verify calculation method"
                        )
        
        passed = len(errors) == 0
        confidence_adjustment = 1.0 if passed else 0.0
        
        if warnings:
            confidence_adjustment *= 0.9  # Reduce confidence by 10% for warnings
        
        return SafeguardReport(
            passed=passed,
            warnings=warnings,
            errors=errors,
            confidence_adjustment=confidence_adjustment,
            details={'check_type': 'temporal_integrity'}
        )
    
    # ==================== DATA LEAKAGE PREVENTION ====================
    
    def check_data_leakage(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str
    ) -> SafeguardReport:
        """
        Check for data leakage between train and test sets.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            target_column: Target variable column
            
        Returns:
            SafeguardReport
        """
        warnings = []
        errors = []
        
        # Check 1: Ensure no overlap in indices
        train_idx = set(train_data.index)
        test_idx = set(test_data.index)
        overlap = train_idx.intersection(test_idx)
        
        if overlap:
            errors.append(
                f"Train/test sets have {len(overlap)} overlapping indices - "
                f"CRITICAL data leakage"
            )
        
        # Check 2: Ensure test data is chronologically after train data
        if len(train_data) > 0 and len(test_data) > 0:
            last_train_date = train_data.index.max()
            first_test_date = test_data.index.min()
            
            if first_test_date <= last_train_date:
                errors.append(
                    f"Test data ({first_test_date}) starts before or at "
                    f"train data end ({last_train_date}) - temporal leakage"
                )
        
        # Check 3: Verify target variable statistics are reasonable
        if target_column in train_data.columns and target_column in test_data.columns:
            train_mean = train_data[target_column].mean()
            test_mean = test_data[target_column].mean()
            train_std = train_data[target_column].std()
            
            if train_std > 0:
                z_score = abs(test_mean - train_mean) / train_std
                if z_score > 5:
                    warnings.append(
                        f"Test set target mean ({test_mean:.4f}) is {z_score:.2f} "
                        f"standard deviations from train mean ({train_mean:.4f}) - "
                        f"possible distribution shift"
                    )
        
        passed = len(errors) == 0
        confidence_adjustment = 1.0 if passed else 0.0
        
        return SafeguardReport(
            passed=passed,
            warnings=warnings,
            errors=errors,
            confidence_adjustment=confidence_adjustment,
            details={'check_type': 'data_leakage'}
        )
    
    # ==================== OVERFITTING PREVENTION ====================
    
    def check_overfitting_risk(
        self,
        train_score: float,
        test_score: float,
        n_features: int,
        n_samples: int,
        model_complexity: str = 'medium'  # 'low', 'medium', 'high'
    ) -> SafeguardReport:
        """
        Assess overfitting risk based on train/test performance gap.
        
        Args:
            train_score: Model score on training data (e.g., R²)
            test_score: Model score on test data
            n_features: Number of features used
            n_samples: Number of training samples
            model_complexity: Complexity level of model
            
        Returns:
            SafeguardReport
        """
        warnings = []
        errors = []
        
        # Check 1: Train/test score gap
        score_gap = train_score - test_score
        
        if score_gap > 0.15:
            errors.append(
                f"Large train/test score gap ({score_gap:.3f}) - "
                f"SEVERE overfitting detected"
            )
        elif score_gap > 0.08:
            warnings.append(
                f"Moderate train/test score gap ({score_gap:.3f}) - "
                f"possible overfitting"
            )
        
        # Check 2: Feature-to-sample ratio
        feature_ratio = n_features / n_samples
        
        complexity_thresholds = {
            'low': 0.1,      # Linear models can handle more features
            'medium': 0.05,  # Tree-based models
            'high': 0.02     # Deep learning / complex models
        }
        
        threshold = complexity_thresholds.get(model_complexity, 0.05)
        
        if feature_ratio > threshold:
            warnings.append(
                f"High feature-to-sample ratio ({feature_ratio:.4f}) - "
                f"overfitting risk. Recommended: < {threshold}"
            )
        
        # Check 3: Test score reasonableness
        if test_score < 0:
            errors.append(
                f"Negative test score ({test_score:.3f}) - "
                f"model performs worse than baseline"
            )
        elif test_score > 0.95:
            warnings.append(
                f"Suspiciously high test score ({test_score:.3f}) - "
                f"verify data integrity"
            )
        
        passed = len(errors) == 0
        
        # Adjust confidence based on overfitting severity
        if score_gap > 0.15:
            confidence_adjustment = 0.5  # Severe overfitting
        elif score_gap > 0.08:
            confidence_adjustment = 0.8  # Moderate overfitting
        else:
            confidence_adjustment = 1.0
        
        return SafeguardReport(
            passed=passed,
            warnings=warnings,
            errors=errors,
            confidence_adjustment=confidence_adjustment,
            details={
                'check_type': 'overfitting',
                'score_gap': score_gap,
                'feature_ratio': feature_ratio
            }
        )
    
    # ==================== NOISE REDUCTION ====================
    
    def detect_noisy_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        correlation_threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        Detect highly correlated (redundant) and noisy features.
        
        Args:
            data: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target variable column
            correlation_threshold: Threshold for high correlation
            
        Returns:
            Dict with 'redundant' and 'noisy' feature lists
        """
        redundant_features = []
        noisy_features = []
        
        # Calculate correlation matrix
        feature_data = data[feature_columns]
        corr_matrix = feature_data.corr().abs()
        
        # Find redundant features (highly correlated pairs)
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        redundant_pairs = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    feat1 = feature_columns[i]
                    feat2 = feature_columns[j]
                    redundant_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))
        
        # For each redundant pair, keep the one more correlated with target
        if target_column in data.columns:
            target_corr = data[feature_columns + [target_column]].corr()[target_column].abs()
            
            for feat1, feat2, pair_corr in redundant_pairs:
                if target_corr[feat1] < target_corr[feat2]:
                    if feat1 not in redundant_features:
                        redundant_features.append(feat1)
                else:
                    if feat2 not in redundant_features:
                        redundant_features.append(feat2)
            
            # Find noisy features (low correlation with target)
            for feat in feature_columns:
                if abs(target_corr[feat]) < 0.05 and feat not in redundant_features:
                    noisy_features.append(feat)
        
        logger.info(f"Noise detection: {len(redundant_features)} redundant, "
                   f"{len(noisy_features)} noisy features")
        
        return {
            'redundant': redundant_features,
            'noisy': noisy_features,
            'redundant_pairs': redundant_pairs
        }
    
    def remove_outliers(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',  # 'iqr' or 'zscore'
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove outliers that could cause overfitting.
        
        Args:
            data: DataFrame
            columns: Columns to check for outliers
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: IQR multiplier or z-score threshold
            
        Returns:
            Tuple of (cleaned_data, outlier_report)
        """
        cleaned_data = data.copy()
        outlier_report = {}
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = pd.Series(False, index=data.index)
                outliers.loc[data[col].notna()] = z_scores > threshold
            
            else:
                continue
            
            n_outliers = outliers.sum()
            if n_outliers > 0:
                outlier_report[col] = {
                    'count': n_outliers,
                    'percentage': (n_outliers / len(data)) * 100,
                    'method': method
                }
                
                # Remove outliers
                cleaned_data = cleaned_data[~outliers]
        
        logger.info(f"Removed {len(data) - len(cleaned_data)} outlier rows")
        
        return cleaned_data, outlier_report
    
    # ==================== COMPREHENSIVE VALIDATION ====================
    
    def run_all_safeguards(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        train_score: Optional[float] = None,
        test_score: Optional[float] = None
    ) -> Dict[str, SafeguardReport]:
        """
        Run all safeguard checks and return comprehensive report.
        
        Args:
            data: Full dataset
            feature_columns: Feature columns
            target_column: Target variable
            train_score: Optional training score
            test_score: Optional test score
            
        Returns:
            Dict of SafeguardReports
        """
        logger.info("="*80)
        logger.info("Running comprehensive safeguard checks")
        logger.info("="*80)
        
        reports = {}
        
        # 1. Temporal integrity
        reports['temporal_integrity'] = self.check_temporal_integrity(
            data, feature_columns
        )
        
        # 2. Noise detection
        noise_result = self.detect_noisy_features(
            data, feature_columns, target_column
        )
        reports['noise_detection'] = SafeguardReport(
            passed=True,
            warnings=[
                f"Found {len(noise_result['redundant'])} redundant features",
                f"Found {len(noise_result['noisy'])} noisy features"
            ],
            errors=[],
            confidence_adjustment=1.0,
            details=noise_result
        )
        
        # 3. Overfitting check (if scores provided)
        if train_score is not None and test_score is not None:
            reports['overfitting'] = self.check_overfitting_risk(
                train_score=train_score,
                test_score=test_score,
                n_features=len(feature_columns),
                n_samples=len(data)
            )
        
        # Calculate overall confidence adjustment
        overall_confidence = 1.0
        all_passed = True
        
        for name, report in reports.items():
            overall_confidence *= report.confidence_adjustment
            if not report.passed:
                all_passed = False
            
            logger.info(f"\n{name.upper()}:")
            logger.info(f"  Passed: {report.passed}")
            if report.warnings:
                for warning in report.warnings:
                    logger.warning(f"  ⚠ {warning}")
            if report.errors:
                for error in report.errors:
                    logger.error(f"  ✗ {error}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Overall safeguard status: {'PASSED' if all_passed else 'FAILED'}")
        logger.info(f"Confidence adjustment: {overall_confidence:.2%}")
        logger.info(f"{'='*80}\n")
        
        return reports


# Global instance
trading_safeguards = TradingSafeguards()
