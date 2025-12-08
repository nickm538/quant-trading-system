"""
Data Validation and Circuit Breaker System
===========================================

This module implements strict data integrity checks and circuit breakers.
NO ESTIMATES OR GUESSES - if confidence < 95%, analysis HALTS.

Author: Institutional Trading System
Date: 2025-11-20
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    source: str
    confidence_score: float  # 0-100%
    completeness: float  # % of expected data points present
    consistency: float  # Cross-source validation score
    timeliness: float  # How recent is the data
    issues: List[str]
    warnings: List[str]
    timestamp: datetime
    
    def is_acceptable(self) -> bool:
        """Check if data quality meets minimum standards"""
        return self.confidence_score >= 95.0
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.is_acceptable() else "✗ FAIL"
        return f"""
Data Quality Report [{status}]
{'='*50}
Source: {self.source}
Confidence Score: {self.confidence_score:.1f}%
Completeness: {self.completeness:.1f}%
Consistency: {self.consistency:.1f}%
Timeliness: {self.timeliness:.1f}%
Issues: {len(self.issues)}
Warnings: {len(self.warnings)}
Timestamp: {self.timestamp}
{'='*50}
"""


class CircuitBreaker:
    """
    Circuit breaker pattern for data integrity.
    Halts analysis if data quality falls below acceptable thresholds.
    """
    
    # Minimum thresholds for production trading
    MIN_CONFIDENCE = 95.0  # Never trade with < 95% confidence
    MIN_COMPLETENESS = 90.0  # Need at least 90% of data points
    MIN_CONSISTENCY = 95.0  # Cross-source validation must agree
    MIN_TIMELINESS = 80.0  # Data must be reasonably fresh
    
    # Price data validation
    MAX_PRICE_DEVIATION = 0.005  # 0.5% max difference between sources
    MIN_VOLUME_THRESHOLD = 1000  # Minimum daily volume
    
    # Technical indicator validation
    MAX_NAN_PERCENTAGE = 5.0  # Max 5% NaN values in indicators
    
    # News sentiment validation
    SENTIMENT_RANGE = (-1.0, 1.0)  # Valid sentiment score range
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_history: List[DataQualityReport] = []
    
    def validate_price_data(
        self,
        prices: pd.DataFrame,
        source: str,
        expected_days: int = 500
    ) -> DataQualityReport:
        """
        Validate OHLCV price data with strict quality checks.
        
        Args:
            prices: DataFrame with OHLCV data
            source: Data source name
            expected_days: Expected number of trading days
            
        Returns:
            DataQualityReport with detailed assessment
        """
        issues = []
        warnings = []
        
        # Check completeness
        actual_days = len(prices)
        completeness = min(100.0, (actual_days / expected_days) * 100)
        
        if actual_days < expected_days * 0.9:
            issues.append(f"Insufficient data: {actual_days}/{expected_days} days")
        
        # Check for missing values
        nan_percentage = (prices.isnull().sum().sum() / prices.size) * 100
        if nan_percentage > self.MAX_NAN_PERCENTAGE:
            issues.append(f"Too many NaN values: {nan_percentage:.1f}%")
        
        # Check for zero/negative prices
        if (prices[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append("Invalid prices detected (zero or negative)")
        
        # Check OHLC logic
        invalid_ohlc = (
            (prices['high'] < prices['low']) |
            (prices['high'] < prices['open']) |
            (prices['high'] < prices['close']) |
            (prices['low'] > prices['open']) |
            (prices['low'] > prices['close'])
        ).sum()
        
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships in {invalid_ohlc} candles")
        
        # Check volume
        if 'volume' in prices.columns:
            zero_volume_days = (prices['volume'] == 0).sum()
            if zero_volume_days > actual_days * 0.1:
                warnings.append(f"High number of zero-volume days: {zero_volume_days}")
        
        # Check for data gaps
        if isinstance(prices.index, pd.DatetimeIndex):
            date_diff = prices.index.to_series().diff()
            max_gap = date_diff.max().days if len(date_diff) > 0 else 0
            if max_gap > 7:  # More than 1 week gap
                warnings.append(f"Large data gap detected: {max_gap} days")
        
        # Check timeliness
        if isinstance(prices.index, pd.DatetimeIndex) and len(prices) > 0:
            latest_date = prices.index[-1]
            days_old = (datetime.now() - latest_date).days
            timeliness = max(0, 100 - (days_old * 5))  # Deduct 5% per day old
        else:
            timeliness = 0
            issues.append("Cannot determine data timeliness")
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            completeness=completeness,
            nan_percentage=nan_percentage,
            invalid_count=invalid_ohlc,
            timeliness=timeliness,
            issues_count=len(issues)
        )
        
        report = DataQualityReport(
            source=source,
            confidence_score=confidence,
            completeness=completeness,
            consistency=100.0,  # Will be updated by cross-validation
            timeliness=timeliness,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )
        
        self.validation_history.append(report)
        return report
    
    def cross_validate_prices(
        self,
        price_sources: Dict[str, float],
        symbol: str
    ) -> Tuple[float, float, List[str]]:
        """
        Cross-validate prices from multiple sources.
        
        Args:
            price_sources: Dict of {source_name: price}
            symbol: Stock symbol
            
        Returns:
            (consensus_price, consistency_score, issues)
        """
        issues = []
        
        if len(price_sources) < 2:
            issues.append("Insufficient sources for cross-validation")
            return list(price_sources.values())[0] if price_sources else 0.0, 50.0, issues
        
        prices = list(price_sources.values())
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        # Check for outliers
        max_deviation = max(abs(p - mean_price) / mean_price for p in prices)
        
        if max_deviation > self.MAX_PRICE_DEVIATION:
            issues.append(
                f"Price deviation {max_deviation*100:.2f}% exceeds threshold "
                f"{self.MAX_PRICE_DEVIATION*100:.2f}%"
            )
            consistency = max(0, 100 - (max_deviation * 10000))  # Harsh penalty
        else:
            consistency = 100.0
        
        # Log discrepancies
        if len(issues) > 0:
            self.logger.warning(
                f"Price validation issues for {symbol}: {issues}\\n"
                f"Sources: {price_sources}"
            )
        
        return mean_price, consistency, issues
    
    def validate_options_data(
        self,
        options_chain: List[Dict[str, Any]],
        min_strikes: int = 5
    ) -> DataQualityReport:
        """
        Validate options chain data.
        
        Args:
            options_chain: List of option contracts
            min_strikes: Minimum number of strikes required
            
        Returns:
            DataQualityReport
        """
        issues = []
        warnings = []
        
        if len(options_chain) < min_strikes:
            issues.append(
                f"Insufficient options data: {len(options_chain)}/{min_strikes} strikes"
            )
        
        # Check for required fields
        required_fields = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for opt in options_chain:
            missing = [f for f in required_fields if f not in opt or opt[f] is None]
            if missing:
                warnings.append(f"Missing fields in option: {missing}")
        
        # Check bid-ask spread
        wide_spreads = 0
        for opt in options_chain:
            if 'bid' in opt and 'ask' in opt and opt['bid'] and opt['ask']:
                spread = (opt['ask'] - opt['bid']) / opt['bid'] if opt['bid'] > 0 else float('inf')
                if spread > 0.10:  # 10% spread is too wide
                    wide_spreads += 1
        
        if wide_spreads > len(options_chain) * 0.3:
            warnings.append(f"Many options have wide bid-ask spreads: {wide_spreads}")
        
        # Check implied volatility
        invalid_iv = sum(
            1 for opt in options_chain
            if 'impliedVolatility' in opt and (
                opt['impliedVolatility'] is None or
                opt['impliedVolatility'] <= 0 or
                opt['impliedVolatility'] > 5.0  # 500% IV is suspicious
            )
        )
        
        if invalid_iv > 0:
            warnings.append(f"Invalid IV values in {invalid_iv} options")
        
        completeness = max(0, 100 - (len(warnings) * 10))
        confidence = max(0, 100 - (len(issues) * 20) - (len(warnings) * 5))
        
        return DataQualityReport(
            source="options_chain",
            confidence_score=confidence,
            completeness=completeness,
            consistency=100.0,
            timeliness=100.0,  # Assume real-time options data
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def validate_technical_indicators(
        self,
        indicators: Dict[str, float],
        min_indicators: int = 50
    ) -> DataQualityReport:
        """
        Validate technical indicators.
        
        Args:
            indicators: Dict of indicator values
            min_indicators: Minimum number of indicators required
            
        Returns:
            DataQualityReport
        """
        issues = []
        warnings = []
        
        if len(indicators) < min_indicators:
            issues.append(
                f"Insufficient indicators: {len(indicators)}/{min_indicators}"
            )
        
        # Check for NaN values
        nan_count = sum(1 for v in indicators.values() if v is None or (isinstance(v, float) and np.isnan(v)))
        nan_percentage = (nan_count / len(indicators)) * 100 if indicators else 100
        
        if nan_percentage > self.MAX_NAN_PERCENTAGE:
            issues.append(f"Too many NaN indicators: {nan_percentage:.1f}%")
        
        # Check for infinite values
        inf_count = sum(1 for v in indicators.values() if isinstance(v, float) and np.isinf(v))
        if inf_count > 0:
            issues.append(f"Infinite values detected in {inf_count} indicators")
        
        completeness = max(0, 100 - nan_percentage)
        confidence = max(0, 100 - (len(issues) * 15) - (len(warnings) * 5))
        
        return DataQualityReport(
            source="technical_indicators",
            confidence_score=confidence,
            completeness=completeness,
            consistency=100.0,
            timeliness=100.0,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def validate_news_sentiment(
        self,
        sentiment_scores: List[float],
        min_articles: int = 5
    ) -> DataQualityReport:
        """
        Validate news sentiment data.
        
        Args:
            sentiment_scores: List of sentiment scores
            min_articles: Minimum number of articles required
            
        Returns:
            DataQualityReport
        """
        issues = []
        warnings = []
        
        if len(sentiment_scores) < min_articles:
            warnings.append(
                f"Limited news coverage: {len(sentiment_scores)} articles"
            )
        
        # Check sentiment range
        invalid_scores = [
            s for s in sentiment_scores
            if s < self.SENTIMENT_RANGE[0] or s > self.SENTIMENT_RANGE[1]
        ]
        
        if invalid_scores:
            issues.append(f"Invalid sentiment scores detected: {len(invalid_scores)}")
        
        completeness = min(100.0, (len(sentiment_scores) / min_articles) * 100)
        confidence = max(0, 100 - (len(issues) * 20) - (len(warnings) * 10))
        
        return DataQualityReport(
            source="news_sentiment",
            confidence_score=confidence,
            completeness=completeness,
            consistency=100.0,
            timeliness=100.0,
            issues=issues,
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _calculate_confidence(
        self,
        completeness: float,
        nan_percentage: float,
        invalid_count: int,
        timeliness: float,
        issues_count: int
    ) -> float:
        """Calculate overall confidence score"""
        # Start with completeness
        confidence = completeness
        
        # Penalize for NaN values
        confidence -= nan_percentage * 2
        
        # Penalize for invalid data
        confidence -= invalid_count * 5
        
        # Factor in timeliness
        confidence = (confidence + timeliness) / 2
        
        # Harsh penalty for critical issues
        confidence -= issues_count * 15
        
        return max(0.0, min(100.0, confidence))
    
    def aggregate_quality_reports(
        self,
        reports: List[DataQualityReport]
    ) -> DataQualityReport:
        """
        Aggregate multiple quality reports into overall assessment.
        
        Args:
            reports: List of DataQualityReport objects
            
        Returns:
            Aggregated DataQualityReport
        """
        if not reports:
            return DataQualityReport(
                source="aggregate",
                confidence_score=0.0,
                completeness=0.0,
                consistency=0.0,
                timeliness=0.0,
                issues=["No data reports available"],
                warnings=[],
                timestamp=datetime.now()
            )
        
        # Aggregate scores (weighted average)
        weights = [1.0] * len(reports)  # Equal weight for now
        total_weight = sum(weights)
        
        avg_confidence = sum(r.confidence_score * w for r, w in zip(reports, weights)) / total_weight
        avg_completeness = sum(r.completeness * w for r, w in zip(reports, weights)) / total_weight
        avg_consistency = sum(r.consistency * w for r, w in zip(reports, weights)) / total_weight
        avg_timeliness = sum(r.timeliness * w for r, w in zip(reports, weights)) / total_weight
        
        # Collect all issues and warnings
        all_issues = []
        all_warnings = []
        for r in reports:
            all_issues.extend([f"[{r.source}] {issue}" for issue in r.issues])
            all_warnings.extend([f"[{r.source}] {warning}" for warning in r.warnings])
        
        return DataQualityReport(
            source="aggregate",
            confidence_score=avg_confidence,
            completeness=avg_completeness,
            consistency=avg_consistency,
            timeliness=avg_timeliness,
            issues=all_issues,
            warnings=all_warnings,
            timestamp=datetime.now()
        )
    
    def check_circuit_breaker(
        self,
        quality_report: DataQualityReport,
        halt_on_failure: bool = True
    ) -> bool:
        """
        Check if circuit breaker should trip.
        
        Args:
            quality_report: Data quality report to check
            halt_on_failure: Whether to raise exception on failure
            
        Returns:
            True if data quality is acceptable, False otherwise
            
        Raises:
            ValueError: If halt_on_failure=True and quality is unacceptable
        """
        if not quality_report.is_acceptable():
            error_msg = f"""
{'='*60}
CIRCUIT BREAKER TRIPPED - DATA QUALITY UNACCEPTABLE
{'='*60}
{quality_report}
CRITICAL ISSUES:
{chr(10).join('  - ' + issue for issue in quality_report.issues)}

WARNINGS:
{chr(10).join('  - ' + warning for warning in quality_report.warnings)}

ANALYSIS HALTED - DO NOT TRADE WITH THIS DATA
Minimum confidence required: {self.MIN_CONFIDENCE}%
Actual confidence: {quality_report.confidence_score:.1f}%
{'='*60}
"""
            self.logger.error(error_msg)
            
            if halt_on_failure:
                raise ValueError(
                    f"Data quality below acceptable threshold: "
                    f"{quality_report.confidence_score:.1f}% < {self.MIN_CONFIDENCE}%"
                )
            
            return False
        
        return True


# Global circuit breaker instance
circuit_breaker = CircuitBreaker()
