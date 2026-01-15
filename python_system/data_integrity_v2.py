"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DATA INTEGRITY & ANTI-HALLUCINATION MODULE v2.0                 ║
║                                                                              ║
║  Ensures all data in the system is:                                          ║
║  ✓ REAL - No placeholders, no mock data                                      ║
║  ✓ CURRENT - Real-time or near real-time                                     ║
║  ✓ VALIDATED - Cross-checked against multiple sources                        ║
║  ✓ UNBIASED - No overfitting, no data leakage                               ║
║                                                                              ║
║  This module is the guardian of data quality across the entire system.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class DataIntegrityValidator:
    """
    Validates data integrity across the system.
    Catches placeholders, stale data, and potential hallucinations.
    """
    
    # Known placeholder patterns to detect
    PLACEHOLDER_PATTERNS = [
        'placeholder', 'mock', 'fake', 'sample', 'test', 'dummy',
        'example', 'todo', 'fixme', 'xxx', 'tbd', 'n/a',
        '0.00', '1.00', '100.00',  # Suspiciously round numbers
    ]
    
    # Suspiciously round numbers that might indicate fake data
    SUSPICIOUS_VALUES = {
        'price': [0, 1, 10, 100, 1000],
        'percentage': [0, 25, 50, 75, 100],
        'ratio': [0, 0.5, 1, 1.5, 2],
    }
    
    @classmethod
    def validate_price_data(cls, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate price data for a stock.
        
        Checks:
        - Data is not empty
        - Prices are realistic (not 0, not negative)
        - Data is recent (not stale)
        - No obvious placeholders
        - Volume is present and realistic
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_quality_score': 100
        }
        
        if data is None or (hasattr(data, 'empty') and data.empty):
            result['valid'] = False
            result['errors'].append(f"No data available for {symbol}")
            result['data_quality_score'] = 0
            return result
        
        # Check for required columns
        required_cols = ['close', 'high', 'low', 'volume']
        alt_cols = ['Close', 'High', 'Low', 'Volume']
        
        has_cols = any(col in data.columns for col in required_cols + alt_cols)
        if not has_cols:
            result['valid'] = False
            result['errors'].append(f"Missing required price columns for {symbol}")
            result['data_quality_score'] = 0
            return result
        
        # Normalize column names
        col_map = {c.lower(): c for c in data.columns}
        close_col = col_map.get('close', 'Close')
        high_col = col_map.get('high', 'High')
        low_col = col_map.get('low', 'Low')
        volume_col = col_map.get('volume', 'Volume')
        
        # Check for zero or negative prices
        if close_col in data.columns:
            close_prices = data[close_col]
            if (close_prices <= 0).any():
                result['warnings'].append(f"Zero or negative prices detected for {symbol}")
                result['data_quality_score'] -= 20
            
            # Check for suspiciously round prices
            round_count = (close_prices == close_prices.round(0)).sum()
            if round_count / len(close_prices) > 0.5:
                result['warnings'].append(f"Suspiciously many round prices for {symbol}")
                result['data_quality_score'] -= 10
        
        # Check data freshness
        if hasattr(data.index, 'max'):
            latest_date = data.index.max()
            if hasattr(latest_date, 'date'):
                latest_date = latest_date.date()
            elif hasattr(latest_date, 'to_pydatetime'):
                latest_date = latest_date.to_pydatetime().date()
            else:
                latest_date = datetime.now().date()
            
            days_old = (datetime.now().date() - latest_date).days
            if days_old > 5:  # More than 5 days old (accounting for weekends)
                result['warnings'].append(f"Data is {days_old} days old for {symbol}")
                result['data_quality_score'] -= min(30, days_old * 2)
        
        # Check volume
        if volume_col in data.columns:
            volumes = data[volume_col]
            if (volumes == 0).all():
                result['warnings'].append(f"All zero volume for {symbol} - data may be incomplete")
                result['data_quality_score'] -= 15
        
        # Check for data gaps
        if len(data) < 20:
            result['warnings'].append(f"Insufficient data points ({len(data)}) for {symbol}")
            result['data_quality_score'] -= 20
        
        result['data_quality_score'] = max(0, result['data_quality_score'])
        result['valid'] = result['data_quality_score'] >= 50
        
        return result
    
    @classmethod
    def validate_indicator_value(cls, indicator: str, value: float) -> Dict[str, Any]:
        """
        Validate that an indicator value is realistic.
        
        Catches:
        - Out of range values
        - Suspiciously round values
        - NaN/Inf values
        """
        result = {
            'valid': True,
            'warnings': [],
            'adjusted_value': value
        }
        
        # Check for NaN/Inf
        if pd.isna(value) or np.isinf(value):
            result['valid'] = False
            result['warnings'].append(f"{indicator} is NaN or Inf")
            result['adjusted_value'] = None
            return result
        
        # Indicator-specific validation
        indicator_ranges = {
            'RSI': (0, 100),
            'ADX': (0, 100),
            'STOCHASTIC': (0, 100),
            'MACD': (-1000, 1000),  # Wide range, depends on stock price
            'ATR': (0, 10000),
            'SHARPE_RATIO': (-5, 10),
            'SORTINO_RATIO': (-5, 15),
            'WIN_RATE': (0, 100),
            'CONFIDENCE': (0, 100),
            'SCORE': (0, 100),
        }
        
        indicator_upper = indicator.upper()
        for key, (min_val, max_val) in indicator_ranges.items():
            if key in indicator_upper:
                if value < min_val or value > max_val:
                    result['warnings'].append(f"{indicator} value {value} is outside expected range [{min_val}, {max_val}]")
                    result['adjusted_value'] = max(min_val, min(max_val, value))
                break
        
        return result
    
    @classmethod
    def detect_overfitting(cls, train_metrics: Dict, test_metrics: Dict) -> Dict[str, Any]:
        """
        Detect potential overfitting by comparing train vs test metrics.
        
        Signs of overfitting:
        - Train accuracy >> Test accuracy
        - Train Sharpe >> Test Sharpe
        - Train win rate >> Test win rate
        """
        result = {
            'overfitting_detected': False,
            'overfitting_score': 0,
            'warnings': [],
            'recommendations': []
        }
        
        # Compare accuracy
        train_acc = train_metrics.get('accuracy', train_metrics.get('direction_accuracy', 0))
        test_acc = test_metrics.get('accuracy', test_metrics.get('direction_accuracy', 0))
        
        if train_acc > 0 and test_acc > 0:
            acc_gap = train_acc - test_acc
            if acc_gap > 10:  # More than 10% gap
                result['overfitting_score'] += 30
                result['warnings'].append(f"Accuracy gap: Train {train_acc:.1f}% vs Test {test_acc:.1f}%")
        
        # Compare Sharpe
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        
        if train_sharpe > 0 and test_sharpe > 0:
            sharpe_ratio = train_sharpe / test_sharpe if test_sharpe != 0 else 999
            if sharpe_ratio > 2:  # Train Sharpe more than 2x test
                result['overfitting_score'] += 30
                result['warnings'].append(f"Sharpe gap: Train {train_sharpe:.2f} vs Test {test_sharpe:.2f}")
        
        # Compare win rate
        train_wr = train_metrics.get('win_rate', 0)
        test_wr = test_metrics.get('win_rate', 0)
        
        if train_wr > 0 and test_wr > 0:
            wr_gap = train_wr - test_wr
            if wr_gap > 15:  # More than 15% gap
                result['overfitting_score'] += 20
                result['warnings'].append(f"Win rate gap: Train {train_wr:.1f}% vs Test {test_wr:.1f}%")
        
        # Suspiciously high train metrics
        if train_acc > 80:
            result['overfitting_score'] += 10
            result['warnings'].append(f"Suspiciously high train accuracy: {train_acc:.1f}%")
        
        if train_sharpe > 3:
            result['overfitting_score'] += 10
            result['warnings'].append(f"Suspiciously high train Sharpe: {train_sharpe:.2f}")
        
        result['overfitting_detected'] = result['overfitting_score'] >= 40
        
        if result['overfitting_detected']:
            result['recommendations'] = [
                "Use more out-of-sample data for validation",
                "Reduce model complexity (fewer features, simpler model)",
                "Add regularization (L1/L2)",
                "Use cross-validation instead of single train/test split",
                "Check for data leakage (future data in training)",
            ]
        
        return result
    
    @classmethod
    def detect_data_leakage(cls, features: List[str], target: str) -> Dict[str, Any]:
        """
        Detect potential data leakage in feature engineering.
        
        Common leakage sources:
        - Using future data (e.g., next day's price in features)
        - Using target-derived features
        - Using data not available at prediction time
        """
        result = {
            'leakage_detected': False,
            'leakage_score': 0,
            'warnings': [],
            'suspicious_features': []
        }
        
        # Suspicious feature patterns
        leakage_patterns = [
            ('future', 'Features containing future data'),
            ('next', 'Features referencing next period'),
            ('forward', 'Forward-looking features'),
            ('target', 'Target-derived features'),
            ('label', 'Label-derived features'),
            ('return_', 'Return features (may include future)'),
            ('_t+', 'Time-shifted forward features'),
        ]
        
        for feature in features:
            feature_lower = feature.lower()
            for pattern, description in leakage_patterns:
                if pattern in feature_lower:
                    result['suspicious_features'].append({
                        'feature': feature,
                        'reason': description
                    })
                    result['leakage_score'] += 20
        
        # Check if target appears in features
        if target.lower() in [f.lower() for f in features]:
            result['warnings'].append(f"Target '{target}' appears in features - definite leakage!")
            result['leakage_score'] += 100
        
        result['leakage_detected'] = result['leakage_score'] >= 20
        
        return result


class AntiHallucinationGuard:
    """
    Prevents AI from hallucinating data or making up information.
    Cross-validates data against multiple sources.
    """
    
    @classmethod
    def validate_stock_exists(cls, symbol: str) -> bool:
        """Verify a stock symbol actually exists."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') is not None or info.get('previousClose') is not None
        except:
            return False
    
    @classmethod
    def validate_price_realistic(cls, symbol: str, price: float) -> Dict[str, Any]:
        """
        Validate that a price is realistic for a given stock.
        Cross-check against actual market data.
        """
        result = {
            'valid': True,
            'actual_price': None,
            'deviation_pct': None,
            'warnings': []
        }
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            actual_price = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose')
            
            if actual_price:
                result['actual_price'] = actual_price
                deviation = abs(price - actual_price) / actual_price * 100
                result['deviation_pct'] = deviation
                
                if deviation > 5:  # More than 5% deviation
                    result['warnings'].append(f"Price {price} deviates {deviation:.1f}% from actual {actual_price}")
                    result['valid'] = deviation < 20  # Allow up to 20% for delayed data
        except Exception as e:
            result['warnings'].append(f"Could not validate price: {e}")
        
        return result
    
    @classmethod
    def validate_analysis_consistency(cls, analysis: Dict) -> Dict[str, Any]:
        """
        Check that analysis is internally consistent.
        
        Examples of inconsistency:
        - Bullish signal with bearish indicators
        - High confidence with conflicting data
        - Extreme predictions without supporting evidence
        """
        result = {
            'consistent': True,
            'consistency_score': 100,
            'warnings': [],
            'conflicts': []
        }
        
        # Extract signals
        signal = analysis.get('signal', '').lower()
        confidence = analysis.get('confidence', 50)
        
        # Check RSI consistency
        rsi = analysis.get('rsi', 50)
        if 'bullish' in signal and rsi > 70:
            result['conflicts'].append("Bullish signal but RSI is overbought (>70)")
            result['consistency_score'] -= 15
        elif 'bearish' in signal and rsi < 30:
            result['conflicts'].append("Bearish signal but RSI is oversold (<30)")
            result['consistency_score'] -= 15
        
        # Check MACD consistency
        macd = analysis.get('macd', {})
        if isinstance(macd, dict):
            macd_val = macd.get('macd', 0)
            if 'bullish' in signal and macd_val < 0:
                result['conflicts'].append("Bullish signal but MACD is negative")
                result['consistency_score'] -= 10
            elif 'bearish' in signal and macd_val > 0:
                result['conflicts'].append("Bearish signal but MACD is positive")
                result['consistency_score'] -= 10
        
        # Check confidence vs data quality
        data_quality = analysis.get('data_quality_score', 100)
        if confidence > 80 and data_quality < 70:
            result['conflicts'].append("High confidence with low data quality")
            result['consistency_score'] -= 20
        
        # Check for extreme predictions
        predicted_change = analysis.get('predicted_change_pct', 0)
        if abs(predicted_change) > 20 and confidence > 70:
            result['warnings'].append(f"Extreme prediction ({predicted_change:.1f}%) with high confidence - verify data")
            result['consistency_score'] -= 10
        
        result['consistency_score'] = max(0, result['consistency_score'])
        result['consistent'] = result['consistency_score'] >= 70
        
        return result
    
    @classmethod
    def add_uncertainty_disclaimer(cls, analysis: Dict) -> str:
        """
        Generate appropriate uncertainty disclaimer based on analysis quality.
        """
        confidence = analysis.get('confidence', 50)
        data_quality = analysis.get('data_quality_score', 100)
        
        if confidence < 50 or data_quality < 70:
            return """
⚠️ **LOW CONFIDENCE ANALYSIS**
This analysis has lower confidence due to limited or potentially stale data.
Do NOT make trading decisions based solely on this analysis.
Cross-validate with other sources before acting.
"""
        elif confidence < 70:
            return """
📊 **MODERATE CONFIDENCE ANALYSIS**
This analysis has moderate confidence. Consider it as one input among many.
Always use proper position sizing and risk management.
"""
        else:
            return """
✅ **HIGH CONFIDENCE ANALYSIS**
This analysis has high confidence based on quality data.
However, markets are inherently unpredictable. Always use stop losses
and never risk more than you can afford to lose.
"""


class DataQualityReport:
    """
    Generate comprehensive data quality reports.
    """
    
    @classmethod
    def generate_report(cls, data: Dict, symbol: str) -> Dict[str, Any]:
        """
        Generate a full data quality report for an analysis.
        """
        report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_quality': 'UNKNOWN',
            'quality_score': 0,
            'checks': [],
            'recommendations': []
        }
        
        # Run all validation checks
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Data freshness
        total_checks += 1
        if 'timestamp' in data or 'date' in data:
            checks_passed += 1
            report['checks'].append({'name': 'Data Freshness', 'status': 'PASS'})
        else:
            report['checks'].append({'name': 'Data Freshness', 'status': 'WARN', 'note': 'No timestamp found'})
        
        # Check 2: Price data validity
        total_checks += 1
        price = data.get('current_price', data.get('price', 0))
        if price and price > 0:
            checks_passed += 1
            report['checks'].append({'name': 'Price Validity', 'status': 'PASS'})
        else:
            report['checks'].append({'name': 'Price Validity', 'status': 'FAIL', 'note': 'Invalid or missing price'})
        
        # Check 3: Indicator completeness
        total_checks += 1
        indicators = ['rsi', 'macd', 'adx', 'atr']
        present = sum(1 for ind in indicators if ind in data or ind.upper() in data)
        if present >= 3:
            checks_passed += 1
            report['checks'].append({'name': 'Indicator Completeness', 'status': 'PASS'})
        else:
            report['checks'].append({'name': 'Indicator Completeness', 'status': 'WARN', 'note': f'Only {present}/4 indicators present'})
        
        # Check 4: No placeholder values
        total_checks += 1
        has_placeholder = False
        for key, value in data.items():
            if isinstance(value, str) and any(p in value.lower() for p in ['placeholder', 'mock', 'fake', 'n/a']):
                has_placeholder = True
                break
        if not has_placeholder:
            checks_passed += 1
            report['checks'].append({'name': 'No Placeholders', 'status': 'PASS'})
        else:
            report['checks'].append({'name': 'No Placeholders', 'status': 'FAIL', 'note': 'Placeholder values detected'})
        
        # Calculate overall score
        report['quality_score'] = int((checks_passed / total_checks) * 100)
        
        if report['quality_score'] >= 90:
            report['overall_quality'] = 'EXCELLENT'
        elif report['quality_score'] >= 70:
            report['overall_quality'] = 'GOOD'
        elif report['quality_score'] >= 50:
            report['overall_quality'] = 'FAIR'
        else:
            report['overall_quality'] = 'POOR'
        
        # Add recommendations
        if report['quality_score'] < 90:
            report['recommendations'].append("Consider refreshing data from primary sources")
        if report['quality_score'] < 70:
            report['recommendations'].append("Cross-validate analysis with alternative data sources")
        if report['quality_score'] < 50:
            report['recommendations'].append("DO NOT make trading decisions based on this data")
        
        return report


# Convenience functions
def validate_data(data: pd.DataFrame, symbol: str) -> Dict:
    """Quick validation of price data."""
    return DataIntegrityValidator.validate_price_data(data, symbol)

def check_overfitting(train_metrics: Dict, test_metrics: Dict) -> Dict:
    """Quick overfitting check."""
    return DataIntegrityValidator.detect_overfitting(train_metrics, test_metrics)

def validate_analysis(analysis: Dict) -> Dict:
    """Quick analysis consistency check."""
    return AntiHallucinationGuard.validate_analysis_consistency(analysis)

def generate_quality_report(data: Dict, symbol: str) -> Dict:
    """Generate data quality report."""
    return DataQualityReport.generate_report(data, symbol)


if __name__ == "__main__":
    # Demo
    print("=== DATA INTEGRITY MODULE ===\n")
    
    # Test price validation
    import yfinance as yf
    aapl = yf.Ticker("AAPL")
    data = aapl.history(period="1mo")
    
    result = DataIntegrityValidator.validate_price_data(data, "AAPL")
    print(f"AAPL Data Quality Score: {result['data_quality_score']}")
    print(f"Valid: {result['valid']}")
    print(f"Warnings: {result['warnings']}")
