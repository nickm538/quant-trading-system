"""
Production Validator & Error Handler
====================================

Comprehensive validation and error handling for real-money trading.
Ensures all inputs, calculations, and outputs are valid before execution.

Author: Institutional Trading System
Date: 2025-11-29
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Validates all inputs and outputs for production trading"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        Validate stock symbol format.
        
        Returns:
            (is_valid, error_message)
        """
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if not isinstance(symbol, str):
            return False, "Symbol must be a string"
        
        symbol = symbol.strip().upper()
        
        if len(symbol) < 1 or len(symbol) > 10:
            return False, "Symbol must be 1-10 characters"
        
        if not symbol.replace('.', '').replace('-', '').replace('^', '').isalnum():
            return False, "Symbol contains invalid characters"
        
        return True, ""
    
    @staticmethod
    def validate_price(price: float, field_name: str = "price") -> Tuple[bool, str]:
        """
        Validate price value.
        
        Returns:
            (is_valid, error_message)
        """
        if price is None:
            return False, f"{field_name} cannot be None"
        
        if not isinstance(price, (int, float)):
            return False, f"{field_name} must be a number"
        
        if np.isnan(price) or np.isinf(price):
            return False, f"{field_name} is NaN or Inf"
        
        if price <= 0:
            return False, f"{field_name} must be positive"
        
        if price > 1000000:
            return False, f"{field_name} is unreasonably high (>{price})"
        
        return True, ""
    
    @staticmethod
    def validate_percentage(pct: float, field_name: str = "percentage", 
                           min_val: float = -100, max_val: float = 1000) -> Tuple[bool, str]:
        """
        Validate percentage value.
        
        Returns:
            (is_valid, error_message)
        """
        if pct is None:
            return False, f"{field_name} cannot be None"
        
        if not isinstance(pct, (int, float)):
            return False, f"{field_name} must be a number"
        
        if np.isnan(pct) or np.isinf(pct):
            return False, f"{field_name} is NaN or Inf"
        
        if pct < min_val or pct > max_val:
            return False, f"{field_name} out of range [{min_val}, {max_val}]: {pct}"
        
        return True, ""
    
    @staticmethod
    def validate_confidence(confidence: float) -> Tuple[bool, str]:
        """
        Validate confidence score (0-100).
        
        Returns:
            (is_valid, error_message)
        """
        return ProductionValidator.validate_percentage(
            confidence, "confidence", min_val=0, max_val=100
        )
    
    @staticmethod
    def validate_analysis_result(analysis: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete analysis result before returning to user.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = [
            'symbol', 'current_price', 'signal', 'confidence',
            'target_price', 'stop_loss', 'position_size'
        ]
        
        for field in required_fields:
            if field not in analysis:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate symbol
        is_valid, error = ProductionValidator.validate_symbol(analysis['symbol'])
        if not is_valid:
            errors.append(f"Invalid symbol: {error}")
        
        # Validate prices
        for price_field in ['current_price', 'target_price', 'stop_loss']:
            if price_field in analysis:
                is_valid, error = ProductionValidator.validate_price(
                    analysis[price_field], price_field
                )
                if not is_valid:
                    errors.append(error)
        
        # Validate confidence
        if 'confidence' in analysis:
            is_valid, error = ProductionValidator.validate_confidence(analysis['confidence'])
            if not is_valid:
                errors.append(error)
        
        # Validate signal
        if 'signal' in analysis:
            if analysis['signal'] not in ['BUY', 'SELL', 'HOLD']:
                errors.append(f"Invalid signal: {analysis['signal']}")
        
        # Validate position size
        if 'position_size' in analysis:
            if analysis['position_size'] < 0:
                errors.append(f"Position size cannot be negative: {analysis['position_size']}")
        
        # Validate stop loss vs current price
        if 'stop_loss' in analysis and 'current_price' in analysis:
            if analysis['signal'] == 'BUY' and analysis['stop_loss'] >= analysis['current_price']:
                errors.append(f"Stop loss ({analysis['stop_loss']}) must be below current price ({analysis['current_price']}) for BUY signal")
            elif analysis['signal'] == 'SELL' and analysis['stop_loss'] <= analysis['current_price']:
                errors.append(f"Stop loss ({analysis['stop_loss']}) must be above current price ({analysis['current_price']}) for SELL signal")
        
        # Validate risk/reward ratio
        if 'risk_assessment' in analysis:
            risk = analysis['risk_assessment']
            if 'risk_reward_ratio' in risk:
                if risk['risk_reward_ratio'] < 0:
                    errors.append(f"Risk/reward ratio cannot be negative: {risk['risk_reward_ratio']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_analysis_result(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize analysis result by replacing NaN/Inf with None.
        
        Returns:
            Sanitized analysis dictionary
        """
        def sanitize_value(val):
            if isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(val, dict):
                return {k: sanitize_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [sanitize_value(v) for v in val]
            return val
        
        return {k: sanitize_value(v) for k, v in analysis.items()}


class CircuitBreaker:
    """Circuit breaker for API calls to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failures = {}
        self.last_failure_time = {}
    
    def record_failure(self, api_name: str):
        """Record an API failure"""
        if api_name not in self.failures:
            self.failures[api_name] = 0
            self.last_failure_time[api_name] = datetime.now()
        
        self.failures[api_name] += 1
        self.last_failure_time[api_name] = datetime.now()
        
        if self.failures[api_name] >= self.failure_threshold:
            logger.error(f"Circuit breaker OPEN for {api_name}: {self.failures[api_name]} failures")
    
    def record_success(self, api_name: str):
        """Record an API success"""
        if api_name in self.failures:
            self.failures[api_name] = 0
    
    def is_open(self, api_name: str) -> bool:
        """Check if circuit breaker is open (blocking calls)"""
        if api_name not in self.failures:
            return False
        
        # Check if timeout has passed
        if api_name in self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time[api_name]).seconds
            if time_since_failure > self.timeout_seconds:
                # Reset after timeout
                self.failures[api_name] = 0
                return False
        
        return self.failures[api_name] >= self.failure_threshold


# Global circuit breaker instance
circuit_breaker = CircuitBreaker()


if __name__ == '__main__':
    # Test validation
    validator = ProductionValidator()
    
    # Test symbol validation
    print("Testing symbol validation:")
    test_symbols = ['AAPL', '', 'INVALID!@#', 'A'*20, '^TNX', 'BRK.B']
    for symbol in test_symbols:
        is_valid, error = validator.validate_symbol(symbol)
        print(f"  {symbol}: {'✓' if is_valid else '✗'} {error}")
    
    # Test price validation
    print("\nTesting price validation:")
    test_prices = [100.5, 0, -10, float('nan'), float('inf'), 1000001]
    for price in test_prices:
        is_valid, error = validator.validate_price(price)
        print(f"  {price}: {'✓' if is_valid else '✗'} {error}")
    
    # Test confidence validation
    print("\nTesting confidence validation:")
    test_confidences = [50, 0, 100, -10, 150, 17.8]
    for conf in test_confidences:
        is_valid, error = validator.validate_confidence(conf)
        print(f"  {conf}: {'✓' if is_valid else '✗'} {error}")
    
    print("\n✅ Validation tests complete")
