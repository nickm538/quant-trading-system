"""
Options Database Saver
======================
Saves institutional options analysis results to MySQL database.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)

class OptionsDBSaver:
    """
    Saves options analysis results to MySQL database.
    """
    
    ENGINE_VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize database connection from environment variables."""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'quant_trading'),
        }
        self.connection = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
        return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def save_analysis_scan(
        self,
        symbol: str,
        scan_stats: Dict[str, Any],
        current_price: float,
        scan_duration_ms: int
    ) -> int:
        """
        Save scan summary to options_analysis_scan table.
        
        Returns:
            Scan ID if successful, None otherwise
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            
            # Convert prices to cents (multiply by 100)
            stock_price_cents = int(current_price * 100)
            
            # Extract top scores (already multiplied by 100 in the data)
            top_call_score = None
            top_put_score = None
            
            if scan_stats.get('calls_above_threshold', 0) > 0:
                top_call_score = int(scan_stats.get('top_call_score', 0) * 100)
            
            if scan_stats.get('puts_above_threshold', 0) > 0:
                top_put_score = int(scan_stats.get('top_put_score', 0) * 100)
            
            query = """
                INSERT INTO options_analysis_scan (
                    stock_symbol,
                    total_options_analyzed,
                    calls_analyzed,
                    puts_analyzed,
                    calls_passed_filters,
                    puts_passed_filters,
                    calls_above_threshold,
                    puts_above_threshold,
                    top_call_score,
                    top_put_score,
                    stock_price,
                    market_volatility,
                    scan_timestamp,
                    scan_duration_ms,
                    engine_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            values = (
                symbol,
                scan_stats.get('total_calls_analyzed', 0) + scan_stats.get('total_puts_analyzed', 0),
                scan_stats.get('total_calls_analyzed', 0),
                scan_stats.get('total_puts_analyzed', 0),
                scan_stats.get('calls_passed_filters', 0),
                scan_stats.get('puts_passed_filters', 0),
                scan_stats.get('calls_above_threshold', 0),
                scan_stats.get('puts_above_threshold', 0),
                top_call_score,
                top_put_score,
                stock_price_cents,
                None,  # market_volatility - can be added later
                datetime.now(),
                scan_duration_ms,
                self.ENGINE_VERSION
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            
            scan_id = cursor.lastrowid
            logger.info(f"Saved scan summary for {symbol}, scan_id={scan_id}")
            
            cursor.close()
            return scan_id
            
        except Error as e:
            logger.error(f"Error saving scan summary: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def save_option_analysis(
        self,
        option: Dict[str, Any],
        current_price: float,
        historical_vol: float = None,
        days_to_earnings: int = None
    ) -> int:
        """
        Save individual option analysis to institutional_options_analysis table.
        
        Args:
            option: Option analysis dict from institutional engine
            current_price: Current stock price
            historical_vol: Historical volatility (optional)
            days_to_earnings: Days until earnings (optional)
        
        Returns:
            Analysis ID if successful, None otherwise
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            
            # Convert all prices to cents (multiply by 100)
            strike_cents = int(option['strike'] * 100)
            last_price_cents = int(option['last_price'] * 100)
            bid_cents = int(option['bid'] * 100)
            ask_cents = int(option['ask'] * 100)
            mid_price_cents = int(option['mid_price'] * 100)
            current_stock_price_cents = int(current_price * 100)
            
            # Convert scores (multiply by 100 for final score, by 10 for category scores)
            final_score = int(option['final_score'] * 100)
            
            scores = option['scores']
            volatility_score = int(scores['volatility'] * 10)
            greeks_score = int(scores['greeks'] * 10)
            technical_score = int(scores['technical'] * 10)
            liquidity_score = int(scores['liquidity'] * 10)
            event_risk_score = int(scores['event_risk'] * 10)
            sentiment_score = int(scores['sentiment'] * 10)
            flow_score = int(scores['flow'] * 10)
            expected_value_score = int(scores['expected_value'] * 10)
            
            # Convert Greeks (multiply by 10000 for precision)
            metrics = option['key_metrics']
            delta = int(metrics['delta'] * 10000)
            gamma = int(metrics['gamma'] * 10000)
            vega = int(metrics['vega'] * 10000)
            theta = int(metrics['theta'] * 10000)
            
            # IV is already percentage (22.29), store as int (2229)
            implied_volatility = int(metrics['iv'] * 100)
            
            # Spread is already percentage, store as int
            spread_pct = int(metrics['spread_pct'] * 100)
            
            # Risk management (Kelly already as decimal, multiply by 10000)
            risk_mgmt = option['risk_management']
            kelly_pct = int(risk_mgmt['kelly_pct'] * 10000)
            conservative_kelly = int(risk_mgmt['conservative_kelly'] * 10000)
            max_position_size_pct = int(risk_mgmt['max_position_size_pct'] * 10000)
            
            # Historical volatility (if provided)
            historical_volatility = None
            if historical_vol is not None:
                historical_volatility = int(historical_vol * 10000)
            
            # Insights as JSON
            insights_json = json.dumps(option.get('insights', []))
            
            # Parse expiration date
            exp_date = datetime.strptime(option['expiration'], '%Y-%m-%d')
            
            query = """
                INSERT INTO institutional_options_analysis (
                    stock_symbol,
                    option_type,
                    strike_price,
                    expiration_date,
                    days_to_expiry,
                    last_price,
                    bid,
                    ask,
                    mid_price,
                    current_stock_price,
                    final_score,
                    rating,
                    volatility_score,
                    greeks_score,
                    technical_score,
                    liquidity_score,
                    event_risk_score,
                    sentiment_score,
                    flow_score,
                    expected_value_score,
                    delta,
                    gamma,
                    vega,
                    theta,
                    implied_volatility,
                    spread_pct,
                    volume,
                    open_interest,
                    kelly_pct,
                    conservative_kelly,
                    max_position_size_pct,
                    historical_volatility,
                    days_to_earnings,
                    insights,
                    analysis_timestamp,
                    engine_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
            """
            
            # Extract stock symbol from option data (should be available in context)
            # For now, we'll need to pass it separately or extract from option dict
            stock_symbol = option.get('symbol', 'UNKNOWN')
            
            values = (
                stock_symbol,
                option['option_type'].lower(),
                strike_cents,
                exp_date,
                option['dte'],
                last_price_cents,
                bid_cents,
                ask_cents,
                mid_price_cents,
                current_stock_price_cents,
                final_score,
                option['rating'],
                volatility_score,
                greeks_score,
                technical_score,
                liquidity_score,
                event_risk_score,
                sentiment_score,
                flow_score,
                expected_value_score,
                delta,
                gamma,
                vega,
                theta,
                implied_volatility,
                spread_pct,
                metrics['volume'],
                metrics['open_interest'],
                kelly_pct,
                conservative_kelly,
                max_position_size_pct,
                historical_volatility,
                days_to_earnings,
                insights_json,
                datetime.now(),
                self.ENGINE_VERSION
            )
            
            cursor.execute(query, values)
            self.connection.commit()
            
            analysis_id = cursor.lastrowid
            logger.info(f"Saved option analysis: {stock_symbol} ${option['strike']} {option['option_type']}, id={analysis_id}")
            
            cursor.close()
            return analysis_id
            
        except Error as e:
            logger.error(f"Error saving option analysis: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def save_full_analysis(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        scan_duration_ms: int
    ) -> Dict[str, Any]:
        """
        Save complete analysis results (scan summary + all options).
        
        Args:
            symbol: Stock symbol
            analysis_result: Full analysis result from institutional engine
            scan_duration_ms: Scan duration in milliseconds
        
        Returns:
            Dict with save statistics
        """
        if not analysis_result.get('success'):
            logger.warning(f"Skipping save for unsuccessful analysis of {symbol}")
            return {'success': False, 'reason': 'analysis_failed'}
        
        # Connect to database
        if not self.connect():
            return {'success': False, 'reason': 'db_connection_failed'}
        
        try:
            # Save scan summary
            scan_stats = {
                'total_calls_analyzed': analysis_result.get('total_calls_analyzed', 0),
                'total_puts_analyzed': analysis_result.get('total_puts_analyzed', 0),
                'calls_passed_filters': analysis_result.get('calls_passed_filters', 0),
                'puts_passed_filters': analysis_result.get('puts_passed_filters', 0),
                'calls_above_threshold': analysis_result.get('calls_above_threshold', 0),
                'puts_above_threshold': analysis_result.get('puts_above_threshold', 0),
            }
            
            # Get top scores
            top_calls = analysis_result.get('top_calls', [])
            top_puts = analysis_result.get('top_puts', [])
            
            if top_calls:
                scan_stats['top_call_score'] = top_calls[0]['final_score']
            if top_puts:
                scan_stats['top_put_score'] = top_puts[0]['final_score']
            
            scan_id = self.save_analysis_scan(
                symbol=symbol,
                scan_stats=scan_stats,
                current_price=analysis_result.get('current_price', 0),
                scan_duration_ms=scan_duration_ms
            )
            
            if not scan_id:
                return {'success': False, 'reason': 'scan_save_failed'}
            
            # Save individual options
            saved_calls = 0
            saved_puts = 0
            
            current_price = analysis_result.get('current_price', 0)
            historical_vol = analysis_result.get('market_context', {}).get('historical_volatility')
            days_to_earnings = analysis_result.get('market_context', {}).get('days_to_earnings')
            
            # Save top calls
            for option in top_calls:
                option['symbol'] = symbol  # Add symbol to option dict
                if self.save_option_analysis(option, current_price, historical_vol, days_to_earnings):
                    saved_calls += 1
            
            # Save top puts
            for option in top_puts:
                option['symbol'] = symbol  # Add symbol to option dict
                if self.save_option_analysis(option, current_price, historical_vol, days_to_earnings):
                    saved_puts += 1
            
            logger.info(f"Saved analysis for {symbol}: scan_id={scan_id}, calls={saved_calls}, puts={saved_puts}")
            
            return {
                'success': True,
                'scan_id': scan_id,
                'saved_calls': saved_calls,
                'saved_puts': saved_puts,
                'total_saved': saved_calls + saved_puts
            }
            
        except Exception as e:
            logger.error(f"Error saving full analysis: {e}")
            return {'success': False, 'reason': str(e)}
        
        finally:
            self.disconnect()
