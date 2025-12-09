"""
Prediction Validation and Continuous Learning System
=====================================================
Validates past predictions against actual outcomes and triggers retraining
Runs daily to maintain model accuracy and adapt to market changes
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_connection():
    """Get database connection"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    # Parse DATABASE_URL (format: mysql://user:password@host:port/database)
    import re
    match = re.match(r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
    if not match:
        raise ValueError(f"Invalid DATABASE_URL format: {database_url}")
    
    user, password, host, port, database = match.groups()
    
    return mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database,
        ssl_disabled=False
    )

def fetch_actual_price(symbol: str, target_date: datetime) -> dict:
    """
    Fetch actual price data (close, high, low) for a given date
    
    Args:
        symbol: Stock symbol
        target_date: Date to fetch price for
    
    Returns:
        Dict with 'close', 'high', 'low' prices, or None if not available
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch data around target date (±5 days to handle weekends/holidays)
        start_date = target_date - timedelta(days=5)
        end_date = target_date + timedelta(days=5)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No price data found for {symbol} around {target_date}")
            return None
        
        # Find closest date to target (prefer exact match or next trading day)
        hist.index = pd.to_datetime(hist.index)
        
        # Normalize target date and make it timezone-aware if needed
        target_date_normalized = pd.Timestamp(target_date.date())
        if hist.index.tz is not None:
            # Make target_date timezone-aware to match hist.index
            target_date_normalized = target_date_normalized.tz_localize(hist.index.tz)
        
        # Try exact date first (compare dates only, ignore time)
        hist_dates = hist.index.normalize()  # Remove time component
        target_date_only = target_date_normalized.normalize()
        
        if target_date_only in hist_dates:
            row = hist.loc[hist_dates == target_date_only].iloc[0]
        else:
            # Find closest trading day (prefer future dates for forward-looking predictions)
            future_dates = hist_dates[hist_dates >= target_date_only]
            if len(future_dates) > 0:
                row = hist.loc[hist_dates == future_dates[0]].iloc[0]
            else:
                # Fall back to closest date
                closest_idx = (hist_dates - target_date_only).abs().argmin()
                row = hist.iloc[closest_idx]
        
        actual_close = float(row['Close'])
        actual_high = float(row['High'])
        actual_low = float(row['Low'])
        actual_date = row.name.date()
        
        logger.info(f"Fetched actual prices for {symbol} on {actual_date}: Close=${actual_close:.2f}, High=${actual_high:.2f}, Low=${actual_low:.2f}")
        
        return {
            'close': actual_close,
            'high': actual_high,
            'low': actual_low,
            'date': actual_date
        }
        
    except Exception as e:
        logger.error(f"Error fetching actual price for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_pending_predictions(conn) -> Dict[str, any]:
    """
    Validate all pending predictions whose target date has passed
    
    Returns:
        Statistics about validation results
    """
    cursor = conn.cursor(dictionary=True)
    
    # Find pending predictions where target_date <= today
    query = """
    SELECT id, stock_symbol, prediction_date, target_date,
           predicted_price, predicted_low, predicted_high, confidence
    FROM model_predictions
    WHERE status = 'pending' AND target_date <= NOW()
    ORDER BY target_date DESC
    LIMIT 100
    """
    
    cursor.execute(query)
    pending_predictions = cursor.fetchall()
    
    logger.info(f"Found {len(pending_predictions)} pending predictions to validate")
    
    validated_count = 0
    failed_count = 0
    total_error = 0
    total_pct_error = 0
    
    for pred in pending_predictions:
        pred_id = pred['id']
        symbol = pred['stock_symbol']
        target_date = pred['target_date']
        predicted_price = pred['predicted_price'] / 100.0  # Convert from cents
        
        # Fetch actual price data (close, high, low)
        price_data = fetch_actual_price(symbol, target_date)
        
        if price_data is None:
            failed_count += 1
            # Mark as failed
            update_query = """
            UPDATE model_predictions
            SET status = 'failed',
                updated_at = NOW()
            WHERE id = %s
            """
            cursor.execute(update_query, (pred_id,))
            logger.warning(f"Failed to fetch price for {symbol} on {target_date.date()}")
            continue
        
        actual_close = price_data['close']
        actual_high = price_data['high']
        actual_low = price_data['low']
        
        # Calculate errors based on closing price
        price_error = abs(actual_close - predicted_price)
        percentage_error = abs((actual_close - predicted_price) / actual_close * 100)
        
        total_error += price_error
        total_pct_error += percentage_error
        
        # Update prediction with actual values
        update_query = """
        UPDATE model_predictions
        SET actual_price = %s,
            actual_low = %s,
            actual_high = %s,
            price_error = %s,
            percentage_error = %s,
            status = 'validated',
            updated_at = NOW()
        WHERE id = %s
        """
        
        values = (
            int(actual_close * 100),  # Convert to cents
            int(actual_low * 100),    # Actual low from yfinance
            int(actual_high * 100),   # Actual high from yfinance
            int(price_error * 100),   # Price error in cents
            int(percentage_error * 100),  # Percentage error in basis points
            pred_id
        )
        
        cursor.execute(update_query, values)
        validated_count += 1
        
        logger.info(f"Validated {symbol}: Predicted ${predicted_price:.2f}, Actual ${actual_close:.2f} (H: ${actual_high:.2f}, L: ${actual_low:.2f}), Error {percentage_error:.2f}%")
    
    conn.commit()
    cursor.close()
    
    avg_error = total_pct_error / validated_count if validated_count > 0 else 0
    
    stats = {
        'validated_count': validated_count,
        'failed_count': failed_count,
        'avg_percentage_error': avg_error,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Validation complete: {validated_count} validated, {failed_count} failed, avg error {avg_error:.2f}%")
    
    return stats

def get_model_performance_metrics(conn, symbol: str = None, days_back: int = 30) -> Dict[str, any]:
    """
    Calculate model performance metrics from validated predictions
    
    Args:
        conn: Database connection
        symbol: Optional stock symbol to filter by
        days_back: Number of days to look back
    
    Returns:
        Performance metrics dict
    """
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        stock_symbol,
        predicted_price,
        actual_price,
        percentage_error,
        confidence,
        prediction_date,
        target_date
    FROM model_predictions
    WHERE status = 'validated'
    AND target_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    
    params = [days_back]
    if symbol:
        query += " AND stock_symbol = %s"
        params.append(symbol)
    
    cursor.execute(query, tuple(params))
    predictions = cursor.fetchall()
    
    if not predictions:
        return {
            'total_predictions': 0,
            'avg_error': 0,
            'direction_accuracy': 0,
            'rmse': 0
        }
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(predictions)
    df['predicted_price'] = df['predicted_price'] / 100.0
    df['actual_price'] = df['actual_price'] / 100.0
    df['percentage_error'] = df['percentage_error'] / 100.0
    
    # Calculate metrics
    total_predictions = len(df)
    avg_error = df['percentage_error'].mean()
    
    # Direction accuracy (did we predict up/down correctly?)
    # We need the price at prediction_date for this
    # For now, use a simplified version
    direction_correct = 0
    
    # RMSE
    rmse = np.sqrt(((df['predicted_price'] - df['actual_price']) ** 2).mean())
    
    metrics = {
        'total_predictions': total_predictions,
        'avg_error': float(avg_error),
        'direction_accuracy': 0,  # TODO: Calculate properly
        'rmse': float(rmse),
        'symbol': symbol,
        'days_back': days_back
    }
    
    cursor.close()
    return metrics

def should_trigger_retraining(conn, symbol: str) -> Tuple[bool, str]:
    """
    Determine if models should be retrained based on performance degradation
    
    Args:
        conn: Database connection
        symbol: Stock symbol to check
    
    Returns:
        (should_retrain, reason)
    """
    # Get recent performance (last 30 days)
    recent_metrics = get_model_performance_metrics(conn, symbol, days_back=30)
    
    # Get older performance (30-60 days ago) for comparison
    # TODO: Implement historical comparison
    
    # Retraining triggers:
    # 1. Average error > 10%
    if recent_metrics['avg_error'] > 10.0:
        return True, f"High error rate: {recent_metrics['avg_error']:.2f}%"
    
    # 2. Not enough recent predictions (model not being used)
    if recent_metrics['total_predictions'] < 5:
        return False, "Not enough recent predictions to evaluate"
    
    # 3. RMSE too high
    if recent_metrics['rmse'] > 5.0:
        return True, f"High RMSE: {recent_metrics['rmse']:.2f}"
    
    return False, "Performance acceptable"

def trigger_model_retraining(symbol: str):
    """
    Trigger model retraining for a specific stock
    
    Args:
        symbol: Stock symbol to retrain
    """
    logger.info(f"Triggering retraining for {symbol}")
    
    # Import training script
    from ml.backtest_and_train import main as train_main
    
    # TODO: Implement selective retraining for single stock
    # For now, log the request
    logger.info(f"Retraining request logged for {symbol}")

def main():
    """Main validation and continuous learning loop"""
    logger.info("=" * 60)
    logger.info("Starting Prediction Validation and Continuous Learning")
    logger.info("=" * 60)
    
    try:
        conn = get_db_connection()
        logger.info("✅ Connected to database")
        
        # Step 1: Validate pending predictions
        logger.info("\n📊 Step 1: Validating pending predictions...")
        validation_stats = validate_pending_predictions(conn)
        
        logger.info(f"\n✅ Validation complete:")
        logger.info(f"   - Validated: {validation_stats['validated_count']}")
        logger.info(f"   - Failed: {validation_stats['failed_count']}")
        logger.info(f"   - Avg Error: {validation_stats['avg_percentage_error']:.2f}%")
        
        # Step 2: Check model performance and trigger retraining if needed
        logger.info("\n🔍 Step 2: Checking model performance...")
        
        # Get list of stocks with predictions
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT stock_symbol 
            FROM model_predictions 
            WHERE status = 'validated'
            AND target_date >= DATE_SUB(NOW(), INTERVAL 60 DAY)
        """)
        stocks = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        logger.info(f"Found {len(stocks)} stocks with recent predictions")
        
        retrain_needed = []
        for symbol in stocks:
            should_retrain, reason = should_trigger_retraining(conn, symbol)
            if should_retrain:
                logger.warning(f"⚠️  {symbol}: {reason}")
                retrain_needed.append(symbol)
            else:
                logger.info(f"✅ {symbol}: {reason}")
        
        if retrain_needed:
            logger.info(f"\n🔄 Retraining needed for {len(retrain_needed)} stocks: {', '.join(retrain_needed)}")
            # TODO: Trigger actual retraining
        else:
            logger.info("\n✅ All models performing within acceptable ranges")
        
        conn.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("Validation and continuous learning complete")
        logger.info("=" * 60)
        
        return {
            'success': True,
            'validation_stats': validation_stats,
            'retrain_needed': retrain_needed
        }
        
    except Exception as e:
        logger.error(f"Error in validation system: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2))
