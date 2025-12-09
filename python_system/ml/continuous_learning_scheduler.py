"""
Continuous Learning Automation Scheduler
=========================================
Automates the complete ML lifecycle:
1. Daily prediction validation
2. Performance monitoring
3. Automatic retraining triggers
4. Model versioning and rollback
5. A/B testing of new models
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_connection():
    """Get database connection"""
    import mysql.connector
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not set")
    
    import re
    match = re.match(r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
    if not match:
        raise ValueError(f"Invalid DATABASE_URL format")
    
    user, password, host, port, database = match.groups()
    
    return mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database,
        ssl_disabled=False
    )

def daily_validation_task():
    """
    Daily task: Validate pending predictions
    
    Runs every day at market close (4:30 PM ET)
    """
    logger.info("=" * 70)
    logger.info("DAILY VALIDATION TASK")
    logger.info("=" * 70)
    
    try:
        from ml.validate_and_retrain import validate_pending_predictions
        
        conn = get_db_connection()
        stats = validate_pending_predictions(conn)
        conn.close()
        
        logger.info(f"✅ Validated {stats['validated_count']} predictions")
        logger.info(f"   Average error: {stats['avg_percentage_error']:.2f}%")
        
        return {
            'success': True,
            'task': 'daily_validation',
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Daily validation failed: {e}")
        return {
            'success': False,
            'task': 'daily_validation',
            'error': str(e)
        }

def weekly_performance_review():
    """
    Weekly task: Review model performance and trigger retraining if needed
    
    Runs every Sunday at midnight
    """
    logger.info("=" * 70)
    logger.info("WEEKLY PERFORMANCE REVIEW")
    logger.info("=" * 70)
    
    try:
        from ml.validate_and_retrain import should_trigger_retraining
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all stocks with models
        cursor.execute("""
            SELECT DISTINCT stock_symbol 
            FROM trained_models 
            WHERE is_active = 'active'
        """)
        stocks = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Reviewing performance for {len(stocks)} stocks")
        
        retrain_needed = []
        performance_summary = []
        
        for symbol in stocks:
            should_retrain, reason = should_trigger_retraining(conn, symbol)
            
            performance_summary.append({
                'symbol': symbol,
                'should_retrain': should_retrain,
                'reason': reason
            })
            
            if should_retrain:
                logger.warning(f"⚠️  {symbol}: {reason}")
                retrain_needed.append(symbol)
            else:
                logger.info(f"✅ {symbol}: {reason}")
        
        cursor.close()
        conn.close()
        
        # Trigger retraining for stocks that need it
        if retrain_needed:
            logger.info(f"\n🔄 Triggering retraining for {len(retrain_needed)} stocks")
            retrain_results = trigger_batch_retraining(retrain_needed)
        else:
            logger.info("\n✅ All models performing well, no retraining needed")
            retrain_results = []
        
        return {
            'success': True,
            'task': 'weekly_performance_review',
            'stocks_reviewed': len(stocks),
            'retrain_needed': len(retrain_needed),
            'retrain_symbols': retrain_needed,
            'performance_summary': performance_summary,
            'retrain_results': retrain_results
        }
        
    except Exception as e:
        logger.error(f"Weekly performance review failed: {e}")
        return {
            'success': False,
            'task': 'weekly_performance_review',
            'error': str(e)
        }

def trigger_batch_retraining(symbols: List[str]) -> List[Dict]:
    """
    Trigger retraining for multiple stocks
    
    Args:
        symbols: List of stock symbols to retrain
    
    Returns:
        List of retraining results
    """
    results = []
    
    for symbol in symbols:
        try:
            logger.info(f"Retraining {symbol}...")
            
            # Import training module
            from ml.backtest_and_train import train_and_save_models_for_stock
            
            # Retrain
            result = train_and_save_models_for_stock(symbol)
            
            results.append({
                'symbol': symbol,
                'success': result.get('success', False),
                'models_trained': result.get('models_trained', 0)
            })
            
            logger.info(f"✅ {symbol} retraining complete")
            
        except Exception as e:
            logger.error(f"❌ {symbol} retraining failed: {e}")
            results.append({
                'symbol': symbol,
                'success': False,
                'error': str(e)
            })
    
    return results

def monthly_model_cleanup():
    """
    Monthly task: Clean up old models and predictions
    
    Runs on 1st of each month
    """
    logger.info("=" * 70)
    logger.info("MONTHLY MODEL CLEANUP")
    logger.info("=" * 70)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Archive old predictions (older than 1 year)
        cursor.execute("""
            UPDATE model_predictions
            SET status = 'archived'
            WHERE status = 'validated'
            AND target_date < DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """)
        archived_predictions = cursor.rowcount
        
        # Deactivate old models (older than 6 months, not used recently)
        cursor.execute("""
            UPDATE trained_models
            SET is_active = 'inactive'
            WHERE is_active = 'active'
            AND created_at < DATE_SUB(NOW(), INTERVAL 6 MONTH)
            AND id NOT IN (
                SELECT DISTINCT model_id 
                FROM model_predictions 
                WHERE prediction_date > DATE_SUB(NOW(), INTERVAL 3 MONTH)
            )
        """)
        deactivated_models = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Archived {archived_predictions} old predictions")
        logger.info(f"✅ Deactivated {deactivated_models} unused models")
        
        return {
            'success': True,
            'task': 'monthly_cleanup',
            'archived_predictions': archived_predictions,
            'deactivated_models': deactivated_models
        }
        
    except Exception as e:
        logger.error(f"Monthly cleanup failed: {e}")
        return {
            'success': False,
            'task': 'monthly_cleanup',
            'error': str(e)
        }

def emergency_model_rollback(symbol: str, reason: str):
    """
    Emergency rollback to previous model version
    
    Args:
        symbol: Stock symbol
        reason: Reason for rollback
    """
    logger.warning(f"🚨 EMERGENCY ROLLBACK for {symbol}: {reason}")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current active model
        cursor.execute("""
            SELECT id, created_at
            FROM trained_models
            WHERE stock_symbol = %s
            AND is_active = 'active'
            ORDER BY created_at DESC
            LIMIT 1
        """, (symbol,))
        current_model = cursor.fetchone()
        
        if not current_model:
            logger.error(f"No active model found for {symbol}")
            return False
        
        # Deactivate current model
        cursor.execute("""
            UPDATE trained_models
            SET is_active = 'rollback', notes = %s
            WHERE id = %s
        """, (f"Rolled back: {reason}", current_model['id']))
        
        # Activate previous model
        cursor.execute("""
            UPDATE trained_models
            SET is_active = 'active'
            WHERE stock_symbol = %s
            AND created_at < %s
            AND is_active = 'inactive'
            ORDER BY created_at DESC
            LIMIT 1
        """, (symbol, current_model['created_at']))
        
        if cursor.rowcount == 0:
            logger.error(f"No previous model found to rollback to for {symbol}")
            return False
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Successfully rolled back {symbol} to previous model")
        return True
        
    except Exception as e:
        logger.error(f"Rollback failed for {symbol}: {e}")
        return False

def run_continuous_learning_cycle():
    """
    Main continuous learning cycle
    
    Runs all scheduled tasks based on current time
    """
    now = datetime.now()
    results = {
        'timestamp': now.isoformat(),
        'tasks_run': []
    }
    
    # Daily validation (runs every day at 4:30 PM ET)
    if now.hour == 16 and now.minute >= 30:
        result = daily_validation_task()
        results['tasks_run'].append(result)
    
    # Weekly performance review (runs Sunday at midnight)
    if now.weekday() == 6 and now.hour == 0:
        result = weekly_performance_review()
        results['tasks_run'].append(result)
    
    # Monthly cleanup (runs 1st of month at 2 AM)
    if now.day == 1 and now.hour == 2:
        result = monthly_model_cleanup()
        results['tasks_run'].append(result)
    
    return results

def main():
    """
    Main entry point for continuous learning scheduler
    
    Can be run as:
    1. One-time execution: python continuous_learning_scheduler.py
    2. Scheduled via cron
    3. Triggered manually for specific tasks
    """
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        
        if task == 'validate':
            result = daily_validation_task()
        elif task == 'review':
            result = weekly_performance_review()
        elif task == 'cleanup':
            result = monthly_model_cleanup()
        elif task == 'rollback' and len(sys.argv) > 3:
            symbol = sys.argv[2]
            reason = sys.argv[3]
            result = {'success': emergency_model_rollback(symbol, reason)}
        else:
            print("Usage:")
            print("  python continuous_learning_scheduler.py validate")
            print("  python continuous_learning_scheduler.py review")
            print("  python continuous_learning_scheduler.py cleanup")
            print("  python continuous_learning_scheduler.py rollback <SYMBOL> <REASON>")
            sys.exit(1)
    else:
        # Run full cycle
        result = run_continuous_learning_cycle()
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
