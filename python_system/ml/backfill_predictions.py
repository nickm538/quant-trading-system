"""
Backfill Script for Model Predictions
======================================
This script validates ALL predictions in the database, not just pending ones.
Use this to populate actual_price, actual_high, actual_low, price_error, and percentage_error
for predictions that were created before the validation system was working.

Usage:
    python3.11 backfill_predictions.py [--limit N] [--status STATUS]
    
Options:
    --limit N       Limit to N predictions (default: all)
    --status STATUS Only process predictions with this status (default: all)
    --dry-run       Show what would be done without making changes
"""

import os
import sys
import argparse
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.validate_and_retrain import get_db_connection, fetch_actual_price

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backfill_predictions(limit: int = None, status_filter: str = None, dry_run: bool = False):
    """
    Backfill actual prices for all predictions in database
    
    Args:
        limit: Maximum number of predictions to process
        status_filter: Only process predictions with this status
        dry_run: If True, don't make any changes
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Build query
    query = """
    SELECT id, stock_symbol, prediction_date, target_date,
           predicted_price, predicted_low, predicted_high, 
           confidence, status, actual_price
    FROM model_predictions
    WHERE target_date <= NOW()
    """
    
    if status_filter:
        query += f" AND status = '{status_filter}'"
    
    query += " ORDER BY target_date DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    predictions = cursor.fetchall()
    
    total = len(predictions)
    logger.info(f"Found {total} predictions to process")
    
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    processed = 0
    updated = 0
    failed = 0
    skipped = 0
    
    for i, pred in enumerate(predictions, 1):
        pred_id = pred['id']
        symbol = pred['stock_symbol']
        target_date = pred['target_date']
        predicted_price = pred['predicted_price'] / 100.0  # Convert from cents
        current_status = pred['status']
        has_actual = pred['actual_price'] is not None
        
        # Skip if already has actual price (unless we're forcing update)
        if has_actual and not dry_run:
            logger.debug(f"[{i}/{total}] Skipping {symbol} (already has actual price)")
            skipped += 1
            continue
        
        logger.info(f"[{i}/{total}] Processing {symbol} - Target: {target_date.date()}, Predicted: ${predicted_price:.2f}")
        
        # Fetch actual price data
        price_data = fetch_actual_price(symbol, target_date)
        
        if price_data is None:
            logger.warning(f"  ❌ Failed to fetch price for {symbol}")
            failed += 1
            
            if not dry_run:
                # Mark as failed
                cursor.execute("""
                    UPDATE model_predictions
                    SET status = 'failed',
                        updated_at = NOW()
                    WHERE id = %s
                """, (pred_id,))
            
            continue
        
        actual_close = price_data['close']
        actual_high = price_data['high']
        actual_low = price_data['low']
        
        # Calculate errors
        price_error = abs(actual_close - predicted_price)
        percentage_error = abs((actual_close - predicted_price) / actual_close * 100)
        
        logger.info(f"  ✅ Actual: ${actual_close:.2f} (H: ${actual_high:.2f}, L: ${actual_low:.2f})")
        logger.info(f"     Error: ${price_error:.2f} ({percentage_error:.2f}%)")
        
        if not dry_run:
            # Update prediction with actual values
            cursor.execute("""
                UPDATE model_predictions
                SET actual_price = %s,
                    actual_low = %s,
                    actual_high = %s,
                    price_error = %s,
                    percentage_error = %s,
                    status = 'validated',
                    updated_at = NOW()
                WHERE id = %s
            """, (
                int(actual_close * 100),
                int(actual_low * 100),
                int(actual_high * 100),
                int(price_error * 100),
                int(percentage_error * 100),
                pred_id
            ))
            
            updated += 1
        
        processed += 1
        
        # Commit every 10 predictions
        if not dry_run and processed % 10 == 0:
            conn.commit()
            logger.info(f"  💾 Committed {processed} updates")
    
    # Final commit
    if not dry_run:
        conn.commit()
    
    cursor.close()
    conn.close()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*70)
    logger.info(f"Total predictions: {total}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Updated: {updated}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    
    if dry_run:
        logger.info("\nThis was a DRY RUN - no changes were made")
        logger.info("Run without --dry-run to apply changes")
    
    return {
        'total': total,
        'processed': processed,
        'updated': updated,
        'failed': failed,
        'skipped': skipped
    }

def main():
    parser = argparse.ArgumentParser(description='Backfill actual prices for model predictions')
    parser.add_argument('--limit', type=int, help='Limit number of predictions to process')
    parser.add_argument('--status', type=str, help='Only process predictions with this status')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("MODEL PREDICTIONS BACKFILL")
    logger.info("="*70)
    logger.info(f"Limit: {args.limit or 'None (all)'}")
    logger.info(f"Status filter: {args.status or 'None (all)'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")
    
    try:
        stats = backfill_predictions(
            limit=args.limit,
            status_filter=args.status,
            dry_run=args.dry_run
        )
        
        sys.exit(0 if stats['failed'] == 0 else 1)
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
