"""
Advanced Model Selector with Continuous Improvement
====================================================
Selects the best model for each stock/ETF based on:
1. Historical test accuracy
2. Backtesting results (Sharpe ratio, win rate, profit factor)
3. Recent prediction accuracy (validated predictions)
4. Model recency (newer models may capture recent market dynamics)

Also provides insights for continuous improvement of training.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import mysql.connector
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


def calculate_composite_score(
    test_accuracy: float,
    sharpe_ratio: float,
    win_rate: float,
    profit_factor: float,
    recent_prediction_accuracy: float,
    model_age_days: int,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate composite score for model selection
    
    Args:
        test_accuracy: Model's test accuracy (0-1 scale)
        sharpe_ratio: Backtest Sharpe ratio
        win_rate: Backtest win rate (0-1 scale)
        profit_factor: Backtest profit factor
        recent_prediction_accuracy: Accuracy of recent validated predictions (0-1 scale)
        model_age_days: Days since model was trained
        weights: Optional custom weights for each component
    
    Returns:
        Composite score (higher is better)
    """
    # Default weights - can be tuned based on what matters most
    if weights is None:
        weights = {
            'test_accuracy': 0.20,        # Training performance
            'sharpe_ratio': 0.25,         # Risk-adjusted returns
            'win_rate': 0.15,             # Direction accuracy
            'profit_factor': 0.15,        # Profitability
            'recent_accuracy': 0.20,      # Real-world performance
            'recency': 0.05               # Model freshness
        }
    
    # Normalize each component to 0-100 scale
    
    # Test accuracy (already 0-1, convert to 0-100)
    test_score = min(100, test_accuracy * 100)
    
    # Sharpe ratio (typical range -2 to 3, normalize to 0-100)
    sharpe_score = min(100, max(0, (sharpe_ratio + 1) * 25))
    
    # Win rate (already 0-1, convert to 0-100)
    win_score = min(100, win_rate * 100)
    
    # Profit factor (typical range 0-3, normalize to 0-100)
    profit_score = min(100, max(0, profit_factor * 33.33))
    
    # Recent prediction accuracy (already 0-1, convert to 0-100)
    recent_score = min(100, recent_prediction_accuracy * 100)
    
    # Recency score (newer models get higher scores)
    # Models < 7 days old get 100, models > 180 days get 0
    recency_score = max(0, min(100, 100 - (model_age_days / 180) * 100))
    
    # Calculate weighted composite score
    composite = (
        weights['test_accuracy'] * test_score +
        weights['sharpe_ratio'] * sharpe_score +
        weights['win_rate'] * win_score +
        weights['profit_factor'] * profit_score +
        weights['recent_accuracy'] * recent_score +
        weights['recency'] * recency_score
    )
    
    return composite


def get_recent_prediction_accuracy(conn, model_id: int, days_back: int = 30) -> float:
    """
    Get the accuracy of recent validated predictions for a model
    
    Args:
        conn: Database connection
        model_id: Model ID to check
        days_back: Number of days to look back
    
    Returns:
        Accuracy as a decimal (0-1), or 0.5 if no data
    """
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN ABS(percentage_error) < 500 THEN 1 ELSE 0 END) as accurate
    FROM model_predictions
    WHERE model_id = %s
    AND status = 'validated'
    AND target_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    
    cursor.execute(query, (model_id, days_back))
    result = cursor.fetchone()
    cursor.close()
    
    if result and result['total'] > 0:
        return result['accurate'] / result['total']
    
    return 0.5  # Default to neutral if no data


def get_best_model_for_stock(conn, symbol: str) -> Optional[Dict]:
    """
    Get the single best model for a stock based on composite scoring
    
    Args:
        conn: Database connection
        symbol: Stock symbol
    
    Returns:
        Best model info dict, or None if no models found
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get all active models with their backtest results
    query = """
    SELECT 
        tm.id,
        tm.stock_symbol,
        tm.model_type,
        tm.version,
        tm.model_data,
        tm.test_accuracy,
        tm.hyperparameters,
        tm.feature_importance,
        tm.trained_at,
        tm.created_at,
        DATEDIFF(NOW(), tm.created_at) as model_age_days,
        COALESCE(br.sharpe_ratio, 0) as sharpe_ratio,
        COALESCE(br.win_rate, 5000) as win_rate,
        COALESCE(br.profit_factor, 10000) as profit_factor,
        COALESCE(br.max_drawdown, 0) as max_drawdown,
        COALESCE(br.total_trades, 0) as total_trades
    FROM trained_models tm
    LEFT JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.stock_symbol = %s 
    AND tm.is_active = 'active'
    ORDER BY tm.created_at DESC
    """
    
    cursor.execute(query, (symbol,))
    models = cursor.fetchall()
    cursor.close()
    
    if not models:
        logger.warning(f"No active models found for {symbol}")
        return None
    
    logger.info(f"Found {len(models)} active models for {symbol}")
    
    # Calculate composite score for each model
    scored_models = []
    for model in models:
        # Get recent prediction accuracy
        recent_accuracy = get_recent_prediction_accuracy(conn, model['id'])
        
        # Convert stored values to proper scale
        test_accuracy = model['test_accuracy'] / 10000.0 if model['test_accuracy'] else 0.5
        sharpe_ratio = model['sharpe_ratio'] / 10000.0 if model['sharpe_ratio'] else 0
        win_rate = model['win_rate'] / 10000.0 if model['win_rate'] else 0.5
        profit_factor = model['profit_factor'] / 10000.0 if model['profit_factor'] else 1.0
        model_age_days = model['model_age_days'] or 0
        
        # Calculate composite score
        composite_score = calculate_composite_score(
            test_accuracy=test_accuracy,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_prediction_accuracy=recent_accuracy,
            model_age_days=model_age_days
        )
        
        scored_models.append({
            **model,
            'composite_score': composite_score,
            'recent_prediction_accuracy': recent_accuracy,
            'test_accuracy_pct': test_accuracy * 100,
            'sharpe_ratio_actual': sharpe_ratio,
            'win_rate_pct': win_rate * 100,
            'profit_factor_actual': profit_factor
        })
        
        logger.info(f"  Model {model['id']} ({model['model_type']}): "
                   f"composite={composite_score:.2f}, "
                   f"test_acc={test_accuracy*100:.1f}%, "
                   f"sharpe={sharpe_ratio:.2f}, "
                   f"win_rate={win_rate*100:.1f}%, "
                   f"recent_acc={recent_accuracy*100:.1f}%")
    
    # Sort by composite score (descending)
    scored_models.sort(key=lambda x: x['composite_score'], reverse=True)
    
    best_model = scored_models[0]
    logger.info(f"✅ Best model for {symbol}: ID {best_model['id']} "
               f"({best_model['model_type']}) with composite score {best_model['composite_score']:.2f}")
    
    return best_model


def get_top_models_for_stock(conn, symbol: str, top_n: int = 3) -> List[Dict]:
    """
    Get top N models for a stock for ensemble prediction
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        top_n: Number of models to return
    
    Returns:
        List of top model info dicts
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get all active models with their backtest results
    query = """
    SELECT 
        tm.id,
        tm.stock_symbol,
        tm.model_type,
        tm.version,
        tm.model_data,
        tm.test_accuracy,
        tm.hyperparameters,
        tm.feature_importance,
        tm.trained_at,
        tm.created_at,
        DATEDIFF(NOW(), tm.created_at) as model_age_days,
        COALESCE(br.sharpe_ratio, 0) as sharpe_ratio,
        COALESCE(br.win_rate, 5000) as win_rate,
        COALESCE(br.profit_factor, 10000) as profit_factor
    FROM trained_models tm
    LEFT JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.stock_symbol = %s 
    AND tm.is_active = 'active'
    """
    
    cursor.execute(query, (symbol,))
    models = cursor.fetchall()
    cursor.close()
    
    if not models:
        return []
    
    # Calculate composite score for each model
    scored_models = []
    for model in models:
        recent_accuracy = get_recent_prediction_accuracy(conn, model['id'])
        
        test_accuracy = model['test_accuracy'] / 10000.0 if model['test_accuracy'] else 0.5
        sharpe_ratio = model['sharpe_ratio'] / 10000.0 if model['sharpe_ratio'] else 0
        win_rate = model['win_rate'] / 10000.0 if model['win_rate'] else 0.5
        profit_factor = model['profit_factor'] / 10000.0 if model['profit_factor'] else 1.0
        model_age_days = model['model_age_days'] or 0
        
        composite_score = calculate_composite_score(
            test_accuracy=test_accuracy,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_prediction_accuracy=recent_accuracy,
            model_age_days=model_age_days
        )
        
        scored_models.append({
            **model,
            'composite_score': composite_score,
            'ensemble_weight': composite_score  # Will be normalized later
        })
    
    # Sort by composite score and take top N
    scored_models.sort(key=lambda x: x['composite_score'], reverse=True)
    top_models = scored_models[:top_n]
    
    # Normalize ensemble weights
    total_weight = sum(m['composite_score'] for m in top_models)
    if total_weight > 0:
        for model in top_models:
            model['ensemble_weight'] = model['composite_score'] / total_weight
    
    return top_models


def get_learning_insights(conn, symbol: str = None) -> Dict:
    """
    Analyze model performance to extract insights for continuous improvement
    
    Args:
        conn: Database connection
        symbol: Optional symbol to filter by
    
    Returns:
        Dict with insights for improving training
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get best performing model configurations
    query = """
    SELECT 
        tm.model_type,
        tm.hyperparameters,
        AVG(br.sharpe_ratio) / 10000.0 as avg_sharpe,
        AVG(br.win_rate) / 10000.0 as avg_win_rate,
        AVG(br.profit_factor) / 10000.0 as avg_profit_factor,
        AVG(tm.test_accuracy) / 10000.0 as avg_test_accuracy,
        COUNT(*) as model_count
    FROM trained_models tm
    LEFT JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.is_active = 'active'
    """
    
    params = []
    if symbol:
        query += " AND tm.stock_symbol = %s"
        params.append(symbol)
    
    query += " GROUP BY tm.model_type, tm.hyperparameters ORDER BY avg_sharpe DESC LIMIT 10"
    
    cursor.execute(query, tuple(params))
    best_configs = cursor.fetchall()
    
    # Get worst performing configurations to avoid
    query_worst = """
    SELECT 
        tm.model_type,
        tm.hyperparameters,
        AVG(br.sharpe_ratio) / 10000.0 as avg_sharpe,
        AVG(br.win_rate) / 10000.0 as avg_win_rate,
        COUNT(*) as model_count
    FROM trained_models tm
    LEFT JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.is_active IN ('active', 'inactive')
    """
    
    if symbol:
        query_worst += " AND tm.stock_symbol = %s"
    
    query_worst += " GROUP BY tm.model_type, tm.hyperparameters HAVING model_count >= 3 ORDER BY avg_sharpe ASC LIMIT 5"
    
    cursor.execute(query_worst, tuple(params))
    worst_configs = cursor.fetchall()
    
    # Get prediction error patterns
    query_errors = """
    SELECT 
        mp.stock_symbol,
        AVG(mp.percentage_error) / 100.0 as avg_error,
        COUNT(*) as prediction_count,
        STDDEV(mp.percentage_error) / 100.0 as error_stddev
    FROM model_predictions mp
    WHERE mp.status = 'validated'
    AND mp.target_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    """
    
    if symbol:
        query_errors += " AND mp.stock_symbol = %s"
    
    query_errors += " GROUP BY mp.stock_symbol ORDER BY avg_error DESC"
    
    cursor.execute(query_errors, tuple(params))
    error_patterns = cursor.fetchall()
    
    cursor.close()
    
    # Extract insights
    insights = {
        'best_model_types': [],
        'best_hyperparameters': [],
        'avoid_configurations': [],
        'high_error_symbols': [],
        'recommendations': []
    }
    
    # Best model types
    if best_configs:
        model_type_performance = {}
        for config in best_configs:
            mt = config['model_type']
            if mt not in model_type_performance:
                model_type_performance[mt] = []
            model_type_performance[mt].append(config['avg_sharpe'] or 0)
        
        for mt, sharpes in model_type_performance.items():
            insights['best_model_types'].append({
                'model_type': mt,
                'avg_sharpe': np.mean(sharpes),
                'count': len(sharpes)
            })
        
        insights['best_model_types'].sort(key=lambda x: x['avg_sharpe'], reverse=True)
    
    # Best hyperparameters
    for config in best_configs[:3]:
        if config['hyperparameters']:
            try:
                hp = json.loads(config['hyperparameters']) if isinstance(config['hyperparameters'], str) else config['hyperparameters']
                insights['best_hyperparameters'].append({
                    'model_type': config['model_type'],
                    'hyperparameters': hp,
                    'avg_sharpe': config['avg_sharpe'],
                    'avg_win_rate': config['avg_win_rate']
                })
            except:
                pass
    
    # Configurations to avoid
    for config in worst_configs:
        if config['avg_sharpe'] and config['avg_sharpe'] < 0:
            insights['avoid_configurations'].append({
                'model_type': config['model_type'],
                'reason': f"Negative Sharpe ratio ({config['avg_sharpe']:.2f})"
            })
    
    # High error symbols that need attention
    for error in error_patterns:
        if error['avg_error'] and error['avg_error'] > 5:  # > 5% average error
            insights['high_error_symbols'].append({
                'symbol': error['stock_symbol'],
                'avg_error_pct': error['avg_error'],
                'prediction_count': error['prediction_count']
            })
    
    # Generate recommendations
    if insights['best_model_types']:
        best_type = insights['best_model_types'][0]['model_type']
        insights['recommendations'].append(
            f"Prioritize {best_type} models - they show the best risk-adjusted returns"
        )
    
    if insights['high_error_symbols']:
        high_error_syms = [s['symbol'] for s in insights['high_error_symbols'][:3]]
        insights['recommendations'].append(
            f"Consider retraining models for: {', '.join(high_error_syms)} - high prediction errors"
        )
    
    if insights['best_hyperparameters']:
        insights['recommendations'].append(
            "Use hyperparameters from top-performing models as starting points for new training"
        )
    
    return insights


def get_optimal_hyperparameters(conn, symbol: str, model_type: str = 'xgboost') -> Dict:
    """
    Get optimal hyperparameters based on historical performance
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        model_type: Type of model (xgboost, lightgbm)
    
    Returns:
        Dict of recommended hyperparameters
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get best performing hyperparameters for this symbol and model type
    query = """
    SELECT 
        tm.hyperparameters,
        br.sharpe_ratio / 10000.0 as sharpe,
        br.win_rate / 10000.0 as win_rate,
        tm.test_accuracy / 10000.0 as test_accuracy
    FROM trained_models tm
    JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.model_type = %s
    AND tm.is_active = 'active'
    AND br.sharpe_ratio > 0
    ORDER BY br.sharpe_ratio DESC
    LIMIT 5
    """
    
    cursor.execute(query, (model_type,))
    results = cursor.fetchall()
    cursor.close()
    
    if not results:
        # Return default hyperparameters
        if model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        else:  # lightgbm
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
    
    # Aggregate hyperparameters from top performers
    all_hyperparams = []
    for result in results:
        if result['hyperparameters']:
            try:
                hp = json.loads(result['hyperparameters']) if isinstance(result['hyperparameters'], str) else result['hyperparameters']
                all_hyperparams.append(hp)
            except:
                pass
    
    if not all_hyperparams:
        return get_optimal_hyperparameters(conn, symbol, model_type)  # Return defaults
    
    # Average the hyperparameters
    optimal = {}
    for key in all_hyperparams[0].keys():
        values = [hp.get(key) for hp in all_hyperparams if hp.get(key) is not None]
        if values:
            if isinstance(values[0], (int, float)):
                optimal[key] = np.mean(values)
                if key in ['n_estimators', 'max_depth']:
                    optimal[key] = int(round(optimal[key]))
            else:
                optimal[key] = values[0]  # Use first value for non-numeric
    
    logger.info(f"Optimal hyperparameters for {model_type}: {optimal}")
    return optimal


def main():
    """Test the model selector"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_selector.py <SYMBOL>")
        print("       python model_selector.py insights [SYMBOL]")
        sys.exit(1)
    
    conn = get_db_connection()
    
    if sys.argv[1] == 'insights':
        symbol = sys.argv[2] if len(sys.argv) > 2 else None
        insights = get_learning_insights(conn, symbol)
        print(json.dumps(insights, indent=2, default=str))
    else:
        symbol = sys.argv[1].upper()
        
        # Get best model
        best = get_best_model_for_stock(conn, symbol)
        if best:
            print(f"\n✅ Best model for {symbol}:")
            print(f"   ID: {best['id']}")
            print(f"   Type: {best['model_type']}")
            print(f"   Composite Score: {best['composite_score']:.2f}")
            print(f"   Test Accuracy: {best['test_accuracy_pct']:.1f}%")
            print(f"   Sharpe Ratio: {best['sharpe_ratio_actual']:.2f}")
            print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
            print(f"   Recent Accuracy: {best['recent_prediction_accuracy']*100:.1f}%")
        
        # Get top 3 for ensemble
        print(f"\n📊 Top 3 models for ensemble:")
        top_models = get_top_models_for_stock(conn, symbol, top_n=3)
        for i, model in enumerate(top_models):
            print(f"   {i+1}. {model['model_type']} (ID: {model['id']}) - "
                  f"weight: {model['ensemble_weight']:.2%}, "
                  f"score: {model['composite_score']:.2f}")
    
    conn.close()


if __name__ == '__main__':
    main()
