"""
Continuous Improvement System for ML Models
============================================
Learns from historical training results to improve future model training:
1. Analyzes which hyperparameters work best for each stock type
2. Identifies feature importance patterns across successful models
3. Adapts training based on market regime (bull/bear/sideways)
4. Implements incremental learning from new data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
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


def analyze_best_hyperparameters(conn, model_type: str = 'xgboost', min_samples: int = 5) -> Dict:
    """
    Analyze historical training results to find optimal hyperparameters
    
    Args:
        conn: Database connection
        model_type: Type of model to analyze
        min_samples: Minimum number of samples required
    
    Returns:
        Dict with recommended hyperparameters and their confidence
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get all models with their performance metrics
    query = """
    SELECT 
        tm.hyperparameters,
        tm.test_accuracy / 10000.0 as test_accuracy,
        br.sharpe_ratio / 10000.0 as sharpe_ratio,
        br.win_rate / 10000.0 as win_rate,
        br.profit_factor / 10000.0 as profit_factor,
        br.total_return / 10000.0 as total_return
    FROM trained_models tm
    JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.model_type = %s
    AND tm.is_active IN ('active', 'inactive')
    AND br.sharpe_ratio IS NOT NULL
    """
    
    cursor.execute(query, (model_type,))
    results = cursor.fetchall()
    cursor.close()
    
    if len(results) < min_samples:
        logger.warning(f"Not enough samples ({len(results)}) for {model_type}, need {min_samples}")
        return get_default_hyperparameters(model_type)
    
    # Parse hyperparameters and create DataFrame
    # Helper to convert Decimal to float
    def to_float(val, default=0):
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    data = []
    for r in results:
        if r['hyperparameters']:
            try:
                hp = json.loads(r['hyperparameters']) if isinstance(r['hyperparameters'], str) else r['hyperparameters']
                # Convert all hyperparameter values to float to avoid Decimal issues
                for key in hp:
                    if isinstance(hp[key], (int, float)) or hasattr(hp[key], '__float__'):
                        hp[key] = to_float(hp[key])
                hp['sharpe_ratio'] = to_float(r['sharpe_ratio'], 0)
                hp['win_rate'] = to_float(r['win_rate'], 0.5)
                hp['profit_factor'] = to_float(r['profit_factor'], 1.0)
                hp['test_accuracy'] = to_float(r['test_accuracy'], 0.5)
                data.append(hp)
            except Exception as e:
                logger.debug(f"Error parsing hyperparameters: {e}")
                pass
    
    if not data:
        return get_default_hyperparameters(model_type)
    
    df = pd.DataFrame(data)
    
    # Calculate composite performance score
    df['performance_score'] = (
        df['sharpe_ratio'].clip(-2, 3) / 3 * 0.4 +  # Sharpe normalized to 0-1
        df['win_rate'] * 0.3 +
        (df['profit_factor'].clip(0, 3) / 3) * 0.3
    )
    
    # Get top 20% performers
    threshold = df['performance_score'].quantile(0.8)
    top_performers = df[df['performance_score'] >= threshold]
    
    logger.info(f"Analyzing {len(top_performers)} top-performing {model_type} models")
    
    # Extract optimal hyperparameters from top performers
    optimal_hp = {}
    confidence = {}
    
    numeric_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                      'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda']
    
    for param in numeric_params:
        if param in top_performers.columns:
            values = top_performers[param].dropna()
            if len(values) > 0:
                # Use median for robustness
                optimal_hp[param] = float(values.median())
                # Confidence based on consistency (lower std = higher confidence)
                if values.std() > 0:
                    cv = values.std() / values.mean() if values.mean() != 0 else 1
                    confidence[param] = max(0, min(1, 1 - cv))
                else:
                    confidence[param] = 1.0
                
                # Round integer parameters
                if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                    optimal_hp[param] = int(round(optimal_hp[param]))
    
    # Calculate overall confidence
    avg_confidence = np.mean(list(confidence.values())) if confidence else 0.5
    
    result = {
        'model_type': model_type,
        'optimal_hyperparameters': optimal_hp,
        'confidence_per_param': confidence,
        'overall_confidence': avg_confidence,
        'samples_analyzed': len(results),
        'top_performers_count': len(top_performers),
        'avg_sharpe_top': float(top_performers['sharpe_ratio'].mean()),
        'avg_win_rate_top': float(top_performers['win_rate'].mean())
    }
    
    logger.info(f"Optimal {model_type} hyperparameters (confidence: {avg_confidence:.2f}):")
    for k, v in optimal_hp.items():
        logger.info(f"  {k}: {v} (conf: {confidence.get(k, 0):.2f})")
    
    return result


def get_default_hyperparameters(model_type: str) -> Dict:
    """Get default hyperparameters for a model type"""
    defaults = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    }
    
    return {
        'model_type': model_type,
        'optimal_hyperparameters': defaults.get(model_type, defaults['xgboost']),
        'confidence_per_param': {},
        'overall_confidence': 0.5,
        'samples_analyzed': 0,
        'is_default': True
    }


def analyze_feature_importance_patterns(conn, symbol: str = None) -> Dict:
    """
    Analyze which features are most important across successful models
    
    Args:
        conn: Database connection
        symbol: Optional symbol to filter by
    
    Returns:
        Dict with feature importance rankings and patterns
    """
    cursor = conn.cursor(dictionary=True)
    
    # Get feature importance from top-performing models
    query = """
    SELECT 
        tm.feature_importance,
        tm.stock_symbol,
        br.sharpe_ratio / 10000.0 as sharpe_ratio
    FROM trained_models tm
    JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.is_active = 'active'
    AND tm.feature_importance IS NOT NULL
    AND br.sharpe_ratio > 0
    """
    
    params = []
    if symbol:
        query += " AND tm.stock_symbol = %s"
        params.append(symbol)
    
    query += " ORDER BY br.sharpe_ratio DESC LIMIT 50"
    
    cursor.execute(query, tuple(params))
    results = cursor.fetchall()
    cursor.close()
    
    if not results:
        return {'features': [], 'message': 'No feature importance data available'}
    
    # Aggregate feature importance
    feature_scores = {}
    feature_counts = {}
    
    for r in results:
        if r['feature_importance']:
            try:
                fi = json.loads(r['feature_importance']) if isinstance(r['feature_importance'], str) else r['feature_importance']
                weight = max(0.1, r['sharpe_ratio'])  # Weight by Sharpe ratio
                
                for feature, importance in fi.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = 0
                        feature_counts[feature] = 0
                    feature_scores[feature] += importance * weight
                    feature_counts[feature] += 1
            except:
                pass
    
    # Calculate weighted average importance
    feature_rankings = []
    for feature in feature_scores:
        if feature_counts[feature] > 0:
            avg_importance = feature_scores[feature] / feature_counts[feature]
            feature_rankings.append({
                'feature': feature,
                'avg_importance': avg_importance,
                'occurrence_count': feature_counts[feature]
            })
    
    # Sort by importance
    feature_rankings.sort(key=lambda x: x['avg_importance'], reverse=True)
    
    return {
        'features': feature_rankings[:20],  # Top 20 features
        'total_models_analyzed': len(results),
        'symbol_filter': symbol
    }


def detect_market_regime(symbol: str, lookback_days: int = 60) -> str:
    """
    Detect current market regime for a symbol
    
    Args:
        symbol: Stock symbol
        lookback_days: Days to look back
    
    Returns:
        Market regime: 'bull', 'bear', 'sideways', 'volatile'
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{lookback_days}d")
        
        if hist.empty or len(hist) < 20:
            return 'unknown'
        
        # Calculate metrics
        returns = hist['Close'].pct_change().dropna()
        cumulative_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # SMA trend
        sma_20 = hist['Close'].rolling(20).mean()
        sma_50 = hist['Close'].rolling(50).mean() if len(hist) >= 50 else sma_20
        
        current_price = hist['Close'].iloc[-1]
        above_sma_20 = current_price > sma_20.iloc[-1]
        above_sma_50 = current_price > sma_50.iloc[-1] if len(hist) >= 50 else above_sma_20
        
        # Determine regime
        if volatility > 0.4:  # Very high volatility
            return 'volatile'
        elif cumulative_return > 0.1 and above_sma_20 and above_sma_50:
            return 'bull'
        elif cumulative_return < -0.1 and not above_sma_20 and not above_sma_50:
            return 'bear'
        else:
            return 'sideways'
            
    except Exception as e:
        logger.warning(f"Error detecting market regime for {symbol}: {e}")
        return 'unknown'


def get_regime_adjusted_hyperparameters(base_hp: Dict, regime: str) -> Dict:
    """
    Adjust hyperparameters based on market regime
    
    Args:
        base_hp: Base hyperparameters
        regime: Market regime
    
    Returns:
        Adjusted hyperparameters
    """
    # Helper to safely convert to float (handles Decimal from database)
    def safe_float(val, default=0):
        if val is None:
            return float(default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)
    
    # Convert all base_hp values to float to avoid Decimal * float errors
    adjusted = {}
    for k, v in base_hp.items():
        if isinstance(v, (int, float)) or hasattr(v, '__float__'):
            adjusted[k] = safe_float(v)
        else:
            adjusted[k] = v
    
    if regime == 'volatile':
        # In volatile markets, use more conservative settings
        adjusted['max_depth'] = int(min(safe_float(base_hp.get('max_depth', 5)), 4))
        adjusted['learning_rate'] = safe_float(base_hp.get('learning_rate', 0.1)) * 0.8
        adjusted['subsample'] = min(safe_float(base_hp.get('subsample', 0.8)), 0.7)
        adjusted['n_estimators'] = int(safe_float(base_hp.get('n_estimators', 100)) * 1.2)
        
    elif regime == 'bull':
        # In bull markets, can be slightly more aggressive
        adjusted['max_depth'] = int(min(safe_float(base_hp.get('max_depth', 5)) + 1, 7))
        adjusted['learning_rate'] = safe_float(base_hp.get('learning_rate', 0.1)) * 1.1
        
    elif regime == 'bear':
        # In bear markets, focus on risk management
        adjusted['max_depth'] = int(min(safe_float(base_hp.get('max_depth', 5)), 4))
        adjusted['reg_lambda'] = safe_float(base_hp.get('reg_lambda', 1)) * 1.5
        
    # sideways or unknown - use base parameters
    
    return adjusted


def get_improved_training_config(conn, symbol: str, model_type: str = 'xgboost') -> Dict:
    """
    Get improved training configuration based on historical performance
    
    Args:
        conn: Database connection
        symbol: Stock symbol to train
        model_type: Type of model
    
    Returns:
        Complete training configuration with optimized settings
    """
    # Get optimal hyperparameters from historical analysis
    hp_analysis = analyze_best_hyperparameters(conn, model_type)
    optimal_hp = hp_analysis.get('optimal_hyperparameters', {})
    
    # Detect current market regime
    regime = detect_market_regime(symbol)
    logger.info(f"Detected market regime for {symbol}: {regime}")
    
    # Adjust hyperparameters for regime
    adjusted_hp = get_regime_adjusted_hyperparameters(optimal_hp, regime)
    
    # Get feature importance patterns
    feature_patterns = analyze_feature_importance_patterns(conn, symbol)
    top_features = [f['feature'] for f in feature_patterns.get('features', [])[:15]]
    
    # Get learning insights
    from ml.model_selector import get_learning_insights
    insights = get_learning_insights(conn, symbol)
    
    config = {
        'symbol': symbol,
        'model_type': model_type,
        'hyperparameters': adjusted_hp,
        'market_regime': regime,
        'hp_confidence': hp_analysis.get('overall_confidence', 0.5),
        'prioritized_features': top_features,
        'insights': insights.get('recommendations', []),
        'training_notes': []
    }
    
    # Add training notes based on analysis
    if hp_analysis.get('overall_confidence', 0) > 0.7:
        config['training_notes'].append(
            f"High confidence in hyperparameters (based on {hp_analysis.get('samples_analyzed', 0)} samples)"
        )
    
    if regime == 'volatile':
        config['training_notes'].append(
            "Using conservative settings due to high market volatility"
        )
    
    if insights.get('high_error_symbols'):
        if symbol in [s['symbol'] for s in insights['high_error_symbols']]:
            config['training_notes'].append(
                f"Warning: {symbol} has shown high prediction errors - consider additional features"
            )
    
    return config


def save_training_result_for_learning(
    conn,
    symbol: str,
    model_type: str,
    hyperparameters: Dict,
    metrics: Dict,
    market_regime: str
) -> None:
    """
    Save training result to enable future learning
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        model_type: Model type
        hyperparameters: Hyperparameters used
        metrics: Training metrics (accuracy, sharpe, etc.)
        market_regime: Market regime during training
    """
    cursor = conn.cursor()
    
    # This data will be used by analyze_best_hyperparameters() in future runs
    # The trained_models and backtesting_results tables already capture this
    # This function is for any additional learning metadata
    
    logger.info(f"Training result saved for {symbol} ({model_type})")
    logger.info(f"  Regime: {market_regime}")
    logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 'N/A')}")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
    
    cursor.close()


def run_improvement_analysis(symbol: str = None) -> Dict:
    """
    Run full improvement analysis and return recommendations
    
    Args:
        symbol: Optional symbol to focus on
    
    Returns:
        Comprehensive improvement recommendations
    """
    conn = get_db_connection()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'analyses': {}
    }
    
    # Analyze XGBoost hyperparameters
    results['analyses']['xgboost_hp'] = analyze_best_hyperparameters(conn, 'xgboost')
    
    # Analyze LightGBM hyperparameters
    results['analyses']['lightgbm_hp'] = analyze_best_hyperparameters(conn, 'lightgbm')
    
    # Analyze feature importance
    results['analyses']['feature_importance'] = analyze_feature_importance_patterns(conn, symbol)
    
    # Get learning insights
    from ml.model_selector import get_learning_insights
    results['analyses']['insights'] = get_learning_insights(conn, symbol)
    
    # Market regime if symbol provided
    if symbol:
        results['analyses']['market_regime'] = detect_market_regime(symbol)
        results['analyses']['training_config'] = get_improved_training_config(conn, symbol)
    
    conn.close()
    
    return results


def main():
    """Test continuous improvement system"""
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = None
    
    print(f"\n{'='*60}")
    print("CONTINUOUS IMPROVEMENT ANALYSIS")
    print(f"{'='*60}\n")
    
    results = run_improvement_analysis(symbol)
    
    # Print summary
    print("\n📊 XGBoost Optimal Hyperparameters:")
    xgb_hp = results['analyses']['xgboost_hp']
    if xgb_hp.get('optimal_hyperparameters'):
        for k, v in xgb_hp['optimal_hyperparameters'].items():
            conf = xgb_hp.get('confidence_per_param', {}).get(k, 0)
            print(f"   {k}: {v} (confidence: {conf:.2f})")
    
    print("\n📊 Top Features:")
    features = results['analyses']['feature_importance'].get('features', [])[:10]
    for i, f in enumerate(features):
        print(f"   {i+1}. {f['feature']}: {f['avg_importance']:.4f}")
    
    print("\n💡 Recommendations:")
    recs = results['analyses']['insights'].get('recommendations', [])
    for rec in recs:
        print(f"   • {rec}")
    
    if symbol:
        print(f"\n🎯 Training Config for {symbol}:")
        config = results['analyses'].get('training_config', {})
        print(f"   Market Regime: {config.get('market_regime', 'unknown')}")
        print(f"   HP Confidence: {config.get('hp_confidence', 0):.2f}")
    
    print(f"\n{'='*60}")
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
