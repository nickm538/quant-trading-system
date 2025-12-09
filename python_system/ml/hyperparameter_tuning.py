"""
Hyperparameter Optimization System
===================================
Uses Optuna for automatic hyperparameter tuning per stock
Stores best parameters in database for future use
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import optuna
    from optuna.samplers import TPESampler
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_OPTUNA = True
except ImportError:
    logger.warning("Optuna not installed. Install with: pip install optuna")
    HAS_OPTUNA = False

def optimize_xgboost_params(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
    
    Returns:
        Best hyperparameters dict
    """
    if not HAS_OPTUNA:
        # Return default parameters
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    logger.info(f"Starting XGBoost hyperparameter optimization ({n_trials} trials)...")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    logger.info(f"Best XGBoost params: {best_params}")
    logger.info(f"Best MSE: {study.best_value:.6f}")
    
    return best_params

def optimize_lightgbm_params(X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
    """
    Optimize LightGBM hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
    
    Returns:
        Best hyperparameters dict
    """
    if not HAS_OPTUNA:
        # Return default parameters
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
        
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    logger.info(f"Starting LightGBM hyperparameter optimization ({n_trials} trials)...")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    logger.info(f"Best LightGBM params: {best_params}")
    logger.info(f"Best MSE: {study.best_value:.6f}")
    
    return best_params

def optimize_ensemble_weights(predictions_list: List[np.ndarray], y_true: np.ndarray, n_trials: int = 100) -> np.ndarray:
    """
    Optimize ensemble weights using Optuna
    
    Args:
        predictions_list: List of prediction arrays from different models
        y_true: True target values
        n_trials: Number of optimization trials
    
    Returns:
        Optimal weights array
    """
    if not HAS_OPTUNA:
        # Return equal weights
        n_models = len(predictions_list)
        return np.ones(n_models) / n_models
    
    n_models = len(predictions_list)
    
    def objective(trial):
        # Suggest weights for each model
        weights = []
        for i in range(n_models):
            weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Normalize weights
        if weights.sum() == 0:
            return float('inf')
        
        weights = weights / weights.sum()
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros_like(y_true)
        for i, pred in enumerate(predictions_list):
            ensemble_pred += weights[i] * pred
        
        # Calculate MSE
        mse = mean_squared_error(y_true, ensemble_pred)
        
        return mse
    
    logger.info(f"Optimizing ensemble weights for {n_models} models...")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Extract best weights
    best_weights = []
    for i in range(n_models):
        best_weights.append(study.best_params[f'weight_{i}'])
    
    best_weights = np.array(best_weights)
    best_weights = best_weights / best_weights.sum()  # Normalize
    
    logger.info(f"Optimal ensemble weights: {best_weights}")
    logger.info(f"Best MSE: {study.best_value:.6f}")
    
    return best_weights

def adaptive_learning_rate_schedule(initial_lr: float, epoch: int, performance_history: List[float]) -> float:
    """
    Adaptive learning rate based on performance history
    
    Args:
        initial_lr: Initial learning rate
        epoch: Current epoch
        performance_history: List of recent performance metrics
    
    Returns:
        Adjusted learning rate
    """
    if len(performance_history) < 5:
        return initial_lr
    
    # Check if performance is plateauing
    recent_performance = performance_history[-5:]
    performance_std = np.std(recent_performance)
    
    if performance_std < 0.001:  # Performance plateaued
        # Reduce learning rate
        new_lr = initial_lr * 0.5
        logger.info(f"Performance plateaued (std={performance_std:.6f}), reducing LR: {initial_lr:.6f} -> {new_lr:.6f}")
        return new_lr
    
    # Check if performance is degrading
    if len(performance_history) >= 10:
        recent_avg = np.mean(performance_history[-5:])
        older_avg = np.mean(performance_history[-10:-5])
        
        if recent_avg > older_avg * 1.1:  # Performance degraded by 10%
            new_lr = initial_lr * 0.7
            logger.info(f"Performance degrading, reducing LR: {initial_lr:.6f} -> {new_lr:.6f}")
            return new_lr
    
    return initial_lr

def save_best_hyperparameters(conn, symbol: str, model_type: str, hyperparameters: Dict):
    """
    Save best hyperparameters to database for future use
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        model_type: Model type (xgboost, lightgbm, etc.)
        hyperparameters: Dict of hyperparameters
    """
    cursor = conn.cursor()
    
    # Store as JSON
    params_json = json.dumps(hyperparameters)
    
    # Check if entry exists
    cursor.execute("""
        SELECT id FROM model_hyperparameters
        WHERE stock_symbol = %s AND model_type = %s
    """, (symbol, model_type))
    
    existing = cursor.fetchone()
    
    if existing:
        # Update existing
        cursor.execute("""
            UPDATE model_hyperparameters
            SET hyperparameters = %s, updated_at = NOW()
            WHERE stock_symbol = %s AND model_type = %s
        """, (params_json, symbol, model_type))
    else:
        # Insert new
        cursor.execute("""
            INSERT INTO model_hyperparameters (stock_symbol, model_type, hyperparameters)
            VALUES (%s, %s, %s)
        """, (symbol, model_type, params_json))
    
    conn.commit()
    cursor.close()
    
    logger.info(f"Saved best hyperparameters for {symbol} ({model_type})")

def load_best_hyperparameters(conn, symbol: str, model_type: str) -> Dict:
    """
    Load best hyperparameters from database
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        model_type: Model type
    
    Returns:
        Hyperparameters dict or None if not found
    """
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT hyperparameters
        FROM model_hyperparameters
        WHERE stock_symbol = %s AND model_type = %s
    """, (symbol, model_type))
    
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        params = json.loads(result['hyperparameters'])
        logger.info(f"Loaded saved hyperparameters for {symbol} ({model_type})")
        return params
    
    return None

def main():
    """Test hyperparameter optimization"""
    from ml.backtest_and_train import load_stock_data, calculate_technical_features, prepare_features_and_target
    
    # Load sample data
    symbol = 'AAPL'
    logger.info(f"Testing hyperparameter optimization for {symbol}")
    
    df = load_stock_data(symbol, years=2)
    df = calculate_technical_features(df)
    X, y, feature_cols = prepare_features_and_target(df, horizon_days=1)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Optimize XGBoost
    best_xgb_params = optimize_xgboost_params(X_train, y_train, X_val, y_val, n_trials=20)
    print("\nBest XGBoost params:")
    print(json.dumps(best_xgb_params, indent=2))
    
    # Optimize LightGBM
    best_lgb_params = optimize_lightgbm_params(X_train, y_train, X_val, y_val, n_trials=20)
    print("\nBest LightGBM params:")
    print(json.dumps(best_lgb_params, indent=2))

if __name__ == '__main__':
    main()
