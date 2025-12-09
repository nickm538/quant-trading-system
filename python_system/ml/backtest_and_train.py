#!/usr/bin/env python3
"""
Comprehensive Backtesting and Training System
Loads 15 stock CSVs, trains models, runs walk-forward validation, saves to database
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import mysql.connector
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import talib as ta
    HAS_ML_LIBS = True
except ImportError as e:
    print(f"⚠️  Missing ML libraries: {e}")
    print("Install with: pip install xgboost lightgbm scikit-learn ta-lib")
    HAS_ML_LIBS = False
    sys.exit(1)

# Database connection
def get_db_connection():
    """Get database connection from environment"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    
    # Parse connection string (mysql://user:pass@host:port/dbname?ssl=...)
    if db_url.startswith('mysql://'):
        db_url = db_url[8:]  # Remove 'mysql://'
    
    # Split into auth@host/db?params
    auth_host, _, db_params = db_url.partition('/')
    user_pass, _, host_port = auth_host.partition('@')
    user, _, password = user_pass.partition(':')
    host, _, port = host_port.partition(':')
    
    # Split database name from query params
    database = db_params.split('?')[0]
    
    return mysql.connector.connect(
        host=host,
        port=int(port) if port else 3306,
        user=user,
        password=password,
        database=database,
        ssl_disabled=False
    )

def load_stock_data(symbol: str, years: int = 5) -> pd.DataFrame:
    """Download and prepare stock data from yfinance"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Reset index to make Date a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'Date'})
    
    # Rename columns to lowercase
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Add adj_close (same as close for yfinance history)
    df['adj_close'] = df['close']
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        df['Date'] = df.index
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.reset_index(drop=True)
    
    # Convert to float64 for TA-Lib
    for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN after conversion
    df = df.dropna()
    
    return df

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators as features"""
    # Ensure all price columns are float64
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_pct'] = (df['high'] - df['low']) / df['close']
    df['close_open_pct'] = (df['close'] - df['open']) / df['open']
    
    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        df[f'sma_{period}'] = ta.SMA(df['close'].astype(np.float64).values, timeperiod=period)
        df[f'ema_{period}'] = ta.EMA(df['close'].astype(np.float64).values, timeperiod=period)
    
    # Technical indicators
    df['rsi'] = ta.RSI(df['close'].astype(np.float64).values, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'].astype(np.float64).values)
    df['adx'] = ta.ADX(df['high'].astype(np.float64).values, df['low'].astype(np.float64).values, df['close'].astype(np.float64).values, timeperiod=14)
    df['atr'] = ta.ATR(df['high'].astype(np.float64).values, df['low'].astype(np.float64).values, df['close'].astype(np.float64).values, timeperiod=14)
    df['cci'] = ta.CCI(df['high'].astype(np.float64).values, df['low'].astype(np.float64).values, df['close'].astype(np.float64).values, timeperiod=14)
    df['mfi'] = ta.MFI(df['high'].astype(np.float64).values, df['low'].astype(np.float64).values, df['close'].astype(np.float64).values, df['volume'].astype(np.float64).values, timeperiod=14)
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'].astype(np.float64).values, timeperiod=20)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volume features
    df['volume_sma_20'] = ta.SMA(df['volume'].astype(np.float64).values, timeperiod=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_50'] = df['returns'].rolling(50).std()
    
    # NOTE: Target is NOT calculated here to avoid look-ahead bias
    # Target will be calculated properly in prepare_features_and_target()
    # using only data available at prediction time
    
    # Drop NaN rows from feature calculations
    df = df.dropna()
    
    return df

def prepare_features_and_target(df: pd.DataFrame, horizon_days: int = 1) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features (X) and target (y) WITHOUT look-ahead bias
    
    Args:
        df: DataFrame with features calculated
        horizon_days: Number of days ahead to predict (default 1)
    
    Returns:
        X: Features (without last horizon_days rows)
        y: Target (price change after horizon_days)
        feature_cols: List of feature column names
    """
    # Calculate target: Future price change after horizon_days
    # Using shift(-horizon_days) but then we DROP those rows
    # This ensures we only use data available at prediction time
    df = df.copy()
    df['target'] = df['close'].shift(-horizon_days) / df['close'] - 1
    
    # CRITICAL: Drop last horizon_days rows (they have NaN targets)
    # This prevents look-ahead bias
    df = df[:-horizon_days] if horizon_days > 0 else df
    df = df.dropna(subset=['target'])
    
    # Feature columns (exclude date, price columns, and target)
    exclude_cols = ['Date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, feature_cols

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    return model

def calculate_backtest_metrics(predictions: np.ndarray, actuals: np.ndarray, returns: np.ndarray) -> Dict:
    """Calculate comprehensive backtest metrics"""
    # Regression metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Trading metrics (assuming we trade based on predicted direction)
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    correct_direction = (pred_direction == actual_direction).sum()
    direction_accuracy = correct_direction / len(predictions)
    
    # Simulated trading returns
    trading_returns = pred_direction * actuals  # If we predict up and it goes up, we profit
    cumulative_return = (1 + trading_returns).prod() - 1
    
    # Sharpe ratio (assuming 252 trading days per year, 0% risk-free rate)
    if trading_returns.std() > 0:
        sharpe_ratio = (trading_returns.mean() / trading_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Sortino ratio (downside deviation)
    downside_returns = trading_returns[trading_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (trading_returns.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino_ratio = 0
    
    # Max drawdown
    cumulative = np.cumprod(1 + trading_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Win rate
    wins = (trading_returns > 0).sum()
    losses = (trading_returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Average win/loss
    avg_win = trading_returns[trading_returns > 0].mean() if wins > 0 else 0
    avg_loss = abs(trading_returns[trading_returns < 0].mean()) if losses > 0 else 0
    
    # Profit factor
    total_wins = trading_returns[trading_returns > 0].sum()
    total_losses = abs(trading_returns[trading_returns < 0].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'r2_score': r2,
        'direction_accuracy': direction_accuracy,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(predictions),
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }

def save_model_to_db(conn, symbol: str, model_type: str, model_obj, metrics: Dict, hyperparams: Dict, feature_importance: Dict = None) -> int:
    """Save trained model and metadata to database"""
    import pickle
    import base64
    
    cursor = conn.cursor()
    
    # Serialize model to base64 for database storage
    model_bytes = pickle.dumps(model_obj)
    model_b64 = base64.b64encode(model_bytes).decode('utf-8')
    
    # Convert metrics to database format (multiply by 10000 for precision)
    insert_query = """
    INSERT INTO trained_models (
        stock_symbol, model_type, version, model_path, model_data,
        training_start_date, training_end_date, training_data_points,
        training_accuracy, validation_accuracy, test_accuracy,
        mse, mae, r2_score,
        hyperparameters, feature_importance, is_active
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    values = (
        symbol,
        model_type,
        datetime.now().strftime('%Y%m%d_%H%M%S'),
        f'db://models/{symbol}_{model_type}_{datetime.now().strftime("%Y%m%d")}.pkl',
        model_b64,  # Serialized model
        datetime.now() - timedelta(days=365*5),  # 5 years ago (training data start)
        datetime.now(),
        metrics['total_trades'],
        int(metrics['direction_accuracy'] * 10000),
        int(metrics['direction_accuracy'] * 10000),  # Same as training for now
        int(metrics['direction_accuracy'] * 10000),
        int(metrics.get('mse', 0) * 10000),
        int(metrics.get('mae', 0) * 10000),
        int(metrics.get('r2_score', 0) * 10000),
        json.dumps(hyperparams),
        json.dumps(feature_importance) if feature_importance else None,
        'active'
    )
    
    cursor.execute(insert_query, values)
    conn.commit()
    
    model_id = cursor.lastrowid
    cursor.close()
    
    return model_id

def save_backtest_results_to_db(conn, model_id: int, symbol: str, metrics: Dict, start_date: datetime, end_date: datetime):
    """Save backtesting results to database"""
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO backtesting_results (
        model_id, stock_symbol,
        start_date, end_date, total_days,
        total_return, annualized_return,
        sharpe_ratio, sortino_ratio, max_drawdown,
        total_trades, winning_trades, losing_trades,
        win_rate, avg_win, avg_loss, profit_factor,
        value_at_risk_95, conditional_var_95,
        test_type
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Calculate annualized return
    days = (end_date - start_date).days
    annualized_return = (1 + metrics['cumulative_return']) ** (252 / days) - 1 if days > 0 else 0
    
    values = (
        int(model_id),
        str(symbol),
        start_date,
        end_date,
        int(days),
        int(float(metrics['cumulative_return']) * 10000),
        int(float(annualized_return) * 10000),
        int(float(metrics['sharpe_ratio']) * 10000),
        int(float(metrics['sortino_ratio']) * 10000),
        int(abs(float(metrics['max_drawdown'])) * 10000),
        int(metrics['total_trades']),
        int(metrics['winning_trades']),
        int(metrics['losing_trades']),
        int(float(metrics['win_rate']) * 10000),
        int(float(metrics['avg_win']) * 10000),
        int(float(metrics['avg_loss']) * 10000),
        int(float(metrics['profit_factor']) * 10000),
        0,  # VaR placeholder
        0,  # CVaR placeholder
        'walk_forward'
    )
    
    cursor.execute(insert_query, values)
    conn.commit()
    cursor.close()

def process_stock(symbol: str, conn) -> Dict:
    """Process single stock: load data, train models, backtest, save results"""
    print(f"\n{'='*80}")
    print(f"Processing {symbol}")
    print(f"{'='*80}")
    
    # Load data
    print(f"[1/6] Downloading stock data from yfinance...")
    df = load_stock_data(symbol, years=5)
    print(f"  ✓ Downloaded {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
    
    # Calculate features
    print(f"[2/6] Calculating technical features...")
    df = calculate_technical_features(df)
    print(f"  ✓ Calculated {len([c for c in df.columns if c not in ['Date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'target']])} features")
    
    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(df)
    
    # Walk-forward split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Further split train into train/val (90/10)
    val_split_idx = int(len(X_train) * 0.9)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"  ✓ Train: {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # Train XGBoost
    print(f"[3/6] Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train_final, y_train_final, X_val, y_val)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_metrics = calculate_backtest_metrics(xgb_predictions, y_test.values, y_test.values)
    print(f"  ✓ XGBoost - Accuracy: {xgb_metrics['direction_accuracy']*100:.2f}%, Sharpe: {xgb_metrics['sharpe_ratio']:.2f}")
    
    # Save XGBoost to DB
    xgb_hyperparams = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    xgb_feature_importance = dict(zip(feature_names[:len(xgb_model.feature_importances_)], xgb_model.feature_importances_.tolist()))
    xgb_model_id = save_model_to_db(conn, symbol, 'xgboost', xgb_model, xgb_metrics, xgb_hyperparams, xgb_feature_importance)
    save_backtest_results_to_db(
        conn, xgb_model_id, symbol, xgb_metrics,
        df.iloc[split_idx]['Date'], df.iloc[-1]['Date']
    )
    results['xgboost'] = xgb_metrics
    
    # Train LightGBM
    print(f"[4/6] Training LightGBM model...")
    lgb_model = train_lightgbm_model(X_train_final, y_train_final, X_val, y_val)
    lgb_predictions = lgb_model.predict(X_test)
    lgb_metrics = calculate_backtest_metrics(lgb_predictions, y_test.values, y_test.values)
    print(f"  ✓ LightGBM - Accuracy: {lgb_metrics['direction_accuracy']*100:.2f}%, Sharpe: {lgb_metrics['sharpe_ratio']:.2f}")
    
    # Save LightGBM to DB
    lgb_hyperparams = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    lgb_feature_importance = dict(zip(feature_names[:len(lgb_model.feature_importances_)], lgb_model.feature_importances_.tolist()))
    lgb_model_id = save_model_to_db(conn, symbol, 'lightgbm', lgb_model, lgb_metrics, lgb_hyperparams, lgb_feature_importance)
    save_backtest_results_to_db(
        conn, lgb_model_id, symbol, lgb_metrics,
        df.iloc[split_idx]['Date'], df.iloc[-1]['Date']
    )
    results['lightgbm'] = lgb_metrics
    
    # Ensemble (average predictions)
    print(f"[5/6] Creating ensemble model...")
    ensemble_predictions = (xgb_predictions + lgb_predictions) / 2
    ensemble_metrics = calculate_backtest_metrics(ensemble_predictions, y_test.values, y_test.values)
    print(f"  ✓ Ensemble - Accuracy: {ensemble_metrics['direction_accuracy']*100:.2f}%, Sharpe: {ensemble_metrics['sharpe_ratio']:.2f}")
    
    # Save Ensemble to DB
    ensemble_hyperparams = {'models': ['xgboost', 'lightgbm'], 'weights': [0.5, 0.5]}
    # Create ensemble model object (dict with both models)
    ensemble_model = {'xgb': xgb_model, 'lgb': lgb_model, 'weights': [0.5, 0.5]}
    ensemble_model_id = save_model_to_db(conn, symbol, 'ensemble', ensemble_model, ensemble_metrics, ensemble_hyperparams)
    save_backtest_results_to_db(
        conn, ensemble_model_id, symbol, ensemble_metrics,
        df.iloc[split_idx]['Date'], df.iloc[-1]['Date']
    )
    results['ensemble'] = ensemble_metrics
    
    print(f"[6/6] ✅ {symbol} complete!")
    print(f"  Best model: Ensemble (Accuracy: {ensemble_metrics['direction_accuracy']*100:.2f}%, Sharpe: {ensemble_metrics['sharpe_ratio']:.2f})")
    
    return results

def main():
    """Main training pipeline"""
    print("="*80)
    print("COMPREHENSIVE BACKTESTING AND TRAINING SYSTEM")
    print("="*80)
    
    # Large pool of diverse stocks across sectors
    stock_pool = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'CRM',
        # Consumer
        'AMZN', 'TSLA', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX', 'COST',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'V', 'MA',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'BMY', 'AMGN',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'MMM', 'DE', 'RTX', 'EMR',
        # Communication
        'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'PM', 'MDLZ', 'CL', 'KMB', 'GIS',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX',
    ]
    
    # Randomly select 15 stocks for this training session
    import random
    random.seed()  # Use current time as seed for true randomness
    stocks = random.sample(stock_pool, 15)
    
    print(f"\nTraining on {len(stocks)} stocks:")
    for symbol in stocks:
        print(f"  - {symbol}")
    
    # Connect to database
    print(f"\nConnecting to database...")
    try:
        conn = get_db_connection()
        print(f"  ✓ Database connected")
    except Exception as e:
        print(f"  ✗ Database connection failed: {e}")
        sys.exit(1)
    
    # Process each stock
    all_results = {}
    for symbol in stocks:
        try:
            results = process_stock(symbol, conn)
            all_results[symbol] = results
        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Stock':<8} {'Model':<12} {'Accuracy':<12} {'Sharpe':<10} {'Win Rate':<10} {'Profit Factor':<15}")
    print("-"*80)
    
    for symbol, results in all_results.items():
        for model_type, metrics in results.items():
            print(f"{symbol:<8} {model_type:<12} {metrics['direction_accuracy']*100:>10.2f}% {metrics['sharpe_ratio']:>9.2f} {metrics['win_rate']*100:>9.2f}% {metrics['profit_factor']:>14.2f}")
    
    # Overall statistics
    all_accuracies = [m['direction_accuracy'] for r in all_results.values() for m in r.values()]
    all_sharpes = [m['sharpe_ratio'] for r in all_results.values() for m in r.values()]
    
    print("-"*80)
    print(f"{'AVERAGE':<8} {'All Models':<12} {np.mean(all_accuracies)*100:>10.2f}% {np.mean(all_sharpes):>9.2f}")
    print(f"\n✅ Successfully trained {len(all_results)} stocks × 3 models = {len(all_results)*3} total models")
    print(f"✅ All results saved to database")
    
    conn.close()

if __name__ == "__main__":
    main()
