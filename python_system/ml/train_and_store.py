#!/usr/bin/env python3
"""
Train ML models on selected stocks and store in database
Reads from stock_analysis_results.csv (top 15 stocks)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import mysql.connector
from typing import Optional

# Add paths
sys.path.append('/opt/.manus/.sandbox-runtime')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_api import ApiClient
import yfinance as yf

# Database connection
def get_db_connection():
    """Get database connection from environment"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not set")
    
    # Parse connection string (mysql://user:pass@host:port/dbname?ssl=...)
    import re
    from urllib.parse import urlparse, parse_qs
    
    # Remove mysql:// prefix and parse
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
        ssl_disabled=False  # Enable SSL
    )

# Import ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_ML_LIBS = True
except ImportError:
    print("⚠️  XGBoost/LightGBM not installed. Install with: pip install xgboost lightgbm")
    HAS_ML_LIBS = False

def fetch_training_data(symbol: str, years: int = 2) -> pd.DataFrame:
    """Fetch historical data for training"""
    print(f"  Fetching {years} years of data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{years}y", interval="1d")
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Rename columns
    df = df.rename(columns={
        'Close': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    
    print(f"  ✓ Fetched {len(df)} data points")
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML models (NO DATA LEAKAGE)"""
    df = df.copy()
    
    # Price-based features (using only past data)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']
    
    # Volatility
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target: Next day's return (THIS IS WHAT WE PREDICT)
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("  Training XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    print("  Training LightGBM...")
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(0)]
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics with macro/micro balance"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Directional accuracy (did we predict the direction correctly?)
    direction_correct = np.sum((y_true > 0) == (y_pred > 0))
    direction_accuracy = direction_correct / len(y_true)
    
    # Macro/Micro Balance Analysis
    # Macro: How well does it predict large movements (>2%)?
    large_moves_mask = np.abs(y_true) > 0.02
    if np.sum(large_moves_mask) > 0:
        macro_accuracy = np.sum((y_true[large_moves_mask] > 0) == (y_pred[large_moves_mask] > 0)) / np.sum(large_moves_mask)
    else:
        macro_accuracy = 0.5  # No large moves to predict
    
    # Micro: How well does it predict small movements (<1%)?
    small_moves_mask = np.abs(y_true) < 0.01
    if np.sum(small_moves_mask) > 0:
        micro_accuracy = np.sum((y_true[small_moves_mask] > 0) == (y_pred[small_moves_mask] > 0)) / np.sum(small_moves_mask)
    else:
        micro_accuracy = 0.5  # No small moves to predict
    
    # Balance score: Both macro and micro should be good (geometric mean)
    balance_score = np.sqrt(macro_accuracy * micro_accuracy)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'direction_accuracy': float(direction_accuracy),
        'macro_accuracy': float(macro_accuracy),  # Large moves (>2%)
        'micro_accuracy': float(micro_accuracy),  # Small moves (<1%)
        'balance_score': float(balance_score)     # Geometric mean of both
    }

def train_models_for_stock(symbol: str) -> dict:
    """Train all models for a single stock"""
    print(f"\n{'='*80}")
    print(f"Training models for {symbol}")
    print(f"{'='*80}")
    
    # Fetch data
    df = fetch_training_data(symbol, years=2)
    
    # Create features
    df_features = create_features(df)
    
    # Define feature columns (exclude target and date)
    feature_cols = [col for col in df_features.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
    
    # Split data (walk-forward: 70% train, 15% val, 15% test)
    n = len(df_features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train = df_features[feature_cols].iloc[:train_end].values
    y_train = df_features['target'].iloc[:train_end].values
    
    X_val = df_features[feature_cols].iloc[train_end:val_end].values
    y_val = df_features['target'].iloc[train_end:val_end].values
    
    X_test = df_features[feature_cols].iloc[val_end:].values
    y_test = df_features['target'].iloc[val_end:].values
    
    print(f"  Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # Train models
    results = {}
    
    if HAS_ML_LIBS:
        # XGBoost
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_pred_train = xgb_model.predict(X_train)
        xgb_pred_val = xgb_model.predict(X_val)
        xgb_pred_test = xgb_model.predict(X_test)
        
        results['xgboost'] = {
            'model': xgb_model,
            'train_metrics': calculate_metrics(y_train, xgb_pred_train),
            'val_metrics': calculate_metrics(y_val, xgb_pred_val),
            'test_metrics': calculate_metrics(y_test, xgb_pred_test),
            'feature_importance': dict(zip(feature_cols, xgb_model.feature_importances_.tolist())),
            'hyperparameters': xgb_model.get_params()
        }
        
        test_metrics = results['xgboost']['test_metrics']
        print(f"  ✓ XGBoost - Test: {test_metrics['direction_accuracy']*100:.2f}% | Macro: {test_metrics['macro_accuracy']*100:.1f}% | Micro: {test_metrics['micro_accuracy']*100:.1f}% | Balance: {test_metrics['balance_score']*100:.1f}%")
        
        # LightGBM
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_pred_train = lgb_model.predict(X_train)
        lgb_pred_val = lgb_model.predict(X_val)
        lgb_pred_test = lgb_model.predict(X_test)
        
        results['lightgbm'] = {
            'model': lgb_model,
            'train_metrics': calculate_metrics(y_train, lgb_pred_train),
            'val_metrics': calculate_metrics(y_val, lgb_pred_val),
            'test_metrics': calculate_metrics(y_test, lgb_pred_test),
            'feature_importance': dict(zip(feature_cols, lgb_model.feature_importances_.tolist())),
            'hyperparameters': lgb_model.get_params()
        }
        
        test_metrics_lgb = results['lightgbm']['test_metrics']
        print(f"  ✓ LightGBM - Test: {test_metrics_lgb['direction_accuracy']*100:.2f}% | Macro: {test_metrics_lgb['macro_accuracy']*100:.1f}% | Micro: {test_metrics_lgb['micro_accuracy']*100:.1f}% | Balance: {test_metrics_lgb['balance_score']*100:.1f}%")
    
    # Add metadata
    results['metadata'] = {
        'symbol': symbol,
        'training_start_date': df_features.index[0].isoformat(),
        'training_end_date': df_features.index[train_end-1].isoformat(),
        'total_data_points': n,
        'train_points': len(X_train),
        'val_points': len(X_val),
        'test_points': len(X_test),
        'features': feature_cols,
        'data': df_features  # Include data for prediction generation
    }
    
    return results

def save_model_to_file(model, symbol: str, model_type: str) -> str:
    """Save model to file and return path"""
    os.makedirs('/home/ubuntu/quant-trading-web/python_system/ml/trained_models', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol}_{model_type}_{timestamp}.pkl"
    filepath = f"/home/ubuntu/quant-trading-web/python_system/ml/trained_models/{filename}"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return filepath

def generate_and_store_predictions(symbol: str, model, model_type: str, model_id: int, latest_data: pd.DataFrame, feature_cols: list):
    """
    Generate 30-day predictions and store in model_predictions table
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Generate predictions for next 30 days
        predictions = []
        current_features = latest_data.iloc[-1].copy()
        
        for day in range(1, 31):
            # Prepare features - only use the feature columns that were used for training
            X = current_features[feature_cols].values.reshape(1, -1)
            
            # Predict
            pred_return = float(model.predict(X)[0])  # Convert numpy.float32 to Python float
            current_close = float(current_features['close'])  # Convert to Python float
            pred_price = current_close * (1 + pred_return)
            pred_direction = 1 if pred_return > 0 else -1
            
            predictions.append({
                'days_ahead': day,
                'predicted_price': float(pred_price),
                'predicted_return': float(pred_return),
                'predicted_direction': int(pred_direction),
                'confidence': float(min(abs(pred_return) * 100, 100.0))  # Simple confidence based on magnitude
            })
            
            # Update features for next prediction (simple approach)
            current_features['close'] = pred_price
            current_features['returns'] = pred_return
        
        # Store predictions in database
        prediction_date = datetime.now()
        for pred in predictions:
            target_date = prediction_date + timedelta(days=pred['days_ahead'])
            
            # Convert prices to cents (schema stores as int)
            predicted_price_cents = int(pred['predicted_price'] * 100)
            # Estimate high/low based on volatility (simple approach: ±2%)
            predicted_low_cents = int(pred['predicted_price'] * 0.98 * 100)
            predicted_high_cents = int(pred['predicted_price'] * 1.02 * 100)
            confidence_int = int(pred['confidence'] * 100)  # Store as basis points
            
            cursor.execute("""
                INSERT INTO model_predictions (
                    model_id, stock_symbol, prediction_date, target_date,
                    predicted_price, predicted_low, predicted_high,
                    confidence, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_id,
                symbol,
                prediction_date,
                target_date,
                predicted_price_cents,
                predicted_low_cents,
                predicted_high_cents,
                confidence_int,
                'pending',
                datetime.now(),
                datetime.now()
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"    ✓ Generated and stored 30-day predictions for {symbol} ({model_type})")
        return True
        
    except Exception as e:
        print(f"    ✗ Error generating predictions for {symbol}: {str(e)}")
        return False

def store_model_in_database(
    symbol: str,
    model_type: str,
    model_path: str,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    feature_importance: dict,
    hyperparameters: dict,
    metadata: dict
) -> int:
    """Store trained model metadata in database for continuous improvement"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if we already have a model for this stock+type
        cursor.execute(
            "SELECT id, test_accuracy FROM trained_models WHERE stock_symbol = %s AND model_type = %s ORDER BY trained_at DESC LIMIT 1",
            (symbol, model_type.lower())
        )
        existing = cursor.fetchone()
        
        # Only store if this model is better OR we don't have one yet
        new_accuracy = int(test_metrics['direction_accuracy'] * 10000)  # Store as basis points
        should_store = True
        
        if existing:
            existing_id, existing_accuracy = existing
            if new_accuracy <= existing_accuracy:
                print(f"    ⚠️  New model ({new_accuracy/100:.2f}%) not better than existing ({existing_accuracy/100:.2f}%), skipping database storage")
                should_store = False
            else:
                print(f"    ✓ New model ({new_accuracy/100:.2f}%) better than existing ({existing_accuracy/100:.2f}%), storing in database")
        
        if should_store:
            # Insert new model
            cursor.execute("""
                INSERT INTO trained_models (
                    stock_symbol, model_type, version, model_path,
                    training_accuracy, validation_accuracy, test_accuracy,
                    mse, mae, r2_score,
                    feature_importance, hyperparameters,
                    training_start_date, training_end_date, training_data_points
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                model_type.lower(),  # Must match enum: 'xgboost', 'lightgbm', 'lstm', 'ensemble'
                datetime.now().strftime('%Y%m%d_%H%M%S'),  # Version timestamp
                model_path,
                int(train_metrics['direction_accuracy'] * 10000),  # Basis points
                int(val_metrics['direction_accuracy'] * 10000),
                int(test_metrics['direction_accuracy'] * 10000),
                int(test_metrics['mse'] * 1000000),  # MSE in micro-units (6 decimals)
                int(test_metrics['mae'] * 1000000),  # MAE in micro-units (6 decimals)
                int(test_metrics['r2'] * 10000),  # R2 in basis points
                json.dumps(feature_importance),
                json.dumps(hyperparameters),
                metadata['training_start_date'],
                metadata['training_end_date'],
                metadata['total_data_points']
            ))
            
            model_id = cursor.lastrowid
            conn.commit()
            
            print(f"    ✓ Stored in database (ID: {model_id}, Accuracy: {new_accuracy/100:.2f}%)")
            return model_id
        else:
            return -1
    
    except Exception as e:
        print(f"    ✗ Database error: {str(e)}")
        conn.rollback()
        return -1
    finally:
        cursor.close()
        conn.close()

def main():
    """Main training loop"""
    print("\n" + "="*100)
    print("ML MODEL TRAINING PIPELINE")
    print("="*100)
    
    if not HAS_ML_LIBS:
        print("\n❌ Cannot train models without XGBoost and LightGBM")
        print("Install with: pip install xgboost lightgbm")
        return
    
    # Read CSV with top stocks
    csv_path = '/home/ubuntu/quant-trading-web/python_system/ml/stock_analysis_results.csv'
    df_stocks = pd.read_csv(csv_path)
    
    # Get top 15 stocks
    top_stocks = df_stocks.head(15)['symbol'].tolist()
    
    print(f"\nTraining models for {len(top_stocks)} stocks:")
    print(f"  {', '.join(top_stocks)}")
    
    all_results = []
    
    for i, symbol in enumerate(top_stocks, 1):
        print(f"\n[{i}/{len(top_stocks)}] Processing {symbol}...")
        
        try:
            results = train_models_for_stock(symbol)
            
            # Save models to files AND database
            for model_type in ['xgboost', 'lightgbm']:
                if model_type in results:
                    model = results[model_type]['model']
                    filepath = save_model_to_file(model, symbol, model_type)
                    results[model_type]['model_path'] = filepath
                    print(f"  ✓ Saved {model_type} model to {filepath}")
                    
                    # Store in database for continuous improvement
                    model_id = store_model_in_database(
                        symbol=symbol,
                        model_type=model_type,
                        model_path=filepath,
                        train_metrics=results[model_type]['train_metrics'],
                        val_metrics=results[model_type]['val_metrics'],
                        test_metrics=results[model_type]['test_metrics'],
                        feature_importance=results[model_type]['feature_importance'],
                        hyperparameters=results[model_type]['hyperparameters'],
                        metadata=results['metadata']
                    )
                    results[model_type]['database_id'] = model_id
                    
                    # Generate 30-day predictions if model was stored
                    if model_id:
                        generate_and_store_predictions(
                            symbol=symbol,
                            model=model,
                            model_type=model_type,
                            model_id=model_id,
                            latest_data=results['metadata']['data'],
                            feature_cols=results['metadata']['features']
                        )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  ✗ Error training {symbol}: {str(e)}")
            continue
    
    # Save summary
    summary_path = '/home/ubuntu/quant-trading-web/python_system/ml/training_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'stocks_trained': len(all_results),
        'results': []
    }
    
    for result in all_results:
        meta = result['metadata']
        summary['results'].append({
            'symbol': meta['symbol'],
            'xgboost_test_accuracy': result.get('xgboost', {}).get('test_metrics', {}).get('direction_accuracy', 0),
            'lightgbm_test_accuracy': result.get('lightgbm', {}).get('test_metrics', {}).get('direction_accuracy', 0),
            'training_points': meta['train_points'],
            'test_points': meta['test_points']
        })
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*100}")
    print(f"  Total stocks trained: {len(all_results)}")
    print(f"  Summary saved to: {summary_path}")
    print(f"\n✓ All models ready for deployment!")

if __name__ == "__main__":
    main()
