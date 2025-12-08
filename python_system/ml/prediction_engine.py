"""
Advanced ML Prediction Engine
Retrieves trained models from database and generates world-class predictions
"""

import os
import sys
import json
import pickle
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.backtest_and_train import load_stock_data, calculate_technical_features, prepare_features_and_target

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

def get_best_models_for_stock(conn, symbol: str, top_n: int = 3) -> List[Dict]:
    """
    Retrieve the best performing models for a stock from database
    Returns top N models sorted by test accuracy
    """
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        tm.id, tm.stock_symbol, tm.model_type, tm.version,
        tm.model_data, tm.test_accuracy, tm.hyperparameters,
        tm.feature_importance, tm.trained_at,
        br.sharpe_ratio, br.win_rate, br.profit_factor
    FROM trained_models tm
    LEFT JOIN backtesting_results br ON tm.id = br.model_id
    WHERE tm.stock_symbol = %s AND tm.is_active = 'active'
    ORDER BY tm.test_accuracy DESC, br.sharpe_ratio DESC
    LIMIT %s
    """
    
    cursor.execute(query, (symbol, top_n))
    models = cursor.fetchall()
    cursor.close()
    
    return models

def deserialize_model(model_data_b64: str):
    """Deserialize model from base64 encoded pickle"""
    try:
        model_bytes = base64.b64decode(model_data_b64)
        model = pickle.loads(model_bytes)
        return model
    except Exception as e:
        print(f"Error deserializing model: {e}")
        return None

def generate_ensemble_prediction(models: List[Dict], features: np.ndarray) -> Dict:
    """
    Generate ensemble prediction using multiple models
    Uses weighted average based on model accuracy
    """
    predictions = []
    weights = []
    
    for model_info in models:
        try:
            # Deserialize model
            if model_info.get('model_data'):
                model = deserialize_model(model_info['model_data'])
                if model is None:
                    continue
                
                # Make prediction
                pred = model.predict(features)
                predictions.append(pred[0])
                
                # Weight by test accuracy
                accuracy = model_info['test_accuracy'] / 10000.0  # Convert from stored format
                weights.append(accuracy)
        except Exception as e:
            print(f"Error making prediction with model {model_info['id']}: {e}")
            continue
    
    if not predictions:
        return None
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted ensemble prediction
    ensemble_pred = np.average(predictions, weights=weights)
    
    # Calculate confidence based on agreement between models
    pred_std = np.std(predictions)
    confidence = max(0, min(100, 100 - (pred_std * 100)))  # Higher agreement = higher confidence
    
    return {
        'prediction': float(ensemble_pred),
        'confidence': float(confidence),
        'individual_predictions': [float(p) for p in predictions],
        'model_weights': weights.tolist(),
        'num_models': len(predictions)
    }

def make_price_prediction(symbol: str, horizon_days: int = 5) -> Dict:
    """
    Make advanced price prediction for a stock
    
    Args:
        symbol: Stock ticker symbol
        horizon_days: Number of days ahead to predict
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Connect to database
        conn = get_db_connection()
        
        # Get best models for this stock
        models = get_best_models_for_stock(conn, symbol, top_n=5)
        
        if not models:
            return {
                'success': False,
                'error': f'No trained models found for {symbol}. Please train models first.'
            }
        
        # Download latest data
        df = load_stock_data(symbol, years=2)
        
        # Calculate features
        df = calculate_technical_features(df)
        
        # Prepare features (use last row for prediction)
        X, _ = prepare_features_and_target(df)
        latest_features = X[-1:, :]  # Last row
        
        # Generate ensemble prediction
        ensemble_result = generate_ensemble_prediction(models, latest_features)
        
        if ensemble_result is None:
            return {
                'success': False,
                'error': 'Failed to generate prediction from models'
            }
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Calculate predicted price change
        predicted_return = ensemble_result['prediction']
        predicted_price = current_price * (1 + predicted_return)
        
        # Calculate price targets
        price_change_pct = (predicted_return * 100)
        
        # Get model performance metrics
        avg_accuracy = np.mean([m['test_accuracy'] / 10000.0 for m in models])
        avg_sharpe = np.mean([m['sharpe_ratio'] / 10000.0 if m['sharpe_ratio'] else 0 for m in models])
        avg_win_rate = np.mean([m['win_rate'] / 10000.0 if m['win_rate'] else 0 for m in models])
        
        # Generate recommendation
        if predicted_return > 0.03:  # > 3% gain
            recommendation = 'STRONG BUY'
            reasoning = f"ML models predict {price_change_pct:.2f}% gain with {ensemble_result['confidence']:.1f}% confidence"
        elif predicted_return > 0.01:  # > 1% gain
            recommendation = 'BUY'
            reasoning = f"ML models predict {price_change_pct:.2f}% gain with {ensemble_result['confidence']:.1f}% confidence"
        elif predicted_return < -0.03:  # < -3% loss
            recommendation = 'STRONG SELL'
            reasoning = f"ML models predict {price_change_pct:.2f}% decline with {ensemble_result['confidence']:.1f}% confidence"
        elif predicted_return < -0.01:  # < -1% loss
            recommendation = 'SELL'
            reasoning = f"ML models predict {price_change_pct:.2f}% decline with {ensemble_result['confidence']:.1f}% confidence"
        else:
            recommendation = 'HOLD'
            reasoning = f"ML models predict minimal movement ({price_change_pct:.2f}%) with {ensemble_result['confidence']:.1f}% confidence"
        
        # Save prediction to database
        save_prediction_to_db(
            conn, 
            models[0]['id'],  # Use best model ID
            symbol,
            predicted_price,
            ensemble_result['confidence'],
            horizon_days
        )
        
        conn.close()
        
        return {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': price_change_pct,
            'confidence': ensemble_result['confidence'],
            'recommendation': recommendation,
            'reasoning': reasoning,
            'horizon_days': horizon_days,
            'prediction_date': datetime.now().isoformat(),
            'target_date': (datetime.now() + timedelta(days=horizon_days)).isoformat(),
            'model_performance': {
                'avg_accuracy': avg_accuracy * 100,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_win_rate': avg_win_rate * 100,
                'num_models_used': ensemble_result['num_models']
            },
            'ensemble_details': {
                'individual_predictions': ensemble_result['individual_predictions'],
                'model_weights': ensemble_result['model_weights']
            }
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def save_prediction_to_db(conn, model_id: int, symbol: str, predicted_price: float, confidence: float, horizon_days: int):
    """Save prediction to database for future validation"""
    cursor = conn.cursor()
    
    query = """
    INSERT INTO model_predictions (
        model_id, stock_symbol, prediction_date, target_date,
        predicted_price, predicted_low, predicted_high, confidence, status
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Calculate price range based on confidence
    price_range = predicted_price * (1 - confidence/100) * 0.1
    
    values = (
        model_id,
        symbol,
        datetime.now(),
        datetime.now() + timedelta(days=horizon_days),
        int(predicted_price * 100),  # Store as cents
        int((predicted_price - price_range) * 100),
        int((predicted_price + price_range) * 100),
        int(confidence * 100),
        'pending'
    )
    
    cursor.execute(query, values)
    conn.commit()
    cursor.close()

def get_prediction_accuracy_report(conn, symbol: Optional[str] = None, days_back: int = 30) -> Dict:
    """
    Generate report on prediction accuracy over time
    Validates predictions against actual prices
    """
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        mp.id, mp.stock_symbol, mp.prediction_date, mp.target_date,
        mp.predicted_price, mp.actual_price, mp.confidence,
        mp.price_error, mp.percentage_error, mp.status
    FROM model_predictions mp
    WHERE mp.target_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
    """
    
    params = [days_back]
    if symbol:
        query += " AND mp.stock_symbol = %s"
        params.append(symbol)
    
    query += " ORDER BY mp.prediction_date DESC"
    
    cursor.execute(query, params)
    predictions = cursor.fetchall()
    cursor.close()
    
    if not predictions:
        return {
            'total_predictions': 0,
            'message': 'No predictions found in the specified period'
        }
    
    # Calculate accuracy metrics
    validated = [p for p in predictions if p['status'] == 'validated']
    
    if validated:
        errors = [abs(p['percentage_error']) / 100.0 for p in validated]
        avg_error = np.mean(errors)
        median_error = np.median(errors)
        
        # Predictions within 5% are considered accurate
        accurate_predictions = sum(1 for e in errors if e < 5.0)
        accuracy_rate = (accurate_predictions / len(validated)) * 100
    else:
        avg_error = median_error = accuracy_rate = 0
    
    return {
        'total_predictions': len(predictions),
        'validated_predictions': len(validated),
        'pending_predictions': len([p for p in predictions if p['status'] == 'pending']),
        'accuracy_rate': accuracy_rate,
        'avg_error_pct': avg_error,
        'median_error_pct': median_error,
        'predictions': predictions[:10]  # Return last 10
    }

if __name__ == "__main__":
    # Test prediction engine
    if len(sys.argv) < 2:
        print("Usage: python prediction_engine.py <SYMBOL> [horizon_days]")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    horizon_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Generating prediction for {symbol}...")
    result = make_price_prediction(symbol, horizon_days)
    
    print(json.dumps(result, indent=2))
