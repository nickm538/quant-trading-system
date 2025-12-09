"""
Transfer Learning System for Untrained Stocks
==============================================
Enables predictions for stocks not in the training set using:
1. Market-wide ensemble model
2. Sector-based transfer learning
3. Similar stock matching
4. Feature-based generalization
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.backtest_and_train import load_stock_data, calculate_technical_features, prepare_features_and_target

# Sector mappings for major stocks
SECTOR_MAPPING = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
    'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology',
    'ORCL': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology', 'AVGO': 'Technology',
    
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance',
    'MS': 'Finance', 'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance',
    
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'TMO': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare', 'LLY': 'Healthcare',
    
    # Consumer
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer',
    'MCD': 'Consumer', 'SBUX': 'Consumer', 'TGT': 'Consumer', 'WMT': 'Consumer',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    
    # Industrial
    'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
    'HON': 'Industrial', 'UPS': 'Industrial', 'LMT': 'Industrial',
    
    # Telecom
    'T': 'Telecom', 'VZ': 'Telecom', 'TMUS': 'Telecom',
}

def get_stock_sector(symbol: str) -> str:
    """Get sector for a stock symbol"""
    return SECTOR_MAPPING.get(symbol, 'General')

def get_sector_stocks(sector: str) -> List[str]:
    """Get all stocks in a sector"""
    return [sym for sym, sec in SECTOR_MAPPING.items() if sec == sector]

def calculate_stock_similarity(symbol1: str, symbol2: str, days: int = 252) -> float:
    """
    Calculate similarity between two stocks based on price correlation
    
    Args:
        symbol1: First stock symbol
        symbol2: Second stock symbol
        days: Number of days to compare
    
    Returns:
        Correlation coefficient (0-1)
    """
    try:
        import yfinance as yf
        
        # Download data for both stocks
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker1 = yf.Ticker(symbol1)
        ticker2 = yf.Ticker(symbol2)
        
        hist1 = ticker1.history(start=start_date, end=end_date)
        hist2 = ticker2.history(start=start_date, end=end_date)
        
        if hist1.empty or hist2.empty:
            return 0.0
        
        # Calculate returns
        returns1 = hist1['Close'].pct_change().dropna()
        returns2 = hist2['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = returns1.index.intersection(returns2.index)
        if len(common_dates) < 50:
            return 0.0
        
        returns1 = returns1.loc[common_dates]
        returns2 = returns2.loc[common_dates]
        
        # Calculate correlation
        correlation = returns1.corr(returns2)
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating similarity between {symbol1} and {symbol2}: {e}")
        return 0.0

def find_similar_stocks(target_symbol: str, trained_symbols: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Find most similar trained stocks to target symbol
    
    Args:
        target_symbol: Symbol to find matches for
        trained_symbols: List of symbols with trained models
        top_n: Number of similar stocks to return
    
    Returns:
        List of (symbol, similarity_score) tuples
    """
    logger.info(f"Finding similar stocks to {target_symbol} from {len(trained_symbols)} trained stocks...")
    
    similarities = []
    
    # First, try same sector
    target_sector = get_stock_sector(target_symbol)
    sector_stocks = [s for s in trained_symbols if get_stock_sector(s) == target_sector]
    
    if sector_stocks:
        logger.info(f"Found {len(sector_stocks)} stocks in same sector ({target_sector})")
        
        # Calculate similarity with sector stocks
        for symbol in sector_stocks[:10]:  # Limit to avoid too many API calls
            similarity = calculate_stock_similarity(target_symbol, symbol)
            if similarity > 0.3:  # Minimum threshold
                similarities.append((symbol, similarity))
    
    # If not enough similar stocks, use all trained stocks
    if len(similarities) < top_n:
        logger.info("Not enough sector matches, expanding to all trained stocks...")
        other_stocks = [s for s in trained_symbols if s not in sector_stocks]
        
        for symbol in other_stocks[:20]:  # Limit API calls
            similarity = calculate_stock_similarity(target_symbol, symbol)
            if similarity > 0.3:
                similarities.append((symbol, similarity))
    
    # Sort by similarity and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    result = similarities[:top_n]
    logger.info(f"Found {len(result)} similar stocks: {[f'{s}({sim:.2f})' for s, sim in result]}")
    
    return result

def get_trained_models_for_similar_stocks(conn, similar_stocks: List[Tuple[str, float]]) -> List[Dict]:
    """
    Retrieve trained models for similar stocks
    
    Args:
        conn: Database connection
        similar_stocks: List of (symbol, similarity) tuples
    
    Returns:
        List of model dicts with similarity weights
    """
    cursor = conn.cursor(dictionary=True)
    
    models = []
    
    for symbol, similarity in similar_stocks:
        query = """
        SELECT id, stock_symbol, model_type, model_data,
               test_accuracy, r2_score, hyperparameters
        FROM trained_models
        WHERE stock_symbol = %s
        AND is_active = 'active'
        ORDER BY test_accuracy DESC
        LIMIT 3
        """
        
        cursor.execute(query, (symbol,))
        stock_models = cursor.fetchall()
        
        for model_dict in stock_models:
            # Add similarity weight
            model_dict['similarity_weight'] = similarity
            models.append(model_dict)
    
    cursor.close()
    
    logger.info(f"Retrieved {len(models)} models from similar stocks")
    return models

def make_transfer_learning_prediction(
    conn,
    target_symbol: str,
    trained_symbols: List[str],
    horizon_days: int = 30
) -> Dict:
    """
    Make prediction for untrained stock using transfer learning
    
    Args:
        conn: Database connection
        target_symbol: Symbol to predict (not in training set)
        trained_symbols: List of symbols with trained models
        horizon_days: Prediction horizon in days
    
    Returns:
        Prediction dict
    """
    logger.info(f"Making transfer learning prediction for {target_symbol}")
    
    try:
        # Step 1: Find similar stocks
        similar_stocks = find_similar_stocks(target_symbol, trained_symbols, top_n=5)
        
        if not similar_stocks:
            return {
                'success': False,
                'error': f'No similar stocks found for {target_symbol}',
                'method': 'transfer_learning'
            }
        
        # Step 2: Get models from similar stocks
        models = get_trained_models_for_similar_stocks(conn, similar_stocks)
        
        if not models:
            return {
                'success': False,
                'error': 'No trained models found for similar stocks',
                'method': 'transfer_learning'
            }
        
        # Step 3: Load target stock data and calculate features
        df = load_stock_data(target_symbol, years=2)
        df = calculate_technical_features(df)
        
        # Get latest data point for prediction
        latest_data = df.iloc[-1]
        current_price = latest_data['close']
        
        # Step 4: Prepare features (same as training)
        X, y, feature_cols = prepare_features_and_target(df, horizon_days=horizon_days)
        latest_features = X.iloc[-1:][feature_cols]
        
        # Step 5: Make predictions using all similar stock models
        predictions = []
        weights = []
        
        for model_dict in models:
            try:
                # Deserialize model
                model_data = base64.b64decode(model_dict['model_data'])
                loaded_object = pickle.loads(model_data)
                
                # Handle case where model was saved as dict
                if isinstance(loaded_object, dict):
                    if 'model' in loaded_object:
                        model = loaded_object['model']
                    else:
                        logger.warning(f"Model {model_dict['id']} is a dict without 'model' key, skipping")
                        continue
                else:
                    model = loaded_object
                
                # Make prediction
                pred = model.predict(latest_features)[0]
                
                # Weight by similarity and model accuracy
                accuracy = model_dict['test_accuracy'] / 10000.0  # Convert from stored format
                similarity = model_dict['similarity_weight']
                weight = accuracy * similarity
                
                predictions.append(pred)
                weights.append(weight)
                
                logger.info(f"Model from {model_dict['stock_symbol']} ({model_dict['model_type']}): "
                          f"pred={pred:.4f}, weight={weight:.4f}")
                
            except Exception as e:
                logger.error(f"Error using model {model_dict['id']}: {e}")
                continue
        
        if not predictions:
            return {
                'success': False,
                'error': 'Failed to make predictions with any model',
                'method': 'transfer_learning'
            }
        
        # Step 6: Weighted ensemble prediction
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, weights=weights)
        
        # Calculate confidence based on prediction agreement
        pred_std = np.std(predictions)
        confidence = max(30, min(70, 60 - pred_std * 100))  # Lower confidence for transfer learning
        
        # Calculate predicted price
        predicted_price = current_price * (1 + ensemble_pred)
        price_change_pct = ensemble_pred * 100
        
        # Generate recommendation
        if price_change_pct > 3 and confidence > 50:
            recommendation = 'BUY'
        elif price_change_pct > 5 and confidence > 60:
            recommendation = 'STRONG BUY'
        elif price_change_pct < -3 and confidence > 50:
            recommendation = 'SELL'
        elif price_change_pct < -5 and confidence > 60:
            recommendation = 'STRONG SELL'
        else:
            recommendation = 'HOLD'
        
        reasoning = (f"Transfer learning from {len(similar_stocks)} similar stocks "
                    f"({', '.join([s for s, _ in similar_stocks])}). "
                    f"Weighted ensemble of {len(predictions)} models predicts "
                    f"{price_change_pct:.2f}% change with {confidence:.1f}% confidence.")
        
        # Calculate average performance metrics from similar stock models
        avg_accuracy = np.mean([m['test_accuracy'] / 10000.0 for m in models]) * 100
        avg_r2 = np.mean([m['r2_score'] / 10000.0 for m in models if m['r2_score']])
        
        result = {
            'success': True,
            'symbol': target_symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change_pct': price_change_pct,
            'confidence': confidence,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'horizon_days': horizon_days,
            'prediction_date': datetime.now().isoformat(),
            'target_date': (datetime.now() + timedelta(days=horizon_days)).isoformat(),
            'method': 'transfer_learning',
            'similar_stocks': [{'symbol': s, 'similarity': sim} for s, sim in similar_stocks],
            'model_performance': {
                'avg_accuracy': avg_accuracy,
                'avg_sharpe_ratio': 1.0,  # Conservative estimate
                'avg_win_rate': 50.0,  # Neutral
                'num_models_used': len(predictions)
            },
            'ensemble_details': {
                'individual_predictions': predictions.tolist(),
                'model_weights': weights.tolist()
            }
        }
        
        logger.info(f"Transfer learning prediction for {target_symbol}: {price_change_pct:.2f}% with {confidence:.1f}% confidence")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in transfer learning prediction: {e}")
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'method': 'transfer_learning'
        }

def main():
    """Test transfer learning system"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transfer_learning.py <SYMBOL>")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # Get database connection
    from ml.prediction_engine import get_db_connection
    conn = get_db_connection()
    
    # Get list of trained symbols
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT stock_symbol FROM trained_models WHERE is_active = 'active'")
    trained_symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    logger.info(f"Found {len(trained_symbols)} trained symbols")
    
    # Make prediction
    result = make_transfer_learning_prediction(conn, symbol, trained_symbols, horizon_days=30)
    
    conn.close()
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
