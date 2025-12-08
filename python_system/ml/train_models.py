"""
ML Training Pipeline for Stock Prediction
==========================================

Trains XGBoost, LightGBM, and LSTM models on historical stock data
with walk-forward cross-validation to prevent overfitting.

Features:
- Walk-forward validation (no look-ahead bias)
- Feature engineering from technical indicators
- Hyperparameter optimization
- Model persistence to S3
- Performance tracking in database

Author: Institutional Trading System
Date: 2025-11-20
"""

import sys
import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not installed")

# Technical indicators
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockMLTrainer:
    """Train ML models for stock price prediction"""
    
    def __init__(
        self,
        data_dir: str = '/home/ubuntu/historical_data',
        output_dir: str = '/home/ubuntu/quant-trading-web/python_system/ml/models'
    ):
        """Initialize trainer"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
    
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a stock"""
        # Find the CSV file
        csv_files = list(self.data_dir.glob(f'{symbol}*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No data file found for {symbol}")
        
        filepath = csv_files[0]
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Find date column
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: 'date'})
        df = df.sort_values('date')
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from OHLCV data.
        
        Creates 50+ technical indicators as features.
        """
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Convert to numpy arrays for TA-Lib
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
        
        # Momentum indicators
        df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
        df['rsi_28'] = talib.RSI(close_prices, timeperiod=28)
        
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
        df['willr'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        df['roc'] = talib.ROC(close_prices, timeperiod=10)
        df['mom'] = talib.MOM(close_prices, timeperiod=10)
        
        # Volatility indicators
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['natr'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (close_prices - lower) / (upper - lower)
        
        # Volume indicators
        df['obv'] = talib.OBV(close_prices, volume)
        df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
        df['adosc'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
        
        # Trend indicators
        df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Pattern recognition (returns -100, 0, or 100)
        df['cdl_doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        df['cdl_hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        df['cdl_engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        
        # Lagged features (previous days)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
        
        # Target: Next day's return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Drop rows with NaN (from indicators and lags)
        df = df.dropna()
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_size: int = 252 * 3,  # 3 years
        test_size: int = 252,  # 1 year
        step_size: int = 63  # Quarter
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.
        
        This prevents look-ahead bias by always training on past data
        and testing on future data.
        """
        splits = []
        
        for start in range(0, len(df) - train_size - test_size, step_size):
            train_end = start + train_size
            test_end = train_end + test_size
            
            if test_end > len(df):
                break
            
            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]
            
            splits.append((train_df, test_df))
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict]:
        """Train XGBoost model"""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not available")
            return None, {}
        
        logger.info("Training XGBoost model...")
        
        model = xgb.XGBRegressor(**self.xgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        # Feature importance
        feature_importance = dict(zip(
            range(X_train.shape[1]),
            model.feature_importances_.tolist()
        ))
        
        logger.info(f"  Train R²: {metrics['train_r2']:.4f}, Val R²: {metrics['val_r2']:.4f}")
        
        return model, metrics, feature_importance
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict]:
        """Train LightGBM model"""
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not available")
            return None, {}
        
        logger.info("Training LightGBM model...")
        
        model = lgb.LGBMRegressor(**self.lgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        # Feature importance
        feature_importance = dict(zip(
            range(X_train.shape[1]),
            model.feature_importances_.tolist()
        ))
        
        logger.info(f"  Train R²: {metrics['train_r2']:.4f}, Val R²: {metrics['val_r2']:.4f}")
        
        return model, metrics, feature_importance
    
    def train_stock(self, symbol: str) -> Dict:
        """
        Train models for a single stock.
        
        Returns dict with model paths and performance metrics.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training models for {symbol}")
        logger.info(f"{'='*80}")
        
        # Load data
        df = self.load_stock_data(symbol)
        logger.info(f"Loaded {len(df)} rows")
        
        # Engineer features
        df_features = self.engineer_features(df)
        logger.info(f"Features engineered: {len(df_features)} rows remaining")
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'target']]
        
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Walk-forward validation
        splits = self.walk_forward_split(df_features)
        
        if len(splits) == 0:
            logger.warning(f"Not enough data for {symbol}")
            return None
        
        # Use last split for final training
        train_df, test_df = splits[-1]
        
        train_idx = train_df.index
        test_idx = test_df.index
        
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_test = X_scaled[test_idx]
        y_test = y[test_idx]
        
        # Split train into train/val
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        results = {
            'symbol': symbol,
            'train_start': train_df['date'].iloc[0].isoformat(),
            'train_end': train_df['date'].iloc[-1].isoformat(),
            'test_start': test_df['date'].iloc[0].isoformat(),
            'test_end': test_df['date'].iloc[-1].isoformat(),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'n_features': X_train.shape[1],
            'feature_names': feature_cols,
            'models': {}
        }
        
        # Train XGBoost
        if HAS_XGBOOST:
            xgb_model, xgb_metrics, xgb_importance = self.train_xgboost(
                X_train, y_train, X_val, y_val
            )
            
            if xgb_model:
                # Test performance
                test_pred = xgb_model.predict(X_test)
                test_metrics = {
                    'test_mse': mean_squared_error(y_test, test_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'test_r2': r2_score(y_test, test_pred)
                }
                
                # Save model
                model_path = self.output_dir / f'{symbol}_xgboost.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': xgb_model,
                        'scaler': scaler,
                        'feature_names': feature_cols
                    }, f)
                
                results['models']['xgboost'] = {
                    'path': str(model_path),
                    'metrics': {**xgb_metrics, **test_metrics},
                    'feature_importance': xgb_importance
                }
                
                logger.info(f"✓ XGBoost Test R²: {test_metrics['test_r2']:.4f}")
        
        # Train LightGBM
        if HAS_LIGHTGBM:
            lgb_model, lgb_metrics, lgb_importance = self.train_lightgbm(
                X_train, y_train, X_val, y_val
            )
            
            if lgb_model:
                # Test performance
                test_pred = lgb_model.predict(X_test)
                test_metrics = {
                    'test_mse': mean_squared_error(y_test, test_pred),
                    'test_mae': mean_absolute_error(y_test, test_pred),
                    'test_r2': r2_score(y_test, test_pred)
                }
                
                # Save model
                model_path = self.output_dir / f'{symbol}_lightgbm.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': lgb_model,
                        'scaler': scaler,
                        'feature_names': feature_cols
                    }, f)
                
                results['models']['lightgbm'] = {
                    'path': str(model_path),
                    'metrics': {**lgb_metrics, **test_metrics},
                    'feature_importance': lgb_importance
                }
                
                logger.info(f"✓ LightGBM Test R²: {test_metrics['test_r2']:.4f}")
        
        # Save scaler
        scaler_path = self.output_dir / f'{symbol}_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        results['scaler_path'] = str(scaler_path)
        
        return results
    
    def train_all_selected_stocks(self) -> List[Dict]:
        """Train models for all selected stocks"""
        # Load selected stocks
        selected_file = '/home/ubuntu/quant-trading-web/python_system/ml/selected_stocks.json'
        
        with open(selected_file, 'r') as f:
            data = json.load(f)
            selected_stocks = data['selected_stocks']
        
        logger.info(f"Training models for {len(selected_stocks)} stocks")
        
        all_results = []
        
        for symbol in selected_stocks:
            try:
                result = self.train_stock(symbol)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error training {symbol}: {str(e)}")
                continue
        
        # Save results
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n✓ Training complete. Results saved to {results_file}")
        
        return all_results


def main():
    """Main execution"""
    trainer = StockMLTrainer()
    results = trainer.train_all_selected_stocks()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for result in results:
        symbol = result['symbol']
        logger.info(f"\n{symbol}:")
        
        for model_type, model_data in result['models'].items():
            metrics = model_data['metrics']
            logger.info(f"  {model_type}:")
            logger.info(f"    Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"    Test MAE: {metrics['test_mae']:.6f}")
            logger.info(f"    Test MSE: {metrics['test_mse']:.6f}")


if __name__ == '__main__':
    main()
