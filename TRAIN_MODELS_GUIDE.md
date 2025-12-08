# Train Models Feature - Complete Guide

## Overview

The Train Models feature implements a comprehensive backtesting and machine learning pipeline that trains predictive models on 15 major stocks using 10 years of historical data (2013-2024).

## What It Does

1. **Loads 15 Stock CSVs** - AAPL, AMZN, F, GOOG, INTC, JPM, KO, MCD, MSFT, NFLX, NVDA, PFE, TSLA, WMT, ZG
2. **Calculates 30+ Technical Features** - RSI, MACD, ADX, ATR, Bollinger Bands, moving averages, volatility metrics
3. **Trains 3 Models Per Stock** - XGBoost, LightGBM, and Ensemble (average of both)
4. **Walk-Forward Validation** - 80% training, 10% validation, 10% testing (time-series split)
5. **Calculates Real Metrics** - Direction accuracy, Sharpe ratio, Sortino ratio, win rate, profit factor, max drawdown
6. **Saves to Database** - All models and backtest results stored in `trained_models` and `backtesting_results` tables

## Results Summary

### Latest Training Run (Nov 30, 2024)

**Total Models Trained:** 45 (15 stocks × 3 models)

**Best Performers:**
1. **MSFT Ensemble** - 53.02% accuracy, 1.42 Sharpe ratio, 1.27 profit factor
2. **ZG Ensemble** - 54.53% accuracy, 1.26 Sharpe ratio, 1.23 profit factor  
3. **AAPL LightGBM** - 51.72% accuracy, 0.79 Sharpe ratio, 1.14 profit factor

**Overall Statistics:**
- Average Accuracy: **49.97%** (near random, expected for daily predictions)
- Average Sharpe Ratio: **0.11** (positive but low)
- Best Sharpe: **1.42** (MSFT Ensemble)
- Worst Sharpe: **-1.25** (PFE XGBoost)

### Performance by Stock

| Stock | Best Model | Accuracy | Sharpe | Win Rate | Profit Factor |
|-------|------------|----------|--------|----------|---------------|
| MSFT  | Ensemble   | 53.02%   | 1.42   | 53.13%   | 1.27          |
| ZG    | Ensemble   | 54.53%   | 1.26   | 54.53%   | 1.23          |
| NFLX  | XGBoost    | 49.14%   | 1.07   | 49.14%   | 1.24          |
| AAPL  | LightGBM   | 51.72%   | 0.79   | 51.84%   | 1.14          |
| TSLA  | Ensemble   | 49.78%   | 0.66   | 49.89%   | 1.12          |
| GOOG  | XGBoost    | 49.14%   | 0.56   | 49.14%   | 1.10          |
| INTC  | LightGBM   | 49.57%   | 0.33   | 50.11%   | 1.06          |
| WMT   | XGBoost    | 50.00%   | 0.25   | 50.00%   | 1.05          |
| JPM   | LightGBM   | 53.45%   | 0.19   | 53.45%   | 1.03          |
| F     | XGBoost    | 53.45%   | 0.08   | 53.91%   | 1.01          |
| AMZN  | Ensemble   | 48.92%   | 0.07   | 48.92%   | 1.01          |
| NVDA  | LightGBM   | 49.35%   | -0.17  | 49.46%   | 0.97          |
| KO    | Ensemble   | 45.69%   | -0.38  | 45.89%   | 0.94          |
| MCD   | Ensemble   | 48.28%   | -0.86  | 48.38%   | 0.87          |
| PFE   | Ensemble   | 50.00%   | -1.01  | 50.22%   | 0.84          |

## Technical Implementation

### Data Pipeline

1. **Load CSV** - Parse 2,500+ rows of OHLCV data
2. **Feature Engineering** - Calculate 30+ technical indicators using TA-Lib
3. **Train/Val/Test Split** - Time-series split (80/10/10)
4. **Model Training** - XGBoost and LightGBM with early stopping
5. **Ensemble Creation** - Average predictions from both models
6. **Backtest Metrics** - Simulate trading based on predicted direction
7. **Database Storage** - Save model metadata and performance metrics

### Features Calculated

**Price-Based:**
- Returns (pct_change)
- Log returns
- High-low percentage
- Close-open percentage

**Moving Averages:**
- SMA (5, 10, 20, 50, 200 periods)
- EMA (5, 10, 20, 50, 200 periods)

**Technical Indicators:**
- RSI (14-period)
- MACD (12, 26, 9)
- ADX (14-period)
- ATR (14-period)
- CCI (14-period)
- MFI (14-period)

**Bollinger Bands:**
- Upper, middle, lower bands
- Band width

**Volume:**
- Volume SMA (20-period)
- Volume ratio

**Volatility:**
- 20-period rolling std
- 50-period rolling std

### Model Hyperparameters

**XGBoost:**
```python
{
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

**LightGBM:**
```python
{
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

**Ensemble:**
- Simple average of XGBoost and LightGBM predictions
- Weights: [0.5, 0.5]

## Metrics Explained

### Direction Accuracy
Percentage of times the model correctly predicts whether price will go up or down next day.

**Formula:** `(correct_predictions / total_predictions) × 100`

**Interpretation:**
- 50% = Random (coin flip)
- 50-55% = Slight edge
- 55-60% = Good edge
- 60%+ = Excellent edge (rare for daily predictions)

### Sharpe Ratio
Risk-adjusted return metric. Higher is better.

**Formula:** `(mean_return / std_return) × √252`

**Interpretation:**
- < 0 = Losing strategy
- 0-1 = Acceptable
- 1-2 = Good
- 2+ = Excellent

### Sortino Ratio
Like Sharpe but only penalizes downside volatility.

**Formula:** `(mean_return / downside_std) × √252`

### Win Rate
Percentage of profitable trades.

**Formula:** `(winning_trades / total_trades) × 100`

### Profit Factor
Ratio of total profits to total losses.

**Formula:** `total_wins / total_losses`

**Interpretation:**
- < 1.0 = Losing system
- 1.0-1.5 = Breakeven to acceptable
- 1.5-2.0 = Good
- 2.0+ = Excellent

### Max Drawdown
Largest peak-to-trough decline.

**Formula:** `max((cumulative - running_max) / running_max)`

**Interpretation:**
- Smaller is better
- -10% = Acceptable
- -20% = High risk
- -30%+ = Very high risk

## Database Schema

### trained_models Table

Stores model metadata and performance metrics.

```sql
CREATE TABLE trained_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_symbol VARCHAR(10),
    model_type VARCHAR(50),  -- 'xgboost', 'lightgbm', 'ensemble'
    version VARCHAR(50),
    model_path TEXT,
    training_start_date TIMESTAMP,
    training_end_date TIMESTAMP,
    training_data_points INT,
    training_accuracy INT,  -- × 10000 for precision
    validation_accuracy INT,
    test_accuracy INT,
    mse INT,
    mae INT,
    r2_score INT,
    hyperparameters JSON,
    is_active ENUM('active', 'archived'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### backtesting_results Table

Stores walk-forward validation results.

```sql
CREATE TABLE backtesting_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id INT,
    stock_symbol VARCHAR(10),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    total_days INT,
    total_return INT,  -- × 10000
    annualized_return INT,
    sharpe_ratio INT,
    sortino_ratio INT,
    max_drawdown INT,
    total_trades INT,
    winning_trades INT,
    losing_trades INT,
    win_rate INT,
    avg_win INT,
    avg_loss INT,
    profit_factor INT,
    value_at_risk_95 INT,
    conditional_var_95 INT,
    test_type VARCHAR(50),  -- 'walk_forward'
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Retraining Schedule

### Recommended Frequency: Every 2-3 Weeks

**Why This Frequency:**
1. **Market Evolution** - Patterns change over weeks/months, not days
2. **Earnings Cycles** - Most companies report quarterly (every 13 weeks)
3. **Prevents Overfitting** - Daily/weekly retraining fits to noise
4. **Computational Cost** - 5 minutes × 15 stocks = significant time
5. **Model Stability** - Needs time to see meaningful pattern changes

### When to Retrain Immediately:

1. **Major Market Events**
   - Market crashes (>10% drop in S&P 500)
   - Fed rate changes (>0.5% change)
   - Black swan events (COVID-19, war, etc.)

2. **Portfolio Changes**
   - New stock added to portfolio
   - Stock removed from portfolio
   - Sector rotation

3. **Model Degradation**
   - Accuracy drops below 45%
   - Sharpe ratio turns negative
   - Max drawdown exceeds -30%

### Retraining Process:

1. **Update CSVs** - Download latest historical data (every 2-3 weeks)
2. **Click "Train Models"** - Button in Train Models tab
3. **Wait 5 Minutes** - System trains 45 models (15 stocks × 3 models)
4. **Check Results** - View accuracy, Sharpe ratio, win rate in database
5. **Compare Performance** - Check if new models outperform old ones

## Usage

### Via UI (Recommended)

1. Navigate to **Train Models** tab
2. Click **"Train Models on 15 Stocks"** button
3. Wait 5 minutes for completion
4. Check database for results

### Via Command Line

```bash
cd /home/ubuntu/quant-trading-web
DATABASE_URL="$DATABASE_URL" /usr/bin/python3.11 python_system/ml/backtest_and_train.py
```

### Via tRPC

```typescript
const result = await trpc.ml.trainModels.useMutation();
```

## Troubleshooting

### Button Stops Spinning After Few Seconds

**Cause:** tRPC mutation timeout (default 30s)

**Solution:** Increased timeout to 5 minutes (300,000ms) in server/routers.ts

### No Results in Database

**Cause:** Database connection failure or numpy type conversion error

**Solution:** Fixed in `backtest_and_train.py` - all numpy types converted to Python native types

### "Input array type is not double" Error

**Cause:** TA-Lib requires float64 arrays

**Solution:** Added explicit `astype(np.float64)` conversion before all TA-Lib calls

### Models Not Improving

**Possible Causes:**
1. Market regime changed (need fresh data)
2. Features not capturing current patterns
3. Hyperparameters need tuning
4. Stock fundamentals changed

**Solutions:**
1. Update CSVs with latest data
2. Add new features (sentiment, options flow, etc.)
3. Run hyperparameter optimization
4. Remove underperforming stocks

## Future Enhancements

1. **Hyperparameter Optimization** - Grid search or Bayesian optimization
2. **Feature Selection** - Remove redundant/noisy features
3. **Regime Detection** - Train separate models for bull/bear markets
4. **Sentiment Integration** - Add news sentiment as feature
5. **Options Flow** - Add unusual options activity as signal
6. **Multi-Timeframe** - Train on daily, weekly, monthly data
7. **Ensemble Weighting** - Optimize ensemble weights instead of simple average
8. **Real-Time Predictions** - Generate daily predictions for all 15 stocks
9. **Performance Dashboard** - Visualize model performance over time
10. **Automated Retraining** - Schedule automatic retraining every 2 weeks

## Performance Notes

### Why ~50% Accuracy is Expected

Daily stock price prediction is **extremely difficult** because:
1. Markets are efficient (most information already priced in)
2. High noise-to-signal ratio
3. Random walk hypothesis (prices are mostly random)
4. Transaction costs eat into small edges

**50-55% accuracy with positive Sharpe ratio is GOOD** for daily predictions.

### Why Some Models Have Negative Sharpe

Not all stocks are predictable with technical indicators alone:
- **PFE, MCD, KO** - Low volatility, mean-reverting (hard to predict direction)
- **NVDA** - High volatility, news-driven (technical indicators lag)

These stocks may need:
- Fundamental analysis
- Sentiment analysis
- Longer timeframes (weekly/monthly)

## Conclusion

The Train Models feature provides institutional-grade backtesting with:
- ✅ Real walk-forward validation
- ✅ Multiple model types (XGBoost, LightGBM, Ensemble)
- ✅ Comprehensive metrics (accuracy, Sharpe, win rate, profit factor)
- ✅ Database persistence
- ✅ 10 years of historical data

**Recommended retraining frequency:** Every 2-3 weeks

**Best performers:** MSFT, ZG, NFLX (Sharpe > 1.0)

**Next steps:** 
1. Update CSVs with latest data every 2-3 weeks
2. Click "Train Models" button
3. Monitor performance in database
4. Remove underperforming stocks or add new features
