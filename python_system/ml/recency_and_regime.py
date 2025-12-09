"""
Recency Weighting and Market Regime Detection
==============================================
Implements:
1. Sample weighting (recent data gets higher weight)
2. Market regime detection (bull, bear, consolidation)
3. Regime-specific model selection
4. Adaptive training windows
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_recency_weights(dates: pd.Series, decay_rate: float = 0.95) -> np.ndarray:
    """
    Calculate exponential recency weights for training samples
    
    More recent samples get higher weight to adapt to current market conditions
    
    Args:
        dates: Series of dates for each sample
        decay_rate: Decay rate (0-1). Higher = more emphasis on recent data
    
    Returns:
        Array of sample weights
    """
    # Convert dates to days from most recent
    most_recent = dates.max()
    days_ago = (most_recent - dates).dt.days
    
    # Exponential decay: weight = decay_rate ^ days_ago
    weights = np.power(decay_rate, days_ago / 30)  # Decay per month
    
    # Normalize to sum to number of samples (for sklearn compatibility)
    weights = weights * len(weights) / weights.sum()
    
    logger.info(f"Recency weights: min={weights.min():.4f}, max={weights.max():.4f}, "
               f"ratio={weights.max()/weights.min():.2f}x")
    
    return weights.values

def detect_market_regime(df: pd.DataFrame, lookback_days: int = 60) -> str:
    """
    Detect current market regime based on recent price action
    
    Regimes:
    - 'bull': Strong uptrend
    - 'bear': Strong downtrend
    - 'volatile': High volatility, no clear trend
    - 'consolidation': Low volatility, sideways
    
    Args:
        df: DataFrame with OHLCV data
        lookback_days: Number of days to analyze
    
    Returns:
        Regime string
    """
    # Get recent data
    recent_df = df.tail(lookback_days).copy()
    
    if len(recent_df) < 20:
        return 'unknown'
    
    # Calculate metrics
    returns = recent_df['close'].pct_change()
    
    # Trend strength (cumulative return)
    cumulative_return = (recent_df['close'].iloc[-1] / recent_df['close'].iloc[0]) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    
    # Trend consistency (% of positive days)
    positive_days = (returns > 0).sum() / len(returns)
    
    # Moving average crossovers
    sma_20 = recent_df['close'].rolling(20).mean().iloc[-1]
    sma_50 = recent_df['close'].rolling(50).mean().iloc[-1] if len(recent_df) >= 50 else sma_20
    current_price = recent_df['close'].iloc[-1]
    
    logger.info(f"Regime detection: return={cumulative_return:.2%}, vol={volatility:.2%}, "
               f"positive_days={positive_days:.2%}")
    
    # Regime classification
    if cumulative_return > 0.10 and positive_days > 0.55:
        regime = 'bull'
    elif cumulative_return < -0.10 and positive_days < 0.45:
        regime = 'bear'
    elif volatility > 0.40:
        regime = 'volatile'
    elif abs(cumulative_return) < 0.05 and volatility < 0.20:
        regime = 'consolidation'
    elif current_price > sma_20 > sma_50:
        regime = 'bull'
    elif current_price < sma_20 < sma_50:
        regime = 'bear'
    else:
        regime = 'neutral'
    
    logger.info(f"Detected regime: {regime}")
    
    return regime

def get_regime_specific_training_window(regime: str) -> int:
    """
    Get optimal training window based on market regime
    
    Args:
        regime: Market regime string
    
    Returns:
        Number of days to use for training
    """
    windows = {
        'bull': 365 * 2,  # 2 years - longer history in stable trends
        'bear': 365 * 2,  # 2 years
        'volatile': 365 * 1,  # 1 year - shorter window for volatile markets
        'consolidation': 365 * 3,  # 3 years - need more data for low-signal environment
        'neutral': 365 * 2,  # 2 years - default
        'unknown': 365 * 2  # 2 years - default
    }
    
    window = windows.get(regime, 365 * 2)
    logger.info(f"Using {window} days training window for {regime} regime")
    
    return window

def filter_training_data_by_regime(df: pd.DataFrame, target_regime: str, similarity_threshold: float = 0.7) -> pd.DataFrame:
    """
    Filter training data to include only periods similar to current regime
    
    Args:
        df: Full DataFrame with OHLCV data
        target_regime: Current market regime
        similarity_threshold: How similar regimes must be (0-1)
    
    Returns:
        Filtered DataFrame
    """
    if target_regime == 'unknown' or target_regime == 'neutral':
        return df  # Use all data
    
    # Detect regime for each rolling window
    window_size = 60
    regimes = []
    
    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i]
        regime = detect_market_regime(window_df, lookback_days=window_size)
        regimes.append(regime)
    
    # Pad beginning
    regimes = ['unknown'] * window_size + regimes
    
    df = df.copy()
    df['regime'] = regimes
    
    # Filter to similar regimes
    if target_regime in ['bull', 'bear']:
        # Include target regime and neutral
        filtered_df = df[df['regime'].isin([target_regime, 'neutral'])]
    else:
        # For volatile/consolidation, be more inclusive
        filtered_df = df[df['regime'] != 'unknown']
    
    pct_retained = len(filtered_df) / len(df) * 100
    logger.info(f"Regime filtering: retained {len(filtered_df)}/{len(df)} samples ({pct_retained:.1f}%) "
               f"for {target_regime} regime")
    
    return filtered_df.drop(columns=['regime'])

def calculate_adaptive_sample_weights(df: pd.DataFrame, regime: str, base_decay: float = 0.95) -> np.ndarray:
    """
    Calculate sample weights combining recency and regime similarity
    
    Args:
        df: DataFrame with Date column
        regime: Current market regime
        base_decay: Base decay rate for recency
    
    Returns:
        Array of sample weights
    """
    # Start with recency weights
    recency_weights = calculate_recency_weights(df['Date'], decay_rate=base_decay)
    
    # Adjust decay rate based on regime
    if regime == 'volatile':
        # In volatile markets, emphasize very recent data more
        recency_weights = calculate_recency_weights(df['Date'], decay_rate=0.90)
    elif regime == 'consolidation':
        # In consolidation, use more historical data
        recency_weights = calculate_recency_weights(df['Date'], decay_rate=0.97)
    
    return recency_weights

def detect_regime_change(historical_regimes: List[Tuple[datetime, str]], lookback_periods: int = 5) -> bool:
    """
    Detect if market regime has changed recently
    
    Args:
        historical_regimes: List of (date, regime) tuples
        lookback_periods: Number of periods to check
    
    Returns:
        True if regime changed
    """
    if len(historical_regimes) < lookback_periods + 1:
        return False
    
    recent_regimes = [r for _, r in historical_regimes[-lookback_periods:]]
    previous_regime = historical_regimes[-lookback_periods-1][1]
    
    # Check if most recent regimes differ from previous
    current_regime = recent_regimes[-1]
    
    if current_regime != previous_regime:
        logger.info(f"Regime change detected: {previous_regime} -> {current_regime}")
        return True
    
    return False

def get_regime_specific_hyperparameters(regime: str, base_params: Dict) -> Dict:
    """
    Adjust hyperparameters based on market regime
    
    Args:
        regime: Market regime
        base_params: Base hyperparameters
    
    Returns:
        Adjusted hyperparameters
    """
    params = base_params.copy()
    
    if regime == 'volatile':
        # In volatile markets, use more regularization
        params['reg_alpha'] = params.get('reg_alpha', 0) * 1.5
        params['reg_lambda'] = params.get('reg_lambda', 1) * 1.5
        # Reduce tree depth to avoid overfitting noise
        params['max_depth'] = min(params.get('max_depth', 5), 4)
        logger.info("Adjusted hyperparameters for volatile regime (more regularization)")
        
    elif regime == 'consolidation':
        # In consolidation, allow more complex models to find subtle patterns
        params['max_depth'] = min(params.get('max_depth', 5) + 1, 8)
        params['n_estimators'] = int(params.get('n_estimators', 100) * 1.2)
        logger.info("Adjusted hyperparameters for consolidation regime (more complexity)")
        
    elif regime in ['bull', 'bear']:
        # In trending markets, reduce regularization to capture trend
        params['reg_alpha'] = params.get('reg_alpha', 0) * 0.8
        params['reg_lambda'] = params.get('reg_lambda', 1) * 0.8
        logger.info(f"Adjusted hyperparameters for {regime} regime (less regularization)")
    
    return params

def calculate_regime_transition_probability(df: pd.DataFrame, current_regime: str) -> Dict[str, float]:
    """
    Calculate probability of transitioning to different regimes
    
    Args:
        df: Historical DataFrame
        current_regime: Current market regime
    
    Returns:
        Dict of {regime: probability}
    """
    # Detect regimes for historical periods
    window_size = 60
    regimes = []
    
    for i in range(window_size, len(df), window_size):
        window_df = df.iloc[i-window_size:i]
        regime = detect_market_regime(window_df, lookback_days=window_size)
        regimes.append(regime)
    
    if len(regimes) < 2:
        return {'bull': 0.25, 'bear': 0.25, 'volatile': 0.25, 'consolidation': 0.25}
    
    # Build transition matrix
    transitions = {}
    for i in range(len(regimes) - 1):
        from_regime = regimes[i]
        to_regime = regimes[i + 1]
        
        if from_regime not in transitions:
            transitions[from_regime] = {}
        
        transitions[from_regime][to_regime] = transitions[from_regime].get(to_regime, 0) + 1
    
    # Calculate probabilities for current regime
    if current_regime in transitions:
        total = sum(transitions[current_regime].values())
        probabilities = {
            regime: count / total
            for regime, count in transitions[current_regime].items()
        }
    else:
        # Default uniform distribution
        probabilities = {'bull': 0.25, 'bear': 0.25, 'volatile': 0.25, 'consolidation': 0.25}
    
    logger.info(f"Regime transition probabilities from {current_regime}: {probabilities}")
    
    return probabilities

def main():
    """Test recency weighting and regime detection"""
    import yfinance as yf
    
    # Download sample data
    symbol = 'AAPL'
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='2y')
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    print(f"\nTesting regime detection for {symbol}")
    print("=" * 60)
    
    # Detect current regime
    regime = detect_market_regime(df)
    print(f"\nCurrent regime: {regime}")
    
    # Calculate recency weights
    weights = calculate_recency_weights(df['date'])
    print(f"\nRecency weights: min={weights.min():.4f}, max={weights.max():.4f}")
    
    # Get regime-specific training window
    window = get_regime_specific_training_window(regime)
    print(f"\nOptimal training window: {window} days")
    
    # Calculate transition probabilities
    probs = calculate_regime_transition_probability(df, regime)
    print(f"\nRegime transition probabilities:")
    for r, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r}: {p:.2%}")

if __name__ == '__main__':
    main()
