"""
Data Quality Validation System
===============================
Ensures training data is clean and reliable:
1. Detect and adjust for stock splits
2. Detect and adjust for dividends
3. Identify and handle outliers
4. Fill missing data properly
5. Validate data integrity
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_stock_splits(df: pd.DataFrame, threshold: float = 0.25) -> List[Tuple[datetime, float]]:
    """
    Detect stock splits in price data
    
    A split is detected when:
    - Price drops/increases dramatically overnight
    - Volume increases significantly
    - Close-to-open gap is large
    
    Args:
        df: DataFrame with OHLCV data
        threshold: Minimum price change ratio to consider a split (default 25%)
    
    Returns:
        List of (date, split_ratio) tuples
    """
    splits = []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else df.index
    
    # Calculate overnight price changes
    df['prev_close'] = df['close'].shift(1)
    df['overnight_change'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Calculate volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Detect potential splits
    for idx in range(1, len(df)):
        change = df.iloc[idx]['overnight_change']
        vol_ratio = df.iloc[idx]['volume_ratio']
        
        # Check for 2:1 split (50% drop)
        if -0.55 < change < -0.45 and vol_ratio > 1.5:
            date = df.iloc[idx]['date']
            splits.append((date, 2.0))
            logger.info(f"Detected 2:1 split on {date.date()}")
        
        # Check for 3:1 split (66% drop)
        elif -0.70 < change < -0.60 and vol_ratio > 1.5:
            date = df.iloc[idx]['date']
            splits.append((date, 3.0))
            logger.info(f"Detected 3:1 split on {date.date()}")
        
        # Check for reverse split (100%+ increase)
        elif change > 0.90 and vol_ratio > 1.5:
            date = df.iloc[idx]['date']
            splits.append((date, 0.5))
            logger.info(f"Detected 1:2 reverse split on {date.date()}")
    
    return splits

def adjust_for_splits(df: pd.DataFrame, splits: List[Tuple[datetime, float]]) -> pd.DataFrame:
    """
    Adjust historical prices for stock splits
    
    Args:
        df: DataFrame with OHLCV data
        splits: List of (date, split_ratio) tuples
    
    Returns:
        Adjusted DataFrame
    """
    if not splits:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else df.index
    
    for split_date, ratio in splits:
        # Adjust all prices before split date
        mask = df['date'] < split_date
        
        df.loc[mask, 'open'] = df.loc[mask, 'open'] / ratio
        df.loc[mask, 'high'] = df.loc[mask, 'high'] / ratio
        df.loc[mask, 'low'] = df.loc[mask, 'low'] / ratio
        df.loc[mask, 'close'] = df.loc[mask, 'close'] / ratio
        
        # Adjust volume (inverse of price adjustment)
        df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * ratio
        
        logger.info(f"Adjusted prices before {split_date.date()} for {ratio}:1 split")
    
    return df

def detect_dividends(df: pd.DataFrame, threshold: float = 0.02) -> List[Tuple[datetime, float]]:
    """
    Detect dividend payments in price data
    
    Dividends cause a gap down on ex-dividend date
    
    Args:
        df: DataFrame with OHLCV data
        threshold: Minimum gap to consider a dividend (default 2%)
    
    Returns:
        List of (date, dividend_amount) tuples
    """
    dividends = []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else df.index
    
    # Calculate overnight gaps
    df['prev_close'] = df['close'].shift(1)
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    # Detect dividend-like gaps
    for idx in range(1, len(df)):
        gap = df.iloc[idx]['gap']
        prev_close = df.iloc[idx]['prev_close']
        
        # Dividend causes small gap down (typically 1-3%)
        if -0.05 < gap < -threshold:
            date = df.iloc[idx]['date']
            dividend_amount = abs(gap * prev_close)
            dividends.append((date, dividend_amount))
            logger.info(f"Detected potential dividend ${dividend_amount:.2f} on {date.date()}")
    
    return dividends

def detect_outliers(df: pd.DataFrame, column: str = 'close', method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in price data
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    n_outliers = outliers.sum()
    if n_outliers > 0:
        logger.warning(f"Detected {n_outliers} outliers in {column} using {method} method")
    
    return outliers

def handle_outliers(df: pd.DataFrame, outliers: pd.Series, method: str = 'winsorize') -> pd.DataFrame:
    """
    Handle outliers in data
    
    Args:
        df: DataFrame
        outliers: Boolean Series indicating outliers
        method: 'remove', 'winsorize', or 'interpolate'
    
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if method == 'remove':
        # Remove outlier rows
        df = df[~outliers]
        logger.info(f"Removed {outliers.sum()} outlier rows")
        
    elif method == 'winsorize':
        # Cap outliers at 5th and 95th percentiles
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                p05 = df[col].quantile(0.05)
                p95 = df[col].quantile(0.95)
                df.loc[outliers, col] = df.loc[outliers, col].clip(p05, p95)
        logger.info(f"Winsorized {outliers.sum()} outlier values")
        
    elif method == 'interpolate':
        # Interpolate outlier values
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate(method='linear')
        logger.info(f"Interpolated {outliers.sum()} outlier values")
    
    return df

def fill_missing_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Fill missing data in DataFrame
    
    Args:
        df: DataFrame with potential missing values
        method: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'
    
    Returns:
        DataFrame with missing data handled
    """
    df = df.copy()
    
    # Count missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        return df
    
    logger.info(f"Found {total_missing} missing values across {(missing_counts > 0).sum()} columns")
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill')
        logger.info("Applied forward fill for missing values")
        
    elif method == 'backward_fill':
        df = df.fillna(method='bfill')
        logger.info("Applied backward fill for missing values")
        
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
        logger.info("Applied linear interpolation for missing values")
        
    elif method == 'drop':
        df = df.dropna()
        logger.info(f"Dropped {total_missing} rows with missing values")
    
    # Final check
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Still have {remaining_missing} missing values after {method}")
        df = df.dropna()  # Drop any remaining
    
    return df

def validate_data_integrity(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive data integrity validation
    
    Checks:
    - No missing values
    - No negative prices
    - High >= Low
    - Close within [Low, High]
    - Volume >= 0
    - Dates are sequential
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dict with validation results
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"{missing} missing values")
    
    # Check for negative prices
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                issues.append(f"{negative} negative values in {col}")
    
    # Check high >= low
    if 'high' in df.columns and 'low' in df.columns:
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            issues.append(f"{invalid_hl} rows where high < low")
    
    # Check close within [low, high]
    if all(col in df.columns for col in ['close', 'low', 'high']):
        invalid_close = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
        if invalid_close > 0:
            issues.append(f"{invalid_close} rows where close outside [low, high]")
    
    # Check volume >= 0
    if 'volume' in df.columns:
        negative_vol = (df['volume'] < 0).sum()
        if negative_vol > 0:
            issues.append(f"{negative_vol} negative volume values")
    
    # Check dates are sequential
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        date_diffs = df['date'].diff()
        negative_diffs = (date_diffs < timedelta(0)).sum()
        if negative_diffs > 0:
            issues.append(f"{negative_diffs} non-sequential dates")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("✅ Data integrity validation passed")
    else:
        logger.warning(f"⚠️ Data integrity issues found: {', '.join(issues)}")
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'total_rows': len(df)
    }

def clean_stock_data(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data cleaning pipeline
    
    Args:
        df: Raw DataFrame with OHLCV data
        symbol: Stock symbol for logging
    
    Returns:
        (cleaned_df, cleaning_report)
    """
    logger.info(f"Starting data cleaning for {symbol}")
    logger.info(f"Initial data shape: {df.shape}")
    
    report = {
        'symbol': symbol,
        'initial_rows': len(df),
        'steps': []
    }
    
    # Step 1: Detect and adjust for splits
    splits = detect_stock_splits(df)
    if splits:
        df = adjust_for_splits(df, splits)
        report['steps'].append(f"Adjusted for {len(splits)} stock splits")
    
    # Step 2: Detect dividends (for information, not adjustment)
    dividends = detect_dividends(df)
    if dividends:
        report['steps'].append(f"Detected {len(dividends)} dividend payments")
    
    # Step 3: Detect and handle outliers
    outliers = detect_outliers(df, column='close', method='iqr', threshold=3.0)
    if outliers.sum() > 0:
        df = handle_outliers(df, outliers, method='winsorize')
        report['steps'].append(f"Handled {outliers.sum()} outliers")
    
    # Step 4: Fill missing data
    df = fill_missing_data(df, method='forward_fill')
    report['steps'].append("Filled missing data")
    
    # Step 5: Validate integrity
    validation = validate_data_integrity(df)
    report['validation'] = validation
    
    if not validation['is_valid']:
        # Try to fix common issues
        if 'high' in df.columns and 'low' in df.columns:
            # Fix high < low
            df['high'] = df[['high', 'low']].max(axis=1)
            df['low'] = df[['high', 'low']].min(axis=1)
        
        # Revalidate
        validation = validate_data_integrity(df)
        report['validation_after_fix'] = validation
    
    report['final_rows'] = len(df)
    report['rows_removed'] = report['initial_rows'] - report['final_rows']
    
    logger.info(f"Data cleaning complete for {symbol}")
    logger.info(f"Final data shape: {df.shape}")
    logger.info(f"Rows removed: {report['rows_removed']}")
    
    return df, report

def main():
    """Test data quality validation"""
    import yfinance as yf
    
    symbol = 'AAPL'
    logger.info(f"Testing data quality validation for {symbol}")
    
    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='5y')
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # Clean data
    cleaned_df, report = clean_stock_data(df, symbol)
    
    print("\n" + "=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)
    print(f"Symbol: {report['symbol']}")
    print(f"Initial rows: {report['initial_rows']}")
    print(f"Final rows: {report['final_rows']}")
    print(f"Rows removed: {report['rows_removed']}")
    print("\nSteps performed:")
    for step in report['steps']:
        print(f"  - {step}")
    print(f"\nValidation: {'✅ PASSED' if report['validation']['is_valid'] else '⚠️ FAILED'}")
    if not report['validation']['is_valid']:
        print("Issues:")
        for issue in report['validation']['issues']:
            print(f"  - {issue}")

if __name__ == '__main__':
    main()
