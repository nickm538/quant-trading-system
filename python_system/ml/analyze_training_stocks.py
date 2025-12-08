"""
Analyze 30 Stocks Historical Data - Select Best 15 for Training
===============================================================

Criteria for selection:
1. Data quality (completeness, no gaps)
2. Liquidity (average volume)
3. Volatility (suitable for training - not too stable, not too chaotic)
4. Price range stability (no extreme outliers)
5. Sector diversity (don't overtrain on one sector)
6. Data length (prefer longer history)

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    """Analyze stocks for ML training suitability"""
    
    # Sector classifications
    SECTORS = {
        'AAPL': 'Technology',
        'AMD': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'ATVI': 'Technology',
        'BABA': 'Consumer Discretionary',
        'BAC': 'Financials',
        'CRM': 'Technology',
        'CSCO': 'Technology',
        'DIS': 'Communication Services',
        'EA': 'Technology',
        'F': 'Consumer Discretionary',
        'GOOG': 'Technology',
        'INTC': 'Technology',
        'JPM': 'Financials',
        'KO': 'Consumer Staples',
        'MCD': 'Consumer Discretionary',
        'META': 'Technology',
        'MSFT': 'Technology',
        'MTCH': 'Technology',
        'NFLX': 'Communication Services',
        'NVDA': 'Technology',
        'PFE': 'Healthcare',
        'PYPL': 'Technology',
        'T': 'Communication Services',
        'TSLA': 'Consumer Discretionary',
        'TTD': 'Technology',
        'WMT': 'Consumer Staples',
        'XOM': 'Energy',
        'YELP': 'Technology',
        'ZG': 'Real Estate'
    }
    
    def __init__(self, data_dir: str = '/home/ubuntu/historical_data'):
        """Initialize analyzer"""
        self.data_dir = Path(data_dir)
        self.results = {}
    
    def analyze_stock(self, filepath: Path) -> Dict:
        """
        Analyze a single stock for training suitability.
        
        Returns dict with quality metrics.
        """
        try:
            # Extract symbol from filename
            filename = filepath.name
            symbol = filename.split('(')[0]
            
            logger.info(f"Analyzing {symbol}...")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Standardize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Convert date column
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Calculate metrics
            total_rows = len(df)
            
            # 1. Data Quality Score (0-100)
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            data_quality_score = max(0, 100 - missing_pct * 10)
            
            # Check for gaps in dates
            df_sorted = df.sort_values(date_col)
            date_diffs = df_sorted[date_col].diff().dt.days
            avg_gap = date_diffs.mean()
            max_gap = date_diffs.max()
            
            # Penalize large gaps
            if max_gap > 7:  # More than 1 week gap
                data_quality_score *= 0.9
            
            # 2. Liquidity Score (0-100)
            if 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                # Score based on average volume (higher is better)
                # 1M+ volume = 100, 100K = 50, 10K = 10
                liquidity_score = min(100, np.log10(max(avg_volume, 1)) * 15)
            else:
                liquidity_score = 50  # Default if no volume data
            
            # 3. Volatility Score (0-100)
            # We want moderate volatility - not too low, not too high
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                daily_vol = returns.std()
                annualized_vol = daily_vol * np.sqrt(252) * 100  # Convert to %
                
                # Ideal volatility: 20-40% annualized
                # Score peaks at 30%
                ideal_vol = 30
                vol_diff = abs(annualized_vol - ideal_vol)
                volatility_score = max(0, 100 - vol_diff * 3)
            else:
                volatility_score = 50
            
            # 4. Price Stability Score (0-100)
            # Check for extreme price movements (circuit breakers, splits, etc.)
            if 'close' in df.columns:
                price_changes = df['close'].pct_change().abs()
                extreme_moves = (price_changes > 0.20).sum()  # >20% moves
                price_stability_score = max(0, 100 - extreme_moves * 5)
            else:
                price_stability_score = 50
            
            # 5. Data Length Score (0-100)
            # Prefer longer history
            years_of_data = (df[date_col].max() - df[date_col].min()).days / 365.25
            data_length_score = min(100, years_of_data * 10)  # 10 years = 100
            
            # 6. Sector (for diversity)
            sector = self.SECTORS.get(symbol, 'Unknown')
            
            # Calculate overall score (weighted average)
            overall_score = (
                data_quality_score * 0.30 +
                liquidity_score * 0.25 +
                volatility_score * 0.20 +
                price_stability_score * 0.15 +
                data_length_score * 0.10
            )
            
            result = {
                'symbol': symbol,
                'sector': sector,
                'total_rows': total_rows,
                'years_of_data': round(years_of_data, 2),
                'avg_volume': int(avg_volume) if 'volume' in df.columns else 0,
                'annualized_volatility': round(annualized_vol, 2) if 'close' in df.columns else 0,
                'data_quality_score': round(data_quality_score, 2),
                'liquidity_score': round(liquidity_score, 2),
                'volatility_score': round(volatility_score, 2),
                'price_stability_score': round(price_stability_score, 2),
                'data_length_score': round(data_length_score, 2),
                'overall_score': round(overall_score, 2),
                'missing_data_pct': round(missing_pct, 2),
                'max_date_gap_days': int(max_gap),
                'extreme_moves_count': int(extreme_moves) if 'close' in df.columns else 0
            }
            
            logger.info(f"  Overall Score: {overall_score:.2f}/100")
            logger.info(f"  Data Quality: {data_quality_score:.2f}, Liquidity: {liquidity_score:.2f}, Volatility: {volatility_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {filepath.name}: {str(e)}")
            return None
    
    def analyze_all_stocks(self) -> pd.DataFrame:
        """Analyze all stocks in the directory"""
        results = []
        
        csv_files = list(self.data_dir.glob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for filepath in csv_files:
            result = self.analyze_stock(filepath)
            if result:
                results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('overall_score', ascending=False)
        
        return df
    
    def select_best_stocks(
        self,
        df: pd.DataFrame,
        n: int = 15,
        max_per_sector: int = 5
    ) -> List[str]:
        """
        Select best N stocks ensuring sector diversity.
        
        Args:
            df: DataFrame with stock analysis results
            n: Number of stocks to select
            max_per_sector: Maximum stocks per sector
            
        Returns:
            List of selected stock symbols
        """
        selected = []
        sector_counts = {}
        
        # Sort by overall score
        df_sorted = df.sort_values('overall_score', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            symbol = row['symbol']
            sector = row['sector']
            
            # Check sector limit
            if sector_counts.get(sector, 0) >= max_per_sector:
                logger.info(f"  Skipping {symbol} - sector {sector} quota full")
                continue
            
            # Add to selected
            selected.append(symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            logger.info(f"✓ Selected {symbol} (Score: {row['overall_score']:.2f}, Sector: {sector})")
            
            if len(selected) >= n:
                break
        
        return selected
    
    def generate_report(self, df: pd.DataFrame, selected: List[str]):
        """Generate selection report"""
        logger.info("\n" + "=" * 80)
        logger.info("STOCK SELECTION REPORT")
        logger.info("=" * 80)
        
        logger.info(f"\nTop 15 Stocks Selected for Training:")
        logger.info("-" * 80)
        
        selected_df = df[df['symbol'].isin(selected)]
        
        for idx, row in selected_df.iterrows():
            logger.info(f"{row['symbol']:6s} | Score: {row['overall_score']:6.2f} | "
                       f"Sector: {row['sector']:25s} | "
                       f"Vol: {row['annualized_volatility']:5.1f}% | "
                       f"Years: {row['years_of_data']:4.1f}")
        
        logger.info("\nSector Distribution:")
        logger.info("-" * 80)
        sector_dist = selected_df['sector'].value_counts()
        for sector, count in sector_dist.items():
            logger.info(f"  {sector:30s}: {count} stocks")
        
        logger.info("\nAverage Metrics for Selected Stocks:")
        logger.info("-" * 80)
        logger.info(f"  Avg Overall Score: {selected_df['overall_score'].mean():.2f}/100")
        logger.info(f"  Avg Data Quality: {selected_df['data_quality_score'].mean():.2f}/100")
        logger.info(f"  Avg Liquidity: {selected_df['liquidity_score'].mean():.2f}/100")
        logger.info(f"  Avg Volatility: {selected_df['annualized_volatility'].mean():.2f}%")
        logger.info(f"  Avg Years of Data: {selected_df['years_of_data'].mean():.2f}")
        
        logger.info("\n" + "=" * 80)


def main():
    """Main execution"""
    analyzer = StockAnalyzer()
    
    # Analyze all stocks
    logger.info("Analyzing 30 stocks for training suitability...")
    df = analyzer.analyze_all_stocks()
    
    # Save full analysis
    output_file = '/home/ubuntu/quant-trading-web/python_system/ml/stock_analysis_results.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Full analysis saved to: {output_file}")
    
    # Select best 15
    selected = analyzer.select_best_stocks(df, n=15, max_per_sector=5)
    
    # Generate report
    analyzer.generate_report(df, selected)
    
    # Save selected stocks list
    selected_file = '/home/ubuntu/quant-trading-web/python_system/ml/selected_stocks.json'
    with open(selected_file, 'w') as f:
        json.dump({
            'selected_stocks': selected,
            'selection_date': datetime.now().isoformat(),
            'total_analyzed': len(df),
            'selection_criteria': {
                'data_quality_weight': 0.30,
                'liquidity_weight': 0.25,
                'volatility_weight': 0.20,
                'price_stability_weight': 0.15,
                'data_length_weight': 0.10,
                'max_per_sector': 5
            }
        }, f, indent=2)
    
    logger.info(f"\n✓ Selected stocks saved to: {selected_file}")
    
    return selected


if __name__ == '__main__':
    main()
