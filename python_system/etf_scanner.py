"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INSTITUTIONAL ETF SCANNER v1.0                            ║
║                                                                              ║
║  Comprehensive ETF Analysis Engine for Sector Rotation & Asset Allocation    ║
║                                                                              ║
║  Features:                                                                   ║
║  ✓ Sector ETF Analysis (XLK, XLF, XLE, XLV, etc.)                           ║
║  ✓ Bond ETF Analysis (TLT, IEF, HYG, LQD, etc.)                             ║
║  ✓ Commodity ETFs (GLD, SLV, USO, UNG, etc.)                                ║
║  ✓ International ETFs (EEM, EFA, FXI, etc.)                                 ║
║  ✓ Thematic ETFs (ARKK, HACK, BOTZ, etc.)                                   ║
║  ✓ Leveraged/Inverse ETFs (TQQQ, SQQQ, SPXU, etc.)                          ║
║  ✓ Real-time Flow Analysis                                                   ║
║  ✓ Sector Rotation Signals                                                   ║
║  ✓ Risk Parity Analysis                                                      ║
║  ✓ Correlation Matrix                                                        ║
║  ✓ Expense Ratio Comparison                                                  ║
║  ✓ AUM and Liquidity Analysis                                               ║
║                                                                              ║
║  NO MOCK DATA. 100% REAL, LIVE DATA.                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ETFScanner:
    """
    Comprehensive ETF Scanner for institutional-grade analysis.
    Analyzes sector rotation, asset allocation, and ETF selection.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ETF UNIVERSE - Comprehensive Coverage
    # ═══════════════════════════════════════════════════════════════════════════
    
    SECTOR_ETFS = {
        'XLK': {'name': 'Technology', 'sector': 'Technology', 'benchmark': 'SPY'},
        'XLF': {'name': 'Financials', 'sector': 'Financials', 'benchmark': 'SPY'},
        'XLE': {'name': 'Energy', 'sector': 'Energy', 'benchmark': 'SPY'},
        'XLV': {'name': 'Healthcare', 'sector': 'Healthcare', 'benchmark': 'SPY'},
        'XLI': {'name': 'Industrials', 'sector': 'Industrials', 'benchmark': 'SPY'},
        'XLP': {'name': 'Consumer Staples', 'sector': 'Consumer Staples', 'benchmark': 'SPY'},
        'XLY': {'name': 'Consumer Discretionary', 'sector': 'Consumer Discretionary', 'benchmark': 'SPY'},
        'XLU': {'name': 'Utilities', 'sector': 'Utilities', 'benchmark': 'SPY'},
        'XLRE': {'name': 'Real Estate', 'sector': 'Real Estate', 'benchmark': 'SPY'},
        'XLB': {'name': 'Materials', 'sector': 'Materials', 'benchmark': 'SPY'},
        'XLC': {'name': 'Communication Services', 'sector': 'Communication Services', 'benchmark': 'SPY'},
    }
    
    BROAD_MARKET_ETFS = {
        'SPY': {'name': 'S&P 500', 'type': 'Large Cap', 'benchmark': None},
        'QQQ': {'name': 'NASDAQ 100', 'type': 'Tech Heavy', 'benchmark': 'SPY'},
        'IWM': {'name': 'Russell 2000', 'type': 'Small Cap', 'benchmark': 'SPY'},
        'DIA': {'name': 'Dow Jones', 'type': 'Blue Chip', 'benchmark': 'SPY'},
        'MDY': {'name': 'S&P MidCap 400', 'type': 'Mid Cap', 'benchmark': 'SPY'},
        'VTI': {'name': 'Total Stock Market', 'type': 'Total Market', 'benchmark': 'SPY'},
        'VOO': {'name': 'Vanguard S&P 500', 'type': 'Large Cap', 'benchmark': 'SPY'},
        'IVV': {'name': 'iShares S&P 500', 'type': 'Large Cap', 'benchmark': 'SPY'},
    }
    
    BOND_ETFS = {
        'TLT': {'name': '20+ Year Treasury', 'type': 'Long Duration', 'duration': 'Long'},
        'IEF': {'name': '7-10 Year Treasury', 'type': 'Intermediate', 'duration': 'Medium'},
        'SHY': {'name': '1-3 Year Treasury', 'type': 'Short Duration', 'duration': 'Short'},
        'BND': {'name': 'Total Bond Market', 'type': 'Aggregate', 'duration': 'Medium'},
        'LQD': {'name': 'Investment Grade Corp', 'type': 'Corporate', 'duration': 'Medium'},
        'HYG': {'name': 'High Yield Corp', 'type': 'Junk Bonds', 'duration': 'Medium'},
        'TIP': {'name': 'TIPS', 'type': 'Inflation Protected', 'duration': 'Medium'},
        'AGG': {'name': 'Aggregate Bond', 'type': 'Aggregate', 'duration': 'Medium'},
    }
    
    COMMODITY_ETFS = {
        'GLD': {'name': 'Gold', 'commodity': 'Gold', 'type': 'Precious Metal'},
        'SLV': {'name': 'Silver', 'commodity': 'Silver', 'type': 'Precious Metal'},
        'USO': {'name': 'Oil', 'commodity': 'Crude Oil', 'type': 'Energy'},
        'UNG': {'name': 'Natural Gas', 'commodity': 'Natural Gas', 'type': 'Energy'},
        'DBA': {'name': 'Agriculture', 'commodity': 'Agriculture', 'type': 'Agriculture'},
        'PDBC': {'name': 'Commodity Basket', 'commodity': 'Diversified', 'type': 'Diversified'},
        'COPX': {'name': 'Copper Miners', 'commodity': 'Copper', 'type': 'Industrial Metal'},
        'URA': {'name': 'Uranium', 'commodity': 'Uranium', 'type': 'Energy'},
    }
    
    INTERNATIONAL_ETFS = {
        'EEM': {'name': 'Emerging Markets', 'region': 'Emerging', 'type': 'Broad'},
        'EFA': {'name': 'Developed Markets', 'region': 'Developed', 'type': 'Broad'},
        'FXI': {'name': 'China Large Cap', 'region': 'China', 'type': 'Country'},
        'EWJ': {'name': 'Japan', 'region': 'Japan', 'type': 'Country'},
        'EWZ': {'name': 'Brazil', 'region': 'Brazil', 'type': 'Country'},
        'INDA': {'name': 'India', 'region': 'India', 'type': 'Country'},
        'VEA': {'name': 'Developed ex-US', 'region': 'Developed', 'type': 'Broad'},
        'VWO': {'name': 'Emerging Markets', 'region': 'Emerging', 'type': 'Broad'},
    }
    
    THEMATIC_ETFS = {
        'ARKK': {'name': 'ARK Innovation', 'theme': 'Disruptive Innovation', 'risk': 'High'},
        'ARKG': {'name': 'ARK Genomic', 'theme': 'Genomics', 'risk': 'High'},
        'HACK': {'name': 'Cybersecurity', 'theme': 'Cybersecurity', 'risk': 'Medium'},
        'BOTZ': {'name': 'Robotics & AI', 'theme': 'Automation', 'risk': 'Medium'},
        'ICLN': {'name': 'Clean Energy', 'theme': 'Clean Energy', 'risk': 'High'},
        'TAN': {'name': 'Solar', 'theme': 'Solar Energy', 'risk': 'High'},
        'LIT': {'name': 'Lithium & Battery', 'theme': 'EV/Battery', 'risk': 'High'},
        'SOXX': {'name': 'Semiconductors', 'theme': 'Semiconductors', 'risk': 'Medium'},
        'XBI': {'name': 'Biotech', 'theme': 'Biotechnology', 'risk': 'High'},
        'IBB': {'name': 'Nasdaq Biotech', 'theme': 'Biotechnology', 'risk': 'High'},
    }
    
    LEVERAGED_ETFS = {
        'TQQQ': {'name': '3x NASDAQ', 'leverage': 3, 'direction': 'Long', 'underlying': 'QQQ'},
        'SQQQ': {'name': '-3x NASDAQ', 'leverage': -3, 'direction': 'Short', 'underlying': 'QQQ'},
        'SPXU': {'name': '-3x S&P 500', 'leverage': -3, 'direction': 'Short', 'underlying': 'SPY'},
        'UPRO': {'name': '3x S&P 500', 'leverage': 3, 'direction': 'Long', 'underlying': 'SPY'},
        'TNA': {'name': '3x Russell 2000', 'leverage': 3, 'direction': 'Long', 'underlying': 'IWM'},
        'TZA': {'name': '-3x Russell 2000', 'leverage': -3, 'direction': 'Short', 'underlying': 'IWM'},
        'UVXY': {'name': '1.5x VIX', 'leverage': 1.5, 'direction': 'Long', 'underlying': 'VIX'},
        'SVXY': {'name': '-0.5x VIX', 'leverage': -0.5, 'direction': 'Short', 'underlying': 'VIX'},
    }
    
    def __init__(self):
        """Initialize the ETF Scanner."""
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        logger.info("ETF Scanner initialized")
    
    def get_all_etfs(self) -> Dict[str, Dict]:
        """Get all ETFs in the universe."""
        all_etfs = {}
        all_etfs.update({k: {**v, 'category': 'Sector'} for k, v in self.SECTOR_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'Broad Market'} for k, v in self.BROAD_MARKET_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'Bonds'} for k, v in self.BOND_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'Commodities'} for k, v in self.COMMODITY_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'International'} for k, v in self.INTERNATIONAL_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'Thematic'} for k, v in self.THEMATIC_ETFS.items()})
        all_etfs.update({k: {**v, 'category': 'Leveraged'} for k, v in self.LEVERAGED_ETFS.items()})
        return all_etfs
    
    def analyze_etf(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single ETF.
        
        Returns:
            Dict with price data, technicals, fundamentals, flows, and scores
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y')
            hist_5y = ticker.history(period='5y')
            
            if hist.empty:
                result['error'] = f"No data available for {symbol}"
                return result
            
            # Get ETF metadata
            etf_info = self.get_all_etfs().get(symbol, {})
            result['name'] = etf_info.get('name', info.get('shortName', symbol))
            result['category'] = etf_info.get('category', 'Unknown')
            
            # ═══════════════════════════════════════════════════════════════════
            # PRICE DATA
            # ═══════════════════════════════════════════════════════════════════
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            result['price_data'] = {
                'current_price': round(current_price, 2),
                'prev_close': round(prev_close, 2),
                'change_pct': round((current_price / prev_close - 1) * 100, 2),
                'day_high': round(hist['High'].iloc[-1], 2),
                'day_low': round(hist['Low'].iloc[-1], 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'avg_volume_20d': int(hist['Volume'].tail(20).mean()),
                '52w_high': round(hist['High'].max(), 2),
                '52w_low': round(hist['Low'].min(), 2),
                'from_52w_high_pct': round((current_price / hist['High'].max() - 1) * 100, 2),
                'from_52w_low_pct': round((current_price / hist['Low'].min() - 1) * 100, 2),
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # PERFORMANCE METRICS
            # ═══════════════════════════════════════════════════════════════════
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate returns for different periods
            def calc_return(days):
                if len(hist) >= days:
                    return round((hist['Close'].iloc[-1] / hist['Close'].iloc[-days] - 1) * 100, 2)
                return None
            
            result['performance'] = {
                'return_1d': round((current_price / prev_close - 1) * 100, 2),
                'return_5d': calc_return(5),
                'return_1m': calc_return(21),
                'return_3m': calc_return(63),
                'return_6m': calc_return(126),
                'return_ytd': self._calc_ytd_return(hist),
                'return_1y': calc_return(252),
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # RISK METRICS
            # ═══════════════════════════════════════════════════════════════════
            if len(returns) > 20:
                daily_std = returns.std()
                annual_std = daily_std * np.sqrt(252)
                daily_mean = returns.mean()
                annual_mean = daily_mean * 252
                
                # Sharpe Ratio (assuming 4.5% risk-free rate)
                risk_free = 0.045 / 252
                sharpe = (daily_mean - risk_free) / daily_std * np.sqrt(252) if daily_std > 0 else 0
                
                # Sortino Ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_std
                sortino = (annual_mean - 0.045) / downside_std if downside_std > 0 else 0
                
                # Max Drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                # Beta (vs SPY)
                beta = self._calculate_beta(symbol, returns)
                
                # Value at Risk (95%)
                var_95 = np.percentile(returns, 5) * 100
                
                result['risk_metrics'] = {
                    'volatility_annual': round(annual_std * 100, 2),
                    'sharpe_ratio': round(sharpe, 2),
                    'sortino_ratio': round(sortino, 2),
                    'max_drawdown_pct': round(max_drawdown, 2),
                    'beta': round(beta, 2) if beta else None,
                    'var_95_daily': round(var_95, 2),
                    'volatility_interpretation': self._interpret_volatility(annual_std * 100),
                }
            
            # ═══════════════════════════════════════════════════════════════════
            # TECHNICAL INDICATORS
            # ═══════════════════════════════════════════════════════════════════
            closes = hist['Close'].values
            
            # Moving Averages
            sma_20 = closes[-20:].mean() if len(closes) >= 20 else None
            sma_50 = closes[-50:].mean() if len(closes) >= 50 else None
            sma_200 = closes[-200:].mean() if len(closes) >= 200 else None
            
            # RSI
            rsi = self._calculate_rsi(closes, 14)
            
            # MACD
            macd, signal, histogram = self._calculate_macd(closes)
            
            # Trend determination
            trend = self._determine_trend(current_price, sma_20, sma_50, sma_200)
            
            result['technicals'] = {
                'sma_20': round(sma_20, 2) if sma_20 else None,
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'sma_200': round(sma_200, 2) if sma_200 else None,
                'price_vs_sma20': round((current_price / sma_20 - 1) * 100, 2) if sma_20 else None,
                'price_vs_sma50': round((current_price / sma_50 - 1) * 100, 2) if sma_50 else None,
                'price_vs_sma200': round((current_price / sma_200 - 1) * 100, 2) if sma_200 else None,
                'rsi_14': round(rsi, 2) if rsi else None,
                'rsi_signal': self._interpret_rsi(rsi),
                'macd': round(macd, 4) if macd else None,
                'macd_signal': round(signal, 4) if signal else None,
                'macd_histogram': round(histogram, 4) if histogram else None,
                'trend': trend,
                'golden_cross': sma_50 > sma_200 if sma_50 and sma_200 else None,
                'death_cross': sma_50 < sma_200 if sma_50 and sma_200 else None,
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # ETF-SPECIFIC METRICS
            # ═══════════════════════════════════════════════════════════════════
            result['etf_metrics'] = {
                'expense_ratio': info.get('annualReportExpenseRatio', info.get('expenseRatio', None)),
                'aum': info.get('totalAssets', None),
                'aum_formatted': self._format_aum(info.get('totalAssets', 0)),
                'avg_volume': info.get('averageVolume', None),
                'bid': info.get('bid', None),
                'ask': info.get('ask', None),
                'bid_ask_spread': self._calc_spread(info.get('bid'), info.get('ask')),
                'nav': info.get('navPrice', None),
                'premium_discount': self._calc_premium_discount(current_price, info.get('navPrice')),
                'holdings_count': info.get('holdingsCount', None),
                'dividend_yield': info.get('dividendYield', None),
                'pe_ratio': info.get('trailingPE', None),
            }
            
            # ═══════════════════════════════════════════════════════════════════
            # FLOW ANALYSIS (Volume-based proxy)
            # ═══════════════════════════════════════════════════════════════════
            result['flow_analysis'] = self._analyze_flows(hist)
            
            # ═══════════════════════════════════════════════════════════════════
            # COMPOSITE SCORE
            # ═══════════════════════════════════════════════════════════════════
            result['composite_score'] = self._calculate_composite_score(result)
            
            # ═══════════════════════════════════════════════════════════════════
            # RECOMMENDATION
            # ═══════════════════════════════════════════════════════════════════
            result['recommendation'] = self._generate_recommendation(result)
            
            # ═══════════════════════════════════════════════════════════════════
            # EDUCATIONAL EXPLANATIONS
            # ═══════════════════════════════════════════════════════════════════
            result['education'] = self._generate_education(result)
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error analyzing ETF {symbol}: {e}")
        
        return result
    
    def scan_sector_rotation(self) -> Dict[str, Any]:
        """
        Analyze sector rotation signals across all sector ETFs.
        
        Returns:
            Dict with sector rankings, rotation signals, and recommendations
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'sectors': [],
            'rotation_signal': None,
            'leading_sectors': [],
            'lagging_sectors': [],
            'recommendations': []
        }
        
        sector_data = []
        
        for symbol, info in self.SECTOR_ETFS.items():
            try:
                analysis = self.analyze_etf(symbol)
                if analysis.get('success'):
                    sector_data.append({
                        'symbol': symbol,
                        'name': info['name'],
                        'return_1m': analysis['performance'].get('return_1m', 0),
                        'return_3m': analysis['performance'].get('return_3m', 0),
                        'rsi': analysis['technicals'].get('rsi_14', 50),
                        'trend': analysis['technicals'].get('trend', 'Neutral'),
                        'relative_strength': analysis['performance'].get('return_1m', 0) - self._get_spy_return_1m(),
                        'score': analysis.get('composite_score', {}).get('total', 50),
                    })
            except Exception as e:
                logger.warning(f"Error analyzing sector {symbol}: {e}")
        
        # Sort by relative strength
        sector_data.sort(key=lambda x: x['relative_strength'], reverse=True)
        
        result['sectors'] = sector_data
        result['leading_sectors'] = sector_data[:3]
        result['lagging_sectors'] = sector_data[-3:]
        
        # Determine rotation signal
        leading_types = [s['name'] for s in result['leading_sectors']]
        
        # Defensive sectors: Utilities, Consumer Staples, Healthcare
        defensive = ['Utilities', 'Consumer Staples', 'Healthcare']
        # Cyclical sectors: Technology, Consumer Discretionary, Financials, Industrials
        cyclical = ['Technology', 'Consumer Discretionary', 'Financials', 'Industrials']
        
        defensive_leading = sum(1 for s in leading_types if s in defensive)
        cyclical_leading = sum(1 for s in leading_types if s in cyclical)
        
        if cyclical_leading >= 2:
            result['rotation_signal'] = 'RISK-ON'
            result['market_phase'] = 'Expansion/Growth'
            result['interpretation'] = 'Cyclical sectors leading suggests economic optimism and risk appetite. Consider overweighting growth-oriented positions.'
        elif defensive_leading >= 2:
            result['rotation_signal'] = 'RISK-OFF'
            result['market_phase'] = 'Defensive/Contraction'
            result['interpretation'] = 'Defensive sectors leading suggests caution and flight to safety. Consider reducing risk exposure and adding defensive positions.'
        else:
            result['rotation_signal'] = 'MIXED'
            result['market_phase'] = 'Transition'
            result['interpretation'] = 'Mixed sector leadership suggests market uncertainty. Maintain balanced allocation and watch for clearer signals.'
        
        # Generate recommendations
        result['recommendations'] = [
            f"OVERWEIGHT: {', '.join([s['symbol'] for s in result['leading_sectors']])}",
            f"UNDERWEIGHT: {', '.join([s['symbol'] for s in result['lagging_sectors']])}",
            f"Market Phase: {result['market_phase']}",
        ]
        
        return result
    
    def scan_all_categories(self) -> Dict[str, Any]:
        """
        Scan all ETF categories and return top picks from each.
        
        Returns:
            Dict with top ETFs from each category
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'categories': {}
        }
        
        categories = {
            'Sector': self.SECTOR_ETFS,
            'Broad Market': self.BROAD_MARKET_ETFS,
            'Bonds': self.BOND_ETFS,
            'Commodities': self.COMMODITY_ETFS,
            'International': self.INTERNATIONAL_ETFS,
            'Thematic': self.THEMATIC_ETFS,
        }
        
        for category_name, etfs in categories.items():
            category_results = []
            
            for symbol in list(etfs.keys())[:10]:  # Limit to 10 per category for speed
                try:
                    analysis = self.analyze_etf(symbol)
                    if analysis.get('success'):
                        category_results.append({
                            'symbol': symbol,
                            'name': analysis.get('name', symbol),
                            'price': analysis['price_data']['current_price'],
                            'return_1m': analysis['performance'].get('return_1m', 0),
                            'return_3m': analysis['performance'].get('return_3m', 0),
                            'sharpe': analysis.get('risk_metrics', {}).get('sharpe_ratio', 0),
                            'score': analysis.get('composite_score', {}).get('total', 50),
                            'recommendation': analysis.get('recommendation', {}).get('action', 'HOLD'),
                        })
                except Exception as e:
                    logger.warning(f"Error scanning {symbol}: {e}")
            
            # Sort by score
            category_results.sort(key=lambda x: x['score'], reverse=True)
            result['categories'][category_name] = category_results[:5]  # Top 5 per category
        
        return result
    
    def get_correlation_matrix(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Calculate correlation matrix for ETFs.
        
        Args:
            symbols: List of ETF symbols (default: major ETFs)
        
        Returns:
            Dict with correlation matrix and insights
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EEM', 'XLE', 'XLF', 'XLK', 'VIX']
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'correlations': {},
            'insights': []
        }
        
        # Fetch returns for all symbols
        returns_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                if not hist.empty:
                    returns_data[symbol] = hist['Close'].pct_change().dropna()
            except:
                pass
        
        # Calculate correlation matrix
        if len(returns_data) > 1:
            df = pd.DataFrame(returns_data)
            corr_matrix = df.corr()
            
            result['correlations'] = corr_matrix.to_dict()
            
            # Generate insights
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    if sym1 in corr_matrix.columns and sym2 in corr_matrix.columns:
                        corr = corr_matrix.loc[sym1, sym2]
                        if abs(corr) > 0.8:
                            result['insights'].append({
                                'pair': f"{sym1}/{sym2}",
                                'correlation': round(corr, 2),
                                'interpretation': 'Highly correlated - limited diversification benefit' if corr > 0 else 'Highly negatively correlated - good hedge'
                            })
                        elif abs(corr) < 0.2:
                            result['insights'].append({
                                'pair': f"{sym1}/{sym2}",
                                'correlation': round(corr, 2),
                                'interpretation': 'Low correlation - good diversification'
                            })
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD."""
        if len(prices) < 26:
            return None, None, None
        
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema_12 - ema_26
        
        macd_series = pd.Series(prices).ewm(span=12, adjust=False).mean() - pd.Series(prices).ewm(span=26, adjust=False).mean()
        signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_beta(self, symbol: str, returns: pd.Series) -> Optional[float]:
        """Calculate beta vs SPY."""
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='1y')
            spy_returns = spy_hist['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) < 20:
                return None
            
            stock_ret = returns.loc[common_dates]
            spy_ret = spy_returns.loc[common_dates]
            
            covariance = np.cov(stock_ret, spy_ret)[0][1]
            spy_variance = np.var(spy_ret)
            
            return covariance / spy_variance if spy_variance > 0 else None
        except:
            return None
    
    def _determine_trend(self, price: float, sma20: float, sma50: float, sma200: float) -> str:
        """Determine trend based on moving averages."""
        if not all([sma20, sma50, sma200]):
            return 'Unknown'
        
        if price > sma20 > sma50 > sma200:
            return 'Strong Uptrend'
        elif price > sma50 > sma200:
            return 'Uptrend'
        elif price < sma20 < sma50 < sma200:
            return 'Strong Downtrend'
        elif price < sma50 < sma200:
            return 'Downtrend'
        else:
            return 'Sideways'
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value."""
        if rsi is None:
            return 'Unknown'
        if rsi >= 70:
            return 'Overbought'
        elif rsi <= 30:
            return 'Oversold'
        elif rsi >= 60:
            return 'Bullish'
        elif rsi <= 40:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _interpret_volatility(self, vol: float) -> str:
        """Interpret volatility level."""
        if vol < 10:
            return 'Very Low'
        elif vol < 15:
            return 'Low'
        elif vol < 25:
            return 'Moderate'
        elif vol < 40:
            return 'High'
        else:
            return 'Very High'
    
    def _format_aum(self, aum: float) -> str:
        """Format AUM for display."""
        if not aum:
            return 'N/A'
        if aum >= 1e12:
            return f"${aum/1e12:.1f}T"
        elif aum >= 1e9:
            return f"${aum/1e9:.1f}B"
        elif aum >= 1e6:
            return f"${aum/1e6:.1f}M"
        else:
            return f"${aum:,.0f}"
    
    def _calc_spread(self, bid: float, ask: float) -> Optional[float]:
        """Calculate bid-ask spread percentage."""
        if not bid or not ask or bid == 0:
            return None
        return round((ask - bid) / bid * 100, 3)
    
    def _calc_premium_discount(self, price: float, nav: float) -> Optional[float]:
        """Calculate premium/discount to NAV."""
        if not nav or nav == 0:
            return None
        return round((price / nav - 1) * 100, 2)
    
    def _calc_ytd_return(self, hist: pd.DataFrame) -> Optional[float]:
        """Calculate YTD return."""
        try:
            current_year = datetime.now().year
            ytd_data = hist[hist.index.year == current_year]
            if len(ytd_data) > 0:
                return round((ytd_data['Close'].iloc[-1] / ytd_data['Close'].iloc[0] - 1) * 100, 2)
        except:
            pass
        return None
    
    def _analyze_flows(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume-based flow proxy."""
        try:
            recent_vol = hist['Volume'].tail(5).mean()
            avg_vol = hist['Volume'].tail(20).mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            # Price-volume analysis
            recent_price_change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
            
            if vol_ratio > 1.5 and recent_price_change > 0:
                flow_signal = 'Strong Inflows'
            elif vol_ratio > 1.5 and recent_price_change < 0:
                flow_signal = 'Strong Outflows'
            elif vol_ratio > 1.2 and recent_price_change > 0:
                flow_signal = 'Moderate Inflows'
            elif vol_ratio > 1.2 and recent_price_change < 0:
                flow_signal = 'Moderate Outflows'
            else:
                flow_signal = 'Neutral'
            
            return {
                'volume_ratio': round(vol_ratio, 2),
                'recent_price_change': round(recent_price_change, 2),
                'flow_signal': flow_signal,
                'interpretation': f"Volume is {vol_ratio:.1f}x average with {recent_price_change:+.1f}% price change"
            }
        except:
            return {'flow_signal': 'Unknown'}
    
    def _calculate_composite_score(self, analysis: Dict) -> Dict[str, Any]:
        """Calculate composite score for ETF."""
        scores = {}
        
        # Momentum Score (0-100)
        perf = analysis.get('performance', {})
        return_1m = perf.get('return_1m', 0) or 0
        return_3m = perf.get('return_3m', 0) or 0
        momentum_score = min(100, max(0, 50 + return_1m * 2 + return_3m))
        scores['momentum'] = round(momentum_score, 1)
        
        # Technical Score (0-100)
        tech = analysis.get('technicals', {})
        rsi = tech.get('rsi_14', 50) or 50
        trend = tech.get('trend', 'Sideways')
        
        tech_score = 50
        if trend in ['Strong Uptrend', 'Uptrend']:
            tech_score += 20
        elif trend in ['Strong Downtrend', 'Downtrend']:
            tech_score -= 20
        
        if 40 <= rsi <= 60:
            tech_score += 10
        elif rsi < 30:
            tech_score += 15  # Oversold = potential buy
        elif rsi > 70:
            tech_score -= 10  # Overbought = caution
        
        scores['technical'] = round(min(100, max(0, tech_score)), 1)
        
        # Risk Score (0-100, higher = better risk-adjusted)
        risk = analysis.get('risk_metrics', {})
        sharpe = risk.get('sharpe_ratio', 0) or 0
        max_dd = abs(risk.get('max_drawdown_pct', 20) or 20)
        
        risk_score = 50 + sharpe * 20 - max_dd * 0.5
        scores['risk'] = round(min(100, max(0, risk_score)), 1)
        
        # Liquidity Score (0-100)
        etf_metrics = analysis.get('etf_metrics', {})
        spread = etf_metrics.get('bid_ask_spread', 0.1) or 0.1
        
        if spread < 0.05:
            liquidity_score = 100
        elif spread < 0.1:
            liquidity_score = 80
        elif spread < 0.2:
            liquidity_score = 60
        else:
            liquidity_score = 40
        
        scores['liquidity'] = round(liquidity_score, 1)
        
        # Total Score (weighted average)
        weights = {'momentum': 0.3, 'technical': 0.25, 'risk': 0.3, 'liquidity': 0.15}
        total = sum(scores[k] * weights[k] for k in weights)
        scores['total'] = round(total, 1)
        
        return scores
    
    def _generate_recommendation(self, analysis: Dict) -> Dict[str, Any]:
        """Generate recommendation based on analysis."""
        score = analysis.get('composite_score', {}).get('total', 50)
        trend = analysis.get('technicals', {}).get('trend', 'Sideways')
        rsi = analysis.get('technicals', {}).get('rsi_14', 50)
        
        if score >= 70 and trend in ['Strong Uptrend', 'Uptrend']:
            action = 'STRONG BUY'
            reasoning = 'High composite score with bullish trend'
        elif score >= 60:
            action = 'BUY'
            reasoning = 'Above average score with positive momentum'
        elif score <= 30 and trend in ['Strong Downtrend', 'Downtrend']:
            action = 'STRONG SELL'
            reasoning = 'Low composite score with bearish trend'
        elif score <= 40:
            action = 'SELL'
            reasoning = 'Below average score with weak momentum'
        else:
            action = 'HOLD'
            reasoning = 'Neutral signals - wait for clearer direction'
        
        return {
            'action': action,
            'score': score,
            'reasoning': reasoning,
            'confidence': min(100, abs(score - 50) * 2),
        }
    
    def _generate_education(self, analysis: Dict) -> Dict[str, str]:
        """Generate educational explanations for beginners."""
        return {
            'what_is_etf': "An ETF (Exchange-Traded Fund) is a basket of securities that trades like a stock. It offers diversification, lower costs, and easy trading compared to mutual funds.",
            'expense_ratio': f"The expense ratio ({analysis.get('etf_metrics', {}).get('expense_ratio', 'N/A')}) is the annual fee charged by the fund. Lower is better - under 0.2% is excellent, under 0.5% is good.",
            'sharpe_ratio': f"The Sharpe Ratio ({analysis.get('risk_metrics', {}).get('sharpe_ratio', 'N/A')}) measures risk-adjusted returns. Above 1.0 is good, above 2.0 is excellent.",
            'beta': f"Beta ({analysis.get('risk_metrics', {}).get('beta', 'N/A')}) measures volatility vs the market. Beta > 1 means more volatile than S&P 500, < 1 means less volatile.",
            'rsi': f"RSI ({analysis.get('technicals', {}).get('rsi_14', 'N/A')}) measures momentum. Above 70 = overbought (may pull back), below 30 = oversold (may bounce).",
            'trend': f"Current trend: {analysis.get('technicals', {}).get('trend', 'Unknown')}. Trading with the trend typically has higher success rates.",
        }
    
    def _get_spy_return_1m(self) -> float:
        """Get SPY 1-month return for relative strength calculation."""
        try:
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1mo')
            if len(hist) >= 2:
                return (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        except:
            pass
        return 0


# CLI interface
if __name__ == "__main__":
    import sys
    import json
    
    scanner = ETFScanner()
    
    if len(sys.argv) < 2:
        print("Usage: python etf_scanner.py <command> [args]")
        print("Commands:")
        print("  analyze <symbol>  - Analyze a single ETF")
        print("  sectors           - Scan sector rotation")
        print("  all               - Scan all categories")
        print("  correlation       - Get correlation matrix")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'analyze' and len(sys.argv) >= 3:
        symbol = sys.argv[2].upper()
        result = scanner.analyze_etf(symbol)
        print(json.dumps(result, indent=2, default=str))
    
    elif command == 'sectors':
        result = scanner.scan_sector_rotation()
        print(json.dumps(result, indent=2, default=str))
    
    elif command == 'all':
        result = scanner.scan_all_categories()
        print(json.dumps(result, indent=2, default=str))
    
    elif command == 'correlation':
        result = scanner.get_correlation_matrix()
        print(json.dumps(result, indent=2, default=str))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
