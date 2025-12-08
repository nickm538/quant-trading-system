"""
INSTITUTIONAL-GRADE QUANTITATIVE TRADING SYSTEM
Complete Integration - All Components Working Synergistically
REAL DATA, REAL COMPUTATIONS - NO SIMULATIONS

For actual money management with real market trading
New World Standard System
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import requests
from dataclasses import dataclass, asdict

# Import our modules
from data.enhanced_data_ingestion import EnhancedDataIngestion
from models.technical_indicators import TechnicalIndicators
from models.stochastic_models import StochasticModels, GARCHResults, MonteCarloResults

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/quant_trading_system/logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with all analysis components"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-100
    
    # Price information
    current_price: float
    target_price: float
    stop_loss: float
    
    # Technical analysis
    technical_score: float
    momentum_score: float
    trend_score: float
    volatility_score: float
    
    # Stochastic analysis
    expected_return: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    
    # Options recommendation
    recommended_option: Optional[Dict] = None
    option_delta: Optional[float] = None
    option_expiry: Optional[str] = None
    
    # News sentiment
    news_sentiment: float = 0.0
    catalyst_events: List[str] = None
    
    # Risk metrics
    position_size: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Bankroll-specific fields
    bankroll: float = 1000.0
    shares: int = 0
    position_value: float = 0.0
    dollar_risk: float = 0.0
    dollar_reward: float = 0.0


class InstitutionalTradingSystem:
    """
    Complete institutional-grade trading system
    Integrates all components synergistically
    """
    
    # Alpha Vantage API key
    ALPHA_VANTAGE_API_KEY = "7EFC2F6EO7HWLF7Q"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        """Initialize the complete trading system"""
        logger.info("=" * 80)
        logger.info("INITIALIZING INSTITUTIONAL-GRADE TRADING SYSTEM")
        logger.info("=" * 80)
        
        self.data_ingestion = EnhancedDataIngestion()
        self.technical_indicators = TechnicalIndicators()
        self.stochastic_models = StochasticModels(random_seed=42)
        
        logger.info("✓ All components initialized")
        logger.info("  - Data Ingestion: Yahoo Finance + Finnhub + Alpha Vantage")
        logger.info("  - Technical Indicators: 50+ indicators with correlation analysis")
        logger.info("  - Stochastic Models: GARCH + Monte Carlo + Jump-Diffusion + Heston")
        logger.info("=" * 80)
    
    # ==================== ALPHA VANTAGE INTEGRATION ====================
    
    def get_intraday_data_alphavantage(
        self,
        symbol: str,
        interval: str = '5min'
    ) -> pd.DataFrame:
        """
        Get real-time intraday data from Alpha Vantage
        REAL INTRADAY DATA
        """
        try:
            logger.info(f"Fetching REAL intraday data for {symbol} from Alpha Vantage")
            
            url = f"{self.ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.ALPHA_VANTAGE_API_KEY,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if f'Time Series ({interval})' in data:
                time_series = data[f'Time Series ({interval})']
                
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                logger.info(f"✓ Fetched {len(df)} intraday data points from Alpha Vantage")
                return df
            else:
                logger.warning(f"No intraday data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage intraday data: {str(e)}")
            return pd.DataFrame()
    
    def get_news_sentiment_alphavantage(self, symbol: str) -> Dict:
        """
        Get real-time news sentiment from Alpha Vantage
        REAL NEWS SENTIMENT FOR CATALYST DETECTION
        """
        try:
            logger.info(f"Fetching REAL news sentiment for {symbol} from Alpha Vantage")
            
            url = f"{self.ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.ALPHA_VANTAGE_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                articles = data['feed']
                
                # Calculate aggregate sentiment
                sentiments = []
                relevance_scores = []
                
                for article in articles:
                    if 'ticker_sentiment' in article:
                        for ticker_sent in article['ticker_sentiment']:
                            if ticker_sent['ticker'] == symbol:
                                sentiment_score = float(ticker_sent.get('ticker_sentiment_score', 0))
                                relevance = float(ticker_sent.get('relevance_score', 0))
                                sentiments.append(sentiment_score)
                                relevance_scores.append(relevance)
                
                if sentiments:
                    # Weighted average by relevance
                    weighted_sentiment = np.average(sentiments, weights=relevance_scores)
                else:
                    weighted_sentiment = 0.0
                
                logger.info(f"✓ Fetched {len(articles)} news articles")
                logger.info(f"  Weighted sentiment: {weighted_sentiment:.3f}")
                
                return {
                    'articles': articles,
                    'sentiment_score': weighted_sentiment,
                    'num_articles': len(articles)
                }
            else:
                logger.warning(f"No news sentiment data for {symbol}")
                return {'sentiment_score': 0.0, 'num_articles': 0}
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news sentiment: {str(e)}")
            return {'sentiment_score': 0.0, 'num_articles': 0}
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    def analyze_stock_comprehensive(
        self,
        symbol: str,
        monte_carlo_sims: int = 20000,  # Increased from 10k to 20k for maximum accuracy
        forecast_days: int = 30,
        bankroll: float = 1000.0  # User's trading bankroll
    ) -> Dict:
        """
        Complete comprehensive analysis of a stock
        ALL REAL DATA AND COMPUTATIONS
        """
        logger.info("=" * 80)
        logger.info(f"COMPREHENSIVE ANALYSIS FOR {symbol}")
        logger.info("=" * 80)
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price_data': None,
            'technical_analysis': {},
            'stochastic_analysis': {},
            'options_analysis': {},
            'sentiment_analysis': {},
            'risk_metrics': {},
            'recommendation': None
        }
        
        # 1. Get complete data
        logger.info("\n1. Fetching COMPLETE REAL DATA...")
        complete_data = self.data_ingestion.get_complete_stock_data(symbol)
        
        if complete_data['price_data'].empty:
            logger.error(f"No price data available for {symbol}")
            return analysis
        
        price_data = complete_data['price_data']
        analysis['price_data'] = price_data
        
        current_price = price_data['close'].iloc[-1]
        logger.info(f"  Current price: ${current_price:.2f}")
        
        # 2. Calculate ALL technical indicators
        logger.info("\n2. Calculating ALL 50+ TECHNICAL INDICATORS...")
        df_with_indicators = self.technical_indicators.calculate_all_indicators(price_data)
        
        # Calculate technical scores
        latest = df_with_indicators.iloc[-1]
        
        # Momentum score (0-100)
        rsi = latest['rsi_14']
        momentum_score = 100 - abs(rsi - 50)  # Best at 50
        
        # Trend score
        if latest['close'] > latest['sma_50'] > latest['sma_200']:
            trend_score = 80 + (latest['adx'] / 100 * 20)  # Strong uptrend
        elif latest['close'] < latest['sma_50'] < latest['sma_200']:
            trend_score = 20 - (latest['adx'] / 100 * 20)  # Strong downtrend
        else:
            trend_score = 50  # Neutral
        
        # Volatility score
        volatility_score = 100 - (latest['hist_vol_20'] * 100)  # Lower vol = higher score
        
        technical_score = (momentum_score + trend_score + volatility_score) / 3
        
        # Extract all indicator values for raw data display
        indicator_values = {}
        for col in df_with_indicators.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume', 'adj_close']:
                val = latest[col]
                if pd.notna(val):
                    indicator_values[col] = float(val)
        
        analysis['technical_analysis'] = {
            'technical_score': technical_score,
            'momentum_score': momentum_score,
            'trend_score': trend_score,
            'volatility_score': volatility_score,
            'rsi': rsi,
            'macd': latest['macd'],
            'adx': latest['adx'],
            'current_volatility': latest['hist_vol_20'],
            'all_indicators': indicator_values
        }
        
        logger.info(f"  Technical Score: {technical_score:.2f}/100")
        logger.info(f"  Momentum: {momentum_score:.2f}, Trend: {trend_score:.2f}, Volatility: {volatility_score:.2f}")
        
        # 3. REAL GARCH volatility modeling
        logger.info("\n3. Fitting REAL GARCH MODEL with FAT-TAIL DISTRIBUTIONS...")
        returns = price_data['close'].pct_change().dropna()
        
        try:
            garch_results = self.stochastic_models.fit_garch_model(
                returns,
                p=1,
                q=1,
                dist='studentst'
            )
            
            analysis['stochastic_analysis']['garch'] = {
                'aic': garch_results.aic,
                'bic': garch_results.bic,
                'current_volatility': garch_results.conditional_vol.iloc[-1],
                'fat_tail_df': garch_results.params.get('nu', None)
            }
            
            logger.info(f"  ✓ GARCH model fitted with fat tails (df={garch_results.params.get('nu', 'N/A')})")
            
        except Exception as e:
            logger.error(f"  Error fitting GARCH: {str(e)}")
            garch_results = None
        
        # 4. REAL Monte Carlo simulation
        logger.info(f"\n4. Running REAL MONTE CARLO SIMULATION ({monte_carlo_sims:,} paths)...")
        logger.info(f"   ⚠️  FULL COMPUTATION MODE - Every path calculated individually")
        logger.info(f"   ⚠️  Fat-tail distributions with Student-t (df=5.0)")
        logger.info(f"   ⚠️  Antithetic variates for variance reduction")
        
        mu = returns.mean() * 252  # Annualized return
        sigma = returns.std() * np.sqrt(252)  # Annualized volatility
        
        mc_results = self.stochastic_models.monte_carlo_gbm(
            S0=current_price,
            mu=mu,
            sigma=sigma,
            T=forecast_days / 252,
            steps=forecast_days,
            n_simulations=monte_carlo_sims,
            use_fat_tails=True,
            df=5.0,
            use_antithetic=True
        )
        
        expected_price = mc_results.mean_path[-1]
        expected_return = (expected_price - current_price) / current_price
        
        analysis['stochastic_analysis']['monte_carlo'] = {
            'expected_price': expected_price,
            'expected_return': expected_return,
            'var_95': mc_results.var_95,
            'cvar_95': mc_results.cvar_95,
            'max_drawdown': mc_results.max_drawdown,
            'ci_95_lower': mc_results.confidence_intervals['95_lower'][-1],
            'ci_95_upper': mc_results.confidence_intervals['95_upper'][-1],
            'mean_path': mc_results.mean_path.tolist(),
            'ci_95_lower_path': mc_results.confidence_intervals['95_lower'].tolist(),
            'ci_95_upper_path': mc_results.confidence_intervals['95_upper'].tolist()
        }
        
        logger.info(f"  Expected price in {forecast_days} days: ${expected_price:.2f}")
        logger.info(f"  Expected return: {expected_return*100:.2f}%")
        logger.info(f"  VaR (95%): {mc_results.var_95*100:.2f}%")
        logger.info(f"  CVaR (95%): {mc_results.cvar_95*100:.2f}%")
        
        # 5. Options analysis
        logger.info("\n5. Analyzing OPTIONS CHAIN (0.3-0.6 delta, >1 week expiry)...")
        
        options_data = complete_data['options']
        
        if options_data:
            best_options = self._find_best_options(options_data, current_price)
            analysis['options_analysis'] = best_options
            
            if best_options['best_call']:
                logger.info(f"  Best Call: Strike ${best_options['best_call']['strike']}, "
                          f"Delta {best_options['best_call']['delta']:.3f}, "
                          f"Expiry {best_options['best_call']['expiration']}")
            
            if best_options['best_put']:
                logger.info(f"  Best Put: Strike ${best_options['best_put']['strike']}, "
                          f"Delta {best_options['best_put']['delta']:.3f}, "
                          f"Expiry {best_options['best_put']['expiration']}")
        else:
            logger.warning("  No options data available")
        
        # 6. News sentiment analysis
        logger.info("\n6. Analyzing NEWS SENTIMENT...")
        
        # Alpha Vantage sentiment
        av_sentiment = self.get_news_sentiment_alphavantage(symbol)
        
        # Finnhub news
        news_articles = complete_data['news']
        
        analysis['sentiment_analysis'] = {
            'alpha_vantage_sentiment': av_sentiment['sentiment_score'],
            'num_articles': av_sentiment['num_articles'] + len(news_articles),
            'recent_news': [
                {'headline': article.get('headline', ''), 'datetime': article.get('datetime', '')}
                for article in news_articles[:5]
            ]
        }
        
        logger.info(f"  Sentiment score: {av_sentiment['sentiment_score']:.3f}")
        logger.info(f"  Total news articles: {av_sentiment['num_articles'] + len(news_articles)}")
                # 7. Generate final trading signal
        logger.info("\n7. Generating FINAL TRADING SIGNAL...")
        signal = self._generate_trading_signal(
            symbol=symbol,
            current_price=current_price,
            technical_analysis=analysis['technical_analysis'],
            stochastic_analysis=analysis['stochastic_analysis'],
            options_analysis=analysis['options_analysis'],
            sentiment_analysis=analysis['sentiment_analysis'],
            bankroll=bankroll,
            max_risk_pct=0.02  # Moderate 2% risk
        )       
        analysis['recommendation'] = asdict(signal)
        
        logger.info(f"  Signal: {signal.signal_type}")
        logger.info(f"  Confidence: {signal.confidence:.2f}%")
        logger.info(f"  Target Price: ${signal.target_price:.2f}")
        logger.info(f"  Stop Loss: ${signal.stop_loss:.2f}")
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE ANALYSIS COMPLETE")
        logger.info("=" * 80)
        
        return analysis
    
    def _find_best_options(
        self,
        options_data: Dict,
        current_price: float
    ) -> Dict:
        """
        Find best options based on delta, OI/Vol ratio, and risk/reward
        Medium risk/reward, 0.3-0.6 delta, >1 week expiry
        """
        best_call = None
        best_put = None
        best_call_score = -np.inf
        best_put_score = -np.inf
        
        for expiration, data in options_data.items():
            calls = data['calls']
            puts = data['puts']
            
            # Analyze calls
            if not calls.empty and 'delta' in calls.columns:
                for idx, row in calls.iterrows():
                    delta = row.get('delta', 0)
                    if 0.3 <= delta <= 0.6:
                        # Score based on delta, OI/Vol ratio, and implied volatility
                        oi_vol_ratio = row.get('oi_vol_ratio', 1)
                        iv = row.get('impliedVolatility', 0.5)
                        
                        # Prefer: mid-range delta, high OI/Vol (liquidity), reasonable IV
                        delta_score = 100 - abs(delta - 0.45) * 200  # Best at 0.45
                        liquidity_score = min(oi_vol_ratio * 10, 100)
                        iv_score = 100 - abs(iv - 0.3) * 200  # Prefer moderate IV
                        
                        score = (delta_score + liquidity_score + iv_score) / 3
                        
                        if score > best_call_score:
                            best_call_score = score
                            best_call = {
                                'strike': row['strike'],
                                'delta': delta,
                                'expiration': expiration,
                                'lastPrice': row.get('lastPrice', 0),
                                'impliedVolatility': iv,
                                'oi_vol_ratio': oi_vol_ratio,
                                'score': score
                            }
            
            # Analyze puts
            if not puts.empty and 'delta' in puts.columns:
                for idx, row in puts.iterrows():
                    delta = row.get('delta', 0)
                    if -0.6 <= delta <= -0.3:
                        oi_vol_ratio = row.get('oi_vol_ratio', 1)
                        iv = row.get('impliedVolatility', 0.5)
                        
                        delta_score = 100 - abs(abs(delta) - 0.45) * 200
                        liquidity_score = min(oi_vol_ratio * 10, 100)
                        iv_score = 100 - abs(iv - 0.3) * 200
                        
                        score = (delta_score + liquidity_score + iv_score) / 3
                        
                        if score > best_put_score:
                            best_put_score = score
                            best_put = {
                                'strike': row['strike'],
                                'delta': delta,
                                'expiration': expiration,
                                'lastPrice': row.get('lastPrice', 0),
                                'impliedVolatility': iv,
                                'oi_vol_ratio': oi_vol_ratio,
                                'score': score
                            }
        
        return {
            'best_call': best_call,
            'best_put': best_put
        }
    
    def _generate_trading_signal(
        self,
        symbol: str,
        current_price: float,
        technical_analysis: Dict,
        stochastic_analysis: Dict,
        options_analysis: Dict,
        sentiment_analysis: Dict,
        bankroll: float = 1000.0,  # Default $1,000 bankroll
        max_risk_pct: float = 0.02  # Moderate 2% max risk per trade
    ) -> TradingSignal:
        """
        Generate trading signal by combining all analysis components
        Synergistic weighting for maximum profitability
        """
        # Weight factors (calibrated for institutional performance)
        TECHNICAL_WEIGHT = 0.30
        STOCHASTIC_WEIGHT = 0.35
        SENTIMENT_WEIGHT = 0.20
        OPTIONS_WEIGHT = 0.15
        
        # Technical score (0-100)
        technical_score = technical_analysis['technical_score']
        
        # Stochastic score (0-100) - OPTIMIZED: Amplified sensitivity
        mc_data = stochastic_analysis.get('monte_carlo', {})
        expected_return = mc_data.get('expected_return', 0)
        # Amplify expected return impact: 10% return = 100 score, -10% return = 0 score
        stochastic_score = 50 + (expected_return * 500)  # 5x amplification for proper weighting
        stochastic_score = max(0, min(100, stochastic_score))
        
        # Sentiment score (0-100)
        sentiment_raw = sentiment_analysis.get('alpha_vantage_sentiment', 0)
        sentiment_score = 50 + (sentiment_raw * 50)  # Convert -1 to 1 scale to 0-100
        
        # Options score (0-100)
        best_call = options_analysis.get('best_call')
        best_put = options_analysis.get('best_put')
        
        if best_call:
            options_score = best_call.get('score', 50)
        elif best_put:
            options_score = best_put.get('score', 50)
        else:
            options_score = 50
        
        # Combined confidence score
        confidence = (
            technical_score * TECHNICAL_WEIGHT +
            stochastic_score * STOCHASTIC_WEIGHT +
            sentiment_score * SENTIMENT_WEIGHT +
            options_score * OPTIONS_WEIGHT
        )
        
        # Determine signal type
        if confidence >= 65:
            signal_type = 'BUY'
        elif confidence <= 35:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        # OPTIMIZED: Adaptive targets/stops based on GARCH volatility
        # Get current volatility from GARCH or technical analysis
        garch_data = stochastic_analysis.get('garch', {})
        current_vol = garch_data.get('current_volatility', technical_analysis.get('current_volatility', 0.02))
        
        # Calculate ATR-equivalent: volatility * price
        atr = current_vol * current_price
        
        # Adaptive targets based on signal type and volatility
        if signal_type == 'BUY':
            # Target = 2.5x volatility upside, Stop = 1.5x volatility downside
            target_price = current_price + (2.5 * atr)
            stop_loss = current_price - (1.5 * atr)
        elif signal_type == 'SELL':
            # Target = 2.5x volatility downside, Stop = 1.5x volatility upside
            target_price = current_price - (2.5 * atr)
            stop_loss = current_price + (1.5 * atr)
        else:  # HOLD
            # For HOLD, use expected return with tighter ranges
            if expected_return > 0.02:  # Bullish HOLD (>2% expected return)
                target_price = current_price + (1.5 * atr)
                stop_loss = current_price - (1.0 * atr)
            elif expected_return < -0.02:  # Bearish HOLD (<-2% expected return)
                target_price = current_price - (1.5 * atr)
                stop_loss = current_price + (1.0 * atr)
            else:  # Neutral HOLD
                # Minimal position for true neutral
                target_price = current_price + (0.5 * atr)
                stop_loss = current_price - (0.5 * atr)
        
        # Risk/reward ratio
        if signal_type == 'BUY':
            risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss) if (current_price - stop_loss) > 0 else 0
        elif signal_type == 'SELL':
            risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price) if (stop_loss - current_price) > 0 else 0
        else:  # HOLD
            # Calculate risk/reward even for HOLD to enable position sizing
            if expected_return > 0:
                # Bullish HOLD - treat like conservative BUY
                risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss) if (current_price - stop_loss) > 0 else 1.5
            else:
                # Bearish HOLD - treat like conservative SELL
                risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price) if (stop_loss - current_price) > 0 else 1.5
        
        # Kelly Criterion position sizing with moderate risk overlay
        var_95 = abs(mc_data.get('var_95', 0.1))
        
        # Calculate position size as percentage of bankroll
        # Kelly = (p * b - q) / b, where p=win prob, q=loss prob, b=win/loss ratio
        win_prob = confidence / 100
        loss_prob = 1 - win_prob
        win_loss_ratio = abs(risk_reward_ratio) if risk_reward_ratio > 0 else 1.5
        
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Apply moderate risk overlay: max 2% risk per trade
        stop_distance_pct = abs((current_price - stop_loss) / current_price)
        max_position_from_risk = max_risk_pct / stop_distance_pct if stop_distance_pct > 0 else 0.1
        
        # Take the more conservative of Kelly and risk-based sizing
        position_size_pct = min(kelly_fraction * 0.5, max_position_from_risk)  # Half-Kelly for safety
        position_size_pct = max(0.01, min(position_size_pct, 0.20))  # Between 1% and 20%
        
        # OPTIMIZED: Reduce position size for HOLD signals
        if signal_type == 'HOLD':
            if abs(expected_return) < 0.02:  # True neutral (<2% expected return)
                position_size_pct = position_size_pct * 0.25  # Quarter position for neutral
            else:  # Slight bias (2-5% expected return)
                position_size_pct = position_size_pct * 0.5  # Half position for weak signals
        
        # Calculate actual dollar amounts and share quantities
        position_value = bankroll * position_size_pct
        shares_to_buy = int(position_value / current_price)
        
        # For small bankrolls, ensure at least 1 share if position_size_pct > 0
        if shares_to_buy == 0 and position_size_pct > 0:
            shares_to_buy = 1
        
        actual_position_value = shares_to_buy * current_price
        
        # Calculate exact dollar risk
        dollar_risk = shares_to_buy * abs(current_price - stop_loss)
        dollar_reward = shares_to_buy * abs(target_price - current_price)
        
        logger.info(f"\n  POSITION SIZING (Bankroll: ${bankroll:,.2f}):")
        logger.info(f"    Kelly Fraction: {kelly_fraction*100:.1f}%")
        logger.info(f"    Risk-Based Max: {max_position_from_risk*100:.1f}%")
        logger.info(f"    Final Position: {position_size_pct*100:.1f}% (${actual_position_value:,.2f})")
        logger.info(f"    Shares: {shares_to_buy}")
        logger.info(f"    Dollar Risk: ${dollar_risk:.2f} ({dollar_risk/bankroll*100:.2f}% of bankroll)")
        logger.info(f"    Dollar Reward: ${dollar_reward:.2f}")
        logger.info(f"    Risk/Reward: 1:{dollar_reward/dollar_risk:.2f}" if dollar_risk > 0 else "    Risk/Reward: N/A")
        
        signal = TradingSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            technical_score=technical_score,
            momentum_score=technical_analysis['momentum_score'],
            trend_score=technical_analysis['trend_score'],
            volatility_score=technical_analysis['volatility_score'],
            expected_return=expected_return,
            var_95=mc_data.get('var_95', 0),
            cvar_95=mc_data.get('cvar_95', 0),
            max_drawdown=mc_data.get('max_drawdown', 0),
            recommended_option=best_call if signal_type == 'BUY' else best_put,
            option_delta=best_call.get('delta') if best_call else (best_put.get('delta') if best_put else None),
            option_expiry=best_call.get('expiration') if best_call else (best_put.get('expiration') if best_put else None),
            news_sentiment=sentiment_raw,
            catalyst_events=[],
            position_size=position_size_pct,
            risk_reward_ratio=risk_reward_ratio,
            bankroll=bankroll,
            shares=shares_to_buy,
            position_value=actual_position_value,
            dollar_risk=dollar_risk,
            dollar_reward=dollar_reward
        )
        
        return signal


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("INSTITUTIONAL-GRADE QUANTITATIVE TRADING SYSTEM")
    print("Real-Life System for Actual Money Management")
    print("=" * 80 + "\n")
    
    # Initialize system
    system = InstitutionalTradingSystem()
    
    # Test with AAPL
    test_symbol = 'AAPL'
    
    print(f"\nAnalyzing {test_symbol}...")
    print("-" * 80)
    
    analysis = system.analyze_stock_comprehensive(
        symbol=test_symbol,
        monte_carlo_sims=10000,
        forecast_days=30
    )
    
    # Save results
    output_file = f'/home/ubuntu/quant_trading_system/logs/analysis_{test_symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Convert DataFrames to dict for JSON serialization
    analysis_copy = analysis.copy()
    if isinstance(analysis_copy['price_data'], pd.DataFrame):
        df_tail = analysis_copy['price_data'].tail(10).reset_index()
        df_tail['timestamp'] = df_tail['timestamp'].astype(str)
        analysis_copy['price_data'] = df_tail.to_dict(orient='records')
    
    with open(output_file, 'w') as f:
        json.dump(analysis_copy, f, indent=2, default=str)
    
    print(f"\n✓ Analysis saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SYSTEM TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
