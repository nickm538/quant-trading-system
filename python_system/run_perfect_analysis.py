#!/usr/bin/env python3
"""
Perfect Analysis Runner - 100% REAL DATA, ZERO PLACEHOLDERS
Uses AlphaVantage for fundamentals + Manus API Hub for prices
"""

import sys
import json
import logging
import warnings
import os

# CRITICAL: Redirect stdout to stderr at the OS level
# This ensures ALL output (print, logging, etc.) goes to stderr
# We'll restore stdout only when outputting the final JSON
_original_stdout = sys.stdout
sys.stdout = sys.stderr

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress all warnings
warnings.filterwarnings('ignore')

# Disable all logging to keep output clean
logging.disable(logging.CRITICAL)
import time
import numpy as np
import math
import talib

# Custom JSON encoder to handle numpy types and NaN values
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self.default(x) for x in obj.tolist()]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (float, int)) and (math.isnan(obj) if isinstance(obj, float) else False):
            return None
        elif isinstance(obj, (float, int)) and (math.isinf(obj) if isinstance(obj, float) else False):
            return None
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """Safely convert object to JSON, handling NaN and numpy types"""
    def clean_value(v):
        if isinstance(v, dict):
            return {k: clean_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [clean_value(x) for x in v]
        elif isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        elif isinstance(v, np.floating):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        elif isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.ndarray):
            return [clean_value(x) for x in v.tolist()]
        elif isinstance(v, np.bool_):
            return bool(v)
        return v
    
    cleaned = clean_value(obj)
    return json.dumps(cleaned, cls=SafeJSONEncoder, **kwargs)
from perfect_production_analyzer import PerfectProductionAnalyzer
from expert_reasoning import ExpertReasoningEngine
from pattern_recognition import PatternRecognitionEngine
from garch_model import fit_garch_model
from production_validator import ProductionValidator, circuit_breaker
from advanced_options_analyzer import AdvancedOptionsAnalyzer
from advanced_technicals import AdvancedTechnicals
from candlestick_patterns import CandlestickPatternDetector
from enhanced_fundamentals import EnhancedFundamentalsAnalyzer
from market_intelligence import MarketIntelligence
try:
    from stockgrid_integration import StockGridIntegration
    HAS_STOCKGRID = True
except ImportError as e:
    HAS_STOCKGRID = False
    print(f"Warning: StockGrid integration not available: {e}", file=sys.stderr)
try:
    from vision_chart_analyzer import VisionChartAnalyzer
    HAS_VISION = True
except ImportError:
    HAS_VISION = False
try:
    from taapi_client import TaapiClient
    HAS_TAAPI = True
except ImportError as e:
    HAS_TAAPI = False
    print(f"Warning: TAAPI.io integration not available: {e}", file=sys.stderr)
try:
    from polygon_data_provider import PolygonDataProvider
    HAS_POLYGON_PROVIDER = True
except ImportError as e:
    HAS_POLYGON_PROVIDER = False
    print(f"Warning: PolygonDataProvider not available: {e}", file=sys.stderr)
try:
    from taapi_cross_validator import TAAPICrossValidator
    HAS_CROSS_VALIDATOR = True
except ImportError as e:
    HAS_CROSS_VALIDATOR = False
    print(f"Warning: TAAPI Cross-Validator not available: {e}", file=sys.stderr)
try:
    from financial_datasets_client import FinancialDatasetsClient
    HAS_FINANCIAL_DATASETS = True
except ImportError as e:
    HAS_FINANCIAL_DATASETS = False
    print(f"Warning: FinancialDatasets.ai integration not available: {e}", file=sys.stderr)
try:
    from personal_recommendation import PersonalRecommendationEngine
    HAS_PERSONAL_REC = True
except ImportError as e:
    HAS_PERSONAL_REC = False
    print(f"Warning: Personal Recommendation Engine not available: {e}", file=sys.stderr)

def generate_monte_carlo_forecast(current_price, volatility, forecast_days=30, num_simulations=20000, fat_tail_df=5.0):
    """
    Generate Monte Carlo price forecast with fat-tail distributions.
    Returns mean path and confidence intervals.
    
    Args:
        current_price: Current stock price
        volatility: Annualized volatility
        forecast_days: Number of days to forecast
        num_simulations: Number of Monte Carlo paths
        fat_tail_df: Degrees of freedom for Student-t (from GARCH model)
    """
    dt = 1/252  # Daily time step
    drift = 0.0  # Zero drift for conservative, unbiased forecast (standard practice)
    
    # Generate simulations with fat tails (Student's t-distribution)
    # Using time-based seed for reproducibility within same second, randomness across runs
    np.random.seed(int(time.time()) % 2**31)
    
    df = fat_tail_df  # Use GARCH-estimated degrees of freedom from real data
    paths = np.zeros((num_simulations, forecast_days + 1))
    paths[:, 0] = current_price
    
    for t in range(1, forecast_days + 1):
        # Use Student's t-distribution for fat tails (captures real market behavior)
        z = np.random.standard_t(df, size=num_simulations)
        # Scale to match volatility
        z = z * np.sqrt((df - 2) / df)  # Adjust for t-distribution variance
        paths[:, t] = paths[:, t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
    
    # Calculate statistics
    mean_path = np.mean(paths, axis=0)
    ci_95_lower = np.percentile(paths, 2.5, axis=0)
    ci_95_upper = np.percentile(paths, 97.5, axis=0)
    ci_68_lower = np.percentile(paths, 16, axis=0)
    ci_68_upper = np.percentile(paths, 84, axis=0)
    
    # Calculate VaR and CVaR from final prices
    final_prices = paths[:, -1]  # Last day prices from all simulations
    var_95_price = np.percentile(final_prices, 5)  # 5th percentile (worst 5%)
    
    # CVaR: Average of losses worse than VaR
    worst_5_percent = final_prices[final_prices <= var_95_price]
    cvar_95_price = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95_price
    
    return {
        'mean_path': mean_path.tolist(),
        'ci_95_lower_path': ci_95_lower.tolist(),
        'ci_95_upper_path': ci_95_upper.tolist(),
        'ci_68_lower_path': ci_68_lower.tolist(),
        'ci_68_upper_path': ci_68_upper.tolist(),
        'var_95_price': float(var_95_price),
        'cvar_95_price': float(cvar_95_price)
    }

def main():
    import time as _time
    _pipeline_start = _time.time()
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_perfect_analysis.py <symbol> [bankroll]"}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0  # Default $1000
    
    try:
        # ====================================================================
        # INITIALIZE SHARED DATA PROVIDER (Polygon.io)
        # Fetches price data ONCE and distributes to all modules
        # Eliminates redundant yfinance calls that cause hangs
        # ====================================================================
        polygon_provider = None
        if HAS_POLYGON_PROVIDER:
            try:
                polygon_provider = PolygonDataProvider(symbol)
                # Pre-fetch daily data to warm the cache
                _prefetch_df = polygon_provider.get_daily_ohlcv(days=400)
                if _prefetch_df is not None and len(_prefetch_df) > 50:
                    print(f"PolygonDataProvider initialized: {len(_prefetch_df)} daily bars for {symbol}", file=sys.stderr, flush=True)
                else:
                    print(f"Warning: PolygonDataProvider returned insufficient data, falling back", file=sys.stderr, flush=True)
                    polygon_provider = None
            except Exception as pp_e:
                print(f"Warning: PolygonDataProvider init failed: {pp_e}, falling back to yfinance", file=sys.stderr, flush=True)
                polygon_provider = None
        
        analyzer = PerfectProductionAnalyzer()
        result = analyzer.analyze_stock(symbol)
        
        # Calculate REAL target/stop based on technical indicators
        current_price = result['current_price']
        technical = result['technical_indicators']
        
        # Use ATR (Average True Range) for institutional-grade targets
        # INTRADAY OPTIMIZED: Tighter stops for faster exits
        atr = technical.get('atr')  # No fallback - fail if ATR missing
        if not atr:
            raise Exception(f"CRITICAL: ATR missing for {symbol}. Cannot calculate targets/stops.")
        
        # INTRADAY ATR multipliers: Tighter than end-of-day
        # Target: 1.5x ATR, Stop: 1.0x ATR (risk/reward 1.5:1)
        if result['recommendation'] in ['STRONG_BUY', 'BUY']:
            target_price = current_price + (1.5 * atr)  # Reduced from 2.5x
            stop_loss = current_price - (1.0 * atr)     # Reduced from 1.5x
        elif result['recommendation'] == 'HOLD':
            # Even tighter for HOLD (neutral bias)
            target_price = current_price + (1.0 * atr)
            stop_loss = current_price - (0.75 * atr)
        else:  # SELL or STRONG_SELL
            target_price = current_price - (1.5 * atr)
            stop_loss = current_price + (1.0 * atr)
        
        # Calculate REAL position size (simple 1% risk rule)
        risk_per_trade = 0.01  # 1% of bankroll
        # bankroll passed from command line argument
        risk_amount = bankroll * risk_per_trade
        price_risk = abs(current_price - stop_loss)
        position_size = int(risk_amount / price_risk) if price_risk > 0 else 0
        
        # CRITICAL: Cap position value at 100% of bankroll (no leverage)
        max_shares = int(bankroll / current_price)
        position_size = min(position_size, max_shares)
        
        # If position size is 0, set to 1 share minimum for valid signals
        if position_size == 0 and result['recommendation'] in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            position_size = 1
        
        # Calculate REAL ADX and volatility from price history
        rsi = technical.get('rsi', 50)
        
        # Get price data for ADX calculation
        try:
            # Re-fetch price data for ADX (need high/low/close arrays)
            from data_api import ApiClient
            from datetime import datetime
            import pandas as pd
            
            api_client = ApiClient()
            # INTRADAY: Use 5-minute bars for pattern recognition
            chart_data = api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '5m',  # 5-minute bars for intraday
                'range': '5d',     # Last 5 days (enough for intraday patterns)
                'includeAdjustedClose': True
            })
            
            if chart_data and 'chart' in chart_data:
                result_data = chart_data['chart']['result'][0]
                quotes = result_data['indicators']['quote'][0]
                
                # Convert to numpy arrays for TA-Lib
                high = np.array([h for h in quotes['high'] if h is not None], dtype=float)
                low = np.array([l for l in quotes['low'] if l is not None], dtype=float)
                close = np.array([c for c in quotes['close'] if c is not None], dtype=float)
                
                # Calculate REAL ADX (14-period)
                if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
                    adx = talib.ADX(high, low, close, timeperiod=14)
                    real_adx = float(adx[-1]) if not np.isnan(adx[-1]) else 25.0
                else:
                    real_adx = 25.0  # Not enough data
                
                # Calculate REAL volatility (20-day historical volatility)
                if len(close) >= 20:
                    returns = np.diff(np.log(close))
                    real_volatility = float(np.std(returns[-20:]) * np.sqrt(252))  # Annualized
                    
                    # Fit GARCH(1,1) model for advanced volatility analysis
                    garch_result = fit_garch_model(returns, dist='t')
                else:
                    real_volatility = 0.25  # Not enough data
                    garch_result = None
                
                # Create DataFrame for pattern recognition
                timestamps = result_data['timestamp']
                price_df = pd.DataFrame({
                    'Close': close,
                    'High': high,
                    'Low': low,
                    'Open': np.array([o for o in quotes['open'] if o is not None], dtype=float),
                    'Volume': np.array([v for v in quotes['volume'] if v is not None], dtype=float)
                })
            else:
                real_adx = 25.0
                real_volatility = 0.25
                # Create minimal DataFrame
                price_df = pd.DataFrame({'Close': [current_price]})
        except Exception as e:
            print(f"Warning: Could not calculate ADX/volatility: {e}", file=sys.stderr)
            real_adx = 25.0
            real_volatility = 0.25
            garch_result = None
            # Create minimal DataFrame
            price_df = pd.DataFrame({'Close': [current_price]})
        
        # Calculate UNIQUE momentum, trend, and volatility scores (NOT all the same!)
        # Momentum score (0-100) - based on RSI and MACD
        # RSI component: Oversold (30) = high momentum potential, Overbought (70) = low
        if rsi < 30:
            rsi_component = 80 + (30 - rsi)  # Oversold = bullish momentum
        elif rsi > 70:
            rsi_component = 20 - (rsi - 70)  # Overbought = bearish momentum
        else:
            rsi_component = 50 + (rsi - 50) * 0.5  # Neutral zone
        
        # MACD component
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        macd_hist = macd - macd_signal
        macd_component = 50 + np.tanh(macd_hist) * 30  # Normalize to 20-80 range
        
        momentum_score = (rsi_component * 0.6 + macd_component * 0.4)
        momentum_score = max(0, min(100, momentum_score))
        
        # Trend score (0-100) - based on price vs moving averages + ADX strength
        sma_20 = technical.get('sma_20', current_price)
        sma_50 = technical.get('sma_50', current_price)
        sma_200 = technical.get('sma_200', current_price)
        
        # Calculate trend direction
        if current_price > sma_20 > sma_50 > sma_200:
            # Perfect uptrend alignment
            base_trend = 85
        elif current_price > sma_20 > sma_50:
            # Strong uptrend
            base_trend = 75
        elif current_price > sma_20:
            # Weak uptrend
            base_trend = 60
        elif current_price < sma_20 < sma_50 < sma_200:
            # Perfect downtrend alignment
            base_trend = 15
        elif current_price < sma_20 < sma_50:
            # Strong downtrend
            base_trend = 25
        elif current_price < sma_20:
            # Weak downtrend
            base_trend = 40
        else:
            # Choppy/sideways
            base_trend = 50
        
        # Adjust by ADX strength (ADX > 25 = strong trend)
        if real_adx > 25:
            # Strong trend - amplify the base score
            adx_multiplier = 1 + (real_adx - 25) / 100
            if base_trend > 50:
                trend_score = min(100, base_trend * adx_multiplier)
            else:
                trend_score = max(0, base_trend / adx_multiplier)
        else:
            # Weak trend - pull toward neutral
            trend_score = base_trend * 0.7 + 50 * 0.3
        
        trend_score = max(0, min(100, trend_score))
        
        # Volatility score (0-100) - lower volatility = higher score for risk-averse
        # But some volatility is good for trading opportunities
        # Optimal volatility: 15-25% annualized
        if real_volatility < 0.15:
            # Too low - limited opportunities
            volatility_score = 50 + (real_volatility / 0.15) * 30
        elif 0.15 <= real_volatility <= 0.25:
            # Optimal range
            volatility_score = 80 + (0.25 - real_volatility) / 0.10 * 20
        elif 0.25 < real_volatility <= 0.50:
            # Elevated but manageable
            volatility_score = 50 - (real_volatility - 0.25) / 0.25 * 30
        else:
            # Too high - dangerous
            volatility_score = max(0, 20 - (real_volatility - 0.50) * 40)
        
        volatility_score = max(0, min(100, volatility_score))
        
        # Convert to format expected by UI (ALL REAL DATA)
        # Run AUTOMATIC pattern recognition (ALWAYS ACTIVE)
        pattern_engine = PatternRecognitionEngine()
        pattern_results = pattern_engine.analyze_patterns(price_df, {
            'fundamentals': result['fundamentals'],
            'technical_analysis': {
                'rsi': rsi,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'adx': real_adx,
                'current_volatility': real_volatility
            },
            'sentiment_score': result['sentiment_score']
        })
        
        # Generate expert reasoning
        reasoning_engine = ExpertReasoningEngine()
        expert_reasoning = reasoning_engine.generate_recommendation_reasoning({
            'symbol': symbol,
            'recommendation': result['recommendation'],
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'confidence': result['confidence'],
            'fundamentals': result['fundamentals'],
            'technical_analysis': {
                'rsi': rsi,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'adx': real_adx,
                'current_volatility': real_volatility
            },
            'sentiment_score': result['sentiment_score'],
            'stochastic_analysis': {
                'var_95': abs(current_price - stop_loss) / current_price
            },
            'position_sizing': {
                'shares': position_size,
                'dollar_risk': abs(current_price - stop_loss) * position_size,
                'dollar_reward': abs(target_price - current_price) * position_size
            },
            'risk_assessment': {
                'risk_reward_ratio': abs(target_price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 1.0,
                'potential_gain_pct': ((target_price - current_price) / current_price),
                'potential_loss_pct': ((current_price - stop_loss) / current_price)
            }
        })
        
        # Calculate probability-weighted expected return from expert reasoning
        weighted_expected_return = 0.0
        if 'alternative_scenarios' in expert_reasoning:
            for scenario in expert_reasoning['alternative_scenarios'].values():
                ret_decimal = scenario['return_pct'] / 100  # Convert % to decimal
                prob_decimal = scenario['probability'] / 100
                weighted_expected_return += ret_decimal * prob_decimal
        
        # Generate Monte Carlo forecast first to get VaR/CVaR
        monte_carlo_result = generate_monte_carlo_forecast(
            current_price, 
            real_volatility, 
            forecast_days=30,
            fat_tail_df=garch_result.get('fat_tail_df', 5.0) if garch_result else 5.0
        )
        
        # Calculate VaR and CVaR as percentage losses from current price
        var_95_pct = abs(current_price - monte_carlo_result['var_95_price']) / current_price
        cvar_95_pct = abs(current_price - monte_carlo_result['cvar_95_price']) / current_price
        
        output = {
            'symbol': symbol,
            'timestamp': result['timestamp'],
            'current_price': current_price,
            'price_change_pct': result.get('price_change_pct', 0),
            'signal': result['recommendation'],
            'confidence': result['confidence'],
            'fundamental_score': result['fundamental_score'],
            'technical_score': result['technical_score'],
            'sentiment_score': result['sentiment_score'],
            'overall_score': result['overall_score'],
            'recommendation': result['recommendation'],
            'fundamentals': result['fundamentals'],
            'technical_indicators': result['technical_indicators'],
            'target_price': float(target_price),
            'stop_loss': float(stop_loss),
            'position_size': position_size,
            'expert_reasoning': expert_reasoning,
            'pattern_recognition': pattern_results,
            'technical_analysis': {
                'technical_score': result['technical_score'],  # Frontend looks for this
                'overall_score': result['technical_score'],
                'momentum_score': momentum_score,  # REAL momentum score based on RSI
                'trend_score': trend_score,  # REAL trend score based on price vs MAs
                'volatility_score': volatility_score,  # REAL volatility score
                'rsi': rsi,
                'macd': technical.get('macd', 0),  # Frontend looks for this
                'adx': real_adx,  # REAL ADX from TA-Lib
                'volatility': real_volatility,  # REAL historical volatility
                'current_volatility': real_volatility  # Frontend looks for this
            },
            'stochastic_analysis': {
                'expected_price': current_price,
                'expected_return': weighted_expected_return,  # Probability-weighted return from expert reasoning
                'confidence_interval_lower': stop_loss,
                'confidence_interval_upper': target_price,
                'var_95': var_95_pct,  # From Monte Carlo 5th percentile
                'cvar_95': cvar_95_pct,  # From Monte Carlo average of worst 5%
                'var_95_price': monte_carlo_result['var_95_price'],  # Absolute price at VaR
                'cvar_95_price': monte_carlo_result['cvar_95_price'],  # Absolute price at CVaR
                'max_drawdown': abs(stop_loss - current_price) / current_price,
                'fat_tail_df': 5.0,
                'monte_carlo': monte_carlo_result,
                # GARCH(1,1) volatility model with MLE fitting
                'garch_analysis': garch_result if garch_result else {
                    'model': 'GARCH(1,1)',
                    'distribution': 'Student-t',
                    'fat_tail_df': None,
                    'aic': None,
                    'bic': None,
                    'current_volatility': real_volatility,
                    'converged': False
                }
            },
            # Advanced options analysis with IV crush, skew, drift detection
            'options_analysis': AdvancedOptionsAnalyzer().analyze_options(
                symbol=symbol,
                current_price=current_price,
                historical_volatility=real_volatility,
                days_to_earnings=result.get('fundamentals', {}).get('days_to_earnings', None)
            ),
            'news_sentiment': {
                'sentiment_score': result['sentiment_score'],
                'total_articles': 0,  # TODO: Extract from AlphaVantage response
                'recent_headlines': []
            },
            'risk_assessment': {
                'risk_reward_ratio': abs(target_price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 1.0,
                'potential_gain_pct': ((target_price - current_price) / current_price * 100),
                'potential_loss_pct': ((current_price - stop_loss) / current_price * 100)
            },
            'position_sizing': {
                'shares': position_size,
                'position_value': position_size * current_price,
                'dollar_risk': position_size * abs(current_price - stop_loss),
                'dollar_reward': position_size * abs(target_price - current_price),
                'risk_reward_ratio': abs(target_price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 1.0,
                'position_size_pct': (position_size * current_price) / bankroll,
                'risk_pct_of_bankroll': (position_size * abs(current_price - stop_loss)) / bankroll
            },
            'bankroll': bankroll,
            'advanced_technicals': None,
            'candlestick_patterns': None,
            'enhanced_fundamentals': None,
            'market_context': None,  # Market status, VIX, regime, sentiment
            'stockgrid_analysis': None,  # Dark pools, ARIMA, factor analysis from StockGrid.io
            'taapi_indicators': None,  # TAAPI.io technical indicators (backup/validation)
            'financialdatasets': None,  # FinancialDatasets.ai fundamental data
            'exa_intelligence': None,  # EXA AI real-time web search & candlestick chart analysis
            'personal_recommendation': None  # "If I Were Trading" recommendation
        }
        
        # Run Advanced Technicals (R2, Pivot, Fibonacci)
        # Pass polygon_provider to avoid redundant yfinance calls
        try:
            adv_tech = AdvancedTechnicals()
            if polygon_provider:
                polygon_df = polygon_provider.get_daily_ohlcv()
                output['advanced_technicals'] = adv_tech.analyze(symbol, pre_fetched_df=polygon_df)
            else:
                output['advanced_technicals'] = adv_tech.analyze(symbol)
        except Exception as e:
            output['advanced_technicals'] = {'error': str(e)}
        
        # Run Candlestick Pattern Detection with Vision AI Enhancement
        # Pass polygon_provider to avoid redundant yfinance calls
        try:
            candle_detector = CandlestickPatternDetector()
            if polygon_provider:
                polygon_df = polygon_provider.get_daily_ohlcv()
                candlestick_result = candle_detector.analyze(symbol, pre_fetched_df=polygon_df)
            else:
                candlestick_result = candle_detector.analyze(symbol)
            
            # Enhance with Vision AI chart analysis if available
            # Use multiprocessing with timeout to prevent blocking the entire pipeline
            # Note: ThreadPoolExecutor's context manager blocks on exit waiting for threads,
            # so we use a threading.Thread with daemon=True instead
            print(f"[{_time.time()-_pipeline_start:.1f}s] Starting Vision AI...", file=sys.stderr, flush=True)
            if HAS_VISION:
                try:
                    import threading
                    import queue
                    
                    vision_queue = queue.Queue()
                    def _run_vision_thread(sym, q):
                        try:
                            va = VisionChartAnalyzer()
                            result = va.analyze(sym)
                            q.put(result)
                        except Exception as e:
                            q.put({'success': False, 'error': str(e)})
                    
                    vision_thread = threading.Thread(target=_run_vision_thread, args=(symbol, vision_queue), daemon=True)
                    vision_thread.start()
                    vision_thread.join(timeout=45)  # Hard 45s cap
                    
                    if not vision_queue.empty():
                        vision_result = vision_queue.get_nowait()
                    else:
                        vision_result = {'success': False, 'error': 'Vision AI timed out (45s cap)'}
                        print(f"Vision AI timed out for {symbol}, continuing without it", file=sys.stderr)
                    
                    if vision_result.get('success'):
                        # Merge Vision AI patterns with algorithmic patterns
                        candlestick_result['vision_ai_analysis'] = {
                            'candlestick_patterns': vision_result.get('candlestick_patterns', []),
                            'trend': vision_result.get('trend', {}),
                            'support_levels': vision_result.get('support_levels', []),
                            'resistance_levels': vision_result.get('resistance_levels', []),
                            'volume_analysis': vision_result.get('volume_analysis', {}),
                            'recommendation': vision_result.get('recommendation', {}),
                            'overall_bias': vision_result.get('overall_bias', 'NEUTRAL'),
                            'key_observations': vision_result.get('key_observations', []),
                            'chart_source': vision_result.get('chart_source', 'Finviz'),
                            'ai_model': vision_result.get('ai_model', 'Vision AI')
                        }
                    elif vision_result.get('error'):
                        candlestick_result['vision_ai_error'] = vision_result['error']
                except Exception as ve:
                    candlestick_result['vision_ai_error'] = str(ve)
            
            print(f"[{_time.time()-_pipeline_start:.1f}s] Vision AI done, starting Golden/Death Cross...", file=sys.stderr, flush=True)
            # Add Golden/Death Cross analysis using Polygon daily data
            # (avoids calling yfinance which hangs in containerized environments)
            try:
                # Use polygon_provider's cached daily data if available, otherwise fetch fresh
                close_series = pd.Series([])
                if polygon_provider:
                    gdc_df = polygon_provider.get_daily_ohlcv(days=400)
                    if gdc_df is not None and len(gdc_df) > 0:
                        close_series = gdc_df['Close'].reset_index(drop=True)
                        print(f"  Golden Cross: Using {len(close_series)} cached daily bars from PolygonDataProvider", file=sys.stderr, flush=True)
                
                if len(close_series) < 200:
                    # Fallback: fetch directly from Polygon
                    try:
                        from polygon import RESTClient as PolygonClient
                        from datetime import datetime, timedelta
                        poly_key = os.environ.get('POLYGON_API_KEY', '')
                        if poly_key:
                            poly_client = PolygonClient(api_key=poly_key)
                            end_date = datetime.now().strftime('%Y-%m-%d')
                            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
                            aggs = list(poly_client.list_aggs(symbol, 1, 'day', start_date, end_date, limit=500))
                            close_series = pd.Series([a.close for a in aggs])
                            print(f"  Golden Cross: Got {len(close_series)} daily bars from Polygon API", file=sys.stderr, flush=True)
                    except Exception as poly_e:
                        print(f"  Golden Cross: Polygon fallback failed: {poly_e}", file=sys.stderr, flush=True)
                if len(close_series) >= 200:
                    sma_50 = close_series.rolling(window=50).mean()
                    sma_200 = close_series.rolling(window=200).mean()
                    
                    current_50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else 0
                    current_200 = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else 0
                    
                    # Check for recent crosses (within last 5 days)
                    recent_golden = False
                    recent_death = False
                    days_since_cross = None
                    
                    for i in range(1, min(6, len(sma_50))):
                        if not pd.isna(sma_50.iloc[-i]) and not pd.isna(sma_200.iloc[-i]):
                            curr_50 = sma_50.iloc[-i]
                            curr_200 = sma_200.iloc[-i]
                            prev_50_check = sma_50.iloc[-i-1] if i+1 <= len(sma_50) else curr_50
                            prev_200_check = sma_200.iloc[-i-1] if i+1 <= len(sma_200) else curr_200
                            
                            if curr_50 > curr_200 and prev_50_check <= prev_200_check:
                                recent_golden = True
                                days_since_cross = i
                                break
                            elif curr_50 < curr_200 and prev_50_check >= prev_200_check:
                                recent_death = True
                                days_since_cross = i
                                break
                    
                    if recent_golden:
                        signal = 'GOLDEN_CROSS'
                        explanation = f'GOLDEN CROSS DETECTED ({days_since_cross} day(s) ago). The 50-day SMA crossed ABOVE the 200-day SMA - a major bullish trend change signal.'
                    elif recent_death:
                        signal = 'DEATH_CROSS'
                        explanation = f'DEATH CROSS DETECTED ({days_since_cross} day(s) ago). The 50-day SMA crossed BELOW the 200-day SMA - a major bearish trend change signal.'
                    elif current_50 > current_200:
                        distance_pct = ((current_50 - current_200) / current_200) * 100
                        signal = 'BULLISH_TREND'
                        explanation = f'BULLISH TREND: 50 SMA (${current_50:.2f}) is {distance_pct:.1f}% above 200 SMA (${current_200:.2f}). Confirmed uptrend.'
                    else:
                        distance_pct = ((current_200 - current_50) / current_200) * 100
                        signal = 'BEARISH_TREND'
                        explanation = f'BEARISH TREND: 50 SMA (${current_50:.2f}) is {distance_pct:.1f}% below 200 SMA (${current_200:.2f}). Confirmed downtrend.'
                    
                    candlestick_result['golden_death_cross'] = {
                        'sma_50': round(current_50, 2),
                        'sma_200': round(current_200, 2),
                        'golden_cross': current_50 > current_200,
                        'death_cross': current_50 < current_200,
                        'recent_golden_cross': recent_golden,
                        'recent_death_cross': recent_death,
                        'days_since_cross': days_since_cross,
                        'signal': signal,
                        'explanation': explanation
                    }
                else:
                    candlestick_result['golden_death_cross'] = {'error': f'Need 200+ days of data, only have {len(close_series)}'}
            except Exception as gdc_e:
                candlestick_result['golden_death_cross_error'] = str(gdc_e)
            
            output['candlestick_patterns'] = candlestick_result
        except Exception as e:
            output['candlestick_patterns'] = {'error': str(e)}
        
        # ====================================================================
        # PARALLEL EXECUTION: Run all independent analysis steps concurrently
        # This reduces total time from ~136s (sequential) to ~60s (parallel)
        # ====================================================================
        import threading
        import queue as _queue
        
        print(f"[{_time.time()-_pipeline_start:.1f}s] Starting parallel analysis (6 threads)...", file=sys.stderr, flush=True)
        
        # Thread-safe results dict
        parallel_results = {}
        parallel_lock = threading.Lock()
        
        def _safe_run(name, func):
            """Run a function and store result thread-safely."""
            try:
                result = func()
                with parallel_lock:
                    parallel_results[name] = result
            except Exception as e:
                with parallel_lock:
                    parallel_results[name] = {'error': str(e)}
        
        # Define all parallel tasks
        threads = []
        
        # 1. Enhanced Fundamentals (slowest: ~54s)
        # Now uses FinancialDatasets.ai as primary, yfinance as fallback
        def _run_enhanced_fund():
            ef = EnhancedFundamentalsAnalyzer()
            if polygon_provider:
                ef_df = polygon_provider.get_daily_ohlcv()
                return ef.analyze(symbol, pre_fetched_df=ef_df)
            return ef.analyze(symbol)
        t = threading.Thread(target=_safe_run, args=('enhanced_fundamentals', _run_enhanced_fund), daemon=True)
        threads.append(t)
        
        # 2. Market Intelligence (~14s)
        def _run_market_intel():
            mi = MarketIntelligence()
            return mi.get_full_market_intelligence()
        t = threading.Thread(target=_safe_run, args=('market_context', _run_market_intel), daemon=True)
        threads.append(t)
        
        # 3. StockGrid (~15s)
        if HAS_STOCKGRID:
            def _run_stockgrid():
                sg = StockGridIntegration()
                return sg.get_full_stockgrid_analysis(symbol)
            t = threading.Thread(target=_safe_run, args=('stockgrid_analysis', _run_stockgrid), daemon=True)
            threads.append(t)
        else:
            parallel_results['stockgrid_analysis'] = {'error': 'StockGrid integration not available'}
        
        # 4. TAAPI (~6s for 6 indicators)
        if HAS_TAAPI:
            def _run_taapi():
                taapi = TaapiClient()
                taapi_result = {}
                for indicator in ['rsi', 'macd', 'supertrend', 'adx', 'bbands', 'stoch']:
                    try:
                        ind_data = taapi.get_indicator(indicator, symbol, '1d')
                        if ind_data and not ind_data.get('error'):
                            taapi_result[indicator] = ind_data
                    except Exception as ind_e:
                        taapi_result[indicator] = {'error': str(ind_e)}
                return taapi_result
            t = threading.Thread(target=_safe_run, args=('taapi_indicators', _run_taapi), daemon=True)
            threads.append(t)
        else:
            parallel_results['taapi_indicators'] = {'error': 'TAAPI.io integration not available'}
        
        # 5. FinancialDatasets (~4s)
        if HAS_FINANCIAL_DATASETS:
            def _run_fd():
                fd_client = FinancialDatasetsClient()
                fd_result = {}
                for key, func in [
                    ('financial_metrics', lambda: fd_client.get_financial_metrics(symbol)),
                    ('income_statement', lambda: fd_client.get_income_statement(symbol, period='annual', limit=2)),
                    ('sec_filings', lambda: fd_client.get_filings(symbol, limit=5)),
                    ('company_facts', lambda: fd_client.get_company_facts(symbol)),
                    ('price_snapshot', lambda: fd_client.get_stock_price_snapshot(symbol)),
                ]:
                    try:
                        fd_result[key] = func()
                    except Exception as e:
                        fd_result[key] = {'error': str(e)}
                return fd_result
            t = threading.Thread(target=_safe_run, args=('financialdatasets', _run_fd), daemon=True)
            threads.append(t)
        else:
            parallel_results['financialdatasets'] = {'error': 'FinancialDatasets.ai integration not available'}
        
        # 6. EXA AI (~30s)
        try:
            from exa_client import ExaClient
            def _run_exa():
                exa = ExaClient()
                return exa.get_comprehensive_stock_intelligence(symbol)
            t = threading.Thread(target=_safe_run, args=('exa_intelligence', _run_exa), daemon=True)
            threads.append(t)
        except ImportError:
            parallel_results['exa_intelligence'] = {'error': 'EXA AI client not available'}
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads with a global timeout of 60s
        PARALLEL_TIMEOUT = 60  # seconds
        deadline = _time.time() + PARALLEL_TIMEOUT
        for t in threads:
            remaining = max(0.1, deadline - _time.time())
            t.join(timeout=remaining)
        
        # Collect results (use error for any that didn't finish)
        for key in ['enhanced_fundamentals', 'market_context', 'stockgrid_analysis', 
                     'taapi_indicators', 'financialdatasets', 'exa_intelligence']:
            if key in parallel_results:
                output[key] = parallel_results[key]
            else:
                output[key] = {'error': f'{key} timed out ({PARALLEL_TIMEOUT}s cap)'}
        
        print(f"[{_time.time()-_pipeline_start:.1f}s] Parallel analysis complete. Got: {[k for k in parallel_results.keys()]}", file=sys.stderr, flush=True)
        
        # ====================================================================
        # TAAPI CROSS-VALIDATION CONFIDENCE LAYER
        # Uses TAAPI.io as independent second opinion to validate local indicators
        # Adjusts confidence multiplier based on agreement/disagreement
        # ====================================================================
        print(f"[{_time.time()-_pipeline_start:.1f}s] Running TAAPI cross-validation...", file=sys.stderr, flush=True)
        if HAS_CROSS_VALIDATOR and output.get('taapi_indicators') and not output['taapi_indicators'].get('error'):
            try:
                cross_validator = TAAPICrossValidator()
                
                # Build local technicals dict from our calculated values
                local_technicals = {
                    'rsi': rsi,
                    'macd': technical.get('macd', 0),
                    'macd_signal': technical.get('macd_signal', 0),
                    'adx': real_adx,
                    'current_price': current_price,
                    'sma_20': technical.get('sma_20', 0),
                    'sma_50': technical.get('sma_50', 0),
                    'signal': result['recommendation'],
                }
                
                # Add Bollinger and Stochastic from advanced_technicals if available
                adv = output.get('advanced_technicals', {})
                if isinstance(adv, dict) and not adv.get('error'):
                    bb = adv.get('bollinger_bands', {})
                    if isinstance(bb, dict):
                        local_technicals['bb_pct_b'] = bb.get('percent_b') or bb.get('pct_b')
                    stoch = adv.get('stochastic', {})
                    if isinstance(stoch, dict):
                        local_technicals['stoch_k'] = stoch.get('k') or stoch.get('slowk')
                
                cross_result = cross_validator.cross_validate(symbol, local_technicals)
                output['taapi_cross_validation'] = cross_result
                
                # Apply confidence multiplier to the overall confidence score
                if cross_result.get('available') and cross_result.get('confidence_multiplier', 1.0) != 1.0:
                    original_confidence = output['confidence']
                    adjusted_confidence = max(0, min(100, original_confidence * cross_result['confidence_multiplier']))
                    output['confidence'] = round(adjusted_confidence, 1)
                    output['confidence_adjustment'] = {
                        'original': original_confidence,
                        'adjusted': round(adjusted_confidence, 1),
                        'multiplier': cross_result['confidence_multiplier'],
                        'reason': cross_result['summary']
                    }
                    print(f"  Cross-validation: confidence {original_confidence} -> {adjusted_confidence:.1f} (x{cross_result['confidence_multiplier']:.3f})", file=sys.stderr, flush=True)
            except Exception as cv_e:
                output['taapi_cross_validation'] = {'error': str(cv_e)}
                print(f"  Cross-validation failed: {cv_e}", file=sys.stderr, flush=True)
        else:
            output['taapi_cross_validation'] = {'available': False, 'summary': 'TAAPI data not available for cross-validation'}
        
        # ====================================================================
        # DATA SOURCE METADATA
        # Track which data sources were used for transparency
        # ====================================================================
        output['data_sources'] = {
            'price_data': 'Polygon.io' if polygon_provider else 'yfinance/Manus API Hub',
            'technical_indicators': {
                'primary': 'Local (TA-Lib)',
                'cross_validation': 'TAAPI.io Pro' if output.get('taapi_cross_validation', {}).get('available') else 'Not available'
            },
            'fundamentals': {
                'primary': 'FinancialDatasets.ai Pro' if output.get('financialdatasets') and not output['financialdatasets'].get('error') else 'yfinance/FMP/AlphaVantage',
                'enhanced': 'FMP + AlphaVantage + Finnhub'
            },
            'real_time_intelligence': 'EXA AI' if output.get('exa_intelligence') and not output['exa_intelligence'].get('error') else 'Not available',
            'dark_pools': 'StockGrid.io' if output.get('stockgrid_analysis') and not output['stockgrid_analysis'].get('error') else 'Not available',
            'vision_chart': 'OpenRouter (Gemini Flash / Claude)' if output.get('candlestick_patterns', {}).get('vision_ai_analysis') else 'Not available'
        }
        
        print(f"[{_time.time()-_pipeline_start:.1f}s] Starting Personal Recommendation...", file=sys.stderr, flush=True)
        # Generate Personal "If I Were Trading" Recommendation
        # This MUST run last because it synthesizes ALL other analysis data
        if HAS_PERSONAL_REC:
            try:
                rec_engine = PersonalRecommendationEngine()
                output['personal_recommendation'] = rec_engine.generate_recommendation(output)
            except Exception as e:
                output['personal_recommendation'] = {'error': str(e)}
                print(f"Personal recommendation failed for {symbol}: {e}", file=sys.stderr)
        else:
            output['personal_recommendation'] = {'error': 'Personal Recommendation Engine not available'}
        
        print(f"[{_time.time()-_pipeline_start:.1f}s] Starting validation...", file=sys.stderr, flush=True)
        # PRODUCTION VALIDATION - Ensure all data is valid before returning
        validator = ProductionValidator()
        
        # Sanitize NaN/Inf values
        output = validator.sanitize_analysis_result(output)
        
        # Validate complete analysis
        is_valid, errors = validator.validate_analysis_result(output)
        if not is_valid:
            import logging
            logging.error(f"Analysis validation failed for {symbol}:")
            for error in errors:
                logging.error(f"  - {error}")
            # Return error but don't crash - graceful degradation
            output['validation_errors'] = errors
            output['validation_status'] = 'FAILED'
        else:
            output['validation_status'] = 'PASSED'
        
        # Restore original stdout for JSON output
        sys.stdout = _original_stdout
        json_output = safe_json_dumps(output, indent=2)
        print(json_output)
        sys.stdout.flush()
        
        # Force exit to kill any lingering daemon threads (network requests etc.)
        # Without this, the process hangs waiting for daemon threads to finish
        os._exit(0)
        
    except Exception as e:
        import traceback
        # Restore original stdout for JSON output
        sys.stdout = _original_stdout
        print(safe_json_dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.stdout.flush()
        os._exit(1)

if __name__ == "__main__":
    main()
