#!/usr/bin/env python3.11
"""
Wrapper script for Institutional Options Engine
Provides clean JSON output for backend integration
"""
import sys
import json
import os
import logging
from datetime import datetime

# Ensure correct Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to stderr (stdout is for JSON output only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for institutional options analysis."""
    try:
        if len(sys.argv) < 2:
            error_result = {
                "success": False,
                "error": "Missing symbol argument",
                "usage": "python run_institutional_options.py <SYMBOL>"
            }
            print(json.dumps(error_result))
            sys.exit(1)
        
        symbol = sys.argv[1].upper()
        start_time = datetime.now()
        logger.info(f"Starting institutional options analysis for {symbol}")
        
        # Import required modules
        import yfinance as yf
        import numpy as np
        import pandas as pd
        from institutional_options_engine import InstitutionalOptionsEngine
        
        # Initialize engine
        engine = InstitutionalOptionsEngine()
        
        # Fetch stock data
        logger.info(f"Fetching stock data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Get current price
        try:
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
        except:
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
            else:
                raise ValueError(f"Could not fetch price for {symbol}")
        
        logger.info(f"Current price: ${current_price:.2f}")
        
        # Get options chain
        logger.info("Fetching options chain...")
        expirations = ticker.options
        
        if not expirations or len(expirations) == 0:
            result = {
                "success": False,
                "error": f"No options available for {symbol}",
                "symbol": symbol,
                "current_price": current_price
            }
            print(json.dumps(result, default=str))
            return
        
        # Collect all options within 90 days
        all_calls = []
        all_puts = []
        
        for expiration in expirations[:8]:  # Check first 8 expirations
            try:
                opt_chain = ticker.option_chain(expiration)
                
                # Add expiration date to each option
                for _, call in opt_chain.calls.iterrows():
                    call_dict = call.to_dict()
                    call_dict['expiration'] = expiration
                    
                    # Calculate DTE
                    exp_date = pd.to_datetime(expiration)
                    dte = (exp_date - pd.Timestamp.now()).days
                    
                    if 7 <= dte <= 90:  # Only include 7-90 day options
                        call_dict['daysToExpiration'] = dte
                        
                        # Extract Greeks
                        call_dict['delta'] = call.get('delta', 0)
                        call_dict['gamma'] = call.get('gamma', 0)
                        call_dict['theta'] = call.get('theta', 0)
                        call_dict['vega'] = call.get('vega', 0)
                        call_dict['rho'] = call.get('rho', 0)
                        
                        all_calls.append(call_dict)
                
                for _, put in opt_chain.puts.iterrows():
                    put_dict = put.to_dict()
                    put_dict['expiration'] = expiration
                    
                    exp_date = pd.to_datetime(expiration)
                    dte = (exp_date - pd.Timestamp.now()).days
                    
                    if 7 <= dte <= 90:
                        put_dict['daysToExpiration'] = dte
                        
                        put_dict['delta'] = put.get('delta', 0)
                        put_dict['gamma'] = put.get('gamma', 0)
                        put_dict['theta'] = put.get('theta', 0)
                        put_dict['vega'] = put.get('vega', 0)
                        put_dict['rho'] = put.get('rho', 0)
                        
                        all_puts.append(put_dict)
                        
            except Exception as e:
                logger.warning(f"Error fetching options for expiration {expiration}: {e}")
                continue
        
        logger.info(f"Collected {len(all_calls)} calls and {len(all_puts)} puts")
        
        if len(all_calls) == 0 and len(all_puts) == 0:
            result = {
                "success": False,
                "error": f"No options found in 7-90 day range for {symbol}",
                "symbol": symbol,
                "current_price": current_price
            }
            print(json.dumps(result, default=str))
            return
        
        # Prepare stock data
        logger.info("Calculating technical indicators...")
        hist = ticker.history(period='60d')
        
        if len(hist) < 20:
            raise ValueError(f"Insufficient historical data for {symbol}")
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = hist['Close'].ewm(span=12).mean()
        ema_26 = hist['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        # Determine MACD signal
        if len(macd) > 0 and len(macd_signal) > 0:
            if macd.iloc[-1] > macd_signal.iloc[-1]:
                macd_signal_str = 'bullish'
            elif macd.iloc[-1] < macd_signal.iloc[-1]:
                macd_signal_str = 'bearish'
            else:
                macd_signal_str = 'neutral'
        else:
            macd_signal_str = 'neutral'
        
        # Calculate ADX (simplified)
        adx = 25.0  # Default moderate trend strength
        
        # Determine trend
        sma_20 = hist['Close'].rolling(window=20).mean()
        sma_50 = hist['Close'].rolling(window=50).mean()
        
        if len(sma_20) > 0 and len(sma_50) > 0:
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend = 'uptrend'
            elif sma_20.iloc[-1] < sma_50.iloc[-1]:
                trend = 'downtrend'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
        
        stock_data = {
            'current_price': current_price,
            'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
            'macd_signal': macd_signal_str,
            'adx': adx,
            'trend': trend
        }
        
        logger.info(f"Stock data: RSI={stock_data['rsi']:.1f}, MACD={macd_signal_str}, Trend={trend}")
        
        # Prepare options data
        options_data = {
            'calls': all_calls,
            'puts': all_puts
        }
        
        # Run institutional analysis
        logger.info("Running institutional-grade options analysis...")
        result = engine.analyze_options_chain(
            symbol=symbol,
            options_data=options_data,
            stock_data=stock_data,
            market_data=None
        )
        
        # Add success flag
        result['success'] = True
        
        # Save to database
        try:
            from options_db_saver import OptionsDBSaver
            
            db_saver = OptionsDBSaver()
            scan_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            save_result = db_saver.save_full_analysis(
                symbol=symbol,
                analysis_result=result,
                scan_duration_ms=scan_duration_ms
            )
            
            if save_result.get('success'):
                logger.info(f"✅ Saved to database: scan_id={save_result['scan_id']}, "
                          f"calls={save_result['saved_calls']}, puts={save_result['saved_puts']}")
                result['database_saved'] = True
                result['scan_id'] = save_result['scan_id']
            else:
                logger.warning(f"⚠️ Database save failed: {save_result.get('reason')}")
                result['database_saved'] = False
        except Exception as db_error:
            logger.error(f"Error saving to database: {db_error}")
            result['database_saved'] = False
        
        # Output JSON to stdout
        print(json.dumps(result, default=str, indent=2))
        
        logger.info(f"Analysis complete: {len(result.get('top_calls', []))} top calls, {len(result.get('top_puts', []))} top puts")
        
    except Exception as e:
        logger.error(f"Error in institutional options analysis: {e}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
            "symbol": sys.argv[1].upper() if len(sys.argv) > 1 else "UNKNOWN",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, default=str))
        sys.exit(1)

if __name__ == "__main__":
    main()
