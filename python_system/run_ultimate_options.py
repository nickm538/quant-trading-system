#!/usr/bin/env python3.11
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ULTIMATE OPTIONS INTELLIGENCE ENGINE - Runner Script               ║
║                                                                              ║
║  This is the ONLY script you need for options analysis.                      ║
║  It replaces:                                                                ║
║  - run_analysis.py (analyze_options command)                                 ║
║  - run_institutional_options.py                                              ║
║  - options_scanner.py (when run directly)                                    ║
║                                                                              ║
║  Usage:                                                                      ║
║    python run_ultimate_options.py scan [--max-results N]                     ║
║    python run_ultimate_options.py analyze SYMBOL                             ║
║    python run_ultimate_options.py single SYMBOL STRIKE EXP TYPE PRICE OPTPX  ║
║                                                                              ║
║  Copyright © 2026 SadieAI - All Rights Reserved                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
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

# Custom JSON encoder for NumPy/Pandas types
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


def main():
    """Main entry point for Ultimate Options Intelligence Engine."""
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                "success": False,
                "error": "Missing command",
                "usage": {
                    "scan": "python run_ultimate_options.py scan [--max-results N]",
                    "analyze": "python run_ultimate_options.py analyze SYMBOL",
                    "single": "python run_ultimate_options.py single SYMBOL STRIKE EXP TYPE PRICE OPTPX"
                }
            }))
            sys.exit(1)
        
        command = sys.argv[1].lower()
        start_time = datetime.now()
        
        # Import the Ultimate Options Engine
        from ultimate_options_engine import UltimateOptionsEngine
        engine = UltimateOptionsEngine()
        
        if command == 'scan':
            # Market-wide scan for best opportunities
            max_results = 10
            option_type = 'both'
            
            # Parse arguments
            i = 2
            while i < len(sys.argv):
                if sys.argv[i] in ['--max-results', '-n'] and i + 1 < len(sys.argv):
                    max_results = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] in ['--type', '-t'] and i + 1 < len(sys.argv):
                    option_type = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1
            
            logger.info(f"Starting market scan for top {max_results} {option_type} opportunities...")
            result = engine.scan_market(max_results=max_results, option_type=option_type)
            
        elif command == 'analyze':
            # Deep analysis of a single symbol
            if len(sys.argv) < 3:
                print(json.dumps({
                    "success": False,
                    "error": "Missing symbol",
                    "usage": "python run_ultimate_options.py analyze SYMBOL"
                }))
                sys.exit(1)
            
            symbol = sys.argv[2].upper()
            option_type = 'both'
            
            # Parse additional arguments
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] in ['--type', '-t'] and i + 1 < len(sys.argv):
                    option_type = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1
            
            logger.info(f"Starting deep analysis for {symbol}...")
            result = engine.analyze_symbol(symbol, option_type=option_type)
            
        elif command == 'single':
            # Analyze a specific option contract
            if len(sys.argv) < 8:
                print(json.dumps({
                    "success": False,
                    "error": "Missing arguments",
                    "usage": "python run_ultimate_options.py single SYMBOL STRIKE EXP TYPE PRICE OPTPX"
                }))
                sys.exit(1)
            
            symbol = sys.argv[2].upper()
            strike = float(sys.argv[3])
            expiration = sys.argv[4]
            option_type = sys.argv[5].lower()
            current_price = float(sys.argv[6])
            option_price = float(sys.argv[7])
            
            logger.info(f"Analyzing {option_type} option: {symbol} ${strike} exp {expiration}...")
            result = engine.analyze_single_option(
                symbol=symbol,
                strike_price=strike,
                expiration_date=expiration,
                option_type=option_type,
                current_price=current_price,
                option_price=option_price
            )
            
        else:
            print(json.dumps({
                "success": False,
                "error": f"Unknown command: {command}",
                "valid_commands": ["scan", "analyze", "single"]
            }))
            sys.exit(1)
        
        # Add timing info
        duration = (datetime.now() - start_time).total_seconds()
        if isinstance(result, dict):
            result['execution_time_seconds'] = round(duration, 2)
        
        # Try to save to database
        try:
            from options_db_saver import OptionsDBSaver
            db_saver = OptionsDBSaver()
            
            if command == 'analyze' and result.get('success'):
                save_result = db_saver.save_full_analysis(
                    symbol=result.get('symbol', 'UNKNOWN'),
                    analysis_result=result,
                    scan_duration_ms=int(duration * 1000)
                )
                if save_result.get('success'):
                    result['database_saved'] = True
                    result['scan_id'] = save_result.get('scan_id')
                    logger.info(f"✅ Saved to database: scan_id={save_result.get('scan_id')}")
                else:
                    result['database_saved'] = False
        except Exception as db_error:
            logger.warning(f"Database save skipped: {db_error}")
            result['database_saved'] = False
        
        # Output JSON to stdout
        print(json.dumps(result, cls=NumpyEncoder, indent=2))
        
    except Exception as e:
        logger.error(f"Error in Ultimate Options Engine: {e}", exc_info=True)
        print(json.dumps({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
