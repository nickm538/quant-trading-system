#!/usr/bin/env python3
"""
CLI interface for the institutional-grade market scanner.
Usage: python run_market_scanner.py [criteria] [max_results]
"""

import sys
import json
from market_scanner import MarketScanner

def main():
    """Run the market scanner with command-line arguments."""
    
    # Parse arguments
    criteria_str = sys.argv[1] if len(sys.argv) > 1 else "momentum,breakout"
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    min_volume = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    min_price = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_price = float(sys.argv[5]) if len(sys.argv) > 5 else 10000.0
    
    # Parse criteria
    criteria = {}
    for criterion in criteria_str.split(','):
        criterion = criterion.strip().lower()
        if criterion in ['momentum', 'breakout', 'value', 'growth', 'volatility']:
            criteria[criterion] = True
    
    # If no valid criteria, use default
    if not criteria:
        criteria = {'momentum': True, 'breakout': True}
    
    # Log to stderr
    print(f"Starting market scanner with criteria: {criteria}", file=sys.stderr)
    print(f"Max results: {max_results}, Min volume: {min_volume}, Price range: ${min_price}-${max_price}", file=sys.stderr)
    
    # Run scanner
    scanner = MarketScanner()
    results = scanner.scan_universe(
        criteria=criteria,
        max_results=max_results,
        min_volume=min_volume,
        min_price=min_price,
        max_price=max_price
    )
    
    # Format output
    output = {
        "success": True,
        "criteria": criteria,
        "total_results": len(results),
        "results": results
    }
    
    # Print JSON to stdout (for parsing by Node.js)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_output = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)
