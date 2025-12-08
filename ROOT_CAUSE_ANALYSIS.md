# Root Cause Analysis: Missing Data in UI

## Problem
After switching from `run_final_analysis.py` to `run_perfect_analysis.py`, the UI shows:
- Technical Score: N/A
- MACD: N/A  
- Current Volatility: NaN%
- Monte Carlo Price Forecast graph is missing

## Root Cause

The new `run_perfect_analysis.py` returns a DIFFERENT JSON structure than the old `run_final_analysis.py`.

### Missing Fields in New Output:

1. **No `garch_analysis` object** - Old version had:
   ```json
   "garch_analysis": {
     "volatility_forecast": [...],
     "conditional_volatility": 0.25
   }
   ```

2. **No `monte_carlo_forecast` object** - Old version had:
   ```json
   "monte_carlo_forecast": {
     "forecast_prices": [...],  // Array of 30 price predictions
     "forecast_dates": [...],   // Array of 30 dates
     "confidence_bands": {
       "upper_95": [...],
       "lower_95": [...]
     }
   }
   ```

3. **No `position_sizing` object** - Old version had detailed breakdown

4. **Different field names** - New version uses:
   - `technical_analysis.overall_score` (old: `technical_score`)
   - `technical_analysis.volatility` (old: `garch_analysis.conditional_volatility`)

## Solution

Need to merge the BEST of both approaches:
1. Keep the REAL fundamentals from `perfect_production_analyzer.py` (P/E, ROE, sentiment)
2. Add back the Monte Carlo forecast array generation
3. Add back the GARCH volatility forecast
4. Ensure all field names match what the frontend expects

## Files to Fix

1. `run_perfect_analysis.py` - Add Monte Carlo forecast generation
2. Verify frontend `StockAnalysis.tsx` field mappings
