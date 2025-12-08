# OptionCharts.io Data Structure Findings

## URLs Tested
1. **Greeks Page**: https://optioncharts.io/options/AAPL/greeks
   - Shows interactive Delta/Gamma/Theta/Vega charts
   - Visual charts only, no data table visible without premium
   - Can switch between Greeks types via buttons
   - Can select expiration dates (20+ available)
   
2. **Option Chain Page**: https://optioncharts.io/options/AAPL/option-chain
   - Shows tabular data with Calls and Puts side-by-side
   - Columns visible: Last Price, Bid, Ask, Volume, Open Interest, Strike
   - **Greeks NOT visible in table** (may require premium or individual contract clicks)
   - **IV NOT visible in table**
   - Has "Download CSV" button (index 20)
   - Can switch expiration dates
   - Can filter strike range

## Data Available (Free Tier)
### Option Chain Table
- **Calls**: Last Price, Bid, Ask, Volume, Open Interest
- **Puts**: Last Price, Bid, Ask, Volume, Open Interest  
- **Strike**: Shared column in middle
- **Expiration**: Selectable via dropdown (20+ dates available)

### Example ATM Data (AAPL @ $278.85)
- 275 strike call: $5.40 last, 11,315 volume, 6,451 OI
- 277.50 strike call: $3.66 last, 19,531 volume, 3,374 OI
- 280 strike call: $2.29 last, 221 volume, 21,998 OI

## Missing Data (Requires Premium or Extraction)
- Delta, Gamma, Theta, Vega (Greeks)
- Implied Volatility (IV)
- Probability of Profit
- Expected Move

## Potential Extraction Strategies
1. **Download CSV** - Click button (index 20) and parse CSV file
2. **Click Individual Contracts** - May show Greeks in detail view
3. **Parse Chart Data** - Greeks charts may have underlying data in JavaScript
4. **Use yfinance as fallback** - Already implemented, works but slower

## Recommendation
Given the limitations of free tier and complexity of extracting Greeks from charts:
1. Keep yfinance as primary data source (already working)
2. Use OptionCharts.io for visual verification only
3. Focus on optimizing yfinance performance (caching, filtering)
4. Consider Massive.com financials API for fundamental data instead
