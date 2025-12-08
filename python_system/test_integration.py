import sys
import json
from main_trading_system import InstitutionalTradingSystem

print('Testing Python system integration...')
system = InstitutionalTradingSystem()
result = system.analyze_stock_comprehensive('AAPL', monte_carlo_sims=5000, forecast_days=30)
print('SUCCESS: Analysis completed')
print(f'Signal: {result["signal"]}')
print(f'Confidence: {result["confidence"]:.2f}%')
print(f'Current Price: ${result["current_price"]:.2f}')
print(f'Expected Price: ${result["stochastic_analysis"]["expected_price"]:.2f}')
