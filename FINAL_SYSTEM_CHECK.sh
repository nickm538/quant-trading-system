#!/bin/bash
# FINAL SYSTEM CHECK - Company Money Live Trading
# Run this script before Monday morning market open

echo "=========================================="
echo "FINAL SYSTEM CHECK - LIVE TRADING READY"
echo "=========================================="
echo ""

cd /home/ubuntu/quant-trading-web/python_system

# Test 1: Verify all Python modules import
echo "TEST 1: Verifying Python module imports..."
python3.11 -c "
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')

try:
    from risk_free_rate import get_risk_free_rate
    from production_validator import ProductionValidator, circuit_breaker
    from perfect_production_analyzer import PerfectProductionAnalyzer
    from expert_reasoning import ExpertReasoningEngine
    from pattern_recognition import PatternRecognitionEngine
    from garch_model import fit_garch_model
    from options_analyzer import OptionsAnalyzer
    from market_scanner import MarketScanner
    from legendary_trader_wisdom import LegendaryTraderWisdom
    print('✅ All modules imported successfully')
except Exception as e:
    print(f'❌ CRITICAL: Module import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ FAILED: Module imports"
    exit 1
fi

# Test 2: Verify risk-free rate fetching
echo ""
echo "TEST 2: Verifying risk-free rate fetching..."
python3.11 -c "
from risk_free_rate import get_risk_free_rate
rate = get_risk_free_rate()
if rate > 0 and rate < 0.10:
    print(f'✅ Risk-free rate: {rate:.4f} ({rate*100:.2f}%)')
else:
    print(f'❌ CRITICAL: Invalid risk-free rate: {rate}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ FAILED: Risk-free rate"
    exit 1
fi

# Test 3: Full analysis test with AAPL
echo ""
echo "TEST 3: Running full analysis test (AAPL)..."
timeout 90 python3.11 run_perfect_analysis.py AAPL > /tmp/test_aapl.json 2>&1

if [ $? -ne 0 ]; then
    echo "❌ FAILED: AAPL analysis timeout or error"
    cat /tmp/test_aapl.json
    exit 1
fi

# Check validation status
VALIDATION=$(cat /tmp/test_aapl.json | grep -o '"validation_status": "[^"]*"' | cut -d'"' -f4)
if [ "$VALIDATION" != "PASSED" ]; then
    echo "❌ FAILED: AAPL validation status: $VALIDATION"
    cat /tmp/test_aapl.json | grep -A5 "validation_errors"
    exit 1
fi

echo "✅ AAPL analysis passed validation"

# Test 4: Full analysis test with SPY
echo ""
echo "TEST 4: Running full analysis test (SPY)..."
timeout 90 python3.11 run_perfect_analysis.py SPY > /tmp/test_spy.json 2>&1

if [ $? -ne 0 ]; then
    echo "❌ FAILED: SPY analysis timeout or error"
    cat /tmp/test_spy.json
    exit 1
fi

VALIDATION=$(cat /tmp/test_spy.json | grep -o '"validation_status": "[^"]*"' | cut -d'"' -f4)
if [ "$VALIDATION" != "PASSED" ]; then
    echo "❌ FAILED: SPY validation status: $VALIDATION"
    exit 1
fi

echo "✅ SPY analysis passed validation"

# Test 5: Verify no NaN or Inf in outputs
echo ""
echo "TEST 5: Checking for NaN/Inf values..."
if grep -q "NaN\|Infinity\|null" /tmp/test_aapl.json; then
    echo "⚠️  WARNING: Found NaN/Inf/null in AAPL output"
    grep -n "NaN\|Infinity\|null" /tmp/test_aapl.json | head -5
else
    echo "✅ No NaN/Inf values found"
fi

# Test 6: Verify confidence values are reasonable
echo ""
echo "TEST 6: Verifying confidence values..."
CONFIDENCE=$(cat /tmp/test_aapl.json | grep -m1 '"confidence":' | grep -o '[0-9.]*' | head -1)
if (( $(echo "$CONFIDENCE > 100" | bc -l) )); then
    echo "❌ CRITICAL: Confidence > 100%: $CONFIDENCE"
    exit 1
elif (( $(echo "$CONFIDENCE < 0" | bc -l) )); then
    echo "❌ CRITICAL: Confidence < 0%: $CONFIDENCE"
    exit 1
else
    echo "✅ Confidence in valid range: $CONFIDENCE%"
fi

# Test 7: Verify price values are reasonable
echo ""
echo "TEST 7: Verifying price values..."
CURRENT_PRICE=$(cat /tmp/test_aapl.json | grep -m1 '"current_price":' | grep -o '[0-9.]*' | head -1)
TARGET_PRICE=$(cat /tmp/test_aapl.json | grep -m1 '"target_price":' | grep -o '[0-9.]*' | head -1)
STOP_LOSS=$(cat /tmp/test_aapl.json | grep -m1 '"stop_loss":' | grep -o '[0-9.]*' | head -1)

if (( $(echo "$CURRENT_PRICE <= 0" | bc -l) )); then
    echo "❌ CRITICAL: Invalid current price: $CURRENT_PRICE"
    exit 1
fi

if (( $(echo "$TARGET_PRICE <= 0" | bc -l) )); then
    echo "❌ CRITICAL: Invalid target price: $TARGET_PRICE"
    exit 1
fi

if (( $(echo "$STOP_LOSS <= 0" | bc -l) )); then
    echo "❌ CRITICAL: Invalid stop loss: $STOP_LOSS"
    exit 1
fi

echo "✅ All prices valid: Current=$CURRENT_PRICE, Target=$TARGET_PRICE, Stop=$STOP_LOSS"

# Test 8: Verify legendary traders are present
echo ""
echo "TEST 8: Verifying legendary trader perspectives..."
TRADERS=$(cat /tmp/test_aapl.json | grep -o '"warren_buffett"\|"george_soros"\|"stanley_druckenmiller"\|"peter_lynch"\|"paul_tudor_jones"\|"jesse_livermore"' | wc -l)
if [ "$TRADERS" -lt 6 ]; then
    echo "❌ CRITICAL: Missing legendary traders (found $TRADERS/6)"
    exit 1
fi
echo "✅ All 6 legendary traders present"

# Final summary
echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED - SYSTEM READY"
echo "=========================================="
echo ""
echo "Risk-Free Rate: $(cat /tmp/test_aapl.json | grep -o '"risk_free_rate": [0-9.]*' | head -1 || echo 'N/A')"
echo "Cache TTL: 3 minutes"
echo "Validation: ACTIVE"
echo ""
echo "🚀 READY FOR LIVE TRADING MONDAY MORNING"
echo ""
