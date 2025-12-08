"""
Verify critical calculations are mathematically correct
"""
import json
import sys

# Load AAPL test results
with open('/tmp/test_aapl.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("CALCULATION VERIFICATION - AAPL")
print("=" * 60)

# Extract key values
current_price = data['current_price']
target_price = data['target_price']
stop_loss = data['stop_loss']
confidence = data['confidence']
position_size = data['position_size']

print(f"\n📊 Core Values:")
print(f"  Current Price: ${current_price:.2f}")
print(f"  Target Price: ${target_price:.2f}")
print(f"  Stop Loss: ${stop_loss:.2f}")
print(f"  Confidence: {confidence:.2f}%")
print(f"  Position Size: {position_size} shares")

# Verify Risk/Reward calculation
risk_assessment = data['risk_assessment']
potential_gain_pct = risk_assessment['potential_gain_pct']
potential_loss_pct = risk_assessment['potential_loss_pct']
risk_reward_ratio = risk_assessment['risk_reward_ratio']

# Manual calculation
manual_gain_pct = ((target_price - current_price) / current_price) * 100
manual_loss_pct = ((current_price - stop_loss) / current_price) * 100
manual_rr = abs(target_price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 1.0

print(f"\n💰 Risk/Reward Verification:")
print(f"  Potential Gain: {potential_gain_pct:.2f}% (manual: {manual_gain_pct:.2f}%)")
print(f"  Potential Loss: {potential_loss_pct:.2f}% (manual: {manual_loss_pct:.2f}%)")
print(f"  R/R Ratio: {risk_reward_ratio:.2f} (manual: {manual_rr:.2f})")

# Check if calculations match
gain_match = abs(potential_gain_pct - manual_gain_pct) < 0.01
loss_match = abs(potential_loss_pct - manual_loss_pct) < 0.01
rr_match = abs(risk_reward_ratio - manual_rr) < 0.01

if gain_match and loss_match and rr_match:
    print("  ✅ All R/R calculations CORRECT")
else:
    print("  ❌ CRITICAL: R/R calculation mismatch!")
    sys.exit(1)

# Verify Position Sizing
position_sizing = data['position_sizing']
shares = position_sizing['shares']
position_value = position_sizing['position_value']
dollar_risk = position_sizing['dollar_risk']
bankroll = data['bankroll']

# Manual calculation (1% risk rule)
manual_risk_amount = bankroll * 0.01
manual_price_risk = abs(current_price - stop_loss)
manual_shares = int(manual_risk_amount / manual_price_risk) if manual_price_risk > 0 else 0
manual_position_value = manual_shares * current_price
manual_dollar_risk = manual_shares * abs(current_price - stop_loss)

print(f"\n📈 Position Sizing Verification:")
print(f"  Shares: {shares} (manual: {manual_shares})")
print(f"  Position Value: ${position_value:.2f} (manual: ${manual_position_value:.2f})")
print(f"  Dollar Risk: ${dollar_risk:.2f} (manual: ${manual_dollar_risk:.2f})")
print(f"  Risk % of Bankroll: {(dollar_risk/bankroll)*100:.2f}%")

shares_match = shares == manual_shares
if shares_match:
    print("  ✅ Position sizing CORRECT")
else:
    print("  ❌ CRITICAL: Position sizing mismatch!")
    sys.exit(1)

# Verify Confidence Range
if confidence < 0 or confidence > 100:
    print(f"\n❌ CRITICAL: Confidence out of range: {confidence}%")
    sys.exit(1)
else:
    print(f"\n✅ Confidence in valid range: {confidence:.2f}%")

# Verify Monte Carlo results exist
stochastic = data['stochastic_analysis']
if 'monte_carlo' in stochastic:
    mc = stochastic['monte_carlo']
    if 'mean_path' in mc and len(mc['mean_path']) > 0:
        print(f"✅ Monte Carlo simulation: {len(mc['mean_path'])} data points")
    else:
        print("❌ CRITICAL: Monte Carlo simulation empty!")
        sys.exit(1)
else:
    print("❌ CRITICAL: Monte Carlo simulation missing!")
    sys.exit(1)

# Verify GARCH results
if 'garch_analysis' in stochastic:
    garch = stochastic['garch_analysis']
    if garch.get('converged', False):
        fat_tail_df = garch.get('fat_tail_df', 0)
        print(f"✅ GARCH model converged (fat-tail df: {fat_tail_df:.2f})")
    else:
        print("⚠️  WARNING: GARCH model did not converge (using fallback)")
else:
    print("❌ CRITICAL: GARCH analysis missing!")
    sys.exit(1)

# Verify Legendary Traders
expert = data['expert_reasoning']
if 'legendary_trader_perspectives' in expert:
    traders = expert['legendary_trader_perspectives']
    required_traders = ['warren_buffett', 'george_soros', 'stanley_druckenmiller', 
                       'peter_lynch', 'paul_tudor_jones', 'jesse_livermore']
    
    all_present = all(trader in traders for trader in required_traders)
    if all_present:
        print(f"✅ All 6 legendary traders present")
    else:
        missing = [t for t in required_traders if t not in traders]
        print(f"❌ CRITICAL: Missing traders: {missing}")
        sys.exit(1)
else:
    print("❌ CRITICAL: Legendary trader perspectives missing!")
    sys.exit(1)

# Final validation status
validation_status = data.get('validation_status', 'UNKNOWN')
if validation_status == 'PASSED':
    print(f"\n✅ Validation Status: {validation_status}")
else:
    print(f"\n❌ CRITICAL: Validation Status: {validation_status}")
    if 'validation_errors' in data:
        for error in data['validation_errors']:
            print(f"  - {error}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CALCULATIONS VERIFIED - MATHEMATICALLY CORRECT")
print("=" * 60)
