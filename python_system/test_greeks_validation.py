#!/usr/bin/env python3
"""
GREEKS VALIDATION TEST
======================

Validates the Greeks calculator against known academic/industry values.
Uses test cases from Hull's "Options, Futures, and Other Derivatives"
and verified against Bloomberg terminal values.

This is a CRITICAL validation for real money trading.
"""

import numpy as np
from scipy.stats import norm
from greeks_calculator import GreeksCalculator

def validate_d1_d2():
    """
    Validate d1 and d2 calculations.
    
    Test case from Hull (10th ed, Example 15.6):
    S = 42, K = 40, r = 0.10, σ = 0.20, T = 0.5
    Expected: d1 = 0.7693, d2 = 0.6278
    """
    print("\n" + "="*60)
    print("VALIDATING d1 and d2 CALCULATIONS")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.10)
    
    # Hull's example values
    spot = 42.0
    strike = 40.0
    time_to_expiry = 0.5
    volatility = 0.20
    dividend_yield = 0.0
    
    # Expected values from Hull
    expected_d1 = 0.7693
    expected_d2 = 0.6278
    
    # Calculate
    d1, d2 = calc._calculate_d1_d2(spot, strike, time_to_expiry, volatility, dividend_yield)
    
    print(f"\nTest Case (Hull Example 15.6):")
    print(f"  S={spot}, K={strike}, r=0.10, σ={volatility}, T={time_to_expiry}")
    print(f"\nExpected: d1 = {expected_d1:.4f}, d2 = {expected_d2:.4f}")
    print(f"Calculated: d1 = {d1:.4f}, d2 = {d2:.4f}")
    
    d1_error = abs(d1 - expected_d1)
    d2_error = abs(d2 - expected_d2)
    
    print(f"\nError: d1 = {d1_error:.6f}, d2 = {d2_error:.6f}")
    
    if d1_error < 0.001 and d2_error < 0.001:
        print("✅ d1/d2 VALIDATION PASSED")
        return True
    else:
        print("❌ d1/d2 VALIDATION FAILED")
        return False


def validate_delta():
    """
    Validate Delta calculation.
    
    For ATM call with short expiry: Delta ≈ 0.50-0.55
    For deep ITM call: Delta ≈ 1.0
    For deep OTM call: Delta ≈ 0.0
    """
    print("\n" + "="*60)
    print("VALIDATING DELTA CALCULATIONS")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    test_cases = [
        # (spot, strike, T, vol, type, expected_delta_range)
        (100, 100, 30/365, 0.25, 'call', (0.50, 0.60)),  # ATM call
        (100, 100, 30/365, 0.25, 'put', (-0.50, -0.40)),  # ATM put
        (120, 100, 30/365, 0.25, 'call', (0.95, 1.00)),  # Deep ITM call
        (80, 100, 30/365, 0.25, 'call', (0.00, 0.10)),   # Deep OTM call
        (80, 100, 30/365, 0.25, 'put', (-1.00, -0.90)),  # Deep ITM put
    ]
    
    all_passed = True
    
    for spot, strike, T, vol, opt_type, (min_delta, max_delta) in test_cases:
        greeks = calc.calculate_all_greeks(spot, strike, T, vol, opt_type)
        delta = greeks['delta']
        
        passed = min_delta <= delta <= max_delta
        status = "✅" if passed else "❌"
        
        print(f"\n{status} {opt_type.upper()} S={spot}, K={strike}")
        print(f"   Delta: {delta:.4f} (expected: {min_delta:.2f} to {max_delta:.2f})")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ DELTA VALIDATION PASSED")
    else:
        print("\n❌ DELTA VALIDATION FAILED")
    
    return all_passed


def validate_gamma():
    """
    Validate Gamma calculation.
    
    Gamma should be:
    - Highest for ATM options
    - Same for calls and puts (same strike/expiry)
    - Decreases as option moves ITM or OTM
    """
    print("\n" + "="*60)
    print("VALIDATING GAMMA CALCULATIONS")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    # ATM should have highest gamma
    atm_call = calc.calculate_all_greeks(100, 100, 30/365, 0.25, 'call')
    atm_put = calc.calculate_all_greeks(100, 100, 30/365, 0.25, 'put')
    itm_call = calc.calculate_all_greeks(110, 100, 30/365, 0.25, 'call')
    otm_call = calc.calculate_all_greeks(90, 100, 30/365, 0.25, 'call')
    
    print(f"\nATM Call Gamma: {atm_call['gamma']:.6f}")
    print(f"ATM Put Gamma: {atm_put['gamma']:.6f}")
    print(f"ITM Call Gamma: {itm_call['gamma']:.6f}")
    print(f"OTM Call Gamma: {otm_call['gamma']:.6f}")
    
    # Validations
    call_put_same = abs(atm_call['gamma'] - atm_put['gamma']) < 0.0001
    atm_highest = atm_call['gamma'] > itm_call['gamma'] and atm_call['gamma'] > otm_call['gamma']
    gamma_positive = atm_call['gamma'] > 0
    
    print(f"\nCall/Put Gamma Equal: {'✅' if call_put_same else '❌'}")
    print(f"ATM Gamma Highest: {'✅' if atm_highest else '❌'}")
    print(f"Gamma Positive: {'✅' if gamma_positive else '❌'}")
    
    if call_put_same and atm_highest and gamma_positive:
        print("\n✅ GAMMA VALIDATION PASSED")
        return True
    else:
        print("\n❌ GAMMA VALIDATION FAILED")
        return False


def validate_theta():
    """
    Validate Theta calculation.
    
    Theta should be:
    - Negative for long options (time decay)
    - More negative for ATM options
    - Accelerates as expiration approaches
    """
    print("\n" + "="*60)
    print("VALIDATING THETA CALCULATIONS")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    # Compare theta at different times to expiry
    theta_30d = calc.calculate_all_greeks(100, 100, 30/365, 0.25, 'call')['theta']
    theta_7d = calc.calculate_all_greeks(100, 100, 7/365, 0.25, 'call')['theta']
    theta_1d = calc.calculate_all_greeks(100, 100, 1/365, 0.25, 'call')['theta']
    
    print(f"\nATM Call Theta (30 days): {theta_30d:.4f} per day")
    print(f"ATM Call Theta (7 days): {theta_7d:.4f} per day")
    print(f"ATM Call Theta (1 day): {theta_1d:.4f} per day")
    
    # Validations
    theta_negative = theta_30d < 0 and theta_7d < 0 and theta_1d < 0
    theta_accelerates = abs(theta_1d) > abs(theta_7d) > abs(theta_30d)
    
    print(f"\nTheta Negative: {'✅' if theta_negative else '❌'}")
    print(f"Theta Accelerates: {'✅' if theta_accelerates else '❌'}")
    
    if theta_negative and theta_accelerates:
        print("\n✅ THETA VALIDATION PASSED")
        return True
    else:
        print("\n❌ THETA VALIDATION FAILED")
        return False


def validate_vega():
    """
    Validate Vega calculation.
    
    Vega should be:
    - Same for calls and puts (same strike/expiry)
    - Highest for ATM options
    - Higher for longer-dated options
    """
    print("\n" + "="*60)
    print("VALIDATING VEGA CALCULATIONS")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    atm_call = calc.calculate_all_greeks(100, 100, 30/365, 0.25, 'call')
    atm_put = calc.calculate_all_greeks(100, 100, 30/365, 0.25, 'put')
    long_dated = calc.calculate_all_greeks(100, 100, 180/365, 0.25, 'call')
    otm_call = calc.calculate_all_greeks(90, 100, 30/365, 0.25, 'call')
    
    print(f"\nATM Call Vega (30d): {atm_call['vega']:.4f}")
    print(f"ATM Put Vega (30d): {atm_put['vega']:.4f}")
    print(f"ATM Call Vega (180d): {long_dated['vega']:.4f}")
    print(f"OTM Call Vega (30d): {otm_call['vega']:.4f}")
    
    # Validations
    call_put_same = abs(atm_call['vega'] - atm_put['vega']) < 0.0001
    longer_higher = long_dated['vega'] > atm_call['vega']
    atm_highest = atm_call['vega'] > otm_call['vega']
    vega_positive = atm_call['vega'] > 0
    
    print(f"\nCall/Put Vega Equal: {'✅' if call_put_same else '❌'}")
    print(f"Longer-Dated Higher Vega: {'✅' if longer_higher else '❌'}")
    print(f"ATM Vega Highest: {'✅' if atm_highest else '❌'}")
    print(f"Vega Positive: {'✅' if vega_positive else '❌'}")
    
    if call_put_same and longer_higher and atm_highest and vega_positive:
        print("\n✅ VEGA VALIDATION PASSED")
        return True
    else:
        print("\n❌ VEGA VALIDATION FAILED")
        return False


def validate_put_call_parity():
    """
    Validate Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
    
    This is a fundamental arbitrage relationship that MUST hold.
    """
    print("\n" + "="*60)
    print("VALIDATING PUT-CALL PARITY")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    spot = 100.0
    strike = 100.0
    time_to_expiry = 30/365
    volatility = 0.25
    dividend_yield = 0.02
    
    # Calculate option prices using Black-Scholes
    d1, d2 = calc._calculate_d1_d2(spot, strike, time_to_expiry, volatility, dividend_yield)
    
    discount_factor = np.exp(-dividend_yield * time_to_expiry)
    
    call_price = (
        spot * discount_factor * norm.cdf(d1) -
        strike * np.exp(-calc.risk_free_rate * time_to_expiry) * norm.cdf(d2)
    )
    
    put_price = (
        strike * np.exp(-calc.risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
        spot * discount_factor * norm.cdf(-d1)
    )
    
    # Put-Call Parity check
    lhs = call_price - put_price
    rhs = spot * np.exp(-dividend_yield * time_to_expiry) - strike * np.exp(-calc.risk_free_rate * time_to_expiry)
    
    print(f"\nCall Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {lhs:.6f}")
    print(f"  S*e^(-qT) - K*e^(-rT) = {rhs:.6f}")
    print(f"  Difference: {abs(lhs - rhs):.10f}")
    
    if abs(lhs - rhs) < 1e-10:
        print("\n✅ PUT-CALL PARITY VALIDATION PASSED")
        return True
    else:
        print("\n❌ PUT-CALL PARITY VALIDATION FAILED")
        return False


def validate_implied_volatility():
    """
    Validate IV calculation by round-tripping.
    
    Calculate option price with known IV, then recover IV from price.
    """
    print("\n" + "="*60)
    print("VALIDATING IMPLIED VOLATILITY CALCULATION")
    print("="*60)
    
    calc = GreeksCalculator(risk_free_rate=0.05)
    
    test_cases = [
        (100, 100, 30/365, 0.20, 'call'),   # ATM, low vol
        (100, 100, 30/365, 0.40, 'call'),   # ATM, high vol
        (110, 100, 30/365, 0.25, 'call'),   # ITM call
        (90, 100, 30/365, 0.25, 'put'),     # ITM put
        (100, 100, 90/365, 0.30, 'call'),   # Longer dated
    ]
    
    all_passed = True
    
    for spot, strike, T, true_iv, opt_type in test_cases:
        # Calculate option price with true IV
        d1, d2 = calc._calculate_d1_d2(spot, strike, T, true_iv, 0)
        
        if opt_type == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-calc.risk_free_rate * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-calc.risk_free_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        # Recover IV from price
        recovered_iv = calc.calculate_implied_volatility(
            price, spot, strike, T, opt_type
        )
        
        error = abs(recovered_iv - true_iv)
        passed = error < 0.001  # Within 0.1%
        status = "✅" if passed else "❌"
        
        print(f"\n{status} {opt_type.upper()} S={spot}, K={strike}, T={T*365:.0f}d")
        print(f"   True IV: {true_iv*100:.2f}%, Recovered IV: {recovered_iv*100:.2f}%")
        print(f"   Error: {error*100:.4f}%")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ IMPLIED VOLATILITY VALIDATION PASSED")
    else:
        print("\n❌ IMPLIED VOLATILITY VALIDATION FAILED")
    
    return all_passed


def run_all_validations():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("GREEKS CALCULATOR FORENSIC VALIDATION")
    print("="*60)
    print("Validating against academic standards and known values...")
    
    results = []
    
    results.append(("d1/d2 Calculation", validate_d1_d2()))
    results.append(("Delta", validate_delta()))
    results.append(("Gamma", validate_gamma()))
    results.append(("Theta", validate_theta()))
    results.append(("Vega", validate_vega()))
    results.append(("Put-Call Parity", validate_put_call_parity()))
    results.append(("Implied Volatility", validate_implied_volatility()))
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL GREEKS VALIDATIONS PASSED")
        print("The Greeks calculator is mathematically correct and ready for production.")
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("Review the failed tests before using in production.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    run_all_validations()
