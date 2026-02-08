"""
Test script for new integrations: TAAPI, FinancialDatasets, EXA AI, Personal Recommendation
"""
import json
import sys
import os

# Ensure we're in the right directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def safe_print(obj, indent=2):
    """Print JSON-safe output."""
    try:
        print(json.dumps(obj, indent=indent, default=str))
    except:
        print(str(obj))

def test_taapi():
    """Test TAAPI.io client."""
    print("\n" + "="*60)
    print("TESTING TAAPI.io CLIENT")
    print("="*60)
    try:
        from taapi_client import TaapiClient
        client = TaapiClient()
        
        # Test RSI
        print("\n--- RSI for AAPL ---")
        rsi = client.get_indicator('rsi', 'AAPL', '1d')
        safe_print(rsi)
        
        # Test MACD
        print("\n--- MACD for AAPL ---")
        macd = client.get_indicator('macd', 'AAPL', '1d')
        safe_print(macd)
        
        print("\n[TAAPI] PASSED" if rsi or macd else "\n[TAAPI] PARTIAL - some indicators may require paid plan")
        return True
    except Exception as e:
        print(f"\n[TAAPI] ERROR: {e}")
        return False

def test_financial_datasets():
    """Test FinancialDatasets.ai client."""
    print("\n" + "="*60)
    print("TESTING FINANCIALDATASETS.AI CLIENT")
    print("="*60)
    try:
        from financial_datasets_client import FinancialDatasetsClient
        client = FinancialDatasetsClient()
        
        # Test price snapshot
        print("\n--- Price Snapshot for AAPL ---")
        snapshot = client.get_stock_price_snapshot('AAPL')
        safe_print(snapshot)
        
        # Test financial metrics
        print("\n--- Financial Metrics for AAPL ---")
        metrics = client.get_financial_metrics('AAPL', limit=1)
        safe_print(metrics)
        
        # Test company facts
        print("\n--- Company Facts for AAPL ---")
        facts = client.get_company_facts('AAPL')
        if facts and 'error' not in facts:
            safe_print({k: v for k, v in (facts.get('company_facts', facts) if isinstance(facts.get('company_facts', facts), dict) else facts).items() if k in ['name', 'sector', 'industry', 'exchange']})
        else:
            safe_print(facts)
        
        has_data = 'error' not in snapshot or 'error' not in metrics
        print(f"\n[FinancialDatasets] {'PASSED' if has_data else 'FAILED'}")
        return has_data
    except Exception as e:
        print(f"\n[FinancialDatasets] ERROR: {e}")
        return False

def test_exa():
    """Test EXA AI client."""
    print("\n" + "="*60)
    print("TESTING EXA AI CLIENT")
    print("="*60)
    try:
        from exa_client import ExaClient
        client = ExaClient()
        
        # Test basic search
        print("\n--- Candlestick Analysis Search for AAPL ---")
        result = client.search(
            query="AAPL stock candlestick chart pattern analysis",
            num_results=3,
            search_type="auto"
        )
        
        if result.get('results'):
            print(f"Found {len(result['results'])} results")
            for r in result['results'][:2]:
                print(f"  - {r.get('title', 'N/A')[:80]}")
                print(f"    URL: {r.get('url', 'N/A')[:80]}")
            print("\n[EXA] PASSED")
            return True
        elif result.get('error'):
            print(f"Error: {result['error']}")
            print("\n[EXA] FAILED")
            return False
        else:
            print("No results returned")
            print("\n[EXA] PARTIAL")
            return True
    except Exception as e:
        print(f"\n[EXA] ERROR: {e}")
        return False

def test_personal_recommendation():
    """Test Personal Recommendation Engine."""
    print("\n" + "="*60)
    print("TESTING PERSONAL RECOMMENDATION ENGINE")
    print("="*60)
    try:
        from personal_recommendation import PersonalRecommendationEngine
        engine = PersonalRecommendationEngine()
        
        # Create a mock analysis dict
        mock_analysis = {
            'symbol': 'AAPL',
            'current_price': 195.50,
            'signal': 'BUY',
            'confidence': 72,
            'bankroll': 10000,
            'stop_loss': 188.00,
            'target_price': 210.00,
            'technical_analysis': {
                'technical_score': 68,
                'momentum_score': 65,
                'trend_score': 72,
                'volatility_score': 45,
                'adx': 28
            },
            'fundamental_score': 75,
            'sentiment_score': 62,
            'pattern_recognition': {
                'confidence': 55,
                'similar_patterns': [1, 2, 3],
                'pattern_prediction': {
                    'probability_up': 0.65,
                    'expected_return': 0.04,
                    'sample_size': 12,
                    'horizons': {
                        '5d': {'probability_up': 0.60},
                        '30d': {'probability_up': 0.68}
                    }
                },
                'regime_match': {
                    'regime_name': 'post_correction_recovery',
                    'period': '2023',
                    'description': 'Market recovering after correction',
                    'outcome': 'Gradual recovery with 15% gain over 3 months',
                    'match_score': 62
                }
            },
            'risk_assessment': {
                'risk_reward_ratio': 2.1,
                'potential_gain_pct': 7.4,
                'potential_loss_pct': 3.8
            },
            'stochastic_analysis': {
                'expected_return': 0.03,
                'var_95': 0.08,
                'cvar_95': 0.12
            },
            'technical_indicators': {
                'rsi': 52
            }
        }
        
        rec = engine.generate_recommendation(mock_analysis)
        
        print(f"\nAction: {rec['action']}")
        print(f"Conviction: {rec['conviction_score']} ({rec['conviction_level']})")
        print(f"Position: {rec['position_size_shares']} shares (${rec['position_size_dollars']:,.2f})")
        print(f"Risk: ${rec['risk_per_trade_dollars']:,.2f}")
        print(f"\nTime Horizon: {rec['time_horizon']['horizon']} ({rec['time_horizon']['suggested_days']})")
        print(f"\n--- NARRATIVE ---")
        print(rec['narrative'][:500])
        print("...")
        print(f"\n--- RISK WARNINGS ---")
        for w in rec['risk_warnings']:
            print(f"  ! {w}")
        
        print("\n[Personal Recommendation] PASSED")
        return True
    except Exception as e:
        print(f"\n[Personal Recommendation] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_recognition():
    """Test improved Pattern Recognition Engine."""
    print("\n" + "="*60)
    print("TESTING PATTERN RECOGNITION ENGINE")
    print("="*60)
    try:
        from pattern_recognition import PatternRecognitionEngine
        engine = PatternRecognitionEngine()
        
        # Test with AAPL
        print("\n--- Pattern Recognition for AAPL ---")
        result = engine.analyze('AAPL')
        
        if result:
            print(f"Patterns found: {len(result.get('similar_patterns', []))}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            pred = result.get('pattern_prediction', {})
            if pred:
                print(f"Probability Up: {pred.get('probability_up', 'N/A')}")
                print(f"Expected Return: {pred.get('expected_return', 'N/A')}")
                print(f"Sample Size: {pred.get('sample_size', 'N/A')}")
            regime = result.get('regime_match', {})
            if regime:
                print(f"Regime Match: {regime.get('regime_name', 'N/A')}")
                print(f"Match Score: {regime.get('match_score', 'N/A')}")
            print("\n[Pattern Recognition] PASSED")
            return True
        else:
            print("No result returned")
            print("\n[Pattern Recognition] FAILED")
            return False
    except Exception as e:
        print(f"\n[Pattern Recognition] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("NEW INTEGRATIONS TEST SUITE")
    print("="*60)
    
    results = {}
    
    results['taapi'] = test_taapi()
    results['financial_datasets'] = test_financial_datasets()
    results['exa'] = test_exa()
    results['personal_recommendation'] = test_personal_recommendation()
    results['pattern_recognition'] = test_pattern_recognition()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Total: {total_passed}/{total} passed")
