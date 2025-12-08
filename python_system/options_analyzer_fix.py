# This is the corrected call analysis section
# Lines 186-290

                # Analyze calls
                calls = data.get('calls', pd.DataFrame())
                if not calls.empty:
                    for idx, row in calls.iterrows():
                        strike = row['strike']
                        last_price = row.get('lastPrice', 0)
                        iv = row.get('impliedVolatility', 0.3)
                        volume = row.get('volume', 0)
                        open_interest = row.get('openInterest', 0)
                        
                        if last_price == 0 or iv == 0:
                            continue
                        
                        # Calculate Greeks (including delta)
                        greeks = self.calculate_greeks(
                            S=current_price,
                            K=strike,
                            T=T,
                            r=r,
                            sigma=iv,
                            option_type='call'
                        )
                        delta = greeks['delta']
                        
                        # Filter by delta range
                        if not (min_delta <= delta <= max_delta):
                            continue
                        
                        # OI/Vol ratio
                        oi_vol_ratio = open_interest / volume if volume > 0 else 0
                        
                        # IV percentile (simplified - compare to 30-day avg vol)
                        returns = price_data['close'].pct_change().dropna()
                        hist_vol = returns.std() * np.sqrt(252)
                        iv_percentile = (iv / hist_vol) * 100 if hist_vol > 0 else 100
                        
                        # Liquidity score
                        liquidity_score = min(volume / 100, 100) + min(open_interest / 1000, 100)
                        
                        # Delta score (prefer 0.45)
                        delta_score = 100 - abs(delta - 0.45) * 200
                        
                        # IV score (prefer moderate IV)
                        iv_score = 100 - abs(iv - 0.35) * 200
                        
                        # Combined score
                        score = (delta_score * 0.4 + liquidity_score * 0.3 + iv_score * 0.3)
                        
                        # Profit/loss scenarios
                        pl_scenarios = self.calculate_profit_loss_scenarios(
                            current_price=current_price,
                            strike=strike,
                            option_price=last_price,
                            option_type='call',
                            expiration_date=exp_date.strftime('%Y-%m-%d')
                        )
                        
                        call_candidates.append({
                            'type': 'CALL',
                            'strike': strike,
                            'expiration': exp_date.strftime('%Y-%m-%d'),
                            'days_to_expiry': days_to_expiry,
                            'last_price': last_price,
                            'bid': row.get('bid', 0),
                            'ask': row.get('ask', 0),
                            'volume': volume,
                            'open_interest': open_interest,
                            'oi_vol_ratio': oi_vol_ratio,
                            'implied_volatility': iv,
                            'iv_percentile': iv_percentile,
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'rho': greeks['rho'],
                            'theoretical_price': greeks['theoretical_price'],
                            'intrinsic_value': max(0, current_price - strike),
                            'extrinsic_value': last_price - max(0, current_price - strike),
                            'breakeven': pl_scenarios['breakeven'],
                            'max_loss': pl_scenarios['max_loss'],
                            'max_profit': pl_scenarios['max_profit'],
                            'profit_loss_scenarios': pl_scenarios['scenarios'],
                            'score': score
                        })
