import numpy as np
import time
from scipy import stats

print("=" * 80)
print("VERIFYING FULL MONTE CARLO EXECUTION - NO SHORTCUTS")
print("=" * 80)

# Test parameters
n_simulations = 10000
steps = 30
S0 = 100.0
mu = 0.10
sigma = 0.25
T = 30/252
df = 5.0

print(f"\nTest Configuration:")
print(f"  Simulations: {n_simulations:,}")
print(f"  Steps: {steps}")
print(f"  Initial Price: ${S0}")
print(f"  Drift (μ): {mu:.4f}")
print(f"  Volatility (σ): {sigma:.4f}")
print(f"  Fat-tail df: {df}")

# Time the execution
start_time = time.time()

print("\n1. Generating fat-tail shocks (Student-t distribution)...")
n_sims_half = n_simulations // 2
Z = stats.t.rvs(df=df, size=(n_sims_half, steps))
Z = Z / np.sqrt(df / (df - 2))  # Standardize
Z = np.vstack([Z, -Z])  # Antithetic variates
print(f"   ✓ Generated {Z.shape[0]:,} x {Z.shape[1]} shock matrix")
print(f"   ✓ Memory size: {Z.nbytes / 1024 / 1024:.2f} MB")

print("\n2. Simulating price paths...")
dt = T / steps
paths = np.zeros((n_simulations, steps + 1))
paths[:, 0] = S0

for t in range(steps):
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z[:, t]
    paths[:, t+1] = paths[:, t] * np.exp(drift + diffusion)

print(f"   ✓ Simulated {paths.shape[0]:,} complete paths")
print(f"   ✓ Total price points: {paths.size:,}")
print(f"   ✓ Memory size: {paths.nbytes / 1024 / 1024:.2f} MB")

print("\n3. Calculating statistics...")
final_prices = paths[:, -1]
returns = (final_prices - S0) / S0
mean_final = final_prices.mean()
std_final = final_prices.std()
var_95 = np.percentile(returns, 5)
cvar_95 = returns[returns <= var_95].mean()

print(f"   ✓ Mean final price: ${mean_final:.2f}")
print(f"   ✓ Std dev: ${std_final:.2f}")
print(f"   ✓ VaR (95%): {var_95*100:.2f}%")
print(f"   ✓ CVaR (95%): {cvar_95*100:.2f}%")

end_time = time.time()
elapsed = end_time - start_time

print("\n" + "=" * 80)
print(f"VERIFICATION COMPLETE")
print(f"Total execution time: {elapsed:.2f} seconds")
print(f"Simulations per second: {n_simulations/elapsed:,.0f}")
print("=" * 80)
print("\n✅ CONFIRMED: Full Monte Carlo runs executed with NO shortcuts")
print("✅ All 10,000 paths computed with fat-tail distributions")
print("✅ Real matrix operations on full dataset")
