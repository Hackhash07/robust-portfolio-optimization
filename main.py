

# this model is created on concepts solely come from A practical guide to robust portfolio optimization
# C. Yin, R. Perchet & F. Soupé research paper.

import pandas as pd
import numpy as np
import cvxpy as cp
import os
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
path = "/Users/vudumulasakethreddy/Desktop/competition/"
file_map = {
    'Reliance': 'reliance2.csv', 'Power': 'powergrid1.csv',
    'NTPC': 'ntpc1.csv', 'HDFC': 'hdfc1.csv',
    'ICICI': 'icici1.csv', 'Sunpharma': 'sunpharma1.csv',
    'ITC': 'itc1.csv'
}

# Sector Groups
sector_groups = {
    'Reliance': 'S1', 'Power': 'S1', 'NTPC': 'S1',
    'HDFC': 'S2', 'ICICI': 'S2',
    'Sunpharma': 'S3', 'ITC': 'S3'
}

tickers = list(file_map.keys())
n = len(tickers)
price_data = pd.DataFrame()

for ticker, file in file_map.items():
    df = pd.read_csv(os.path.join(path, file))
    df.columns = df.columns.str.strip()
    date_col = [c for c in df.columns if 'Date' in c][0]
    price_col = [c for c in df.columns if 'EQN' in c][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    price_data[ticker] = df[price_col]

returns = np.log(price_data / price_data.shift(1)).dropna()
train_size = int(len(returns) * 0.7)
train_ret = returns.iloc[:train_size]
test_ret = returns.iloc[train_size:]

# ==========================================
# 2. AIC GRID SEARCH & GARCH
# ==========================================
mu_vec, vols = [], []
best_orders = {}

for t in tickers:
    best_aic = np.inf
    best_order = (0, 0, 0)
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(train_ret[t], order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, 0, q)
            except:
                continue

    best_orders[t] = best_order
    final_arima = ARIMA(train_ret[t], order=best_order).fit()
    mu_vec.append(final_arima.forecast(1).values[0])

    am = arch_model(train_ret[t], vol='Garch', p=1,
                    q=1, rescale=False).fit(disp="off")
    vols.append(np.sqrt(am.forecast(horizon=1).variance.values[-1, 0]))

mu, vols = np.array(mu_vec), np.array(vols)

# ==========================================
# 3. MID-GROUND CCM (70% Intra-Sector Rule)
# ==========================================


def get_mid_ground_ccm(p):

    C = np.eye(n)

    #  You provide these
    intra_dict = {
        'S1': 0.60,
        'S2': 0.80,
        'S3': 0.40
    }

    for i in range(n):
        for j in range(i+1, n):

            s1, s2 = sector_groups[tickers[i]], sector_groups[tickers[j]]

            # Same sector → fixed values
            if s1 == s2:
                val = intra_dict[s1]

            # Cross-sector → optimize
            elif {s1, s2} == {'S1', 'S2'}:
                val = p[0]
            elif {s1, s2} == {'S1', 'S3'}:
                val = p[1]
            else:
                val = p[2]  # S2-S3

            C[i, j] = C[j, i] = val

    return C


# Optimize only 3 cross-sector parameters instead of 6
res_ccm = minimize(
    lambda p: np.linalg.norm(train_ret.corr().values -
                             get_mid_ground_ccm(p), 'fro'),
    x0=[0.3, 0.3, 0.3],
    bounds=[(0, 0.7)] * 3

)
mid_ground_corr = get_mid_ground_ccm(res_ccm.x)
Sigma_base = np.outer(vols, vols) * mid_ground_corr

# ==========================================
# 4. ROBUST OPTIMIZATION
# ==========================================
ann_mu, ann_sigma = train_ret.mean() * 252, train_ret.std() * np.sqrt(252)
avg_sharpe = (ann_mu / ann_sigma).mean()
k = (abs(avg_sharpe) / 2)
Omega = np.diag(np.diag(Sigma_base))

# Omega^(1/2) — safe since Omega is diagonal with positive entries
Omega_sqrt = np.diag(np.sqrt(np.diag(Omega)))

w = cp.Variable(n)

#  DCP-compliant: norm2 is convex, subtracting it in maximization is valid
robust_penalty = k * cp.norm(Omega_sqrt @ w, 2)   # = k * sqrt(w^T Ω w)

prob = cp.Problem(
    cp.Maximize(mu @ w - 5 * cp.quad_form(w, Sigma_base) - robust_penalty),
    [cp.sum(w) == 1]
)
prob.solve()

final_weights = w.value

# Gradient of robust penalty (matches Formula 16)
wTOw = np.sqrt(final_weights @ Omega @ final_weights)
mu_ro = mu - (k / wTOw) * (Omega @ final_weights)
Sigma_RO = Sigma_base + (k / wTOw) * Omega


# ==========================================
# 5. OUTPUT
# ==========================================
print("\n" + "="*55 + "\nOUTPUT\n" + "="*55)
print("\n--- Arima expected returns (AIC Detected) ---")
for i, t in enumerate(tickers):
    print(f"{t:<12} (Order {str(best_orders[t]):<7}): {mu[i]:.6f}")


print("\n--- Mid-Ground CCM matrix (Intra-Sector = 0.70) ---")
print(pd.DataFrame(mid_ground_corr, index=tickers, columns=tickers).round(4))

print("\n--- RO weights (5% < w < 30%) ---")
for i, t in enumerate(tickers):
    print(f"{t:<12}: {final_weights[i]:.4f}")

eigvals = np.sort(np.real(np.linalg.eigvals(Sigma_RO)))[::-1]

# Condition Number 1 → λ1 / λn
cond1 = np.sqrt(eigvals[0] / eigvals[-1])

# Condition Number 2 → λ1 / λ(n-1)
cond2 = np.sqrt(eigvals[0] / eigvals[-2])

# Condition Number 3 → λ1 / λ(n-2)
cond3 = np.sqrt(eigvals[0] / eigvals[-3])

print("\n--- Condition Numbers ---")
print(f"Condition Number 1 (λ1/λn):   {cond1:.4f}")
print(f"Condition Number 2 (λ1/λn-1): {cond2:.4f}")
print(f"Condition Number 3 (λ1/λn-2): {cond3:.4f}")

print("\n--- out-of sample performance ---")
test_p = test_ret @ final_weights
print(f"Mean Return: {test_p.mean():.6f}")
print(f"Volatility:  {test_p.std() * np.sqrt(252):.4%}")
print(f"Sharpe Ratio: {(test_p.mean()*252) / (test_p.std()*np.sqrt(252)):.4f}")
