import numpy as np
import pandas as pd


alpha_hat = 13.527722472498198
beta_hat = -9.525035530742965
delta_hat = 0.20862148989303406
mu_hat = 0.0
S0 = 7623.68
q = 1.06 / 100

# Échéance  : 6 mois
T_lookback = 0.5
n_paths = 4000
n_steps = 180
dt = T_lookback / n_steps

# Taux sans risque 
maturities_days = np.array([28, 56, 84, 175, 329])
r_simple = np.array([3.93, 4.03, 4.08, 4.20, 4.37]) / 100

# Convertion 
tau_curve = maturities_days / 365
discount_curve = 1 / (1 + r_simple * maturities_days / 360)
r_continuous_curve = -np.log(discount_curve) / tau_curve

# Interpolation 
r = float(np.interp(T_lookback, tau_curve, r_continuous_curve))

print(f"alpha = {alpha_hat:.8f}")
print(f"beta  = {beta_hat:.8f}")
print(f"delta = {delta_hat:.8f}")
print(f"S0    = {S0:.2f}")
print(f"q     = {q:.4%}")
print(f"r à 6 mois = {r:.4%}")

# Simulation et Valorisation 
def simulate_ig_increments(alpha, beta, delta, dt, N):
    """Simule N incréments inverse gaussiens selon la paramétrisation du notebook."""
    a = 1.0
    b = delta * np.sqrt(alpha**2 - beta**2)

    v = np.random.standard_normal(size=N)
    y = v**2

    x = (
        (a * dt / b)
        + y / (2 * b**2)
        - np.sqrt(4 * a * dt * b * y + y**2) / (2 * b**2)
    )

    u = np.random.uniform(size=N)

    ig_increments = np.where(
        u <= (a * dt) / (a * dt + b * x),
        x,
        (a * dt)**2 / (b**2 * x)
    )

    return ig_increments

def price_lookback_mc(alpha, beta, delta, seed=2026):

    # Vérification 
    if alpha <= 0 or delta <= 0:
        raise ValueError("alpha et delta doivent être positifs.")

    if abs(beta) >= alpha:
        raise ValueError("Il faut respecter |beta| < alpha.")

    if abs(beta + 1) >= alpha:
        raise ValueError("Condition risque-neutre non respectée : |beta + 1| < alpha.")

    # Paramètres auxiliaires NIG
    gamma = np.sqrt(alpha**2 - beta**2)
    gamma_tilt = np.sqrt(alpha**2 - (beta + 1)**2)

    # Correction 

    omega = -(mu_hat + delta * (gamma - gamma_tilt))

    np.random.seed(seed)

    log_s = np.full(n_paths, np.log(S0))
    min_s = np.full(n_paths, S0)

    # Steps 180 
    for step in range(n_steps):

        # Incrémentation NIG
        ig_increments = simulate_ig_increments(
            alpha=alpha,
            beta=beta,
            delta=delta,
            dt=dt,
            N=n_paths
        )

        dX = (
            beta * delta**2 * ig_increments
            + delta * np.sqrt(ig_increments)
            * np.random.standard_normal(n_paths)
        )

        # Évolution 
        log_s += (r - q + omega) * dt + dX

        s = np.exp(log_s)

        # update 
        min_s = np.minimum(min_s, s)

    # Payoff 
    payoff = np.exp(log_s) - min_s

    discounted_payoff = np.exp(-r * T_lookback) * payoff

    # Estimation Monte-Carlo 
    price = discounted_payoff.mean()
    standard_error = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)

    return {
        "Price": price,
        "Standard_Error": standard_error,
        "CI_95_Lower": price - 1.96 * standard_error,
        "CI_95_Upper": price + 1.96 * standard_error,
        "Omega": omega
    }


# Prices
baseline = price_lookback_mc(
    alpha=alpha_hat,
    beta=beta_hat,
    delta=delta_hat
)

print(f"Prix Lookback Call : {baseline['Price']:.4f}")
print(f"Erreur-type MC     : {baseline['Standard_Error']:.4f}")
print(
    f"IC 95 %            : "
    f"[{baseline['CI_95_Lower']:.4f}, {baseline['CI_95_Upper']:.4f}]"
)
print(f"Correction omega   : {baseline['Omega']:.8f}")

## Sensibility
def price_lookback_mc(alpha, beta, delta, seed=2026):
    """
    Prix d'un Lookback Call de payoff:
    S_T - min(S_t), observé sur 180 pas de temps.

    Réutilise simulate_ig_increments définie dans le notebook précédent.
    """

    # Vérification
    if alpha <= 0 or delta <= 0:
        raise ValueError("alpha et delta doivent être positifs.")

    if abs(beta) >= alpha:
        raise ValueError("Il faut respecter |beta| < alpha.")

    if abs(beta + 1) >= alpha:
        raise ValueError("Condition risque-neutre non respectée : |beta + 1| < alpha.")

    gamma = np.sqrt(alpha**2 - beta**2)
    gamma_tilt = np.sqrt(alpha**2 - (beta + 1)**2)

    # Correction 
    omega = -(mu_hat + delta * (gamma - gamma_tilt))

    np.random.seed(seed)

    log_s = np.full(n_paths, np.log(S0))
    min_s = np.full(n_paths, S0)

    # Simulation 180 steps
    for step in range(n_steps):

        ig_increments = simulate_ig_increments(
            alpha=alpha,
            beta=beta,
            delta=delta,
            dt=dt,
            N=n_paths
        )

        dX = (
            beta * delta**2 * ig_increments
            + delta * np.sqrt(ig_increments)
            * np.random.standard_normal(n_paths)
        )

        log_s += (r - q + omega) * dt + dX

        s = np.exp(log_s)

        min_s = np.minimum(min_s, s)

    # Payoff 
    payoff = np.exp(log_s) - min_s

    discounted_payoff = np.exp(-r * T_lookback) * payoff

    # Estimation Monte-Carlo price 
    price = discounted_payoff.mean()
    standard_error = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)

    return {
        "Price": price,
        "Standard_Error": standard_error,
        "CI_95_Lower": price - 1.96 * standard_error,
        "CI_95_Upper": price + 1.96 * standard_error,
        "Omega": omega
    }


# prices 
baseline = price_lookback_mc(
    alpha=alpha_hat,
    beta=beta_hat,
    delta=delta_hat
)

print(f"Prix Lookback Call : {baseline['Price']:.4f}")
print(f"Erreur-type MC     : {baseline['Standard_Error']:.4f}")
print(
    f"IC 95 %            : "
    f"[{baseline['CI_95_Lower']:.4f}, {baseline['CI_95_Upper']:.4f}]"
)
print(f"Correction omega   : {baseline['Omega']:.8f}")

scenarios = [
    ("Référence", alpha_hat, delta_hat),
    ("Alpha -10 %", 0.90 * alpha_hat, delta_hat),
    ("Alpha +10 %", 1.10 * alpha_hat, delta_hat),
    ("Delta -10 %", alpha_hat, 0.90 * delta_hat),
    ("Delta +10 %", alpha_hat, 1.10 * delta_hat),
]

results = []

for name, alpha_scenario, delta_scenario in scenarios:
    result = price_lookback_mc(
        alpha=alpha_scenario,
        beta=beta_hat,
        delta=delta_scenario,
        seed=2026
    )

    results.append({
        "Scénario": name,
        "Alpha": alpha_scenario,
        "Beta": beta_hat,
        "Delta": delta_scenario,
        "Omega": result["Omega"],
        "Prix MC": result["Price"],
        "Erreur-type": result["Standard_Error"],
        "Variation vs référence (%)":
            100 * (result["Price"] / baseline["Price"] - 1)
    })

sensitivity_table = pd.DataFrame(results)
print(sensitivity_table.round(4).to_string(index=False))
