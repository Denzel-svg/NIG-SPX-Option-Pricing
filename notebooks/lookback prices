# ============================================================
# LOOKBACK CALL: BASELINE MONTE CARLO PRICE
# ============================================================

# Contract and simulation settings
T_lookback = 0.5
n_steps_lookback = 180
n_paths_lookback = 4000
dt_lookback = T_lookback / n_steps_lookback

# Use the calibrated risk-neutral NIG parameters
alpha_hat = calibration_result.parameters.alpha
beta_hat = calibration_result.parameters.beta
delta_hat = calibration_result.parameters.delta

# Read S0 and the six-month market inputs from the calibration dataset
S0_lookback = float(calibration_data["Spot"].iloc[0])
q_lookback = float(calibration_data["Dividend_Yield"].iloc[0])

# Interpolate the continuously compounded risk-free rate at six months
rate_curve = (
    calibration_data[["tau", "r"]]
    .drop_duplicates()
    .sort_values("tau")
)
r_lookback = float(np.interp(
    T_lookback,
    rate_curve["tau"].to_numpy(),
    rate_curve["r"].to_numpy()
))

def price_lookback_mc(alpha, beta, delta, seed=2026):
    """
    Monte Carlo price of a discretely monitored lookback call.
    Uses the existing simulate_ig_increments function from this notebook.
    """
    if alpha <= 0 or delta <= 0:
        raise ValueError("alpha and delta must be positive.")
    if abs(beta) >= alpha or abs(beta + 1) >= alpha:
        raise ValueError("The NIG parameters violate the risk-neutral constraints.")

    gamma = np.sqrt(alpha**2 - beta**2)
    gamma_tilt = np.sqrt(alpha**2 - (beta + 1)**2)

    # Risk-neutral correction for the zero-location NIG process
    omega = -delta * (gamma - gamma_tilt)

    # Resetting the seed gives comparable random draws across scenarios
    np.random.seed(seed)

    log_s = np.full(n_paths_lookback, np.log(S0_lookback))
    min_s = np.full(n_paths_lookback, S0_lookback)

    for _ in range(n_steps_lookback):
        ig_increments = simulate_ig_increments(
            alpha=alpha,
            beta=beta,
            delta=delta,
            dt=dt_lookback,
            N=n_paths_lookback
        )

        # NIG increment under the parameterisation used in the previous notebook
        dX = (
            beta * delta**2 * ig_increments
            + delta * np.sqrt(ig_increments)
            * np.random.standard_normal(n_paths_lookback)
        )

        # Risk-neutral log-price increment
        log_s += (r_lookback - q_lookback + omega) * dt_lookback + dX

        s = np.exp(log_s)
        min_s = np.minimum(min_s, s)

    payoff = np.exp(log_s) - min_s
    discounted_payoff = np.exp(-r_lookback * T_lookback) * payoff

    price = discounted_payoff.mean()
    standard_error = discounted_payoff.std(ddof=1) / np.sqrt(n_paths_lookback)

    return {
        "Price": price,
        "Standard_Error": standard_error,
        "CI_95_Lower": price - 1.96 * standard_error,
        "CI_95_Upper": price + 1.96 * standard_error,
        "Omega": omega
    }

baseline_result = price_lookback_mc(
    alpha=alpha_hat,
    beta=beta_hat,
    delta=delta_hat
)

print(f"S0: {S0_lookback:.4f}")
print(f"Risk-free rate (continuous): {r_lookback:.6%}")
print(f"Dividend yield: {q_lookback:.6%}")
print(f"Lookback call price: {baseline_result['Price']:.4f}")
print(f"Monte Carlo standard error: {baseline_result['Standard_Error']:.4f}")
print(
    "95% confidence interval: "
    f"[{baseline_result['CI_95_Lower']:.4f}, "
    f"{baseline_result['CI_95_Upper']:.4f}]"
)
print(f"Risk-neutral drift correction omega: {baseline_result['Omega']:.8f}")
# ============================================================
# LOOKBACK CALL: SENSITIVITY TO ALPHA AND DELTA
# ============================================================

sensitivity_scenarios = [
    ("Baseline", alpha_hat, delta_hat),
    ("Alpha -10%", 0.90 * alpha_hat, delta_hat),
    ("Alpha +10%", 1.10 * alpha_hat, delta_hat),
    ("Delta -10%", alpha_hat, 0.90 * delta_hat),
    ("Delta +10%", alpha_hat, 1.10 * delta_hat),
]

sensitivity_results = []

for scenario, alpha_value, delta_value in sensitivity_scenarios:
    result = price_lookback_mc(
        alpha=alpha_value,
        beta=beta_hat,
        delta=delta_value,
        seed=2026
    )

    sensitivity_results.append({
        "Scenario": scenario,
        "Alpha": alpha_value,
        "Beta": beta_hat,
        "Delta": delta_value,
        "Omega": result["Omega"],
        "Lookback_Price": result["Price"],
        "Standard_Error": result["Standard_Error"],
        "Change_vs_Baseline_%": (
            100 * (result["Price"] / baseline_result["Price"] - 1)
        )
    })

sensitivity_table = pd.DataFrame(sensitivity_results)
display(sensitivity_table.round(6))
