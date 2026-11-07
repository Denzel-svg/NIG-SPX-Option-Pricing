# Methodology and conventions

Short summary of the choices made in the code. The derivations are in the report.

## Data

- Bloomberg closing quotes (OMON) for SPX calls on 23 October 2026, spot S0 = 7,623.68, five expiries: 20 Nov 2026, 18 Dec 2026, 15 Jan 2027, 16 Apr 2027 and 17 Sep 2027.
- Selection: strikes with 0.9 ≤ K/S0 ≤ 1.1, then strictly positive traded volume (1,333 → 916 → 477 quotes). There is no spread filter; spreads are only used to describe the data.
- Market price: midpoint of bid and ask. The last traded price is not used.
- Expiries are written as ISO dates (YYYY-MM-DD) so that alphabetical order is also chronological order.

## Market inputs

- Maturity T: number of calendar days divided by 365.
- Risk-free rate: the OIS rates are simple ACT/360 rates. Each one is turned into a discount factor D = 1/(1 + R·d/360) and then into a continuous rate r = −ln(D)/T.
- Dividend yield: the quoted yield is 1.06%, and we use the continuous equivalent q = ln(1.0106) ≈ 1.0544% for every maturity.

## Calibration

- Mean-correcting risk-neutral measure. The parameter μ cancels out, so we fix μ = 0 and estimate (α, β, δ).
- Admissible region: α > 0, δ > 0, |β| < α and |β + 1| < α.
- Call price: C = S0·e^(−qT)·Π1 − K·e^(−rT)·Π2, computed by Gauss–Legendre quadrature (upper limit 1,000, 512 nodes).
- Objective: unweighted mean squared price error, minimised with `scipy.optimize.least_squares` (method `trf`, `x_scale="jac"`, `max_nfev=500`) on transformed parameters.
- Starting points: (8, −2, 0.25), (10, −3, 0.20) and (15, −5, 0.15).
- Benchmark: one common Black–Scholes volatility fitted on the same quotes with the same objective.
