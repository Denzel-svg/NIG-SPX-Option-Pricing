# NIG–SPX Option Pricing

A group project on risk-neutral calibration of a Normal Inverse Gaussian (NIG) Lévy model to European S&P 500 index options, followed by a Monte Carlo lookback-option application.

## Repository structure

- `data/cleaned/`: cleaned calibration candidates and data-quality outputs.
- `notebooks/`: step-by-step explanations, diagnostics and presentation of results.
- `src/`: reusable object-oriented model, pricing, calibration and simulation code.
- `tests/`: automated numerical and mathematical checks.
- `results/`: tables and figures produced by the analysis.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Open the repository in Jupyter and run notebooks in numerical order. The notebooks explain and validate the methods; reusable implementation belongs in `src/`.

Run automated tests from the repository root:

```bash
python -m pytest -q
```

## Current implementation status

- Cleaned calibration sample: prepared, with 477 positive-volume candidates across five expiries.
- NIG Lévy-process primitives: implemented.
- Mean-correcting risk-neutral NIG characteristic function: implemented.
- European call Fourier pricing using the Bakshi–Madan / Schoutens representation: implemented and checked against the Black–Scholes analytical formula.
- Vectorised risk-neutral NIG calibration engine: implemented; numerical validation and final fit diagnostics are in progress.
- Monte Carlo lookback pricing: work in progress.

## Risk-neutral convention

The course material uses a mean-correcting equivalent martingale measure. With continuous dividend yield (q), the implementation uses

`log(S_T)=log(S_0)+(r-q)T+X_T-T\kappa(1)`.

Thus the risk-neutral log-price characteristic function is

`phi_T(u)=exp(iu[log(S_0)+(r-q)T]+T[kappa(iu)-iu*kappa(1)])`.

The exponential moment condition `abs(beta + 1) < alpha` is required. Under this convention the linear location parameter `mu` cancels after martingale correction, so European option prices identify the three shape parameters `alpha, beta, delta`, not `mu`.

## Collaboration

Development is performed on `setup/project-structure`. The `main` branch remains the stable project version and is updated only after review and validation.
