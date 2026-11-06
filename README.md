# NIG–SPX Option Pricing

A group project on risk-neutral calibration of a Normal Inverse Gaussian (NIG) Lévy model to European S&P 500 index options.

## Repository structure

- `data/raw/`: original market-data workbook (not committed).
- `data/cleaned/`: cleaned calibration sample and data-quality outputs.
- `notebooks/`: focused workflow notebooks for market data, validation, calibration and simulation.
- `src/`: reusable NIG model and calibration implementations.
- `scripts/`: reproducible pipeline and diagnostics.
- `tests/`: automated numerical and mathematical checks.
- `results/`: generated tables and figures.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run automated tests from the repository root:

```bash
python -m pytest -q
```

## Current status

- Market-data cleaning: implemented in `notebooks/01_Market_Data_Cleaning.ipynb`; the documented selection is 90%–110% moneyness followed by positive trading volume.
- Data validation: implemented separately in `notebooks/01_Data_Validation.ipynb`.
- NIG model and risk-neutral characteristic function: implemented in `src/nig_model.py`.
- NIG calibration: implemented in `src/calibration.py`, with the calibration workflow in `notebooks/03_NIG_Calibration.ipynb`.
- Simulation: available in `notebooks/04_NIG_simulation.ipynb`; to be reviewed as a separate next stage.

## Risk-neutral convention

The mean-correcting equivalent martingale measure uses

`log(S_T)=log(S_0)+(r-q)T+X_T-T\kappa(1)`.

Thus,

`phi_T(u)=exp(iu[log(S_0)+(r-q)T]+T[kappa(iu)-iu*kappa(1)])`.

The exponential-moment condition `abs(beta + 1) < alpha` is required. Under this convention, the location parameter `mu` cancels after martingale correction, so European option prices identify the three shape parameters `alpha, beta, delta`, not `mu`.

## Collaboration

Development is performed on `setup/project-structure`. The `main` branch remains the stable project version and is updated only after review and validation.
