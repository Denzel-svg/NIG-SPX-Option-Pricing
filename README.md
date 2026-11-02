# NIG–SPX Option Pricing

A research project on the Normal Inverse Gaussian (NIG) Lévy model, calibration to S&P 500 (SPX) European call options, simulation, and exotic-option pricing.

## Project workflow

- Validate the cleaned SPX option dataset.
- Derive and test the risk-neutral NIG characteristic function using the professor's prescribed convention.
- Implement European option pricing using the course reference methodology.
- Calibrate NIG parameters and a Black–Scholes benchmark.
- Compare fit quality, including ARPE where required.
- Simulate NIG paths and price a lookback option.

## Repository layout

```text
.
├── data/          # raw/ and cleaned/ datasets
├── notebooks/     # explained workflow, diagnostics and results
├── src/           # reusable object-oriented Python implementation
├── tests/         # unit and numerical validation tests
├── results/       # figures and tables
├── docs/          # methodology notes and references
├── requirements.txt
└── README.md
```

## Development principles

1. Reusable model, pricing and calibration logic belongs in `src/`, not in notebooks.
2. Notebooks explain methodology, inspect inputs, run experiments and present results.
3. Every notebook code cell must be preceded by a short English Markdown explanation stating its purpose and what to check.
4. Use descriptive names, type hints, docstrings, useful comments, explicit validation and readable errors.
5. Do not implement a pricing convention until it has been checked against the professor's instructions and cited reference.
6. Keep raw market data, local environments, generated files and credentials out of Git unless explicitly agreed by the team.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the validated long-format calibration dataset under `data/cleaned/`. Validate its schema before any model calculation.

## Collaboration

Work on feature branches and use pull requests. Keep commits focused and descriptive. Do not commit raw data or secrets.
