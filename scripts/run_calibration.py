"""Run the full-sample NIG calibration and print reproducible diagnostics."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import NIGCalibrator
DATA_PATH = ROOT / "data" / "cleaned" / "calibration_candidates.csv.gz"
OUTPUT_DIR = ROOT / "results" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

quotes = pd.read_csv(DATA_PATH)
print(f"Loaded {len(quotes)} call quotes across {quotes['Expiry'].nunique()} expiries.")

# The first run uses the production quadrature configuration.
calibrator = NIGCalibrator(
    integration_limit=1000.0,
    quadrature_points=512,
    max_nfev=500,
)
starting_points = [
    (8.0, -2.0, 0.25),
    (10.0, -3.0, 0.20),
    (15.0, -5.0, 0.15),
]
records = []
for start in starting_points:
    print(f"Starting calibration from {start}", flush=True)
    result = calibrator.calibrate(quotes, initial_parameters=start)
    row = {
        "initial_alpha": start[0],
        "initial_beta": start[1],
        "initial_delta": start[2],
        "alpha": result.parameters.alpha,
        "beta": result.parameters.beta,
        "delta": result.parameters.delta,
        "mu": result.parameters.mu,
        "MSE": result.mean_squared_error,
        "RMSE": result.root_mean_squared_error,
        "success": result.success,
        "message": result.message,
        "nfev": result.function_evaluations,
        "n_quotes": result.number_of_quotes,
    }
    records.append(row)
    print(json.dumps(row, indent=2), flush=True)

results = pd.DataFrame(records)
results.to_csv(OUTPUT_DIR / "initialisation_sensitivity.csv", index=False)
best = results.loc[results["MSE"].idxmin()]
best_parameters = (float(best["alpha"]), float(best["beta"]), float(best["delta"]))
model_prices = calibrator.model_prices(quotes, *best_parameters)

fit = quotes.copy()
fit["Model_Price"] = model_prices
fit["Pricing_Error"] = fit["Model_Price"] - fit["Market_Price"]
fit["Absolute_Error"] = fit["Pricing_Error"].abs()
fit["Absolute_Percentage_Error"] = (
    fit["Absolute_Error"] / fit["Market_Price"].replace(0.0, np.nan) * 100.0
)
fit.to_csv(OUTPUT_DIR / "full_sample_fit.csv", index=False)

summary = fit.groupby("Expiry").agg(
    Number_of_Quotes=("Pricing_Error", "size"),
    Mean_Error=("Pricing_Error", "mean"),
    MAE=("Absolute_Error", "mean"),
    RMSE=("Pricing_Error", lambda x: float(np.sqrt(np.mean(x**2)))),
    Mean_ARPE=("Absolute_Percentage_Error", "mean"),
)
summary.to_csv(OUTPUT_DIR / "fit_by_expiry.csv")

print("\nBest initialisation:", json.dumps(best.to_dict(), indent=2), flush=True)
print("\nFit diagnostics by expiry:\n", summary.to_string(), flush=True)
print(f"Saved reproducibility outputs to {OUTPUT_DIR}", flush=True)
