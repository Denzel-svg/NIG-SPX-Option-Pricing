"""Compare the calibrated NIG fit with a single-volatility Black-Scholes fit."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks import BlackScholesBenchmark

RESULTS_DIR = ROOT / "results" / "calibration"
FIT_PATH = RESULTS_DIR / "full_sample_fit.csv"
if not FIT_PATH.exists():
    raise FileNotFoundError(
        f"{FIT_PATH} not found. Run scripts/run_calibration.py first."
    )

fit = pd.read_csv(FIT_PATH)
benchmark = BlackScholesBenchmark()
result = benchmark.calibrate(fit)
fit["BS_Model_Price"] = benchmark.model_prices(fit, result.volatility)
fit["BS_Pricing_Error"] = fit["BS_Model_Price"] - fit["Market_Price"]
fit["BS_Absolute_Error"] = fit["BS_Pricing_Error"].abs()
fit["BS_Absolute_Percentage_Error"] = (
    fit["BS_Absolute_Error"] / fit["Market_Price"] * 100.0
)
fit.to_csv(RESULTS_DIR / "nig_vs_black_scholes_fit.csv", index=False)

summary = fit.groupby("Expiry").agg(
    Number_of_Quotes=("Market_Price", "size"),
    NIG_RMSE=("Pricing_Error", lambda values: float(np.sqrt(np.mean(values**2)))),
    NIG_ARPE=("Absolute_Percentage_Error", "mean"),
    BS_RMSE=("BS_Pricing_Error", lambda values: float(np.sqrt(np.mean(values**2)))),
    BS_ARPE=("BS_Absolute_Percentage_Error", "mean"),
)
summary.to_csv(RESULTS_DIR / "nig_vs_black_scholes_by_expiry.csv")

comparison = pd.DataFrame([{
    "Number_of_Quotes": result.number_of_quotes,
    "BS_Implied_Common_Volatility": result.volatility,
    "BS_MSE": result.mean_squared_error,
    "BS_RMSE": result.root_mean_squared_error,
    "BS_ARPE_Percent": result.average_relative_percentage_error,
    "NIG_MSE": float(np.mean(fit["Pricing_Error"] ** 2)),
    "NIG_RMSE": float(np.sqrt(np.mean(fit["Pricing_Error"] ** 2))),
    "NIG_ARPE_Percent": float(fit["Absolute_Percentage_Error"].mean()),
}])
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("Black-Scholes benchmark:", comparison.to_string(index=False), flush=True)
print("Comparison by expiry:\n", summary.to_string(), flush=True)
