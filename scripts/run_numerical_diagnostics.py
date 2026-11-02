"""Residual-structure and Fourier-quadrature diagnostics for the fitted NIG model."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import NIGCalibrator

RESULTS = ROOT / "results" / "calibration"
FIT = pd.read_csv(RESULTS / "full_sample_fit.csv")
STARTS = pd.read_csv(RESULTS / "initialisation_sensitivity.csv")
best = STARTS.loc[STARTS["MSE"].idxmin()]
parameters = (float(best["alpha"]), float(best["beta"]), float(best["delta"]))

# Residuals by maturity and moneyness reveal systematic smile/surface patterns.
FIT["Moneyness"] = FIT["Strike"] / FIT["Spot"]
FIT["Moneyness_Bucket"] = pd.cut(
    FIT["Moneyness"],
    bins=[0.0, 0.95, 0.98, 1.00, 1.02, 1.05, np.inf],
    labels=[
        "Deep ITM <95%", "ITM 95-98%", "Near ATM 98-100%",
        "Near OTM 100-102%", "OTM 102-105%", "Far OTM >105%",
    ],
    right=False,
)
FIT["Signed_ARPE"] = FIT["Pricing_Error"] / FIT["Market_Price"] * 100.0
by_moneyness = FIT.groupby("Moneyness_Bucket", observed=True).agg(
    Number_of_Quotes=("Pricing_Error", "size"),
    Mean_Moneyness=("Moneyness", "mean"),
    Mean_Error=("Pricing_Error", "mean"),
    MAE=("Absolute_Error", "mean"),
    RMSE=("Pricing_Error", lambda values: float(np.sqrt(np.mean(values**2)))),
    Mean_ARPE=("Absolute_Percentage_Error", "mean"),
    Mean_Signed_RPE=("Signed_ARPE", "mean"),
)
by_moneyness.to_csv(RESULTS / "residuals_by_moneyness.csv")

by_expiry = FIT.groupby("Expiry").agg(
    Number_of_Quotes=("Pricing_Error", "size"),
    Mean_Moneyness=("Moneyness", "mean"),
    Mean_Error=("Pricing_Error", "mean"),
    MAE=("Absolute_Error", "mean"),
    RMSE=("Pricing_Error", lambda values: float(np.sqrt(np.mean(values**2)))),
    Mean_ARPE=("Absolute_Percentage_Error", "mean"),
    Mean_Signed_RPE=("Signed_ARPE", "mean"),
)
by_expiry.to_csv(RESULTS / "residuals_by_expiry.csv")

worst = FIT.nlargest(25, "Absolute_Error")[
    ["Expiry", "Strike", "Spot", "Moneyness", "Market_Price",
     "Model_Price", "Pricing_Error", "Absolute_Error",
     "Absolute_Percentage_Error"]
]
worst.to_csv(RESULTS / "largest_nig_residuals.csv", index=False)

# Compare price outputs at different integration limits and quadrature orders.
# This holds the fitted parameters fixed: it tests numerical pricing stability,
# not the sensitivity of the optimizer's fitted parameters.
configurations = [
    (500.0, 256), (500.0, 512), (1000.0, 256),
    (1000.0, 512), (1000.0, 1024), (1500.0, 1024),
    (2000.0, 1024), (2000.0, 2048),
]
reference = NIGCalibrator(
    integration_limit=2000.0, quadrature_points=2048
).model_prices(FIT, *parameters)
records = []
for limit, points in configurations:
    calculator = NIGCalibrator(
        integration_limit=limit, quadrature_points=points
    )
    prices = calculator.model_prices(FIT, *parameters)
    differences = prices - reference
    records.append({
        "Integration_Limit": limit,
        "Quadrature_Points": points,
        "Mean_Absolute_Difference_vs_Reference": float(np.mean(np.abs(differences))),
        "RMSE_vs_Reference": float(np.sqrt(np.mean(differences**2))),
        "Maximum_Absolute_Difference_vs_Reference": float(np.max(np.abs(differences))),
        "Reference_Limit": 2000.0,
        "Reference_Quadrature_Points": 2048,
    })
pd.DataFrame(records).to_csv(RESULTS / "fourier_quadrature_stability.csv", index=False)

print("Residuals by moneyness:\\n", by_moneyness.to_string(), flush=True)
print("\\nResiduals by expiry:\\n", by_expiry.to_string(), flush=True)
print("\\nLargest absolute residuals:\\n", worst.head(10).to_string(index=False), flush=True)
print("\\nFourier quadrature stability:\\n", pd.DataFrame(records).to_string(index=False), flush=True)
