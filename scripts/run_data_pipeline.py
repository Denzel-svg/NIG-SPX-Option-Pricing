"""Execute the market-data pipeline and full NIG calibration reproducibly.

The canonical Bloomberg workbook must be available at
data/raw/SPX_options_data.xlsx. The script executes the two data notebooks
in order, validates the exported sample, then runs calibration diagnostics.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "raw" / "SPX_options_data.xlsx"
DATASET = ROOT / "data" / "cleaned" / "calibration_candidates.csv.gz"
EXECUTED_DIR = ROOT / "results" / "executed_notebooks"
NOTEBOOK_DIR = ROOT / "notebooks"

EXPECTED_ROWS = 477
EXPECTED_EXPIRIES = 5


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def execute_notebook(name: str) -> None:
    source = NOTEBOOK_DIR / name
    if not source.is_file():
        raise FileNotFoundError(f"Required notebook not found: {source}")
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    output = EXECUTED_DIR / name
    run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            "--output",
            str(output),
            str(source),
        ],
        cwd=NOTEBOOK_DIR,
    )


def validate_fresh_export() -> None:
    if not DATASET.is_file():
        raise FileNotFoundError(
            "The cleaning notebook did not create data/cleaned/"
            "calibration_candidates.csv.gz."
        )

    data = pd.read_csv(DATASET)
    required = {
        "Expiry",
        "Days_To_Maturity",
        "Strike",
        "Market_Price",
        "Spot",
        "Tau",
        "Risk_Free_Rate_CC",
        "Dividend_Yield",
        "Trading_Volume",
        "Implied_Volatility",
        "Relative_Spread",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Fresh export is missing columns: {sorted(missing)}")

    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} positive-volume quotes, found {len(data)}."
        )
    if data["Expiry"].nunique() != EXPECTED_EXPIRIES:
        raise ValueError(
            f"Expected {EXPECTED_EXPIRIES} expiries, found {data['Expiry'].nunique()}."
        )
    if not (data["Trading_Volume"] > 0).all():
        raise ValueError("The exported sample contains non-positive trading volume.")
    if data[list(required - {"Expiry"})].isna().any().any():
        raise ValueError("The exported calibration dataset contains missing values.")

    print(
        f"Fresh export verified: {len(data)} quotes, "
        f"{data['Expiry'].nunique()} expiries.",
        flush=True,
    )
    print(
        data.groupby("Expiry").size().to_string(),
        flush=True,
    )


def main() -> None:
    if not WORKBOOK.is_file():
        raise FileNotFoundError(
            "Bloomberg workbook not found. Put the file "
            f"at: {WORKBOOK.relative_to(ROOT)}. "
            "The pipeline deliberately refuses to reuse a stale cleaned CSV."
        )

    print(f"Using canonical workbook: {WORKBOOK}", flush=True)
    execute_notebook("01_Market_Data_Cleaning.ipynb")
    validate_fresh_export()
    execute_notebook("01_Data_Validation.ipynb")

    # Reuse the same calibration and diagnostics entry points as the established
    # full-sample workflow, but only after a fresh export has passed validation.
    for script in (
        "run_calibration.py",
        "run_benchmark.py",
        "run_numerical_diagnostics.py",
        "plot_calibration_results.py",
    ):
        run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)


if __name__ == "__main__":
    main()
