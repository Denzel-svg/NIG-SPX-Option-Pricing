"""Integrity checks for the committed cleaned SPX calibration dataset."""
from pathlib import Path
import gzip

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "cleaned" / "calibration_candidates.csv.gz"


def test_cleaned_calibration_dataset_is_a_valid_gzip_csv():
    assert DATA_FILE.exists(), f"Missing cleaned dataset: {DATA_FILE}"
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as compressed_file:
        data = pd.read_csv(compressed_file)

    required_columns = {
        "Strike", "Tau", "Spot", "Dividend_Yield",
        "Risk_Free_Rate_CC", "Market_Price",
    }
    assert required_columns.issubset(data.columns)
    assert len(data) == 477
    assert data["Expiry"].nunique() == 5
    assert (data["Strike"] > 0).all()
    assert (data["Tau"] > 0).all()
    assert (data["Market_Price"] >= 0).all()
