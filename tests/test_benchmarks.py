import numpy as np
import pandas as pd
import pytest

from src.benchmarks import (
    BlackScholesBenchmark,
    average_relative_percentage_error,
)


def _synthetic_quotes():
    return pd.DataFrame({
        "Strike": [95.0, 100.0, 105.0, 110.0],
        "Tau": [0.5, 0.5, 1.0, 1.0],
        "Spot": [100.0] * 4,
        "Dividend_Yield": [0.01] * 4,
        "Risk_Free_Rate_CC": [0.03] * 4,
        "Market_Price": [0.0] * 4,
    })


def test_black_scholes_benchmark_recovers_common_volatility():
    quotes = _synthetic_quotes()
    quotes["Market_Price"] = BlackScholesBenchmark.model_prices(quotes, 0.22)
    result = BlackScholesBenchmark().calibrate(quotes)
    assert result.success
    assert result.volatility == pytest.approx(0.22, abs=1e-5)
    assert result.mean_squared_error < 1e-8


def test_arpe_definition():
    assert average_relative_percentage_error(
        np.array([100.0, 200.0]), np.array([110.0, 180.0])
    ) == pytest.approx(10.0)


def test_arpe_rejects_zero_market_prices():
    with pytest.raises(ValueError, match="strictly positive"):
        BlackScholesBenchmark().calibrate(
            _synthetic_quotes().assign(Market_Price=[0.0, 1.0, 2.0, 3.0])
        )
