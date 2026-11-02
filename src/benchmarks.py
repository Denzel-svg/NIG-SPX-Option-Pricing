"""Black-Scholes benchmark and relative pricing-error metrics.

The volatility is estimated as one common parameter by minimising the same
unweighted mean squared price error used in the NIG calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesResult:
    volatility: float
    mean_squared_error: float
    root_mean_squared_error: float
    average_relative_percentage_error: float
    success: bool
    number_of_quotes: int


def average_relative_percentage_error(
    market_prices: np.ndarray, model_prices: np.ndarray
) -> float:
    """Return mean(|model-market| / market) * 100; reject zero market prices."""
    market = np.asarray(market_prices, dtype=float)
    model = np.asarray(model_prices, dtype=float)
    if market.shape != model.shape or market.ndim != 1 or market.size == 0:
        raise ValueError("Market and model prices must be non-empty 1D arrays of equal shape.")
    if not np.isfinite(market).all() or not np.isfinite(model).all():
        raise ValueError("Prices must contain only finite values.")
    if (market <= 0).any():
        raise ValueError("ARPE is undefined for zero or negative market prices.")
    return float(np.mean(np.abs(model - market) / market) * 100.0)


class BlackScholesBenchmark:
    """Fit a single Black-Scholes volatility to a panel of European calls."""

    REQUIRED_COLUMNS = (
        "Strike", "Tau", "Spot", "Dividend_Yield",
        "Risk_Free_Rate_CC", "Market_Price",
    )

    def __init__(self, volatility_bounds: tuple[float, float] = (0.001, 3.0)) -> None:
        lower, upper = volatility_bounds
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("Volatility bounds must be finite and satisfy 0 < lower < upper.")
        self.volatility_bounds = volatility_bounds

    def _validate_quotes(self, quotes: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.REQUIRED_COLUMNS).difference(quotes.columns)
        if missing:
            raise ValueError(f"Benchmark data is missing columns: {sorted(missing)}")
        data = quotes.loc[:, list(self.REQUIRED_COLUMNS)].copy()
        for column in self.REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError("Benchmark data contains missing or non-finite numeric values.")
        if (data[["Strike", "Spot", "Tau"]] <= 0).any().any():
            raise ValueError("Strike, Spot and Tau must be strictly positive.")
        if (data["Market_Price"] <= 0).any():
            raise ValueError("Market prices must be strictly positive for ARPE comparison.")
        return data.reset_index(drop=True)

    @staticmethod
    def model_prices(quotes: pd.DataFrame, volatility: float) -> np.ndarray:
        """Price European calls using continuous rates and dividend yield."""
        if not math.isfinite(volatility) or volatility <= 0:
            raise ValueError("Volatility must be finite and strictly positive.")
        strike = quotes["Strike"].to_numpy(dtype=float)
        maturity = quotes["Tau"].to_numpy(dtype=float)
        spot = quotes["Spot"].to_numpy(dtype=float)
        rate = quotes["Risk_Free_Rate_CC"].to_numpy(dtype=float)
        dividend = quotes["Dividend_Yield"].to_numpy(dtype=float)
        sigma_root_t = volatility * np.sqrt(maturity)
        d1 = (
            np.log(spot / strike)
            + (rate - dividend + 0.5 * volatility**2) * maturity
        ) / sigma_root_t
        d2 = d1 - sigma_root_t
        return (
            spot * np.exp(-dividend * maturity) * norm.cdf(d1)
            - strike * np.exp(-rate * maturity) * norm.cdf(d2)
        )

    def calibrate(self, quotes: pd.DataFrame) -> BlackScholesResult:
        """Estimate one common volatility by minimising unweighted price MSE."""
        data = self._validate_quotes(quotes)
        market = data["Market_Price"].to_numpy(dtype=float)

        def objective(volatility: float) -> float:
            residuals = self.model_prices(data, volatility) - market
            return float(np.mean(residuals**2))

        fit = minimize_scalar(
            objective,
            bounds=self.volatility_bounds,
            method="bounded",
            options={"xatol": 1e-12},
        )
        fitted = self.model_prices(data, float(fit.x))
        mse = float(np.mean((fitted - market) ** 2))
        return BlackScholesResult(
            volatility=float(fit.x),
            mean_squared_error=mse,
            root_mean_squared_error=math.sqrt(mse),
            average_relative_percentage_error=average_relative_percentage_error(market, fitted),
            success=bool(fit.success),
            number_of_quotes=len(data),
        )
