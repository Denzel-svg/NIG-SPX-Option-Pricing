"""Risk-neutral NIG parameter calibration to European call prices.

Calibration follows the professor's stated mean-square price-error objective.
Under the mean-correcting martingale measure, mu cancels, so only
(alpha, beta, delta) are estimated. A vectorised Gauss-Legendre integration
routine makes repeated objective evaluations practical for the full sample.\nThe production run is reproducible with scripts/run_calibration.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

from src.nig_model import NIGParameters


@dataclass(frozen=True)
class CalibrationResult:
    """Summary of a fitted risk-neutral NIG parameter set."""
    parameters: NIGParameters
    mean_squared_error: float
    root_mean_squared_error: float
    success: bool
    message: str
    function_evaluations: int
    number_of_quotes: int


class NIGCalibrator:
    """Calibrate the identifiable NIG shape parameters to European call prices.

    The objective is the unweighted mean squared difference between market
    mid-prices and model prices, matching the course assignment's MSE
    specification. Integration uses Gauss-Legendre quadrature on [0, U].
    """
    REQUIRED_COLUMNS = (
        "Strike", "Tau", "Spot", "Dividend_Yield",
        "Risk_Free_Rate_CC", "Market_Price",
    )

    def __init__(
        self,
        integration_limit: float = 1000.0,
        quadrature_points: int = 512,
        max_nfev: int = 500,
        mu: float = 0.0,
    ) -> None:
        if not math.isfinite(integration_limit) or integration_limit <= 0:
            raise ValueError("Integration limit must be finite and strictly positive.")
        if quadrature_points < 32:
            raise ValueError("At least 32 quadrature points are required.")
        if max_nfev < 1:
            raise ValueError("max_nfev must be positive.")
        if not math.isfinite(mu):
            raise ValueError("mu must be finite.")
        self.integration_limit = integration_limit
        self.quadrature_points = quadrature_points
        self.max_nfev = max_nfev
        self.mu = mu
        nodes, weights = leggauss(quadrature_points)
        self._frequencies = 0.5 * integration_limit * (nodes + 1.0)
        self._weights = 0.5 * integration_limit * weights

    @staticmethod
    def _decode_parameters(transformed: np.ndarray) -> tuple[float, float, float]:
        """Map unconstrained coordinates into the NIG risk-neutral domain.

        alpha = 0.5 + softplus(a) ensures alpha > 0.5.
        beta is mapped into (-alpha, alpha-1), enforcing both
        |beta| < alpha and |beta+1| < alpha.
        delta = exp(d) ensures delta > 0.
        """
        a, b, d = transformed
        alpha = 0.5 + np.logaddexp(0.0, a)
        if b >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-min(b, 700.0)))
        else:
            exp_b = math.exp(max(b, -700.0))
            sigmoid = exp_b / (1.0 + exp_b)
        beta = -alpha + (2.0 * alpha - 1.0) * sigmoid
        delta = math.exp(float(np.clip(d, -700.0, 700.0)))
        return float(alpha), float(beta), float(delta)

    @staticmethod
    def _encode_parameters(alpha: float, beta: float, delta: float) -> np.ndarray:
        """Map a valid NIG risk-neutral parameter triple to optimizer coordinates."""
        if alpha <= 0.5 or delta <= 0:
            raise ValueError("Initial values require alpha > 0.5 and delta > 0.")
        if not (abs(beta) < alpha and abs(beta + 1.0) < alpha):
            raise ValueError("Initial beta must satisfy both NIG moment constraints.")
        softplus_argument = alpha - 0.5
        a = math.log(math.expm1(softplus_argument)) if softplus_argument < 40 else softplus_argument
        fraction = (beta + alpha) / (2.0 * alpha - 1.0)
        b = math.log(fraction / (1.0 - fraction))
        return np.array([a, b, math.log(delta)], dtype=float)

    def _validate_quotes(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a numeric calibration frame."""
        missing = set(self.REQUIRED_COLUMNS).difference(quotes.columns)
        if missing:
            raise ValueError(f"Calibration data is missing columns: {sorted(missing)}")
        data = quotes.loc[:, list(self.REQUIRED_COLUMNS)].copy()
        for column in self.REQUIRED_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError("Calibration data contains missing or non-finite numeric values.")
        if (data["Strike"] <= 0).any() or (data["Spot"] <= 0).any():
            raise ValueError("Strike and Spot must be strictly positive.")
        if (data["Tau"] <= 0).any():
            raise ValueError("Calibration maturities must be strictly positive.")
        if (data["Market_Price"] < 0).any():
            raise ValueError("Market call prices must be non-negative.")
        if len(data) < 1:
            raise ValueError("At least one option quote is required.")
        return data.reset_index(drop=True)

    def model_prices(
        self, quotes: pd.DataFrame, alpha: float, beta: float, delta: float
    ) -> np.ndarray:
        """Vectorised Fourier prices for a quote frame and one NIG parameter set."""
        if alpha <= 0.5 or delta <= 0:
            raise ValueError("Risk-neutral NIG parameters require alpha > 0.5 and delta > 0.")
        if not (abs(beta) < alpha and abs(beta + 1.0) < alpha):
            raise ValueError("Risk-neutral NIG parameters violate an exponential-moment condition.")
        data = self._validate_quotes(quotes)

        strike = data["Strike"].to_numpy(dtype=float)[:, None]
        maturity = data["Tau"].to_numpy(dtype=float)[:, None]
        spot = data["Spot"].to_numpy(dtype=float)[:, None]
        rate = data["Risk_Free_Rate_CC"].to_numpy(dtype=float)[:, None]
        dividend = data["Dividend_Yield"].to_numpy(dtype=float)[:, None]
        frequency = self._frequencies[None, :]
        weights = self._weights[None, :]

        base_root = math.sqrt(alpha**2 - beta**2)
        positive_moment_root = math.sqrt(alpha**2 - (beta + 1.0)**2)

        def cumulant_at(z: np.ndarray) -> np.ndarray:
            return delta * (
                base_root - np.sqrt(alpha**2 - (beta + z)**2 + 0j)
            )

        log_forward = np.log(spot) + (rate - dividend) * maturity
        log_strike = np.log(strike)

        # phi_T(u-i): the stock-numeraire probability integral Pi1.
        z_stock = 1.0 + 1j * frequency
        k_stock = cumulant_at(z_stock) - (1.0 + 1j * frequency) * delta * (
            base_root - positive_moment_root
        )
        phi_stock = np.exp(
            (1.0 + 1j * frequency) * log_forward + maturity * k_stock
        )
        phi_minus_i = spot * np.exp((rate - dividend) * maturity)
        integrand_1 = (
            np.exp(-1j * frequency * log_strike)
            * phi_stock / phi_minus_i
            / (1j * frequency)
        ).real

        # phi_T(u): the cash-numeraire probability integral Pi2.
        z_cash = 1j * frequency
        k_cash = cumulant_at(z_cash) - 1j * frequency * delta * (
            base_root - positive_moment_root
        )
        phi_cash = np.exp(
            1j * frequency * log_forward + maturity * k_cash
        )
        integrand_2 = (
            np.exp(-1j * frequency * log_strike)
            * phi_cash / (1j * frequency)
        ).real

        pi1 = 0.5 + np.sum(weights * integrand_1, axis=1) / math.pi
        pi2 = 0.5 + np.sum(weights * integrand_2, axis=1) / math.pi
        prices = (
            data["Spot"].to_numpy(dtype=float)
            * np.exp(-data["Dividend_Yield"].to_numpy(dtype=float)
                    * data["Tau"].to_numpy(dtype=float)) * pi1
            - data["Strike"].to_numpy(dtype=float)
            * np.exp(-data["Risk_Free_Rate_CC"].to_numpy(dtype=float)
                    * data["Tau"].to_numpy(dtype=float)) * pi2
        )
        if not np.isfinite(prices).all():
            raise ArithmeticError("Fourier pricing returned non-finite model prices.")
        return np.maximum(prices, 0.0)

    def calibrate(
        self,
        quotes: pd.DataFrame,
        initial_parameters: tuple[float, float, float] = (8.0, -2.0, 0.25),
    ) -> CalibrationResult:
        """Estimate alpha, beta and delta by nonlinear least squares on prices."""
        data = self._validate_quotes(quotes)
        market_prices = data["Market_Price"].to_numpy(dtype=float)
        initial = self._encode_parameters(*initial_parameters)

        def residuals(transformed: np.ndarray) -> np.ndarray:
            alpha, beta, delta = self._decode_parameters(transformed)
            try:
                fitted_prices = self.model_prices(data, alpha, beta, delta)
            except (ValueError, ArithmeticError, FloatingPointError):
                return np.full(len(data), 1e8, dtype=float)
            return fitted_prices - market_prices

        fit = least_squares(
            residuals,
            x0=initial,
            method="trf",
            max_nfev=self.max_nfev,
            x_scale="jac",
        )
        alpha, beta, delta = self._decode_parameters(fit.x)
        final_residuals = residuals(fit.x)
        mse = float(np.mean(final_residuals**2))
        parameters = NIGParameters(
            alpha=alpha, beta=beta, delta=delta, mu=self.mu
        )
        return CalibrationResult(
            parameters=parameters,
            mean_squared_error=mse,
            root_mean_squared_error=math.sqrt(mse),
            success=bool(fit.success),
            message=str(fit.message),
            function_evaluations=int(fit.nfev),
            number_of_quotes=len(data),
        )
