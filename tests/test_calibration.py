"""Tests for the NIG risk-neutral calibration engine."""
import cmath
import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from src.calibration import NIGCalibrator


def reference_call_price(spot, strike, rate, dividend, maturity, alpha, beta, delta):
    """Scalar reference price: probability representation with adaptive quadrature.

    C = S*exp(-qT)*Pi1 - K*exp(-rT)*Pi2, where Pi1 and Pi2 follow from
    Gil-Pelaez inversion of the mean-corrected risk-neutral characteristic function.
    """
    def kappa(z):
        return delta * (
            math.sqrt(alpha**2 - beta**2) - cmath.sqrt(alpha**2 - (beta + z) ** 2)
        )

    def phi(u):
        return cmath.exp(
            1j * u * (math.log(spot) + (rate - dividend) * maturity)
            + maturity * (kappa(1j * u) - 1j * u * kappa(1.0))
        )

    log_strike = math.log(strike)
    phi_minus_i = phi(-1j)

    def integrand_1(u):
        return (cmath.exp(-1j * u * log_strike) * phi(u - 1j) / (1j * u * phi_minus_i)).real

    def integrand_2(u):
        return (cmath.exp(-1j * u * log_strike) * phi(u) / (1j * u)).real

    pi1 = 0.5 + quad(integrand_1, 1e-12, 1000.0, limit=2000, epsabs=1e-12, epsrel=1e-12)[0] / math.pi
    pi2 = 0.5 + quad(integrand_2, 1e-12, 1000.0, limit=2000, epsabs=1e-12, epsrel=1e-12)[0] / math.pi
    return (
        spot * math.exp(-dividend * maturity) * pi1
        - strike * math.exp(-rate * maturity) * pi2
    )


def small_quote_frame():
    return pd.DataFrame({
        "Strike": [95.0, 100.0, 105.0],
        "Tau": [0.5, 0.5, 0.5],
        "Spot": [100.0, 100.0, 100.0],
        "Dividend_Yield": [0.01, 0.01, 0.01],
        "Risk_Free_Rate_CC": [0.03, 0.03, 0.03],
        "Market_Price": [8.0, 5.0, 2.5],
    })


def test_parameter_transform_enforces_both_exponential_moment_constraints():
    calibrator = NIGCalibrator()
    for coordinates in ([0.0, -5.0, -1.0], [1.0, 0.0, -2.0], [2.0, 5.0, 0.0]):
        alpha, beta, delta = calibrator._decode_parameters(np.asarray(coordinates))
        assert alpha > 0.5
        assert delta > 0.0
        assert abs(beta) < alpha
        assert abs(beta + 1.0) < alpha


def test_vectorised_fourier_price_matches_quad_pricer():
    """The fast calibration integrator should agree with the adaptive-quadrature reference."""
    calibrator = NIGCalibrator(integration_limit=1000.0, quadrature_points=512)
    quotes = small_quote_frame()
    alpha, beta, delta = 8.0, -2.0, 0.25
    vector_prices = calibrator.model_prices(quotes, alpha, beta, delta)

    reference_prices = [
        reference_call_price(
            row.Spot, row.Strike, row.Risk_Free_Rate_CC, row.Dividend_Yield,
            row.Tau, alpha, beta, delta,
        )
        for row in quotes.itertuples(index=False)
    ]
    assert vector_prices == pytest.approx(reference_prices, abs=2e-4)


def test_calibrator_rejects_missing_required_columns():
    calibrator = NIGCalibrator()
    with pytest.raises(ValueError, match="missing columns"):
        calibrator.calibrate(pd.DataFrame({"Strike": [100.0]}))
