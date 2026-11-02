"""Tests for the NIG risk-neutral calibration engine."""
import numpy as np
import pandas as pd
import pytest

from src.nig_model import NIGParameters
from src.option_pricing import EuropeanCallPricer, MarketInputs, NIGRiskNeutralModel
from src.calibration import NIGCalibrator


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
    """The fast calibration integrator should agree with the reference quad engine."""
    calibrator = NIGCalibrator(integration_limit=180.0, quadrature_points=512)
    quotes = small_quote_frame()
    alpha, beta, delta = 8.0, -2.0, 0.25
    vector_prices = calibrator.model_prices(quotes, alpha, beta, delta)

    model = NIGRiskNeutralModel(
        NIGParameters(alpha=alpha, beta=beta, delta=delta, mu=0.0)
    )
    reference_pricer = EuropeanCallPricer(epsabs=1e-9, epsrel=1e-9)
    reference_prices = []
    for row in quotes.itertuples(index=False):
        market = MarketInputs(
            spot=row.Spot,
            risk_free_rate=row.Risk_Free_Rate_CC,
            dividend_yield=row.Dividend_Yield,
            maturity=row.Tau,
        )
        reference_prices.append(reference_pricer.price_nig(row.Strike, market, model))
    assert vector_prices == pytest.approx(reference_prices, abs=2e-4)


def test_calibrator_rejects_missing_required_columns():
    calibrator = NIGCalibrator()
    with pytest.raises(ValueError, match="missing columns"):
        calibrator.calibrate(pd.DataFrame({"Strike": [100.0]}))
