"""Tests for risk-neutral NIG dynamics and Fourier call pricing."""
import cmath
import math

import pytest
from scipy.stats import norm

from src.nig_model import NIGParameters
from src.option_pricing import EuropeanCallPricer, MarketInputs, NIGRiskNeutralModel


def black_scholes_characteristic_function(spot, rate, dividend, maturity, volatility):
    """Return the risk-neutral characteristic function of log(S_T) in Black-Scholes."""
    def characteristic_function(frequency):
        drift = math.log(spot) + (rate - dividend - 0.5 * volatility**2) * maturity
        exponent = 1j * frequency * drift - 0.5 * volatility**2 * frequency**2 * maturity
        return cmath.exp(exponent)
    return characteristic_function


def black_scholes_call(spot, strike, rate, dividend, maturity, volatility):
    """Closed-form Black-Scholes European call price with continuous dividends."""
    root_time = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    return (
        spot * math.exp(-dividend * maturity) * norm.cdf(d1)
        - strike * math.exp(-rate * maturity) * norm.cdf(d2)
    )


def test_nig_risk_neutral_characteristic_function_has_correct_first_moment():
    """The mean-corrected model must satisfy phi(-i)=S0*exp((r-q)T)."""
    market = MarketInputs(5000.0, 0.04, 0.015, 1.5)
    model = NIGRiskNeutralModel(
        NIGParameters(alpha=8.0, beta=-2.0, delta=0.25, mu=0.1)
    )
    first_moment = model.log_price_characteristic_function(-1j, market)
    expected = market.spot * math.exp(
        (market.risk_free_rate - market.dividend_yield) * market.maturity
    )
    assert first_moment.real == pytest.approx(expected, rel=1e-10)
    assert first_moment.imag == pytest.approx(0.0, abs=1e-10)


def test_risk_neutral_martingale_correction_removes_location_parameter():
    """European option CF is invariant to mu under this mean-correcting convention."""
    market = MarketInputs(5000.0, 0.04, 0.015, 1.0)
    first = NIGRiskNeutralModel(
        NIGParameters(alpha=8.0, beta=-2.0, delta=0.25, mu=-0.4)
    )
    second = NIGRiskNeutralModel(
        NIGParameters(alpha=8.0, beta=-2.0, delta=0.25, mu=0.7)
    )
    assert first.log_price_characteristic_function(1.2, market) == pytest.approx(
        second.log_price_characteristic_function(1.2, market)
    )


def test_nig_model_rejects_missing_stock_first_moment():
    """The exponential moment at one must exist for risk-neutral stock pricing."""
    with pytest.raises(ValueError, match=r"abs\(beta \+ 1\)"):
        NIGRiskNeutralModel(
            NIGParameters(alpha=2.0, beta=1.5, delta=0.2, mu=0.0)
        )


def test_fourier_pricer_matches_black_scholes_closed_form():
    """Generic Fourier pricing must reproduce the analytical BS call price."""
    spot, strike = 100.0, 105.0
    rate, dividend, maturity, volatility = 0.03, 0.01, 0.8, 0.22
    market = MarketInputs(spot, rate, dividend, maturity)
    cf = black_scholes_characteristic_function(
        spot, rate, dividend, maturity, volatility
    )
    pricer = EuropeanCallPricer(epsabs=1e-9, epsrel=1e-9)
    fourier_price = pricer.price(strike, market, cf)
    analytical_price = black_scholes_call(
        spot, strike, rate, dividend, maturity, volatility
    )
    assert fourier_price == pytest.approx(analytical_price, abs=2e-5)


def test_zero_maturity_call_price_is_intrinsic_value():
    """At expiry the call price equals its intrinsic value."""
    market = MarketInputs(100.0, 0.03, 0.01, 0.0)
    pricer = EuropeanCallPricer()
    assert pricer.price(95.0, market, lambda frequency: 1.0) == pytest.approx(5.0)
