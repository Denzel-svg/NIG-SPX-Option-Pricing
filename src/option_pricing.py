"""Risk-neutral NIG dynamics and Fourier pricing of European call options.

Mean-correcting martingale measure:
    log(S_T) = log(S_0) + (r-q)T + X_T - T*kappa(1)
    phi_T(u) = exp(i*u*(log(S_0)+(r-q)T) + T*(kappa(i*u)-i*u*kappa(1))).

The location parameter mu cancels under this convention. European calls are
priced with the Bakshi-Madan / Schoutens probability integrals, extended to
continuous dividend yield q.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable

from scipy.integrate import quad

from src.nig_model import NIGLevyProcess, NIGParameters


@dataclass(frozen=True)
class MarketInputs:
    """Market inputs required to price one European option."""
    spot: float
    risk_free_rate: float
    dividend_yield: float
    maturity: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"Market input '{name}' must be finite.")
        if self.spot <= 0:
            raise ValueError("Spot must be strictly positive.")
        if self.maturity < 0:
            raise ValueError("Maturity must be non-negative.")


class NIGRiskNeutralModel:
    """NIG log-return model under the mean-correcting risk-neutral measure.

    The location parameter mu cancels after martingale correction. It is
    therefore not identifiable from European option prices under this
    particular risk-neutral convention.
    """
    def __init__(self, parameters: NIGParameters) -> None:
        self.process = NIGLevyProcess(parameters)
        self.parameters = parameters
        if abs(parameters.beta + 1.0) >= parameters.alpha:
            raise ValueError(
                "Risk-neutral stock pricing requires the NIG moment condition "
                "abs(beta + 1) < alpha."
            )
        self._martingale_cumulant = self.process.cumulant_exponent(1.0).real

    def log_price_characteristic_function(
        self, frequency: complex, market: MarketInputs
    ) -> complex:
        """Return E^Q[exp(i*u*log(S_T))], including complex Fourier arguments."""
        if not (
            math.isfinite(complex(frequency).real)
            and math.isfinite(complex(frequency).imag)
        ):
            raise ValueError("Frequency must be finite.")
        time = market.maturity
        log_forward = math.log(market.spot) + (
            market.risk_free_rate - market.dividend_yield
        ) * time
        exponent = (
            1j * frequency * log_forward
            + time * (
                self.process.cumulant_exponent(1j * frequency)
                - 1j * frequency * self._martingale_cumulant
            )
        )
        return cmath.exp(exponent)

    def discounted_stock_expectation(self, market: MarketInputs) -> float:
        """Return exp(-rT) E^Q[S_T], the dividend-adjusted martingale diagnostic."""
        return market.spot * math.exp(-market.dividend_yield * market.maturity)


class EuropeanCallPricer:
    """Price European calls from the log-price characteristic function.

    C = S0*exp(-qT)*Pi1 - K*exp(-rT)*Pi2.
    """
    def __init__(
        self,
        integration_limit: float = math.inf,
        epsabs: float = 1e-8,
        epsrel: float = 1e-8,
        limit: int = 250,
    ) -> None:
        if integration_limit <= 0:
            raise ValueError("Integration limit must be strictly positive.")
        if epsabs <= 0 or epsrel <= 0:
            raise ValueError("Integration tolerances must be strictly positive.")
        if limit < 1:
            raise ValueError("Quadrature subdivision limit must be positive.")
        self.integration_limit = integration_limit
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit

    def probabilities(
        self,
        strike: float,
        market: MarketInputs,
        characteristic_function: Callable[[complex], complex],
    ) -> tuple[float, float]:
        """Calculate Pi1 and Pi2 using the course's Fourier representation."""
        if not math.isfinite(strike) or strike <= 0:
            raise ValueError("Strike must be finite and strictly positive.")
        if market.maturity == 0:
            raise ValueError("Probability integrals are not needed at zero maturity.")

        log_strike = math.log(strike)
        phi_minus_i = characteristic_function(-1j)
        if abs(phi_minus_i) == 0 or not (
            math.isfinite(phi_minus_i.real) and math.isfinite(phi_minus_i.imag)
        ):
            raise ValueError("Characteristic function returned an invalid first moment.")

        # The real parts have finite limits at u=0; a tiny lower cutoff avoids 0/0.
        lower_bound = 1e-8

        def pi1_integrand(frequency: float) -> float:
            numerator = (
                cmath.exp(-1j * frequency * log_strike)
                * characteristic_function(frequency - 1j)
                / phi_minus_i
            )
            return (numerator / (1j * frequency)).real

        def pi2_integrand(frequency: float) -> float:
            numerator = (
                cmath.exp(-1j * frequency * log_strike)
                * characteristic_function(frequency)
            )
            return (numerator / (1j * frequency)).real

        pi1_integral, _ = quad(
            pi1_integrand, lower_bound, self.integration_limit,
            epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit,
        )
        pi2_integral, _ = quad(
            pi2_integrand, lower_bound, self.integration_limit,
            epsabs=self.epsabs, epsrel=self.epsrel, limit=self.limit,
        )
        if not (math.isfinite(pi1_integral) and math.isfinite(pi2_integral)):
            raise ArithmeticError("Fourier integration did not return finite probabilities.")

        pi1 = 0.5 + pi1_integral / math.pi
        pi2 = 0.5 + pi2_integral / math.pi
        tolerance = 5e-6
        if not (-tolerance <= pi1 <= 1.0 + tolerance):
            raise ArithmeticError(f"Pi1 is outside [0, 1]: {pi1:.8f}.")
        if not (-tolerance <= pi2 <= 1.0 + tolerance):
            raise ArithmeticError(f"Pi2 is outside [0, 1]: {pi2:.8f}.")
        return min(1.0, max(0.0, pi1)), min(1.0, max(0.0, pi2))

    def price(
        self,
        strike: float,
        market: MarketInputs,
        characteristic_function: Callable[[complex], complex],
    ) -> float:
        """Return the Fourier price of a European call option."""
        if not math.isfinite(strike) or strike <= 0:
            raise ValueError("Strike must be finite and strictly positive.")
        if market.maturity == 0:
            return max(market.spot - strike, 0.0)
        pi1, pi2 = self.probabilities(strike, market, characteristic_function)
        call_price = (
            market.spot * math.exp(-market.dividend_yield * market.maturity) * pi1
            - strike * math.exp(-market.risk_free_rate * market.maturity) * pi2
        )
        if not math.isfinite(call_price):
            raise ArithmeticError("The Fourier call price is not finite.")
        return max(call_price, 0.0)

    def price_nig(
        self, strike: float, market: MarketInputs, model: NIGRiskNeutralModel
    ) -> float:
        """Convenience wrapper for pricing with an NIG risk-neutral model."""
        return self.price(
            strike,
            market,
            lambda frequency: model.log_price_characteristic_function(frequency, market),
        )
