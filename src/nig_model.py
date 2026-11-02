"""Normal Inverse Gaussian (NIG) model primitives.

Convention
----------
The NIG process is parameterised by (alpha, beta, delta, mu), with
alpha > 0, delta > 0 and |beta| < alpha. Its unit-time cumulant exponent is

    kappa(z) = mu*z + delta * (
        sqrt(alpha**2 - beta**2)
        - sqrt(alpha**2 - (beta + z)**2)
    ).

This module deliberately implements the distribution/process building block
only. The risk-neutral drift adjustment and option-pricing convention must be
derived and validated against the professor's prescribed reference before
they are added to the calibration workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math


@dataclass(frozen=True)
class NIGParameters:
    """Immutable parameter set for an NIG Lévy process.

    Attributes:
        alpha: Tail steepness parameter; must be strictly positive.
        beta: Skewness parameter; must satisfy abs(beta) < alpha.
        delta: Positive scale/activity parameter per unit time.
        mu: Location/drift parameter per unit time.
    """

    alpha: float
    beta: float
    delta: float
    mu: float

    def __post_init__(self) -> None:
        """Reject parameter values outside the NIG admissible region."""
        parameter_values = {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "mu": self.mu,
        }
        for parameter_name, parameter_value in parameter_values.items():
            if not math.isfinite(parameter_value):
                raise ValueError(
                    f"NIG parameter '{parameter_name}' must be finite; "
                    f"received {parameter_value!r}."
                )

        if self.alpha <= 0:
            raise ValueError("NIG parameter 'alpha' must be strictly positive.")
        if self.delta <= 0:
            raise ValueError("NIG parameter 'delta' must be strictly positive.")
        if abs(self.beta) >= self.alpha:
            raise ValueError(
                "NIG parameters must satisfy abs(beta) < alpha; "
                f"received alpha={self.alpha}, beta={self.beta}."
            )


class NIGLevyProcess:
    """Evaluate the cumulant and characteristic functions of an NIG process.

    The process is defined through stationary independent increments. For a
    time horizon T, its characteristic function is exp(T * kappa(i*u)).
    """

    def __init__(self, parameters: NIGParameters) -> None:
        """Store a validated immutable NIG parameter set."""
        self.parameters = parameters

    def cumulant_exponent(self, argument: complex) -> complex:
        """Return the unit-time cumulant exponent kappa(argument).

        The complex square root uses Python's principal branch. The argument
        must lie in the analytic domain of the chosen NIG cumulant.
        """
        alpha = self.parameters.alpha
        beta = self.parameters.beta
        delta = self.parameters.delta
        mu = self.parameters.mu

        base_radicand = alpha**2 - beta**2
        shifted_radicand = alpha**2 - (beta + argument) ** 2

        base_root = math.sqrt(base_radicand)
        shifted_root = cmath.sqrt(shifted_radicand)

        return mu * argument + delta * (base_root - shifted_root)

    def characteristic_function(self, frequency: float, maturity: float) -> complex:
        """Return E[exp(i * frequency * X_T)] for the NIG process."""
        if not math.isfinite(frequency):
            raise ValueError("Frequency must be finite.")
        if not math.isfinite(maturity) or maturity < 0:
            raise ValueError("Maturity must be finite and non-negative.")

        exponent = maturity * self.cumulant_exponent(1j * frequency)
        return cmath.exp(exponent)

    def moment_generating_function(self, argument: float, maturity: float) -> float:
        """Return E[exp(argument * X_T)] when the exponential moment exists.

        The moment exists only when abs(beta + argument) < alpha.
        """
        if not math.isfinite(argument):
            raise ValueError("MGF argument must be finite.")
        if not math.isfinite(maturity) or maturity < 0:
            raise ValueError("Maturity must be finite and non-negative.")

        alpha = self.parameters.alpha
        beta = self.parameters.beta
        if abs(beta + argument) >= alpha:
            raise ValueError(
                "The requested exponential moment does not exist: "
                "it requires abs(beta + argument) < alpha."
            )

        exponent = maturity * self.cumulant_exponent(argument)
        return math.exp(exponent.real)
