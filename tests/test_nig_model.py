"""Numerical tests for the NIG process building blocks."""

import math

import pytest

from src.nig_model import NIGLevyProcess, NIGParameters


def make_reference_process() -> NIGLevyProcess:
    """Create a valid parameter set for deterministic unit tests."""
    return NIGLevyProcess(
        NIGParameters(alpha=5.0, beta=-1.0, delta=0.2, mu=0.05)
    )


def test_nig_parameter_constraints_are_enforced() -> None:
    """The NIG parameter vector must remain inside its admissible region."""
    with pytest.raises(ValueError, match="alpha"):
        NIGParameters(alpha=0.0, beta=0.0, delta=0.2, mu=0.0)

    with pytest.raises(ValueError, match="delta"):
        NIGParameters(alpha=5.0, beta=0.0, delta=0.0, mu=0.0)

    with pytest.raises(ValueError, match="abs"):
        NIGParameters(alpha=1.0, beta=1.0, delta=0.2, mu=0.0)


def test_characteristic_function_at_zero_frequency_is_one() -> None:
    """Every valid characteristic function satisfies phi(0) = 1."""
    process = make_reference_process()

    assert process.characteristic_function(frequency=0.0, maturity=0.75) == pytest.approx(1.0 + 0.0j)


def test_characteristic_function_modulus_does_not_exceed_one() -> None:
    """A characteristic function must have modulus at most one."""
    process = make_reference_process()

    for frequency in (-10.0, -2.0, 0.5, 3.0, 12.0):
        value = process.characteristic_function(frequency, maturity=1.25)
        assert abs(value) <= 1.0 + 1e-12


def test_zero_maturity_returns_degenerate_characteristic_function() -> None:
    """At T=0, the increment is zero and its characteristic function is one."""
    process = make_reference_process()

    assert process.characteristic_function(4.2, maturity=0.0) == pytest.approx(1.0 + 0.0j)


def test_mgf_at_zero_is_one() -> None:
    """Every valid moment-generating function satisfies M(0) = 1."""
    process = make_reference_process()

    assert process.moment_generating_function(0.0, maturity=0.5) == pytest.approx(1.0)


def test_mgf_rejects_argument_outside_exponential_moment_domain() -> None:
    """The MGF must reject arguments outside |beta + z| < alpha."""
    process = make_reference_process()

    with pytest.raises(ValueError, match="does not exist"):
        process.moment_generating_function(6.1, maturity=1.0)


def test_cumulant_first_derivative_matches_nig_mean() -> None:
    """The first cumulant derivative at zero equals E[X_1] = mu + delta*beta/sqrt(alpha²-beta²)."""
    process = make_reference_process()
    parameters = process.parameters
    step = 1e-5

    numerical_mean = (
        process.cumulant_exponent(step).real
        - process.cumulant_exponent(-step).real
    ) / (2.0 * step)

    theoretical_mean = (
        parameters.mu
        + parameters.delta * parameters.beta
        / math.sqrt(parameters.alpha**2 - parameters.beta**2)
    )

    assert numerical_mean == pytest.approx(theoretical_mean, rel=1e-7, abs=1e-8)
