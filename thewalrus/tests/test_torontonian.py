# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Torontonian"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np

from itertools import product

from scipy.special import poch, factorial
from thewalrus.quantum import density_matrix_element, reduced_gaussian, Qmat, Xmat, Amat
from thewalrus.random import random_covariance
from thewalrus import tor, ltor, numba_ltor, threshold_detection_prob, numba_tor, numba_vac_prob
from thewalrus.symplectic import two_mode_squeezing


def gen_omats(l, nbar):
    r"""Generates the matrix O that enters inside the Torontonian for an l mode system
    in which the first mode is prepared in a thermal state with mean photon number nbar
    and the rest in vacuum and are later sent into a Fourier interferometer, i.e. one described
    by a DFT unitary matrix

    Args:
        l (int): number of modes
        nbar (float): mean photon number of the first mode (the only one not prepared in vacuum)

    Returns:
        array: An O matrix whose Torontonian can be calculated analytically.
    """
    A = (nbar / (l * (1.0 + nbar))) * np.ones([l, l])
    O = np.block([[A, 0 * A], [0 * A, A]])
    return O


def torontonian_analytical(l, nbar):
    r"""Return the value of the Torontonian of the O matrices generated by gen_omats

    Args:
        l (int): number of modes
        nbar (float): mean photon number of the first mode (the only one not prepared in vacuum)

    Returns:
        float: Value of the torontonian of gen_omats(l,nbar)
    """
    if np.allclose(l, nbar, atol=1e-14, rtol=0.0):
        return 1.0
    beta = -(nbar / (l * (1 + nbar)))
    pref = factorial(l) / beta
    p1 = pref * l / poch(1 / beta, l + 2)
    p2 = pref * beta / poch(2 + 1 / beta, l)
    return (p1 + p2) * (-1) ** l


def test_torontonian_tmsv():
    """Calculates the torontonian of a two-mode squeezed vacuum
    state squeezed with mean photon number 1.0"""

    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    Omat = np.tanh(r) * np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    tor_val = tor(Omat)
    assert np.allclose(tor_val.real, 1.0)


def test_torontonian_tmsv_complex_zero_imag_part():
    """Calculates the torontonian of a two-mode squeezed vacuum
    state squeezed with mean photon number 1.0"""

    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    Omat = np.tanh(r) * np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    Omat = np.complex128(Omat)
    tor_val = tor(Omat)
    assert np.allclose(tor_val.real, 1.0)


def test_torontonian_tmsv_complex():
    """Calculates the torontonian of a two-mode squeezed vacuum
    state squeezed with mean photon number 1.0"""

    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    phase = np.exp(1j * 0.3)
    phasec = np.conj(phase)
    Omat = np.tanh(r) * np.array(
        [[0, 0, 0, phase], [0, 0, phase, 0], [0, phasec, 0, 0], [phasec, 0, 0, 0]]
    )
    tor_val = tor(Omat)
    assert np.allclose(tor_val.real, 1.0)


def test_torontonian_vacuum():
    """Calculates the torontonian of a vacuum in n modes"""
    n_modes = 5
    Omat = np.zeros([2 * n_modes, 2 * n_modes])
    tor_val = tor(Omat)
    assert np.allclose(tor_val.real, 0.0)


@pytest.mark.parametrize("l", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nbar", np.arange(0.25, 3, 0.25))
def test_torontonian_analytical_mats(l, nbar):
    """Checks the correct value of the torontonian for the analytical family described by gen_omats"""
    assert np.allclose(torontonian_analytical(l, nbar), tor(gen_omats(l, nbar)))


@pytest.mark.parametrize("r", [0.5, 0.5, -0.8, 1, 0])
@pytest.mark.parametrize("alpha", [0.5, 2, -0.5, 0.0, -0.5])
def test_disp_torontonian(r, alpha):
    """Calculates click probabilities of displaced two mode squeezed state"""

    p00a = np.exp(-2 * (abs(alpha) ** 2 - abs(alpha) ** 2 * np.tanh(r))) / (np.cosh(r) ** 2)

    fact_0 = np.exp(-(abs(alpha) ** 2) / (np.cosh(r) ** 2))
    p01a = fact_0 / (np.cosh(r) ** 2) - p00a

    fact_0 = np.cosh(r) ** 2
    fact_1 = -2 * np.exp(-(abs(alpha) ** 2) / (np.cosh(r) ** 2))
    fact_2 = np.exp(-2 * (abs(alpha) ** 2 - abs(alpha) ** 2.0 * np.tanh(r)))
    p11a = (fact_0 + fact_1 + fact_2) / (np.cosh(r) ** 2)

    cov = two_mode_squeezing(abs(2 * r), np.angle(2 * r))
    mu = 2 * np.array([alpha.real, alpha.real, alpha.imag, alpha.imag])

    p00n = threshold_detection_prob(mu, cov, np.array([0, 0]))
    p01n = threshold_detection_prob(mu, cov, np.array([0, 1]))
    p11n = threshold_detection_prob(mu, cov, np.array([1, 1]))

    assert np.isclose(p00a, p00n)
    assert np.isclose(p01a, p01n)
    assert np.isclose(p11a, p11n)


@pytest.mark.parametrize("scale", [0, 1, 2, 3])
def test_disp_torontonian_single_mode(scale):
    """Calculates the probability of clicking for a single mode state"""
    cv = random_covariance(1)
    mu = scale * (2 * np.random.rand(2) - 1)
    prob_click = threshold_detection_prob(mu, cv, np.array([1]))
    expected = 1 - density_matrix_element(mu, cv, [0], [0])
    assert np.allclose(prob_click, expected)


@pytest.mark.parametrize("scale", [0, 1, 2, 3])
def test_disp_torontonian_two_mode(scale):
    """Calculates the probability of clicking for a two mode state"""
    cv = random_covariance(2)
    mu = scale * (2 * np.random.rand(4) - 1)
    prob_click = threshold_detection_prob(mu, cv, [1, 1])
    mu0, cv0 = reduced_gaussian(mu, cv, [0])
    mu1, cv1 = reduced_gaussian(mu, cv, [1])
    expected = (
        1
        - density_matrix_element(mu0, cv0, [0], [0])
        - density_matrix_element(mu1, cv1, [0], [0])
        + density_matrix_element(mu, cv, [0, 0], [0, 0])
    )
    assert np.allclose(expected, prob_click)


@pytest.mark.parametrize("n_modes", range(1, 10))
def test_tor_and_threshold_displacement_prob_agree(n_modes):
    """Tests that threshold_detection_prob, ltor and the usual tor expression all agree
    when displacements are zero"""
    cv = random_covariance(n_modes)
    mu = np.zeros([2 * n_modes])
    Q = Qmat(cv)
    O = Xmat(n_modes) @ Amat(cv)
    expected = tor(O) / np.sqrt(np.linalg.det(Q))
    prob = threshold_detection_prob(mu, cv, np.array([1] * n_modes))
    prob2 = numba_ltor(O, mu) / np.sqrt(np.linalg.det(Q))
    prob3 = numba_vac_prob(mu, Q) * ltor(O, mu)
    assert np.isclose(expected, prob)
    assert np.isclose(expected, prob2)
    assert np.isclose(expected, prob3)


@pytest.mark.parametrize("N", range(1, 10))
def test_numba_tor(N):
    """Tests numba implementation of the torontonian against the default implementation"""
    cov = random_covariance(N)
    O = Xmat(N) @ Amat(cov)
    t1 = tor(O)
    t2 = numba_tor(O)
    assert np.isclose(t1, t2)


def test_tor_exceptions():
    """test that correct exceptions are raised for tor function"""
    with pytest.raises(TypeError):
        tor("hello")

    with pytest.raises(ValueError):
        tor(np.zeros((4, 2)))

    with pytest.raises(ValueError):
        tor(np.zeros((3, 3)))


def test_ltor_exceptions():
    """test that correct exceptions are raised for ltor function"""

    with pytest.raises(TypeError):
        ltor("hello", np.zeros(4))

    with pytest.raises(TypeError):
        ltor(np.zeros((4, 4)), "hello")

    with pytest.raises(ValueError):
        ltor(np.zeros((4, 2)), np.zeros(4))

    with pytest.raises(ValueError):
        ltor(np.zeros((3, 3)), np.zeros(3))

    with pytest.raises(ValueError):
        ltor(np.zeros((4, 4)), np.zeros(6))


@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("scale", [0, 1, 1.4])
def test_probs_sum_to_1(n, scale):
    """test that threshold probabilities sum to 1"""
    cov = random_covariance(n)
    mu = scale * (2 * np.random.rand(2 * n) - 1)

    p_total = 0
    for det_pattern in product([0, 1], repeat=n):
        p = threshold_detection_prob(mu, cov, det_pattern)
        p_total += p

    assert np.isclose(p_total, 1)
