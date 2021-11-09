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
"""Tests for the batch hafnian wrapper function"""
# pylint: disable=no-self-use,redefined-outer-name
from itertools import product
import pytest

import numpy as np

from scipy.special import eval_hermitenorm, eval_hermite

from thewalrus import (
    hermite_multidimensional,
    interferometer,
    hafnian_batched,
    hafnian_repeated,
    grad_hermite_multidimensional,
)
from thewalrus.random import random_interferometer


def test_hermite_multidimensional_renorm():
    """This tests the renormalized batchhafnian wrapper function to compute photon number statistics for a fixed gaussian state."""
    B = np.sqrt(0.5) * np.array([[0, 1], [1, 0]]) + 0 * 1j
    res = 10
    expected = np.diag(0.5 ** (np.arange(0, res) / 2))
    array = hermite_multidimensional(-B, res, renorm=True)

    assert np.allclose(array, expected)


def test_reduction_to_physicists_polys():
    """Tests that the multidimensional hermite polynomials reduce to the regular physicists' hermite polynomials in the appropriate limit"""
    x = np.arange(-1, 1, 0.1)
    init = 1
    n_max = 5
    A = np.ones([init, init], dtype=complex)
    vals = np.array(
        [hermite_multidimensional(2 * A, n_max, y=np.array([x0], dtype=complex)) for x0 in x]
    ).T
    expected = np.array([eval_hermite(i, x) for i in range(len(vals))])
    assert np.allclose(vals, expected)


def test_reduction_to_probabilist_polys():
    """Tests that the multidimensional hermite polynomials reduce to the regular probabilist' hermite polynomials in the appropriate limit"""
    x = np.arange(-1, 1, 0.1)
    init = 1
    n_max = 5
    A = np.ones([init, init], dtype=complex)
    vals = np.array(
        [hermite_multidimensional(A, n_max, y=np.array([x0], dtype=complex)) for x0 in x]
    ).T
    expected = np.array([eval_hermitenorm(i, x) for i in range(len(vals))])
    assert np.allclose(vals, expected)


def test_hafnian_batched():
    """Test hafnian_batched against hafnian_repeated for a random symmetric matrix"""
    n_modes = 4
    A = np.random.rand(n_modes, n_modes) + 1j * np.random.rand(n_modes, n_modes)
    A += A.T
    n_photon = 5
    v1 = np.array([hafnian_repeated(A, q) for q in product(np.arange(n_photon), repeat=n_modes)])
    assert np.allclose(hafnian_batched(A, n_photon, make_tensor=False), v1)


def test_hafnian_batched_loops():
    """Test hafnian_batched with loops against hafnian_repeated with loops for a random symmetric matrix
    and a random vector of loops
    """
    n_modes = 4
    A = np.random.rand(n_modes, n_modes) + 1j * np.random.rand(n_modes, n_modes)
    A += A.T
    mu = np.random.rand(n_modes) + 1j * np.random.rand(n_modes)
    n_photon = 5
    v1 = np.array(
        [
            hafnian_repeated(A, q, mu=mu, loop=True)
            for q in product(np.arange(n_photon), repeat=n_modes)
        ]
    )
    expected = hafnian_batched(A, n_photon, mu=mu, make_tensor=False)
    assert np.allclose(expected, v1)


def test_hafnian_batched_loops_no_edges():
    """Test hafnian_batched with loops against hafnian_repeated with loops for a random symmetric matrix
    and a random vector of loops
    """
    n_modes = 4
    A = np.zeros([n_modes, n_modes], dtype=complex)
    mu = np.random.rand(n_modes) + 1j * np.random.rand(n_modes)
    n_photon = 5
    v1 = np.array(
        [
            hafnian_repeated(A, q, mu=mu, loop=True)
            for q in product(np.arange(n_photon), repeat=n_modes)
        ]
    )
    expected = hafnian_batched(A, n_photon, mu=mu, make_tensor=False)

    assert np.allclose(expected, v1)


def test_hafnian_batched_zero_loops_no_edges():
    """Test hafnian_batched with loops against hafnian_repeated with loops for a the zero matrix
    and a loops
    """
    n_modes = 4
    A = np.zeros([n_modes, n_modes], dtype=complex)
    n_photon = 5
    v1 = np.array(
        [hafnian_repeated(A, q, loop=True) for q in product(np.arange(n_photon), repeat=n_modes)]
    )
    expected = hafnian_batched(A, n_photon, make_tensor=False)

    assert np.allclose(expected, v1)


def test_hermite_vs_hermite_modified():
    """Test the relation hermite and hermite modified"""
    n_modes = 2
    A = np.zeros([n_modes, n_modes], dtype=complex)
    mu = np.random.rand(n_modes) + 1j * np.random.rand(n_modes)
    cutoff = 3
    assert np.allclose(
        hermite_multidimensional(A, cutoff, y=A @ mu, modified=True),
        hermite_multidimensional(A, cutoff, y=mu),
    )


def test_hermite_cutoffs():
    """Test that the cutoff is correctly set"""
    R = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    R = R + R.T
    y = np.random.rand(3) + 1j * np.random.rand(3)
    cutoff = (1, 2, 3)
    hm = hermite_multidimensional(R, cutoff, y=y)
    assert hm.shape == cutoff


@pytest.mark.parametrize("renorm", [True, False])
def test_grad_hermite_multidimensional_vs_finite_differences(tol, renorm):
    """Tests the gradients of hermite polynomials. The gradients of parameters are tested by finite differences."""
    d = 4
    R = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    R += R.T
    y = np.random.rand(d) + 1j * np.random.rand(d)
    C = 0.5
    cutoff = [3, 3, 3, 3]
    gate = hermite_multidimensional(R, cutoff, y, C, renorm=renorm, modified=True)
    grad_C, grad_R, grad_y = grad_hermite_multidimensional(gate, R, y, C, dtype=np.complex128)

    delta = 0.000001 + 1j * 0.000001
    expected_grad_C = (
        hermite_multidimensional(R, cutoff, y, C + delta, renorm=renorm, modified=True)
        - hermite_multidimensional(R, cutoff, y, C - delta, renorm=renorm, modified=True)
    ) / (2 * delta)
    assert np.allclose(grad_C, expected_grad_C, atol=tol, rtol=0)

    for i in range(y.shape[0]):
        y[i] += delta
        plus = hermite_multidimensional(R, cutoff, y, C, renorm=renorm, modified=True)
        y[i] -= 2 * delta
        minus = hermite_multidimensional(R, cutoff, y, C, renorm=renorm, modified=True)
        expected_grad_y = (plus - minus) / (2 * delta)
        y[i] += delta
        assert np.allclose(grad_y[..., i], expected_grad_y, atol=tol, rtol=0)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] += delta
            plus = hermite_multidimensional(R, cutoff, y, C, renorm=renorm, modified=True)
            R[i, j] -= 2 * delta
            minus = hermite_multidimensional(R, cutoff, y, C, renorm=renorm, modified=True)
            expected_grad_R = (plus - minus) / (2 * delta)
            R[i, j] += delta
            assert np.allclose(grad_R[..., i, j], expected_grad_R, atol=tol, rtol=0)


def test_auto_dtype_multidim_herm():
    """Tests that auto-dtype detection works"""
    d = 4
    R = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    R += R.T
    y = np.random.rand(d) + 1j * np.random.rand(d)
    C = 0.5
    cutoff = 3

    R = R.astype("complex64")
    y = y.astype("complex128")
    poly = hermite_multidimensional(R, cutoff, y, C)
    assert poly.dtype == y.dtype

    R = R.astype("complex128")
    y = y.astype("complex64")
    poly = poly.astype("complex64")
    grad = grad_hermite_multidimensional(poly, R, y, C, dtype=None)
    assert all(g.dtype == R.dtype for g in grad)


def test_multi_cutoffs_multidim_herm():
    """Tests that different cutoffs give the right result"""
    d = 4
    R = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    R += R.T
    y = np.random.rand(d) + 1j * np.random.rand(d)
    C = 0.5

    cutoffs = [np.random.randint(3, 7) for _ in range(d)]
    poly = hermite_multidimensional(R, cutoffs, y, C)
    poly_expected = hermite_multidimensional(R, 3, y, C)
    assert np.allclose(poly[:3, :3, :3, :3], poly_expected)
    poly_expected = hermite_multidimensional(R, np.array(3), y, C)  # testing ndarrary cutoff int
    assert np.allclose(poly[:3, :3, :3, :3], poly_expected)


@pytest.mark.parametrize("renorm", [True, False])
@pytest.mark.parametrize("cutoff", [6, np.array([4, 5, 6]), [4, 5, 6]])
def test_interferometer_vs_hermite_multidimensional(renorm, cutoff):
    """Test that intertferometer and hermite_multidimensional give the same result"""
    U = random_interferometer(2)
    R = np.block([[0 * U, -U], [-U.T, 0 * U]])

    hermite_renorm = hermite_multidimensional(R, cutoff, y=None, renorm=renorm)
    interf_renorm = interferometer(R, cutoff, renorm=renorm)
    assert np.allclose(hermite_renorm, interf_renorm)
