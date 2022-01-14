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
"""Tests for the Python permanent wrapper function"""
# pylint: disable=no-self-use

from itertools import chain

import pytest

import numpy as np

from scipy.special import factorial as fac
from scipy.linalg import sqrtm
from scipy.stats import unitary_group

from thewalrus import perm, permanent_repeated, brs, ubrs
from thewalrus._permanent import fock_prob, fock_threshold_prob

perm_real = perm
perm_complex = perm
perm_BBFG_real = lambda x: perm(x, method="bbfg")
perm_BBFG_complex = lambda x: perm(x, method="bbfg")


class TestPermanentWrapper:
    """Tests for the Permanent function"""

    def test_array_exception(self):
        """Check exception for non-matrix argument"""
        with pytest.raises(TypeError):
            perm(1)

    def test_square_exception(self):
        """Check exception for non-square argument"""
        A = np.zeros([2, 3])
        with pytest.raises(ValueError):
            perm(A)

    def test_nan(self):
        """Check exception for non-finite matrix"""
        A = np.array([[2, 1], [1, np.nan]])
        with pytest.raises(ValueError):
            perm(A)

    def test_2x2(self, random_matrix):
        """Check 2x2 permanent"""
        A = random_matrix(2)
        p = perm(A, method="ryser")
        expected = A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]
        assert p == expected

        p = perm(A, method="bbfg")
        assert p == expected

    def test_3x3(self, random_matrix):
        """Check 3x3 permanent"""
        A = random_matrix(3)
        p = perm(A, method="ryser")
        expected = (
            A[0, 2] * A[1, 1] * A[2, 0]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[0, 0] * A[1, 2] * A[2, 1]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2]
        )
        assert p == expected

        p = perm(A, method="bbfg")
        assert p == expected

    @pytest.mark.parametrize("dtype", [np.float64])
    def test_real(self, random_matrix):
        """Check perm(A) == perm_real(A) and perm(A, method="bbfg") == perm_BBFG_real(A) for a random real matrix."""
        A = random_matrix(6)
        p = perm(A, method="ryser")
        expected = perm_real(A)
        assert np.allclose(p, expected)

        A = random_matrix(6)
        A = np.array(A, dtype=np.complex128)
        p = perm(A, method="ryser")
        expected = perm_real(np.float64(A.real))
        assert np.allclose(p, expected)

        A = random_matrix(6)
        p = perm(A, method="bbfg")
        expected = perm_BBFG_real(A)
        assert np.allclose(p, expected)

        A = random_matrix(6)
        A = np.array(A, dtype=np.complex128)
        p = perm(A, method="bbfg")
        expected = perm_BBFG_real(np.float64(A.real))
        assert np.allclose(p, expected)

    @pytest.mark.parametrize("dtype", [np.complex128])
    def test_complex(self, random_matrix):
        """Check perm(A) == perm_complex(A) and perm(A) == perm_BBFG_complex(A) for a complex."""
        A = random_matrix(6)
        p = perm(A, method="ryser")
        expected = perm_complex(A)
        assert np.allclose(p, expected)

        A = random_matrix(6)
        p = perm(A, method="ryser")
        expected = perm_BBFG_complex(A)
        assert np.allclose(p, expected)

    @pytest.mark.parametrize("dtype", [np.float64])
    def test_complex_no_imag(self, random_matrix):
        """Check perm(A) == perm_real(A) and perm(A) == perm_BBFG_real(A) for a complex random matrix with zero imaginary parts."""
        A = np.complex128(random_matrix(6))
        p = perm(A, method="ryser")
        expected = perm_real(A.real)
        assert np.allclose(p, expected)

        A = np.complex128(random_matrix(6))
        p = perm(A, method="ryser")
        expected = perm_BBFG_real(A.real)
        assert np.allclose(p, expected)


class TestPermanentRepeated:
    """Tests for the repeated permanent"""

    def test_rpt_zero(self):
        """Check 2x2 permanent when rpt is all 0"""
        A = np.array([[2, 1], [1, 3]])
        rpt = [0, 0]
        res = permanent_repeated(A, rpt)
        assert res == 1.0

    def test_2x2(self, random_matrix):
        """Check 2x2 permanent"""
        A = random_matrix(2)
        p = permanent_repeated(A, [1] * 2)
        assert np.allclose(p, A[0, 0] * A[1, 1] + A[1, 0] * A[0, 1])

    def test_3x3(self, random_matrix):
        """Check 3x3 permanent"""
        A = random_matrix(3)
        p = permanent_repeated(A, [1] * 3)
        exp = (
            A[0, 0] * A[1, 1] * A[2, 2]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[2, 0] * A[1, 1] * A[0, 2]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 2] * A[2, 1]
        )
        assert np.allclose(p, exp)

    @pytest.mark.parametrize("n", [6, 8, 10, 15, 20])
    def test_ones(self, n):
        """Check all ones matrix has perm(J_n)=n!"""
        A = np.array([[1]])
        p = permanent_repeated(A, [n])
        assert np.allclose(p, fac(n))


def test_brs_HOM():
    """HOM test"""

    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    n = [1, 1]
    d = [1, 1]

    assert np.isclose(fock_threshold_prob(n, d, U), fock_prob(n, d, U))

    d = [1, 0]
    m = [2, 0]

    assert np.isclose(fock_threshold_prob(n, d, U), fock_prob(n, m, U))


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.9, 1])
def test_brs_HOM_lossy(eta):
    """lossy HOM dip test"""
    T = np.sqrt(eta / 2) * np.array([[1, 1], [1, -1]])

    n = [1, 1]
    d = [1, 1]

    assert np.isclose(fock_prob(n, d, T), fock_threshold_prob(n, d, T))


def test_brs_ZTL():
    """test 3-mode ZTL suppression"""

    U = np.fft.fft(np.eye(3)) / np.sqrt(3)

    n = [1, 1, 1]
    d = [1, 1, 0]

    p1 = fock_threshold_prob(n, d, U)
    p2 = fock_prob(n, [1, 2, 0], U) + fock_prob(n, [2, 1, 0], U)
    assert np.isclose(p1, p2)

    n = [1, 1, 1]
    d = [1, 1, 1]

    p1 = fock_threshold_prob(n, d, U)
    p2 = fock_prob(n, d, U)
    assert np.isclose(p1, p2)

    T = U[:2, :]
    d = [1, 1]

    p1 = fock_threshold_prob(n, d, T)
    p2 = fock_prob(n, [1, 1, 1], U)

    assert np.isclose(p1, p2)

    d = [1, 0, 0]

    p1 = fock_threshold_prob(n, d, U)
    p2 = fock_prob(n, [3, 0, 0], U)

    assert np.isclose(p1, p2)

    n = [1, 2, 0]
    d = [0, 1, 1]

    p1 = fock_threshold_prob(n, d, U)
    p2 = fock_prob(n, [0, 2, 1], U) + fock_prob(n, [0, 1, 2], U)

    assert np.isclose(p1, p2)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.9, 1])
def test_brs_ZTL_lossy(eta):
    """test lossy 3-mode ZTL suppression"""
    T = np.sqrt(eta) * np.fft.fft(np.eye(3)) / np.sqrt(3)

    n = [1, 1, 1]
    d = [1, 1, 0]

    p1 = eta ** 2 * (1 - eta) / 3
    p2 = fock_threshold_prob(n, d, T)

    assert np.allclose(p1, p2)


@pytest.mark.parametrize("d", [[1, 1, 1], [1, 1, 0], [1, 0, 0]])
def test_brs_ubrs(d):
    """test that brs and ubrs give same results for unitary transformation"""

    U = np.fft.fft(np.eye(3)) / np.sqrt(3)

    n = np.array([2, 1, 0])
    d = np.array(d)

    in_modes = np.array(list(chain(*[[i] * j for i, j in enumerate(n) if j > 0])))
    click_modes = np.where(d > 0)[0]

    U_dn = U[np.ix_(click_modes, in_modes)]

    b1 = ubrs(U_dn)

    R = sqrtm(np.eye(U.shape[1]) - U.conj().T @ U)[:, in_modes]
    E = R.conj().T @ R

    b2 = brs(U_dn, E)

    assert np.allclose(b1, b2)


@pytest.mark.parametrize("M", range(2, 7))
def test_brs_random(M):
    """test that brs and per agree for random matices"""

    n = np.ones(M, dtype=int)
    n[np.random.randint(0, M)] = 0
    d = np.ones(M, dtype=int)
    d[np.random.randint(0, M)] = 0

    loss_in = np.random.random(M)
    loss_out = np.random.random(M)
    U = unitary_group.rvs(M)
    T = np.diag(loss_in) @ U @ np.diag(loss_out)

    p1 = fock_threshold_prob(n, d, T)
    p2 = fock_prob(n, d, T)

    assert np.isclose(p1, p2)
