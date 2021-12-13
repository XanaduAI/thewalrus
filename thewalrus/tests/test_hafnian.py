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
"""Tests for the Python hafnian wrapper function"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np
from scipy.special import factorial as fac

import thewalrus as hf
from thewalrus import hafnian, reduction, hafnian_sparse, hafnian_banded, matched_reps

from thewalrus._hafnian import haf as jhaf
from thewalrus._hafnian import loop_hafnian
from thewalrus._hafnian import bandwidth


# the first 11 telephone numbers
T = [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]


def random_banded(n, bw):
    """Generates a random matrix of a given size and bandwidth.

    Args:
      n (int): Size of the matrix
      bw (int): Bandwidth of the matrix

    Returns:
      (array): a matrix with the given properties
    """

    M = np.zeros([n, n], dtype=np.complex128)
    for j in range(bw + 1):
        M += np.diag(np.random.rand(n - j) + 1j * np.random.rand(n - j), k=j)
    M += M.T
    return M


class TestReduction:
    """Tests for the reduction function"""

    @pytest.mark.parametrize("n", [6, 8])
    def test_reduction(self, n):
        """Check kron reduced returns correct result"""
        res = reduction(np.array([[0, 1], [1, 0]]), [n, n])

        O = np.zeros([n, n])
        B = np.ones([n, n])
        ex = np.vstack([np.hstack([O, B]), np.hstack([B, O])])

        assert np.all(res == ex)

    @pytest.mark.parametrize("n", [6, 8])
    def test_reduction_vector(self, n):
        """Check kron reduced returns correct result"""
        res = reduction(np.array([0, 1]), [n, n])

        O = np.zeros([n, n])
        J = np.ones([n, n])
        ex = np.hstack([O, J])
        assert np.all(res == ex)


class TestHafnianWrapper:
    """Tests for the Python hafnian wrapper function.
    These tests should only test for:

    * exceptions
    * validation
    * that the wrapper returns the same
      value as the C++ functions
    """

    def test_version_number(self):
        """returns true if returns a string"""
        res = hf.version()
        assert isinstance(res, str)

    def test_array_exception(self):
        """Check exception for non-matrix argument"""
        with pytest.raises(TypeError):
            hafnian(1)

    def test_square_exception(self):
        """Check exception for non-square argument"""
        A = np.zeros([2, 3])
        with pytest.raises(ValueError):
            hafnian(A)

    def test_odd_dim(self):
        """Check hafnian for matrix with odd dimensions"""
        A = np.ones([3, 3])
        assert hafnian(A) == 0

    def test_non_symmetric_exception(self):
        """Check exception for non-symmetric matrix"""
        A = np.ones([4, 4])
        A[0, 1] = 0.0
        with pytest.raises(ValueError):
            hafnian(A)

    def test_nan(self):
        """Check exception for non-finite matrix"""
        A = np.array([[2, 1], [1, np.nan]])
        with pytest.raises(ValueError):
            hafnian(A)

    def test_empty_matrix(self):
        """Check empty matrix returns 1"""
        A = np.ndarray((0, 0))
        res = hafnian(A)
        assert res == 1

    def test_real_wrapper(self):
        """Check hafnian(A)=haf_real(A) for a random
        real matrix.
        """
        A = np.random.random([6, 6])
        A += A.T
        haf = hafnian(A)
        expected = jhaf(A)
        assert np.allclose(haf, expected)

        haf = hafnian(A, loop=True)
        expected = loop_hafnian(A)
        assert np.allclose(haf, expected)

        A = np.random.random([6, 6])
        A += A.T
        A = np.array(A, dtype=np.complex128)
        haf = hafnian(A)
        expected = jhaf(np.float64(A.real))
        assert np.allclose(haf, expected)

    def test_int_wrapper(self):
        """Check hafnian(A)=haf_int(A) for a random
        integer matrix.
        """
        A = np.int64(np.ones([6, 6]))
        haf = hafnian(A)
        expected = jhaf(np.int64(A))
        assert np.allclose(haf, expected)

    def test_int_wrapper_loop(self):
        """Check hafnian(A, loop=True)=haf_real(A, loop=True) for a random
        integer matrix.
        """
        A = np.int64(np.ones([6, 6]))
        haf = hafnian(A, loop=True)
        expected = loop_hafnian(np.float64(A))
        assert np.allclose(haf, expected)

    def test_complex_wrapper(self):
        """Check hafnian(A)=haf_complex(A) for a random
        real matrix.
        """
        A = np.complex128(np.random.random([6, 6]))
        A += 1j * np.random.random([6, 6])
        A += A.T
        haf = hafnian(A)
        expected = jhaf(A)
        assert np.allclose(haf, expected)

        haf = hafnian(A, loop=True)
        expected = loop_hafnian(A)
        assert np.allclose(haf, expected)


class TestHafnian:
    """Various Hafnian consistency checks."""

    def test_2x2(self, random_matrix):
        """Check 2x2 hafnian"""
        A = random_matrix(2)
        haf = hafnian(A)
        assert np.allclose(haf, A[0, 1])

    def test_3x3(self, dtype):
        """Check 3x3 hafnian"""
        A = dtype(np.ones([3, 3]))
        haf = hafnian(A)
        assert haf == 0.0

    def test_4x4(self, random_matrix):
        """Check 4x4 hafnian"""
        A = random_matrix(4)
        haf = hafnian(A)
        expected = A[0, 1] * A[2, 3] + A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n, dtype):
        """Check hafnian(I)=0"""
        A = dtype(np.identity(n))
        haf = hafnian(A)
        assert np.allclose(haf, 0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n, dtype):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = dtype(np.ones([2 * n, 2 * n]))
        haf = hafnian(A)
        expected = fac(2 * n) / (fac(n) * (2 ** n))
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n, dtype):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]), np.hstack([B, O])])
        A = dtype(A)
        haf = hafnian(A)
        expected = float(fac(n))
        assert np.allclose(haf, expected)


class TestLoopHafnian:
    """Various loop Hafnian consistency checks.
    The loop hafnian is currently only defined for
    the trace algorithm.
    """

    def test_2x2(self, random_matrix):
        """Check 2x2 loop hafnian"""
        A = random_matrix(2)
        haf = hafnian(A, loop=True)
        assert np.allclose(haf, A[0, 1] + A[0, 0] * A[1, 1])

    def test_3x3(self, dtype):
        """Check 3x3 loop hafnian"""
        A = dtype(np.ones([3, 3]))
        haf = hafnian(A, loop=True)
        assert haf == 4.0

    def test_4x4(self, random_matrix):
        """Check 4x4 loop hafnian"""
        A = random_matrix(4)
        haf = hafnian(A, loop=True)
        expected = (
            A[0, 1] * A[2, 3]
            + A[0, 2] * A[1, 3]
            + A[0, 3] * A[1, 2]
            + A[0, 0] * A[1, 1] * A[2, 3]
            + A[0, 1] * A[2, 2] * A[3, 3]
            + A[0, 2] * A[1, 1] * A[3, 3]
            + A[0, 0] * A[2, 2] * A[1, 3]
            + A[0, 0] * A[3, 3] * A[1, 2]
            + A[0, 3] * A[1, 1] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3]
        )
        assert np.allclose(haf, expected)

    def test_4x4_zero_diag(self, random_matrix):
        """Check 4x4 loop hafnian with zero diagonals"""
        A = random_matrix(4)
        A = A - np.diag(np.diag(A))
        haf = hafnian(A, loop=True)
        expected = A[0, 1] * A[2, 3] + A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    @pytest.mark.parametrize("dtype", [np.complex128, np.float64])
    def test_identity(self, n, dtype):
        """Check loop hafnian(I)=1"""
        A = dtype(np.identity(n))
        haf = hafnian(A, loop=True)
        assert np.allclose(haf, 1)

    @pytest.mark.parametrize("n", [6, 7, 8])
    @pytest.mark.parametrize("dtype", [np.complex128, np.float64])
    def test_ones(self, n, dtype):
        """Check loop hafnian(J_n)=T(n)"""
        A = dtype(np.ones([n, n]))
        haf = hafnian(A, loop=True)
        expected = T[n]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_diag(self, n):
        """Check loophafnian of diagonal matrix is product of diagonals"""
        v = np.random.rand(n)
        A = np.diag(v)
        haf = hafnian(A, loop=True)
        expected = np.prod(v)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 15, 20])
    @pytest.mark.parametrize("fill", [0.5, 0.2, 0.1, 0.05])
    def test_valid_output(self, random_matrix, n, fill):
        """Tests that sparse loop hafnian matches full implementation"""
        A = random_matrix(n, fill_factor=fill)
        assert np.allclose(
            hafnian_sparse(A, loop=True),
            hafnian_sparse(A, D=set(range(len(A))), loop=True),
        )
        assert np.allclose(
            hafnian_sparse(A, loop=False),
            hafnian_sparse(A, D=set(range(len(A))), loop=False),
        )
        assert np.allclose(hafnian(A, loop=True), hafnian_sparse(A, loop=True))
        assert np.allclose(hafnian(A, loop=False), hafnian_sparse(A, loop=False))


@pytest.mark.parametrize("n", [7, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("w", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("loop", [True, False])
def test_hafnian_banded(n, w, loop):
    """Check banded loop hafnian is correct"""
    M = random_banded(n, w)
    result = hafnian_banded(M, loop=loop)
    expected = hafnian(M, loop=loop)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("N", [19, 20])
@pytest.mark.parametrize("bw", [18, 17, 16])
def test_bandwidth(N, bw):
    """Check bandwidth is correct"""
    A = random_banded(N, 3)
    A[0, bw] = 1.0
    A[bw, 0] = 1.0
    result = bandwidth(A)
    expected = bw
    assert np.allclose(result, expected)


@pytest.mark.parametrize("N", [19, 20])
def test_bandwidth_zero(N):
    """Check bandwidth is correct for zero matrix"""
    A = np.zeros([N, N])
    result = bandwidth(A)
    expected = 0
    assert np.allclose(result, expected)


def haf_edge_cases():
    """Tests hafnian and loop hafnian with no repetitions"""
    A = np.ones([2, 2])
    D = np.ones([2])
    assert np.allclose(loop_hafnian(A, D=D, reps=[0, 0]), 1)
    assert np.allclose(jhaf(A, reps=[0, 0]), 1)


def haf_odd_size():
    """Tests hafnian and loop hafnian with no repetitions"""
    A = np.ones([2, 2])
    assert np.allclose(jhaf(A, reps=[1, 2]), 0.0)


def test_empty_matched_reps():
    """Tests that when the repetitions are zero an empty array is returned"""
    reps = [0, 0, 0, 0]
    x, y, z = matched_reps(reps)
    assert x.shape == (0,)
    assert y.shape == (0,)
    assert z is None
