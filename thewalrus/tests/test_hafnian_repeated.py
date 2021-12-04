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
"""Tests for the lhaf Python function, which calls lhafnian.so"""
# pylint: disable=no-self-use,redefined-outer-name
from math import factorial as fac

import pytest

import numpy as np
from thewalrus import hafnian_repeated
from j_hafnian import haf as jhaf


# the first 11 telephone numbers
T = [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]


class TestHafnianRepeatedWrapper:
    """Tests for the Python hafnian repeated wrapper function.
    These tests should only test for:

    * exceptions
    * validation
    * values computed by the wrapper
    * that the wrapper returns the same
      value as the C++ functions
    """

    def test_array_exception(self):
        """Check exception for non-matrix argument"""
        with pytest.raises(TypeError):
            hafnian_repeated(1, [1])

    def test_square_exception(self):
        """Check exception for non-square argument"""
        A = np.zeros([2, 3])
        with pytest.raises(ValueError):
            hafnian_repeated(A, [1] * 2)

    def test_non_symmetric_exception(self):
        """Check exception for non-symmetric matrix"""
        A = np.ones([4, 4])
        A[0, 1] = 0.0
        with pytest.raises(ValueError):
            hafnian_repeated(A, [1] * 4)

    def test_nan(self):
        """Check exception for non-finite matrix"""
        A = np.array([[2, 1], [1, np.nan]])
        with pytest.raises(ValueError):
            hafnian_repeated(A, [1, 1])

    def test_rpt_length(self):
        """Check exception for rpt having incorrect length"""
        A = np.array([[2, 1], [1, 3]])
        with pytest.raises(ValueError):
            hafnian_repeated(A, [1])

    def test_rpt_valid(self):
        """Check exception for rpt having invalid values"""
        A = np.array([[2, 1], [1, 3]])

        with pytest.raises(ValueError):
            hafnian_repeated(A, [1, -1])

        with pytest.raises(ValueError):
            hafnian_repeated(A, [1.1, 1])

    def test_rpt_zero(self):
        """Check 2x2 hafnian when rpt is all 0"""
        A = np.array([[2, 1], [1, 3]])
        rpt = [0, 0]

        res = hafnian_repeated(A, rpt)
        assert res == 1.0

    def test_3x3(self):
        """Check 3x3 hafnian"""
        A = np.ones([3, 3])
        haf = hafnian_repeated(A, [1] * 3)
        assert haf == 0.0

    def test_real(self):
        """Check hafnian_repeated(A)=haf_real(A) for a random
        real matrix.
        """
        A = np.random.random([6, 6])
        A += A.T
        haf = hafnian_repeated(A, [1] * 6)
        expected = jhaf(np.float64(A), np.ones([6], dtype=np.int32))
        assert np.allclose(haf, expected)

        A = np.random.random([6, 6])
        A += A.T
        haf = hafnian_repeated(np.complex128(A), [1] * 6)
        expected = jhaf(np.float64(A), np.ones([6], dtype=np.int32))
        assert np.allclose(haf, expected)

    def test_complex(self):
        """Check hafnian_repeated(A)=haf_complex(A) for a random
        real matrix.
        """
        A = np.complex128(np.random.random([6, 6]))
        A += 1j * np.random.random([6, 6])
        A += A.T
        haf = hafnian_repeated(A, [1] * 6)
        expected = jhaf(np.complex128(A), np.ones([6], dtype=np.int32))
        assert np.allclose(haf, expected)


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
class TestHafnianRepeated:
    """Various Hafnian repeated consistency checks"""

    def test_2x2(self, random_matrix):
        """Check 2x2 hafnian"""
        A = random_matrix(2)
        rpt = np.ones([2], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        assert np.allclose(haf, A[0, 1])

    def test_2x2_loop(self, random_matrix):
        """Check 2x2 loop hafnian"""
        A = random_matrix(2)
        rpt = np.ones([2], dtype=np.int32)
        haf = hafnian_repeated(A, rpt, loop=True)
        assert np.allclose(haf, A[0, 1] + A[0, 0] * A[1, 1])

    def test_4x4(self, random_matrix):
        """Check 4x4 hafnian"""
        A = random_matrix(4)
        rpt = np.ones([4], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        expected = A[0, 1] * A[2, 3] + A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]
        assert np.allclose(haf, expected)

    def test_4x4_loop(self, random_matrix):
        """Check 4x4 loop hafnian"""
        A = random_matrix(4)
        rpt = np.ones([4], dtype=np.int32)
        haf = hafnian_repeated(A, rpt, loop=True)
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

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n, dtype):
        """Check hafnian(I)=0"""
        A = dtype(np.identity(n))
        rpt = np.ones([n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        assert np.allclose(haf, 0)

        haf = hafnian_repeated(A, rpt, loop=True)
        assert np.allclose(haf, 1)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n, dtype):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = dtype(np.ones([2 * n, 2 * n]))
        rpt = np.ones([2 * n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        expected = fac(2 * n) / (fac(n) * (2 ** n))
        assert np.allclose(haf, expected)

        A = dtype([[1]])
        rpt = [2 * n]
        haf = hafnian_repeated(A, rpt)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 7, 8])
    def test_ones_loop(self, n, dtype):
        """Check loop hafnian(J_n)=T(n)"""
        A = dtype(np.ones([n, n]))
        rpt = np.ones([n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt, loop=True)
        expected = T[n]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n, dtype):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]), np.hstack([B, O])])
        A = dtype(A)
        rpt = np.ones([2 * n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        expected = float(fac(n))
        assert np.allclose(haf, expected)

        A = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rpt = np.array([n, n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [3, 5])
    def test_outer_product(self, n, dtype):
        r"""Check that hafnian(x \otimes x) = hafnian(J_2n)*prod(x)"""
        x = np.random.rand(2 * n) + 1j * np.random.rand(2 * n)

        if not np.iscomplex(dtype()):
            x = x.real

        x = dtype(x)
        A = np.outer(x, x)

        rpt = np.ones([2 * n], dtype=np.int32)
        haf = hafnian_repeated(A, rpt)
        expected = np.prod(x) * fac(2 * n) / (fac(n) * (2 ** n))
        assert np.allclose(haf, expected)
