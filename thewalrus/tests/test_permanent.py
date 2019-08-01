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
import pytest

import numpy as np
from scipy.special import factorial as fac

from thewalrus import perm, perm_real, perm_complex, permanent_repeated


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
        p = perm(A)
        assert p == A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]

    def test_3x3(self, random_matrix):
        """Check 3x3 permanent"""
        A = random_matrix(3)
        p = perm(A)
        expected = (
            A[0, 2] * A[1, 1] * A[2, 0]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[0, 0] * A[1, 2] * A[2, 1]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2]
        )
        assert p == expected

    @pytest.mark.parametrize("dtype", [np.float64])
    def test_real(self, random_matrix):
        """Check permanent(A)=perm_real(A) for a random
        real matrix.
        """
        A = random_matrix(6)
        p = perm(A)
        expected = perm_real(A)
        assert p == expected

        A = random_matrix(6)
        A = np.array(A, dtype=np.complex128)
        p = perm(A)
        expected = perm_real(np.float64(A.real))
        assert p == expected

    @pytest.mark.parametrize("dtype", [np.complex128])
    def test_complex(self, random_matrix):
        """Check perm(A)=perm_complex(A) for a random matrix.
        """
        A = random_matrix(6)
        p = perm(A)
        expected = perm_complex(A)
        assert np.allclose(p, expected)

    @pytest.mark.parametrize("dtype", [np.float64])
    def test_complex_no_imag(self, random_matrix):
        """Check perm(A)=perm_real(A) for a complex random matrix
        with zero imaginary parts.
        """
        A = np.complex128(random_matrix(6))
        p = perm(A)
        expected = perm_real(A.real)
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
