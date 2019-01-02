# Copyright 2018 Xanadu Quantum Technologies Inc.

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
from math import factorial as fac

import pytest

import numpy as np
from hafnian.lib.libhaf import haf_complex as hafnian


hyp1f1 = {
    1: 2.0,
    2: 3.3333333333333335,
    3: 5.0666666666666664,
    4: 7.2761904761904761,
    5: 10.048677248677249,
    6: 13.482635882635883,
    7: 17.689569689569687,
    8: 22.795345888679229,
    9: 28.941685010704621,
    10: 36.287779179502607,
    11: 45.012048825000804,
    12: 55.314048000042611,
    13: 67.416529128084463,
    14: 81.567678700572273,
    15: 98.043536174028574,
    16: 117.1506090894388,
    17: 139.22869825254688,
    18: 164.65394767535315,
    19: 193.84213488876887,
    20: 227.25221819642439
    }


class TestComplexHaf:
    """Various Hafnian consistency checks"""

    def test_2x2(self):
        """Check 2x2 hafnian"""
        A = np.complex128(np.random.random([2, 2])) + 1j*np.random.random([2, 2])
        A = A + A.T
        haf = hafnian(A, recursive=False)
        assert np.allclose(haf, A[0, 1])

    def test_4x4(self):
        """Check 4x4 hafnian"""
        A = np.complex128(np.random.random([4, 4]))
        A += 1j*np.random.random([4, 4])
        A += A.T
        haf = hafnian(A, recursive=False)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n):
        """Check hafnian(I)=0"""
        A = np.complex128(np.identity(n))
        haf = hafnian(A, recursive=False)
        assert np.allclose(haf, 0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.complex128(np.ones([2*n, 2*n]))
        haf = hafnian(A, recursive=False)
        expected = fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        A = np.complex128(A)
        haf = hafnian(A, recursive=False)
        expected = float(fac(n))
        assert np.allclose(haf, expected)


class TestComplexHafRecursive:
    """Various Hafnian consistency checks"""

    def test_2x2(self):
        """Check 2x2 hafnian"""
        A = np.complex128(np.random.random([2, 2])) + 1j*np.random.random([2, 2])
        A = A + A.T
        haf = hafnian(A, recursive=True)
        assert np.allclose(haf, A[0, 1])

    def test_4x4(self):
        """Check 4x4 hafnian"""
        A = np.complex128(np.random.random([4, 4]))
        A += 1j*np.random.random([4, 4])
        A += A.T
        haf = hafnian(A, recursive=True)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n):
        """Check hafnian(I)=0"""
        A = np.complex128(np.identity(n))
        haf = hafnian(A, recursive=True)
        assert np.allclose(haf, 0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.complex128(np.ones([2*n, 2*n]))
        haf = hafnian(A, recursive=True)
        expected = fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        A = np.complex128(A)
        haf = hafnian(A, recursive=True)
        expected = float(fac(n))
        assert np.allclose(haf, expected)


class TestComplexHafLoops:
    """Various Hafnian consistency checks"""

    def test_2x2(self):
        """Check 2x2 loop hafnian"""
        A = np.complex128(np.random.random([2, 2])) + 1j*np.random.random([2, 2])
        A = A + A.T
        haf = hafnian(A, loop=True)
        assert np.allclose(haf, A[0, 1]+A[0, 0]*A[1, 1])

    def test_4x4(self):
        """Check 4x4 loop hafnian"""
        A = np.complex128(np.random.random([4, 4]))
        A += 1j*np.random.random([4, 4])
        A += A.T
        haf = hafnian(A, loop=True)
        expected = A[0, 1]*A[2, 3] \
            + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2] \
            + A[0, 0]*A[1, 1]*A[2, 3] + A[0, 1]*A[2, 2]*A[3, 3] \
            + A[0, 2]*A[1, 1]*A[3, 3] + A[0, 0]*A[2, 2]*A[1, 3] \
            + A[0, 0]*A[3, 3]*A[1, 2] + A[0, 3]*A[1, 1]*A[2, 2] \
            + A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3]
        assert np.allclose(haf, expected)

    def test_4x4_zero_diag(self):
        """Check 4x4 loop hafnian with zero diagonals"""
        A = np.complex128(np.random.random([4, 4]))
        A += 1j*np.random.random([4, 4])
        A += A.T
        A -= np.diag(np.diag(A))
        haf = hafnian(A, loop=True)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n):
        """Check loop hafnian(I)=1"""
        A = np.complex128(np.identity(n))
        haf = hafnian(A, loop=True)
        assert np.allclose(haf, 1)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n):
        """Check loop hafnian(J_2n)=hyp1f1(-2n/2,1/2,-1/2)*(2n)!/(n!2^n)"""
        A = np.complex128(np.ones([2*n, 2*n]))
        haf = hafnian(A, loop=True)
        expected = fac(2*n)/(fac(n)*(2**n))*hyp1f1[n]
        assert np.allclose(haf, expected)
