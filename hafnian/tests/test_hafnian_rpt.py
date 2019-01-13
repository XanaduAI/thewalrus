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
from math import factorial as fac

import pytest

import numpy as np
from hafnian.lib.libhaf import haf_rpt_complex, haf_rpt_real


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


@pytest.mark.parametrize("eigen", [True, False])
class TestComplexHafRpt:
    """Various Hafnian consistency checks"""

    def test_2x2(self, eigen):
        """Check 2x2 hafnian"""
        A = np.complex128(np.random.random([2, 2])) + 1j*np.random.random([2, 2])
        A = A + A.T
        rpt = np.ones([2], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, A[0, 1])

    def test_4x4(self, eigen):
        """Check 4x4 hafnian"""
        A = np.complex128(np.random.random([4, 4]))
        A += 1j*np.random.random([4, 4])
        A += A.T
        rpt = np.ones([4], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n, eigen):
        """Check hafnian(I)=0"""
        A = np.complex128(np.identity(n))
        rpt = np.ones([n], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, 0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n, eigen):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.complex128(np.ones([2*n, 2*n]))
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        expected = fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)

        A = np.complex128([[1]])
        rpt = np.int32([2*n])
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n, eigen):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        A = np.complex128(A)
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        expected = float(fac(n))
        assert np.allclose(haf, expected)

        A = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rpt = np.array([n, n], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [3, 5])
    def test_outer_product(self, n, eigen):
        r"""Check that hafnian(x \otimes x) = hafnian(J_2n)*prod(x)"""
        x = np.random.rand(2*n)+1j*np.random.rand(2*n)
        A = np.outer(x, x)
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_complex(A, rpt, use_eigen=eigen)
        expected = np.prod(x)*fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)


@pytest.mark.parametrize("eigen", [True, False])
class TestRealHafRpt:
    """Various Hafnian consistency checks"""

    def test_2x2(self, eigen):
        """Check 2x2 hafnian"""
        A = np.float64(np.random.random([2, 2]))
        A = A + A.T
        rpt = np.ones([2], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, A[0, 1])

    def test_4x4(self, eigen):
        """Check 4x4 hafnian"""
        A = np.float64(np.random.random([4, 4]))
        A += A.T
        rpt = np.ones([4], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n, eigen):
        """Check hafnian(I)=0"""
        A = np.float64(np.identity(n))
        rpt = np.ones([n], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, 0)

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n, eigen):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.float64(np.ones([2*n, 2*n]))
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        expected = fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)

        A = np.float64([[1]])
        rpt = np.int32([2*n])
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n, eigen):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n])
        B = np.ones([n, n])
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        A = np.float64(A)
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        expected = float(fac(n))
        assert np.allclose(haf, expected)

        A = np.array([[0, 1], [1, 0]], dtype=np.float64)
        rpt = np.array([n, n], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [3, 5])
    def test_outer_product(self, n, eigen):
        r"""Check that hafnian(x \otimes x) = hafnian(J_2n)*prod(x)"""
        x = np.random.rand(2*n)
        A = np.outer(x, x)
        rpt = np.ones([2*n], dtype=np.int32)
        haf = haf_rpt_real(A, rpt, use_eigen=eigen)
        expected = np.prod(x)*fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)
