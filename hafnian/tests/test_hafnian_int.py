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
"""Tests for the haf_int Python function, which calls libhaf.so"""
from math import factorial as fac
import pytest

import numpy as np
from hafnian.lib.libhaf import haf_int as hafnian


class TestIntHaf:
    """Various Hafnian consistency checks"""

    def test_2x2(self):
        """Check 2x2 hafnian"""
        A = np.int64(np.random.random([2, 2]))
        A = A + A.T
        haf = hafnian(A)
        assert np.allclose(haf, A[0, 1])

    def test_4x4(self):
        """Check 4x4 hafnian"""
        A = np.int64(np.random.random([4, 4]))
        A += A.T
        haf = hafnian(A)
        expected = A[0, 1]*A[2, 3] + \
            A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_identity(self, n):
        """Check hafnian(I)=0"""
        A = np.identity(n, dtype=np.int64)
        haf = hafnian(A)
        assert haf == 0

    @pytest.mark.parametrize("n", [6, 8])
    def test_ones(self, n):
        """Check hafnian(J_2n)=(2n)!/(n!2^n)"""
        A = np.int64(np.ones([2*n, 2*n]))
        haf = hafnian(A)
        expected = fac(2*n)/(fac(n)*(2**n))
        assert np.allclose(haf, expected)

    @pytest.mark.parametrize("n", [6, 8])
    def test_block_ones(self, n):
        """Check hafnian([[0, I_n], [I_n, 0]])=n!"""
        O = np.zeros([n, n], dtype=np.int64)
        B = np.ones([n, n], dtype=np.int64)
        A = np.vstack([np.hstack([O, B]),
                       np.hstack([B, O])])
        haf = hafnian(A)
        expected = fac(n)
        assert haf == expected
