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
"""Tests for the Python permanent repeated wrapper function"""
from math import factorial as fac
import pytest

import numpy as np
from hafnian import permanent_repeated


def test_rpt_zero():
    """Check 2x2 permanent when rpt is all 0"""
    A = np.array([[2, 1], [1, 3]])
    rpt = [0, 0]

    res = permanent_repeated(A, rpt)
    assert res == 1.0

def test_2x2():
    """Check 2x2 permanent"""
    A = np.random.random([2, 2])
    p = permanent_repeated(A, [1]*2)
    assert np.allclose(p, A[0, 0]*A[1, 1]+A[1, 0]*A[0, 1])

def test_3x3():
    """Check 3x3 permanent"""
    A = np.random.random([3, 3])
    p = permanent_repeated(A, [1]*3)
    exp = A[0, 0]*A[1, 1]*A[2, 2]+A[0, 1]*A[1, 2]*A[2, 0]+A[0, 2]*A[1, 0]*A[2, 1] \
        + A[2, 0]*A[1, 1]*A[0, 2]+A[0, 1]*A[1, 0]*A[2, 2]+A[0, 0]*A[1, 2]*A[2, 1]
    assert np.allclose(p, exp)

@pytest.mark.parametrize('n', [6, 8])
def test_ones(n):
    """Check all ones matrix has perm(J_n)=n!"""
    A = np.array([[1]])
    p = permanent_repeated(A, [n])
    assert np.allclose(p, fac(n))
