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
"""Tests for the Python pnian wrapper function"""
import pytest

import numpy as np

from hafnian import perm
from hafnian.lib import libperm

perm_real = libperm.perm.re
perm_complex = libperm.perm.comp


def test_array_exception():
    """Check exception for non-matrix argument"""
    with pytest.raises(TypeError):
        perm(1)


def test_square_exception():
    """Check exception for non-square argument"""
    A = np.zeros([2, 3])
    with pytest.raises(ValueError):
        perm(A)


def test_nan():
    """Check exception for non-finite matrix"""
    A = np.array([[2, 1], [1, np.nan]])
    with pytest.raises(ValueError):
        perm(A)


def test_2x2():
    """Check 2x2 permanent"""
    A = np.random.random([2, 2])
    A += A.T
    p = perm(A)
    assert p == A[0, 0]*A[1, 1]+A[0, 1]*A[1, 0]


def test_3x3():
    """Check 3x3 permanent"""
    A = np.random.random([3, 3])
    A += A.T
    p = perm(A)
    expected = A[0, 2]*A[1, 1]*A[2, 0] + A[0, 1]*A[1, 2]*A[2, 0] \
        + A[0, 2]*A[1, 0]*A[2, 1] + A[0, 0]*A[1, 2]*A[2, 1] \
        +  A[0, 1]*A[1, 0]*A[2, 2] + A[0, 0]*A[1, 1]*A[2, 2]
    assert p == expected


def test_real():
    """Check permanent(A)=perm_real(A) for a random
    real matrix.
    """
    A = np.random.random([6, 6])
    A += A.T
    p = perm(A)
    expected = perm_real(A)
    assert p == expected

    A = np.random.random([6, 6])
    A += A.T
    A = np.array(A, dtype=np.complex128)
    p = perm(A)
    expected = perm_real(np.float64(A.real))
    assert p == expected

def test_complex():
    """Check perm(A)=perm_complex(A) for a random matrix.
    """
    A = np.complex128(np.random.random([6, 6]))
    A += 1j*np.random.random([6, 6])
    A += A.T
    p = perm(A)
    expected = perm_complex(A)
    assert np.allclose(p, expected)
