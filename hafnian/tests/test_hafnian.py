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
import pytest

import numpy as np
import hafnian as hf
from hafnian import hafnian
from hafnian.lib.libhaf import haf_complex, haf_real, haf_int


def test_version_number():
    """returns true if returns a string"""
    res = hf.version()
    assert isinstance(res, str)


def test_array_exception():
    """Check exception for non-matrix argument"""
    with pytest.raises(TypeError):
        hafnian(1)


def test_square_exception():
    """Check exception for non-square argument"""
    A = np.zeros([2, 3])
    with pytest.raises(ValueError):
        hafnian(A)


def test_odd_dim():
    """Check hafnian for matrix with odd dimensions"""
    A = np.zeros([3, 3])
    assert hafnian(A) == 0


def test_non_symmetric_exception():
    """Check exception for non-symmetric matrix"""
    A = np.ones([4, 4])
    A[0, 1] = 0.
    with pytest.raises(ValueError):
        hafnian(A)


def test_nan():
    """Check exception for non-finite matrix"""
    A = np.array([[2, 1], [1, np.nan]])
    with pytest.raises(ValueError):
        hafnian(A)


def test_2x2():
    """Check 2x2 hafnian"""
    A = np.random.random([2, 2])
    A += A.T
    haf = hafnian(A)
    assert haf == A[0, 1]

    haf = hafnian(A, loop=True)
    assert haf == A[0, 1]+A[0, 0]*A[1, 1]


def test_3x3():
    """Check 3x3 hafnian"""
    A = np.ones([3, 3])
    haf = hafnian(A)
    assert haf == 0.0

    haf = hafnian(A, loop=True)
    assert haf == 4.0


def test_4x4():
    """Check 4x4 hafnian"""
    A = np.random.random([4, 4])
    A += A.T
    haf = hafnian(A)
    expected = A[0, 1]*A[2, 3] + \
        A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
    assert haf == expected

    haf = hafnian(A, loop=True)
    expected = A[0, 1]*A[2, 3] \
        + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2] \
        + A[0, 0]*A[1, 1]*A[2, 3] + A[0, 1]*A[2, 2]*A[3, 3] \
        + A[0, 2]*A[1, 1]*A[3, 3] + A[0, 0]*A[2, 2]*A[1, 3] \
        + A[0, 0]*A[3, 3]*A[1, 2] + A[0, 3]*A[1, 1]*A[2, 2] \
        + A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3]
    assert haf == expected


def test_real():
    """Check hafnian(A)=haf_real(A) for a random
    real matrix.
    """
    A = np.random.random([6, 6])
    A += A.T
    haf = hafnian(A)
    expected = haf_real(A)
    assert np.allclose(haf, expected)

    haf = hafnian(A, loop=True)
    expected = haf_real(A, loop=True)
    assert np.allclose(haf, expected)

    A = np.random.random([6, 6])
    A += A.T
    A = np.array(A, dtype=np.complex128)
    haf = hafnian(A)
    expected = haf_real(np.float64(A.real))
    assert np.allclose(haf, expected)


def test_complex():
    """Check hafnian(A)=haf_complex(A) for a random
    real matrix.
    """
    A = np.complex128(np.random.random([6, 6]))
    A += 1j*np.random.random([6, 6])
    A += A.T
    haf = hafnian(A)
    expected = haf_complex(A)
    assert np.allclose(haf, expected)

    haf = hafnian(A, loop=True)
    expected = haf_complex(A, loop=True)
    assert np.allclose(haf, expected)


def test_int():
    """Check hafnian(A)=haf_int(A) for a random
    real matrix.
    """
    A = np.ones([6, 6])
    haf = hafnian(A)
    expected = haf_int(np.int64(A))
    assert np.allclose(haf, expected)
