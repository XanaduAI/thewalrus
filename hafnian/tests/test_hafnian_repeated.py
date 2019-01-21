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
from hafnian import hafnian_repeated
from hafnian.lib.libhaf import haf_rpt_complex, haf_rpt_real


def test_array_exception():
    """Check exception for non-matrix argument"""
    with pytest.raises(TypeError):
        hafnian_repeated(1, [1])


def test_square_exception():
    """Check exception for non-square argument"""
    A = np.zeros([2, 3])
    with pytest.raises(ValueError):
        hafnian_repeated(A, [1]*2)


def test_non_symmetric_exception():
    """Check exception for non-symmetric matrix"""
    A = np.ones([4, 4])
    A[0, 1] = 0.
    with pytest.raises(ValueError):
        hafnian_repeated(A, [1]*4)


def test_nan():
    """Check exception for non-finite matrix"""
    A = np.array([[2, 1], [1, np.nan]])
    with pytest.raises(ValueError):
        hafnian_repeated(A, [1, 1])


def test_rpt_length():
    """Check exception for rpt having incorrect length"""
    A = np.array([[2, 1], [1, 3]])
    with pytest.raises(ValueError):
        hafnian_repeated(A, [1])


def test_rpt_valid():
    """Check exception for rpt having invalid values"""
    A = np.array([[2, 1], [1, 3]])

    with pytest.raises(ValueError):
        hafnian_repeated(A, [1, -1])

    with pytest.raises(ValueError):
        hafnian_repeated(A, [1.1, 1])

def test_rpt_zero():
    """Check 2x2 hafnian when rpt is all 0"""
    A = np.array([[2, 1], [1, 3]])
    rpt = [0, 0]

    res = hafnian_repeated(A, rpt)
    assert res == 1.0

def test_2x2():
    """Check 2x2 hafnian"""
    A = np.random.random([2, 2])
    A += A.T
    haf = hafnian_repeated(A, [1]*2)
    assert np.allclose(haf, A[0, 1])


def test_3x3():
    """Check 3x3 hafnian"""
    A = np.ones([3, 3])
    haf = hafnian_repeated(A, [1]*3)
    assert haf == 0.0


def test_4x4():
    """Check 4x4 hafnian"""
    A = np.random.random([4, 4])
    A += A.T
    haf = hafnian_repeated(A, [1]*4)
    expected = A[0, 1]*A[2, 3] + \
        A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
    assert np.allclose(haf, expected)


def test_real():
    """Check hafnian_repeated(A)=haf_real(A) for a random
    real matrix.
    """
    A = np.random.random([6, 6])
    A += A.T
    haf = hafnian_repeated(A, [1]*6)
    expected = haf_rpt_real(np.float64(A), np.ones([6], dtype=np.int32))
    assert np.allclose(haf, expected)

    A = np.random.random([6, 6])
    A += A.T
    haf = hafnian_repeated(np.complex128(A), [1]*6)
    expected = haf_rpt_real(np.float64(A), np.ones([6], dtype=np.int32))
    assert np.allclose(haf, expected)


def test_complex():
    """Check hafnian_repeated(A)=haf_complex(A) for a random
    real matrix.
    """
    A = np.complex128(np.random.random([6, 6]))
    A += 1j*np.random.random([6, 6])
    A += A.T
    haf = hafnian_repeated(A, [1]*6)
    expected = haf_rpt_complex(np.complex128(A), np.ones([6], dtype=np.int32))
    assert np.allclose(haf, expected)


def test_int():
    """Check hafnian_repeated(A)=haf_int(A) for a random
    real matrix.
    """
    A = np.ones([6, 6])
    haf = hafnian_repeated(A, [1]*6)
    assert isinstance(haf, int)
    expected = haf_rpt_real(np.float64(A), np.ones([6], dtype=np.int32))
    assert np.allclose(haf, expected)
