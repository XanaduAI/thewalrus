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
"""Tests for the Python determinant wrapper function"""
import pytest

import numpy as np

# from hafnian import det
from hafnian.lib import libtor


# det_real = libtor.torontonian.det_real
# det_complex = libtor.torontonian.det_complex


# def test_2x2_real():
#     """Check 2x2 determinant"""
#     A = np.random.random([2, 2])
#     d = det(A)
#     assert np.allclose(d, A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0])


# def test_3x3_real():
#     """Check 3x3 determinant"""
#     A = np.random.random([3, 3])
#     d = det(A)
#     expected = -A[0, 2]*A[1, 1]*A[2, 0] + A[0, 1]*A[1, 2]*A[2, 0] \
#         + A[0, 2]*A[1, 0]*A[2, 1] - A[0, 0]*A[1, 2]*A[2, 1] \
#         - A[0, 1]*A[1, 0]*A[2, 2] + A[0, 0]*A[1, 1]*A[2, 2]
#     assert np.allclose(d, expected)


# def test_2x2_complex():
#     """Check 2x2 determinant"""
#     A = np.random.random([2, 2]) + np.random.random([2, 2])*1j
#     d = det(A)
#     assert np.allclose(d, A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0])


# def test_3x3_complex():
#     """Check 3x3 determinant"""
#     A = np.random.random([3, 3]) + np.random.random([3, 3])*1j
#     d = det(A)
#     expected = -A[0, 2]*A[1, 1]*A[2, 0] + A[0, 1]*A[1, 2]*A[2, 0] \
#         + A[0, 2]*A[1, 0]*A[2, 1] - A[0, 0]*A[1, 2]*A[2, 1] \
#         - A[0, 1]*A[1, 0]*A[2, 2] + A[0, 0]*A[1, 1]*A[2, 2]
#     assert np.allclose(d, expected)


# def test_real():
#     """Check determinant(A)=det_real(A) for a random
#     real matrix.
#     """
#     A = np.random.random([6, 6])
#     d = det(A)
#     expected = det_real(A)
#     assert d == expected


# def test_zero_complex():
#     """Check determinant(A)=det_real(A) for a random complex matrix with 0 imaginary part.
#     """
#     A = np.random.random([6, 6])
#     A = np.array(A, dtype=np.complex128)
#     d = det(A)
#     expected = det_real(np.float64(A.real))
#     assert d == expected


# def test_complex():
#     """Check determinant(A)=det_complex(A) for a random matrix.
#     """
#     A = np.complex128(np.random.random([6, 6]))
#     A += 1j*np.random.random([6, 6])
#     d = det(A)
#     expected = det_complex(A)
#     assert np.allclose(d, expected)
