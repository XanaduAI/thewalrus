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
"""Tests for the Python reference hafnian functions"""
# pylint: disable=redefined-outer-name
import pytest
import numpy as np

np.random.seed(137)

# defaults
TOL = 1e-3

@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return TOL

@pytest.fixture(params=[0.5, 1, 2])
def hbar(request):
    """The value of hbar to use in tests"""
    return request.param

@pytest.fixture(params=[np.complex128, np.float64, np.int64])
def dtype(request):
    """Fixture that iterates through all numpy types"""
    return request.param

@pytest.fixture
def random_matrix(dtype):
    """Returns a random symmetric matrix of type dtype
    and of size n x n"""

    def _wrapper(n):
        """wrapper function"""
        A = np.complex128(np.random.random([n, n]))
        A += 1j * np.random.random([n, n])
        A += A.T

        if not np.issubdtype(dtype, np.complexfloating):
            A = A.real

        return dtype(A)

    return _wrapper
