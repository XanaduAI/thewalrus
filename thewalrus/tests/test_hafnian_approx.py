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
"""Tests for the hafnian_approx Python function"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np
from scipy.special import factorial2, factorial as fac

from thewalrus import hafnian, haf_real


@pytest.mark.parametrize("n", [6, 8, 10])
def test_rank_one(n):
    """ Test the hafnian of rank one matrices so that it is within
    10% of the exact value """
    x = np.random.rand(n)
    A = np.outer(x, x)
    exact = factorial2(n - 1) * np.prod(x)
    approx = haf_real(A, approx=True, nsamples=10000)
    assert np.allclose(approx, exact, rtol=2e-1, atol=0)


def test_approx_complex_error():
    """Check exception raised if matrix is complex"""
    A = 1j * np.ones([6, 6])
    with pytest.raises(ValueError, match="Input matrix must be real"):
        hafnian(A, approx=True)


def test_approx_negative_error():
    """Check exception raised if matrix is negative"""
    A = np.ones([6, 6])
    A[0, 0] = -1
    with pytest.raises(ValueError, match="Input matrix must not have negative entries"):
        hafnian(A, approx=True)


@pytest.mark.parametrize("n", [6, 8])
def test_ones_approx(n):
    """Check hafnian_approx(J_2n)=(2n)!/(n!2^n)"""
    A = np.float64(np.ones([2 * n, 2 * n]))
    haf = hafnian(A, approx=True, num_samples=1e4)
    expected = fac(2 * n) / (fac(n) * (2 ** n))
    assert np.abs(haf - expected) / expected < 0.15
