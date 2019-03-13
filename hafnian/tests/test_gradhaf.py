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
"""Tests for the gradhaf function"""
import pytest

import numpy as np
from scipy.special import factorial2

from hafnian import gradhaf


def test_gradhaf_homogeneous():
    """ Test the graf haf for a homogenoeus matrix for which A_{i,j} = f(x)
    In this cade d haf(A)/dx = (n-1)!! (n/2) (f(x))**(n/2-1) * d f(x)/dx
    where n is the size of A
    """
    n = 10
    f = np.random.rand()
    df = np.random.rand()

    exact_grad = factorial2(n-1)*n/2*(f)**((n/2)-1)*df
    A = f*np.ones([n, n])
    dA = df*np.ones([n, n])
    num_grad = gradhaf(A, dA)

    assert np.allclose(num_grad, exact_grad)


def test_gradhad_2x2():
    """ Test the grad haf for a 2 x 2 matrix """
    da12 = np.random.rand()
    da11 = np.random.rand()
    da22 = np.random.rand()

    A = np.random.rand(2,2)
    A = A + A.T
    dA = np.array([[da11, da12], [da12,da22]])
    num_grad = gradhaf(A, dA)

    assert num_grad == da12
