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
"""Tests for the recursive_hafnian Python function"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np

from thewalrus import hafnian


@pytest.mark.parametrize("n", [6, 8, 10])
def test_equality(n):
    """Test if recursive_hafnian gives the same as non recursive"""
    A = np.random.rand((n, n))
    A += A.T
    exact = hafnian(A)
    recursive = hafnian(A, recursive = True)
    assert np.allclose(recursive, exact, rtol=2e-1, atol=0)


def test_recursive_or_loop():
    """Check exception raised if chosen loop and recursive"""
    loop = True
    recursive = True
    A = np.random.rand((3,3))
    with pytest.raises(TypeError, match="Recursive algorithm cannot support loop"):
        hafnian(A, recursive = recursive, loop = loop)
