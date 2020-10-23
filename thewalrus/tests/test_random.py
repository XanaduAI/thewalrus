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
"""Random tests"""
import pytest

import numpy as np
from thewalrus.random import random_block_interferometer, random_banded_interferometer


def bandwidth(A):
    """Calculates the upper bandwidth of the matrix A.

    Args:
        A (array): input matrix

    Returns:
        (int): bandwidth of matrix
    """
    n, _ = A.shape
    for i in range(n):
        vali = np.diag(A, i)
        if np.allclose(vali, 0):
            return i - 1
    return n - 1


@pytest.mark.parametrize("n", [5, 7, 8, 9])
@pytest.mark.parametrize("top_one", [True, False])
@pytest.mark.parametrize("real", [True, False])
def test_random_block(n, top_one, real):
    """Test that random_block_interferometer produces a unitary with the right structure."""
    U = random_block_interferometer(n, top_one=top_one, real=real)
    assert np.allclose(U @ U.T.conj(), np.identity(n))
    if top_one:
        assert np.allclose(U[0, 1], 0)
        assert np.allclose(U[1, 0], 0)


@pytest.mark.parametrize("n", [5, 7, 8, 9])
@pytest.mark.parametrize("top_one_init", [True, False])
@pytest.mark.parametrize("real", [True, False])
def test_random_banded(n, top_one_init, real):
    """Test that random_banded_interferometer produces a unitary with the right structure."""
    for w in range(n):
        U = random_banded_interferometer(n, w, top_one_init=top_one_init, real=real)
        assert np.allclose(U @ U.T.conj(), np.identity(n))
        assert bandwidth(U) == w

def test_wrong_bandwidth():
    """Test that the correct error is raised if w > n-1."""
    n = 10
    w = 10
    with pytest.raises(ValueError, match="The bandwidth can be at most one minus the size of the matrix."):
        random_banded_interferometer(n, w)
