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
"""Tests for the low_rank_hafnian function"""

import pytest
import numpy as np
from thewalrus import low_rank_hafnian, hafnian
from thewalrus._low_rank_haf import partitions
from scipy.special import binom


@pytest.mark.parametrize("n", [9, 11, 13])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_number_of_partitions(n, r):
    r"""Test that the number of r-partitions of n elements is {n + r - 1 \choose r - 1}"""
    parts = partitions(r, n)
    assert np.allclose(len(parts), binom(n + r - 1, r - 1))


@pytest.mark.parametrize("n", [9, 11, 13])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_partitions_correctness(n, r):
    """Check that the r-partitions of n do sum up to n"""
    part = partitions(r, n)
    sums = list(map(sum, part))
    assert np.allclose(sums, n)


@pytest.mark.parametrize("n", [9, 11, 13])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_odd_n(n, r):
    """Test that if n is odd one gets zero"""
    G = np.random.rand(n, r) + 1j * np.random.rand(n, r)
    haf = low_rank_hafnian(G)
    assert np.allclose(haf, 0)


@pytest.mark.parametrize("n", [8, 10, 12])
@pytest.mark.parametrize("r", [1, 2, 3])
def test_rank_r(r, n):
    """Test rank-r matrices"""
    G = np.random.rand(n, r) + 1j * np.random.rand(n, r)
    A = G @ G.T
    haf = low_rank_hafnian(G)
    expected = hafnian(A)
    assert np.allclose(haf, expected)
