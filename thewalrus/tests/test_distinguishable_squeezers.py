# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distinguishable squeezers tests"""

import numpy as np
import pytest
from scipy.stats import unitary_group

from thewalrus.quantum.distinguishable_squeezers import number_cov, number_means, sample


@pytest.mark.parametrize("M", range(2, 8, 2))
def test_moments_of_distinguishable_distribution(M):
    """
    Test that means and covariance matrices calculated from samples match those calculated directly
    """
    T = (unitary_group.rvs(M) * np.random.rand(M)) @ unitary_group.rvs(M)
    rs = np.random.rand(M)
    num_samples = 100000
    samples = sample(rs, T, n_samples=num_samples)
    means = samples.mean(axis=0)
    cov = np.cov(samples.T)
    expected_means = number_means(T, rs)
    expected_cov = number_cov(T, rs)
    assert np.allclose(expected_means, means, atol=4 / np.sqrt(num_samples))
    assert np.allclose(expected_cov, cov, atol=4 / np.sqrt(num_samples))
