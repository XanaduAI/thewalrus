# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
tests for code in thewalrus.internal_modes
"""

import pytest

import numpy as np 

from thewalrus.random import random_covariance
from thewalrus.internal_modes import pnr_prob
from thewalrus.quantum import density_matrix_element

@pytest.mark.parametrize("M", [3,4,5,6])
def test_pnr_prob_single_internal_mode(M):
    """
    test internal modes functionality against standard method for pnr probabilities
    """

    cov = random_covariance(M)
    mu = np.zeros(2 * M)

    pattern = [2,3,0] + [1] * (M - 3)
        
    p1 = pnr_prob(cov, pattern)
    p2 = density_matrix_element(mu, cov, pattern, pattern).real
    
    assert np.isclose(p1, p2)
