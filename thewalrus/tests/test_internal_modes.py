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

from scipy.stats import unitary_group

from thewalrus.internal_modes import pnr_prob, distinguishable_pnr_prob

from thewalrus.random import random_covariance
from thewalrus.quantum import density_matrix_element
from thewalrus.symplectic import squeezing, passive_transformation

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

@pytest.mark.parametrize("M", [3,4,5,6])
def test_distinguishable_pnr_prob(M):
    hbar = 2

    pattern = [3,2,0] + [1] * (M - 3)

    mu = np.zeros(2 * M)

    rs = [1] * M
    T = 0.5 * unitary_group.rvs(M)

    big_cov = np.zeros((2*M**2, 2*M**2))
    covs = []
    for i, r in enumerate(rs):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, T)
        covs.append(cov)
        big_cov[i::M,i::M] = cov

    p1 = pnr_prob(covs, pattern, hbar=hbar)
    p2 = pnr_prob(big_cov, pattern, hbar=hbar)
    p3 = distinguishable_pnr_prob(pattern, rs, T)

    assert np.isclose(p1,p2)
    assert np.isclose(p1,p3)
    assert np.isclose(p2,p3)
