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
"""Tests for The Walrus quantum functions"""
# pylint: disable=no-self-use,redefined-outer-name

import numpy as np
import pytest
from scipy.linalg import block_diag
from thewalrus.quantum import density_matrix, state_vector, probabilities, update_probabilities_with_loss
from thewalrus.symplectic import expand, interferometer, two_mode_squeezing, loss


@pytest.mark.parametrize("hbar", [0.1, 0.5, 1, 2, 1.0/137])
def test_cubic_phase(hbar):
    """Test that all the possible ways of obtaining a cubic phase state using the different methods agree"""
    mu = np.sqrt(hbar/2.0) * np.array([-0.50047867, 0.37373598, 0.01421683, 0.26999427, 0.04450994, 0.01903583])

    cov = (hbar/2.0) * np.array(
        [
            [1.57884241, 0.81035494, 1.03468307, 1.14908791, 0.09179507, -0.11893174],
            [0.81035494, 1.06942863, 0.89359234, 0.20145142, 0.16202296, 0.4578259],
            [1.03468307, 0.89359234, 1.87560498, 0.16915661, 1.0836528, -0.09405278],
            [1.14908791, 0.20145142, 0.16915661, 2.37765137, -0.93543385, -0.6544286],
            [0.09179507, 0.16202296, 1.0836528, -0.93543385, 2.78903152, -0.76519088],
            [-0.11893174, 0.4578259, -0.09405278, -0.6544286, -0.76519088, 1.51724222],
        ]
    )

    cutoff = 7
    # the Fock state measurement of mode 0 to be post-selected
    m1 = 1
    # the Fock state measurement of mode 1 to be post-selected
    m2 = 2

    psi = state_vector(mu, cov, post_select={0: m1, 1: m2}, cutoff=cutoff, hbar=hbar)
    psi_c = state_vector(mu, cov, cutoff=cutoff, hbar=hbar)[m1, m2, :]
    rho = density_matrix(mu, cov, post_select={0: m1, 1: m2}, cutoff=cutoff, hbar=hbar)
    rho_c = density_matrix(mu, cov, cutoff=cutoff, hbar=hbar)[m1, m1, m2, m2, :, :]
    assert np.allclose(np.outer(psi, psi.conj()), rho)
    assert np.allclose(np.outer(psi_c, psi_c.conj()), rho)
    assert np.allclose(rho_c, rho)


@pytest.mark.parametrize("hbar", [2.0, 1.0/137])
def test_four_modes(hbar):
    """ Test that probabilities are correctly updates for a four modes system under loss"""
    # All this block is to generate the correct covariance matrix.
    # It correnponds to num_modes=4 modes that undergo two mode squeezing between modes i and i + (num_modes / 2).
    # Then they undergo displacement.
    # The signal and idlers see and interferometer with unitary matrix u2x2.
    # And then they see loss by amount etas[i].
    num_modes = 4
    theta = 0.45
    phi = 0.7
    u2x2 = np.array([[np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
                    [-np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)]])

    u4x4 = block_diag(u2x2, u2x2)

    cov = np.identity(2 * num_modes) * hbar / 2
    means = 0.5 * np.random.rand(2 * num_modes) * np.sqrt(hbar / 2)
    rs = [0.1, 0.9]
    n_half = num_modes // 2

    for i, r_val in enumerate(rs):
        Sexpanded = expand(two_mode_squeezing(r_val, 0.0), [i, n_half + i], num_modes)
        cov = Sexpanded @ cov @ (Sexpanded.T)

    Su = expand(interferometer(u4x4), range(num_modes), num_modes)
    cov = Su @ cov @ (Su.T)
    cov_lossless = np.copy(cov)
    means_lossless = np.copy(means)
    etas = [0.9, 0.7, 0.9, 0.1]

    for i, eta in enumerate(etas):
        means, cov = loss(means, cov, eta, i, hbar=hbar)

    cutoff = 3
    probs_lossless = probabilities(means_lossless, cov_lossless, 4 * cutoff, hbar=hbar)
    probs = probabilities(means, cov, cutoff, hbar=hbar)
    probs_updated = update_probabilities_with_loss(etas, probs_lossless)
    assert np.allclose(probs, probs_updated[:cutoff, :cutoff, :cutoff, :cutoff], atol=1e-6)
