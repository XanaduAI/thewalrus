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
from thewalrus.quantum import (
    density_matrix,
    state_vector,
    probabilities,
    update_probabilities_with_loss,
    photon_number_cumulant,
)
from thewalrus.symplectic import expand, interferometer, two_mode_squeezing, loss, squeezing
from thewalrus.random import random_interferometer


@pytest.mark.parametrize("hbar", [0.1, 0.5, 1, 2, 1.0 / 137])
def test_cubic_phase(hbar):
    """Test that all the possible ways of obtaining a cubic phase state using the different methods agree"""
    mu = np.sqrt(hbar / 2.0) * np.array(
        [-0.50047867, 0.37373598, 0.01421683, 0.26999427, 0.04450994, 0.01903583]
    )

    cov = (hbar / 2.0) * np.array(
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


@pytest.mark.parametrize("hbar", [2.0, 1.0 / 137])
def test_four_modes(hbar):
    """Test that probabilities are correctly updates for a four modes system under loss"""
    # All this block is to generate the correct covariance matrix.
    # It correnponds to num_modes=4 modes that undergo two mode squeezing between modes i and i + (num_modes / 2).
    # Then they undergo displacement.
    # The signal and idlers see and interferometer with unitary matrix u2x2.
    # And then they see loss by amount etas[i].
    num_modes = 4
    theta = 0.45
    phi = 0.7
    u2x2 = np.array(
        [
            [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
            [-np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )

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


@pytest.mark.parametrize("hbar", [0.5, 1.0, 1.7, 2.0])
def test_cumulants_three_mode_random_state(hbar):  # pylint: disable=too-many-statements
    """Tests third order cumulants for a random state"""
    M = 3
    O = interferometer(random_interferometer(3))
    mu = np.random.rand(2 * M) - 0.5
    hbar = 2
    cov = 0.5 * hbar * O @ squeezing(np.random.rand(M)) @ O.T
    cutoff = 50
    probs = probabilities(mu, cov, cutoff, hbar=hbar)
    n = np.arange(cutoff)
    probs0 = np.sum(probs, axis=(1, 2))
    probs1 = np.sum(probs, axis=(0, 2))
    probs2 = np.sum(probs, axis=(0, 1))

    # Check one body cumulants
    n0_1 = n @ probs0
    n1_1 = n @ probs1
    n2_1 = n @ probs2
    assert np.allclose(photon_number_cumulant(mu, cov, [0], hbar=hbar), n0_1)
    assert np.allclose(photon_number_cumulant(mu, cov, [1], hbar=hbar), n1_1)
    assert np.allclose(photon_number_cumulant(mu, cov, [2], hbar=hbar), n2_1)

    n0_2 = n ** 2 @ probs0
    n1_2 = n ** 2 @ probs1
    n2_2 = n ** 2 @ probs2
    var0 = n0_2 - n0_1 ** 2
    var1 = n1_2 - n1_1 ** 2
    var2 = n2_2 - n2_1 ** 2
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 0], hbar=hbar), var0)
    assert np.allclose(photon_number_cumulant(mu, cov, [1, 1], hbar=hbar), var1)
    assert np.allclose(photon_number_cumulant(mu, cov, [2, 2], hbar=hbar), var2)

    n0_3 = n ** 3 @ probs0 - 3 * n0_2 * n0_1 + 2 * n0_1 ** 3
    n1_3 = n ** 3 @ probs1 - 3 * n1_2 * n1_1 + 2 * n1_1 ** 3
    n2_3 = n ** 3 @ probs2 - 3 * n2_2 * n2_1 + 2 * n2_1 ** 3
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 0, 0], hbar=hbar), n0_3)
    assert np.allclose(photon_number_cumulant(mu, cov, [1, 1, 1], hbar=hbar), n1_3)
    assert np.allclose(photon_number_cumulant(mu, cov, [2, 2, 2], hbar=hbar), n2_3)

    # Check two body cumulants
    probs01 = np.sum(probs, axis=(2))
    probs02 = np.sum(probs, axis=(1))
    probs12 = np.sum(probs, axis=(0))

    n0n1 = n @ probs01 @ n
    n0n2 = n @ probs02 @ n
    n1n2 = n @ probs12 @ n
    covar01 = n0n1 - n0_1 * n1_1
    covar02 = n0n2 - n0_1 * n2_1
    covar12 = n1n2 - n1_1 * n2_1

    assert np.allclose(photon_number_cumulant(mu, cov, [0, 1], hbar=hbar), covar01)
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 2], hbar=hbar), covar02)
    assert np.allclose(photon_number_cumulant(mu, cov, [1, 2], hbar=hbar), covar12)

    kappa001 = n ** 2 @ probs01 @ n - 2 * n0n1 * n0_1 - n0_2 * n1_1 + 2 * n0_1 ** 2 * n1_1
    kappa011 = n @ probs01 @ n ** 2 - 2 * n0n1 * n1_1 - n1_2 * n0_1 + 2 * n1_1 ** 2 * n0_1
    kappa002 = n ** 2 @ probs02 @ n - 2 * n0n2 * n0_1 - n0_2 * n2_1 + 2 * n0_1 ** 2 * n2_1
    kappa022 = n @ probs02 @ n ** 2 - 2 * n0n2 * n2_1 - n2_2 * n0_1 + 2 * n2_1 ** 2 * n0_1
    kappa112 = n ** 2 @ probs12 @ n - 2 * n1n2 * n1_1 - n1_2 * n2_1 + 2 * n1_1 ** 2 * n2_1
    kappa122 = n @ probs12 @ n ** 2 - 2 * n1n2 * n2_1 - n2_2 * n1_1 + 2 * n2_1 ** 2 * n1_1

    assert np.allclose(photon_number_cumulant(mu, cov, [0, 0, 1], hbar=hbar), kappa001)
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 1, 1], hbar=hbar), kappa011)
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 0, 2], hbar=hbar), kappa002)
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 2, 2], hbar=hbar), kappa022)
    assert np.allclose(photon_number_cumulant(mu, cov, [1, 1, 2], hbar=hbar), kappa112)
    assert np.allclose(photon_number_cumulant(mu, cov, [1, 2, 2], hbar=hbar), kappa122)

    # Finally, the three body cumulant
    n0n1n2 = np.einsum("ijk, i, j, k", probs, n, n, n)
    kappa012 = n0n1n2 - n0n1 * n2_1 - n0n2 * n1_1 - n1n2 * n0_1 + 2 * n0_1 * n1_1 * n2_1
    print(kappa012, photon_number_cumulant(mu, cov, [0, 1, 2], hbar=hbar))
    assert np.allclose(photon_number_cumulant(mu, cov, [0, 1, 2], hbar=hbar), kappa012)
