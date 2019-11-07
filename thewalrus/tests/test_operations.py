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
"""Operations tests"""
# pylint: disable=no-self-use, assignment-from-no-return
import pytest

import numpy as np

from thewalrus import operations
from thewalrus.symplectic import rotation, squeezing, interferometer

# make test deterministic
np.random.seed(137)


@pytest.mark.parametrize("cutoff", [4, 5, 6, 7])
def test_single_mode_identity(cutoff, tol):
    """Tests the correct construction of the single mode identity operation"""
    nmodes = 1
    S = np.identity(2 * nmodes)
    alphas = np.zeros([nmodes])
    T = operations.fock_tensor(S, alphas, cutoff)
    expected = np.identity(cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("cutoff", [7])
def test_single_mode_rotation(cutoff, tol):
    """Tests the correct construction of the single mode rotation operation"""
    nmodes = 1
    theta = 2 * np.pi * np.random.rand()
    S = rotation(theta)
    alphas = np.zeros([nmodes])
    T = operations.fock_tensor(S, alphas, cutoff)
    expected = np.diag(np.exp(1j * theta * np.arange(cutoff)))
    assert np.allclose(T, expected, atol=tol, rtol=0)


def test_single_mode_displacement(tol):
    """Tests the correct construction of the single mode displacement operation"""
    nmodes = 1
    cutoff = 5
    alphas = (0.3 + 0.5 * 1j) * np.ones([nmodes])
    S = np.identity(2 * nmodes)
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.84366482 + 0.00000000e00j, -0.25309944 + 4.21832408e-01j, -0.09544978 - 1.78968334e-01j, 0.06819609 + 3.44424719e-03j, -0.01109048 + 1.65323865e-02j],
            [0.25309944 + 4.21832408e-01j, 0.55681878 + 0.00000000e00j, -0.29708743 + 4.95145724e-01j, -0.14658716 - 2.74850926e-01j, 0.12479885 + 6.30297236e-03j],
            [-0.09544978 + 1.78968334e-01j, 0.29708743 + 4.95145724e-01j, 0.31873657 + 0.00000000e00j, -0.29777767 + 4.96296112e-01j, -0.18306015 - 3.43237787e-01j],
            [-0.06819609 + 3.44424719e-03j, -0.14658716 + 2.74850926e-01j, 0.29777767 + 4.96296112e-01j, 0.12389162 + 1.10385981e-17j, -0.27646677 + 4.60777945e-01j],
            [-0.01109048 - 1.65323865e-02j, -0.12479885 + 6.30297236e-03j, -0.18306015 + 3.43237787e-01j, 0.27646677 + 4.60777945e-01j, -0.03277289 + 1.88440656e-17j],
        ]
    )
    T = operations.fock_tensor(S, alphas, cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


def test_single_mode_squeezing(tol):
    """Tests the correct construction of the single mode squeezing operation"""
    nmodes = 1
    r = 1.0
    cutoff = 5
    S = squeezing(r, 0.0)
    alphas = np.zeros([nmodes])
    # This data is obtained by using qutip
    # np.array(squeeze(40,r).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.80501818 + 0.0j, 0.0 + 0.0j, 0.43352515 + 0.0j, 0.0 + 0.0j, 0.2859358 + 0.0j],
            [0.0 + 0.0j, 0.52169547 + 0.0j, 0.0 + 0.0j, 0.48661591 + 0.0j, 0.0 + 0.0j],
            [-0.43352515 + 0.0j, 0.0 + 0.0j, 0.10462138 + 0.0j, 0.0 + 0.0j, 0.29199268 + 0.0j],
            [0.0 + 0.0j, -0.48661591 + 0.0j, 0.0 + 0.0j, -0.23479643 + 0.0j, 0.0 + 0.0j],
            [0.2859358 + 0.0j, 0.0 + 0.0j, -0.29199268 + 0.0j, 0.0 + 0.0j, -0.34474749 + 0.0j],
        ]
    )
    T = operations.fock_tensor(S, alphas, cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


def test_single_mode_displacement_squeezing(tol):
    """Tests the correct construction of the single mode squeezing operation followed by the single mode displacement operation"""
    nmodes = 1
    r = 1.0
    cutoff = 5
    S = squeezing(r, 0.0)
    alphas = (0.5 + 0.4 * 1j) * np.ones([nmodes])
    # This data is obtained by using qutip
    # np.array((displace(40,alpha)*squeeze(40,r)).data.todense())[0:10,0:10]
    expected = np.array(
        [
            [0.6263739 + 0.09615331j, -0.22788717 + 0.13121343j, 0.36548296 - 0.0200537j, -0.20708137 + 0.14004403j, 0.25645667 - 0.06275564j],
            [0.5425389 + 0.14442404j, 0.19268911 + 0.15615312j, 0.11497303 + 0.13744549j, 0.21448948 + 0.08109308j, -0.03652914 + 0.15069359j],
            [-0.00915607 + 0.07475267j, 0.48081922 + 0.10576742j, -0.00961086 + 0.20535144j, 0.33089303 + 0.09864247j, 0.02555522 + 0.19950786j],
            [-0.34614367 - 0.05229875j, 0.11543956 + 0.01112537j, 0.16672961 + 0.07439407j, 0.02323121 + 0.15103267j, 0.27233637 + 0.08297028j],
            [-0.14390852 - 0.08884069j, -0.37898007 - 0.07630228j, 0.12911863 - 0.08963054j, -0.12164023 + 0.04431394j, 0.1141808 + 0.01581529j],
        ]
    )
    T = operations.fock_tensor(S, alphas, cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("cutoff", [5])
def test_hong_ou_mandel_interference(cutoff, tol):
    """Tests the the properties of a 50-50 beamsplitter"""
    nmodes = 2
    U = np.sqrt(0.5) * np.array([[1, 1j], [1j, 1]])
    alphas = np.zeros([nmodes])
    S = interferometer(U)
    T = operations.fock_tensor(S, alphas, cutoff)

    items = []
    for i in range(cutoff):
        for j in range(i + 1):
            items.append([i - j, j])

    mat = np.empty([len(items), len(items)], dtype=np.complex128)
    for i, indi in enumerate(items):
        for j, indj in enumerate(items):
            mat[i, j] = T[tuple(indj + indi)]
            if sum(indj) != sum(indi):
                assert np.allclose(T[tuple(indj + indi)], 0)
                # Checking that different excitation manifolds are not mixed
            if indj == [1, 1] and indi == [1, 1]:
                assert np.allclose(T[tuple(indj + indi)], 0)
                # This is Hong-Ou-Mandel interference
            if indj == [0, 0] and indi == [0, 0]:
                assert np.allclose(T[tuple(indj + indi)], 1.0)
                # Checking that the vacuum-vacuum amplitude is 1
    assert np.allclose(mat[1 : 1 + nmodes, 1 : 1 + nmodes], U, atol=tol, rtol=0)


@pytest.mark.parametrize("cutoff", [2])
def test_single_excitation_manifold_unitary(cutoff, tol):
    """Tests the the properties of a 3 mode interferometer"""
    nmodes = 3
    U = np.array(
        [
            [0.25962161 - 0.08744841j, -0.63742098 + 0.55640489j, -0.35243833 + 0.29128117j],
            [0.73811606 - 0.32606317j, 0.33062786 - 0.32070986j, -0.28449121 + 0.23614116j],
            [-0.46985047 - 0.23034198j, 0.26455752 + 0.04413416j, -0.79944267 - 0.12302853j],
        ]
    )
    alphas = np.zeros([nmodes])
    S = interferometer(U)
    T = operations.fock_tensor(S, alphas, cutoff)

    items = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mat = np.zeros([len(items), len(items)], dtype=np.complex128)
    for i, indi in enumerate(items):
        for j, indj in enumerate(items):
            mat[i, j] = T[tuple(indi + indj)]
    np.allclose(U, mat[1 : 1 + nmodes, 1 : 1 + nmodes], atol=tol, rtol=0)
