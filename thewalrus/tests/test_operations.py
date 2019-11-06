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


@pytest.mark.parametrize("cutoff", [5])
def test_single_mode_displacement(cutoff, tol):
    """Tests the correct construction of the single mode displacement operation"""
    nmodes = 1
    alphas = (0.3 + 0.5 * 1j) * np.ones([nmodes])
    S = np.identity(2 * nmodes)
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:10,0:10]
    data = np.load("displacement.npy")
    if cutoff <= len(data):
        expected = data[0:cutoff, 0:cutoff]
        T = operations.fock_tensor(S, alphas, cutoff)
        assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("cutoff", [5])
def test_single_mode_squeezing(cutoff, tol):
    """Tests the correct construction of the single mode squeezing operation"""
    nmodes = 1
    r = 1.0
    S = squeezing(r, 0.0)
    alphas = np.zeros([nmodes])
    # This data is obtained by using qutip
    # np.array(squeeze(40,1.0).data.todense())[0:10,0:10]
    data = np.load("squeeze.npy")
    if cutoff <= len(data):
        expected = data[0:cutoff, 0:cutoff]
        T = operations.fock_tensor(S, alphas, cutoff)
        assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("cutoff", [5])
def test_single_mode_displacement_squeezing(cutoff, tol):
    """Tests the correct construction of the single mode squeezing operation followed by the single mode displacement operation"""
    nmodes = 1
    r = 1.0
    S = squeezing(r, 0.0)
    alphas = (0.5 + 0.4 * 1j) * np.ones([nmodes])
    # This data is obtained by using qutip
    # np.array((displace(40,alpha)*squeeze(40,r)).data.todense())[0:10,0:10]
    data = np.load("displace-squeeze.npy")
    if cutoff <= len(data):
        expected = data[0:cutoff, 0:cutoff]
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
