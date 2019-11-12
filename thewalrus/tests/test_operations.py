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

from itertools import product

import pytest

import numpy as np
from scipy.linalg import qr

from thewalrus import operations
from thewalrus.symplectic import rotation, squeezing, interferometer, two_mode_squeezing, beam_splitter

# make test deterministic
np.random.seed(137)


def random_interferometer(N, real=False):
    r"""Random unitary matrix representing an interferometer.

    For more details, see :cite:`mezzadri2006`.

    Args:
        N (int): number of modes
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    if real:
        z = np.random.randn(N, N)
    else:
        z = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2.0)
    q, r = qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    U = np.multiply(q, ph, q)
    return U

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_identity(r, tol):
    """Tests the correct construction of the single mode identity operation"""
    nmodes = 1
    cutoff = 7
    S = np.identity(2 * nmodes)
    alphas = np.zeros([nmodes])
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    expected = np.identity(cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_rotation(r, tol):
    """Tests the correct construction of the single mode rotation operation"""
    nmodes = 1
    cutoff = 7
    theta = 2 * np.pi * np.random.rand()
    S = rotation(theta)
    alphas = np.zeros([nmodes])
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    expected = np.diag(np.exp(1j * theta * np.arange(cutoff)))
    assert np.allclose(T, expected, atol=tol, rtol=0)

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_displacement(r, tol):
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
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    assert np.allclose(T, expected, atol=tol, rtol=0)

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_squeezing(r, tol):
    """Tests the correct construction of the single mode squeezing operation"""
    nmodes = 1
    s = 1.0
    cutoff = 5
    S = squeezing(s, 0.0)
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
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    assert np.allclose(T, expected, atol=tol, rtol=0)

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_displacement_squeezing(r, tol):
    """Tests the correct construction of the single mode squeezing operation followed by the single mode displacement operation"""
    nmodes = 1
    s = 1.0
    cutoff = 5
    S = squeezing(s, 0.0)
    alphas = (0.5 + 0.4 * 1j) * np.ones([nmodes])
    # This data is obtained by using qutip
    # np.array((displace(40,alpha)*squeeze(40,r)).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.6263739 + 0.09615331j, -0.22788717 + 0.13121343j, 0.36548296 - 0.0200537j, -0.20708137 + 0.14004403j, 0.25645667 - 0.06275564j],
            [0.5425389 + 0.14442404j, 0.19268911 + 0.15615312j, 0.11497303 + 0.13744549j, 0.21448948 + 0.08109308j, -0.03652914 + 0.15069359j],
            [-0.00915607 + 0.07475267j, 0.48081922 + 0.10576742j, -0.00961086 + 0.20535144j, 0.33089303 + 0.09864247j, 0.02555522 + 0.19950786j],
            [-0.34614367 - 0.05229875j, 0.11543956 + 0.01112537j, 0.16672961 + 0.07439407j, 0.02323121 + 0.15103267j, 0.27233637 + 0.08297028j],
            [-0.14390852 - 0.08884069j, -0.37898007 - 0.07630228j, 0.12911863 - 0.08963054j, -0.12164023 + 0.04431394j, 0.1141808 + 0.01581529j],
        ]
    )
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("nmodes", [2, 3, 4])
@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_interferometer_selection_rules(r, nmodes, tol):
    r"""Test the selection rules of an interferometer.
    If one writes the interferometer gate of k modes as :math:`U` and its matrix elements as
    :math:`\langle p_0 p_1 \ldots p_{k-1} |U|q_0 q_1 \ldots q_{k-1}\rangle` then these elements
    are nonzero if and only if :math:`\sum_{i=0}^k p_i = \sum_{i=0}^k q_i`. This test checks
    that this selection rule holds.
    """
    U = random_interferometer(nmodes)
    S = interferometer(U)
    alphas = np.zeros([nmodes])
    cutoff = 4
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if sum(p) != sum(q): #Check that there are the same total number of photons in the bra and the ket
                r = tuple(list(p) + list(q))
                np.allclose(T[r], 0.0, atol=tol, rtol=0)


@pytest.mark.parametrize("nmodes", [2, 3, 4])
@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_interferometer_single_excitation(r, nmodes, tol):
    r"""Test that the representation of an interferometer in the single
    excitation manifold is precisely the unitary matrix that represents it
    mode in space.
    Let :math:`V` be a unitary matrix in N modes and let :math:`U` be its Fock representation
    Also let :math:`|i \rangle = |0_0,\ldots, 1_i, 0_{N-1} \rangle`, i.e a single photon in mode :math:`i`.
    Then it must hold that :math:`V_{i,j} = \langle i | U | j \rangle`.
    """
    U = random_interferometer(nmodes)
    S = interferometer(U)
    alphas = np.zeros([nmodes])
    cutoff = 2
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    # Construct a list with all the indices corresponding to |i \rangle
    vec_list = np.identity(nmodes, dtype=int).tolist()
    # Calculate the matrix \langle i | U | j \rangle = T[i+j]
    U_rec = np.empty([nmodes, nmodes], dtype=complex)
    for i, vec_i in enumerate(vec_list):
        for j, vec_j in enumerate(vec_list):
            U_rec[i, j] = T[tuple(vec_i + vec_j)]
    assert np.allclose(U_rec, U, atol=tol, rtol=0)


@pytest.mark.parametrize("phi", list(np.arange(0, np.pi, 0.1)))
@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_hong_ou_mandel_interference(r, phi, tol):
    r"""Tests Hong-Ou-Mandel interference for a 50:50 beamsplitter.
    If one writes :math:`U` for the Fock representation of a 50-50 beamsplitter
    then it must hold that :math:`\langle 1,1|U|1,1 \rangle = 0`.
    """
    S = beam_splitter(np.pi / 4, phi)  # a 50-50 beamsplitter with phase phi
    cutoff = 2
    nmodes = 2
    alphas = np.zeros([nmodes])
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    assert np.allclose(T[1, 1, 1, 1], 0.0, atol=tol, rtol=0)

@pytest.mark.parametrize("r", [0.5, np.arcsinh(1.0), 2])
def test_two_mode_squeezing(r, tol):
    r"""Tests the selection rules of a two mode squeezing operation.
    If one writes the squeezing gate as :math:`S_2` and its matrix elements as
    :math:`\langle p_0 p_1|S_2|q_0 q_1 \rangle` then these elements are nonzero
    if and only if :math:`p_0 - q_0 = p_1 - q_1`. This test checks that this
    selection rule holds.
    """
    cutoff = 5
    nmodes = 2
    s = np.arcsinh(1.0)
    phi = np.pi / 6
    alphas = np.zeros([nmodes])
    S = two_mode_squeezing(s, phi)
    T = operations.fock_tensor(S, alphas, cutoff, r=r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if p[0] - q[0] != p[1] - q[1]:
                t = tuple(list(p) + list(q))
                assert np.allclose(T[t], 0, atol=tol, rtol=0)
