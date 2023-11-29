# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain adj copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Montrealer tests
Yanic Cardin and Nicolás Quesada. "Photon-number moments and cumulants of Gaussian states"
`arxiv:12212.06067 (2023) <https://arxiv.org/abs/2212.06067>`_
"""

import pytest
import numpy as np
from thewalrus import mtl, lmtl
from thewalrus.reference import mapper
from thewalrus.quantum import Qmat, Xmat
from thewalrus.reference import rspm, rpmp, mtl as mtl_symb
from thewalrus.random import random_covariance
from scipy.special import factorial2
from scipy.stats import unitary_group


@pytest.mark.parametrize("n", range(1, 8))
def test_montrealer_all_ones(n):
    """Test that the Montrealer of a matrix of ones gives (2n-2)!!"""
    adj = np.ones([2 * n, 2 * n])
    mtl_val = mtl(adj)
    mtl_expect = factorial2(2 * n - 2)
    assert np.allclose(mtl_val, mtl_expect)


@pytest.mark.parametrize("n", range(1, 8))
def test_loop_montrealer_all_ones(n):
    """Test that the loop Montrealer of a matrix of ones gives (n+1)(2n-2)!!"""
    adj = np.ones([2 * n, 2 * n])
    lmtl_val = lmtl(adj, zeta=np.diag(adj))
    lmtl_expect = (n + 1) * factorial2(2 * n - 2)
    assert np.allclose(lmtl_val, lmtl_expect)


@pytest.mark.parametrize("n", range(1, 8))
def test_size_of_rpmp(n):
    """rpmp(2n) should have (2n-2)!! elements"""
    terms_rpmp = len(list(rpmp(range(2 * n))))
    terms_theo = factorial2(2 * n - 2)
    assert terms_rpmp == terms_theo


@pytest.mark.parametrize("n", range(1, 8))
def test_size_of_rspm(n):
    """rspm(2n) should have (n+1)(2n-2)!! elements"""
    terms_rspm = sum(1 for _ in rspm(range(2 * n)))
    terms_theo = (n + 1) * factorial2(2 * n - 2)
    assert terms_rspm == terms_theo


@pytest.mark.parametrize("n", range(2, 8))
def test_rpmp_alternating_walk(n):
    """The rpmp must form a Y-alternating walk without loops"""
    test = True
    for perfect in rpmp(range(1, 2 * n + 1)):
        last = perfect[0][1]  # starting point
        reduced_last = last - n if last > n else last
        # different mode in every tuple
        if reduced_last == 1:
            test = False

        for i in perfect[1:]:
            reduced = i[0] - n if i[0] > n else i[0], i[1] - n if i[1] > n else i[1]
            # different mode in every tuple
            if reduced[0] == reduced[1]:
                test = False
            # consecutive tuple contain the same mode
            if reduced_last not in reduced:
                test = False

            last = i[0] if reduced[1] == reduced_last else i[1]
            reduced_last = last - n if last > n else last

        # last mode most coincide with the first one
        if reduced_last != 1:
            test = False

    assert test


@pytest.mark.parametrize("n", range(1, 8))
def test_mtl_functions_agree(n):
    """Make sure both mtl functions agree with one another"""
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    assert np.allclose(mtl_symb(Aad), mtl(Aad))


@pytest.mark.parametrize("n", range(1, 8))
def test_lmtl_functions_agree(n):
    """Make sure both lmtl functions agree with one another"""
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    zeta = np.diag(Aad).conj()
    assert np.allclose(lmtl(Aad, zeta), mtl_symb(Aad, loop=True))


@pytest.mark.parametrize("n", range(1, 8))
def test_mtl_lmtl_agree(n):
    """Make sure mtl and lmtl give the same result if zeta = 0"""
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    zeta = np.zeros(2 * n, dtype=np.complex128)
    assert np.allclose(lmtl(Aad, zeta), lmtl(Aad, zeta))


@pytest.mark.parametrize("n", range(1, 8))
def test_mtl_lmtl_reference_agree(n):
    """Make sure mtl and lmtl from .reference give the same result if zeta = 0"""
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    zeta = np.zeros(2 * n, dtype=np.complex128)
    np.fill_diagonal(Aad, zeta)
    assert np.allclose(mtl_symb(Aad, loop=True), mtl_symb(Aad))


@pytest.mark.parametrize("n", range(1, 8))
def test_mtl_permutation(n):
    """Make sure the mtl is invariant under permutation
    cf. Eq. 44 of `arxiv:12212.06067 (2023) <https://arxiv.org/abs/arxiv:2212.06067v2>`_"""
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    perm = np.random.permutation(n)
    perm = np.concatenate((perm, [i + n for i in perm]))
    assert np.allclose(mtl(Aad), mtl(Aad[perm][:, perm]))


@pytest.mark.parametrize("n", range(2, 5))
def test_mtl_associated_adjacency(n):
    """Make sure the mtl of a matrix in which each block is block diaognal is zero.
    cf. Eq. 45 of `arxiv:12212.06067 (2023) <https://arxiv.org/abs/arxiv:2212.06067v2>`_"""
    u_zero = np.zeros((n, n), dtype=np.complex128)

    u_n1 = unitary_group.rvs(n)
    u_n2 = unitary_group.rvs(n)
    u_n = np.block([[u_n1, u_zero], [u_zero, u_n2]])
    u_n = u_n + u_n.conj().T

    u_m1 = unitary_group.rvs(n)
    u_m2 = unitary_group.rvs(n)
    u_m = np.block([[u_m1, u_zero], [u_zero, u_m2]])
    u_m_r = u_m + u_m.T

    u_m3 = unitary_group.rvs(n)
    u_m4 = unitary_group.rvs(n)
    u_m = np.block([[u_m3, u_zero], [u_zero, u_m4]])
    u_m_l = u_m + u_m.T

    adj = np.block([[u_m_r, u_n], [u_n.T, u_m_l]])

    assert np.allclose(mtl(adj), 0)


@pytest.mark.parametrize("n", range(1, 8))
def test_mtl_diagonal_trace(n):
    """Make sure the mtl of A times a diagonal matrix gives the product of the norms of the diagonal matrix times the mtl of A
    cf. Eq. 41 of `arxiv:12212.06067 (2023) <https://arxiv.org/abs/arxiv:2212.06067v2>`_"""
    gamma = np.random.uniform(-1, 1, n) + 1.0j * np.random.uniform(-1, 1, n)
    product = np.prod([abs(i) ** 2 for i in gamma])
    gamma = np.diag(np.concatenate((gamma, gamma.conj())))
    V = random_covariance(n)
    Aad = Xmat(n) @ (Qmat(V) - np.identity(2 * n))
    assert np.allclose(mtl(gamma @ Aad @ gamma), product * mtl(Aad))


def test_mapper_hard_coded():
    """Tests the the mapper function for a particular hardcoded value"""
    assert mapper(((1, 2, 3), "0000"), (0, 1, 2, 3, 4, 5, 6, 7)) == ((0, 5), (1, 6), (2, 7), (4, 3))
