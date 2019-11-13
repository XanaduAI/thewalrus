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
from itertools import product

import pytest

import numpy as np
from scipy.linalg import qr

from thewalrus.symplectic import rotation, squeezing, interferometer, two_mode_squeezing, beam_splitter

from thewalrus.quantum import (
    reduced_gaussian,
    Xmat,
    Qmat,
    Amat,
    Beta,
    prefactor,
    density_matrix_element,
    density_matrix,
    find_scaling_adjacency_matrix,
    Covmat,
    gen_Qmat_from_graph,
    Means,
    Sympmat,
    is_valid_cov,
    is_pure_cov,
    pure_state_amplitude,
    state_vector,
    is_classical_cov,
    total_photon_num_dist_pure_state,
    gen_single_mode_dist,
    fock_tensor,
)


# make tests deterministic
np.random.seed(137)


def TMS_cov(r, phi):
    """returns the covariance matrix of a TMS state"""
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array(
        [
            [ch, cp * sh, 0, sp * sh],
            [cp * sh, ch, sp * sh, 0],
            [0, sp * sh, ch, -cp * sh],
            [sp * sh, 0, -cp * sh, ch],
        ]
    )

    return S @ S.T


@pytest.mark.parametrize("n", [0, 1, 2])
def test_reduced_gaussian(n):
    """test that reduced gaussian returns the correct result"""
    m = 5
    N = 2 * m
    mu = np.arange(N)
    cov = np.arange(N ** 2).reshape(N, N)
    res = reduced_gaussian(mu, cov, n)
    assert np.all(res[0] == np.array([n, n + m]))
    assert np.all(
        res[1]
        == np.array(
            [[(N + 1) * n, (N + 1) * n + m], [(N + 1) * n + N * m, (N + 1) * n + N * m + m]]
        )
    )


def test_reduced_gaussian_two_mode():
    """test that reduced gaussian returns the correct result"""
    m = 5
    N = 2 * m
    mu = np.arange(N)
    cov = np.arange(N ** 2).reshape(N, N)
    res = reduced_gaussian(mu, cov, [0, 2])
    assert np.all(res[0] == np.array([0, 2, m, m + 2]))


def test_reduced_gaussian_full_state():
    """returns the arguments"""
    mu = np.array([0, 0, 0, 0])
    cov = np.identity(4)
    res = reduced_gaussian(mu, cov, list(range(2)))
    assert np.all(mu == res[0])
    assert np.all(cov == res[1])


def test_reduced_gaussian_exceptions():
    """raises an exception"""
    mu = np.array([0, 0, 0, 0])
    cov = np.identity(4)

    with pytest.raises(ValueError, match="Provided mode is larger than the number of subsystems."):
        reduced_gaussian(mu, cov, [0, 5])


@pytest.mark.parametrize("n", [1, 2, 4])
def test_xmat(n):
    """test X_n = [[0, I], [I, 0]]"""
    I = np.identity(n)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    res = Xmat(n)
    assert np.all(X == res)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_sympmat(n):
    """test X_n = [[0, I], [I, 0]]"""
    I = np.identity(n)
    O = np.zeros_like(I)
    X = np.block([[O, I], [-I, O]])
    res = Sympmat(n)
    assert np.all(X == res)


def test_Qmat_vacuum():
    """test Qmat returns correct result for a vacuum state"""
    V = np.identity(2)
    res = Qmat(V)
    ex = np.identity(2)
    assert np.allclose(res, ex)


def test_Qmat_TMS():
    """test Qmat returns correct result for a two-mode squeezed state"""
    V = TMS_cov(np.arcsinh(1), 0)
    res = Qmat(V)

    q = np.fliplr(np.diag([2.0] * 4))
    np.fill_diagonal(q, np.sqrt(2))
    ex = np.fliplr(q)
    assert np.allclose(res, ex)


def test_Amat_vacuum_using_cov():
    """test Amat returns correct result for a vacuum state"""
    V = np.identity(2)
    res = Amat(V)
    ex = np.zeros([2, 2])
    assert np.allclose(res, ex)


def test_Amat_vacuum_using_Q():
    """test Amat returns correct result for a vacuum state"""
    Q = np.identity(2)
    res = Amat(Q, cov_is_qmat=True)
    ex = np.zeros([2, 2])
    assert np.allclose(res, ex)


def test_Amat_TMS_using_cov():
    """test Amat returns correct result for a two-mode squeezed state"""
    V = TMS_cov(np.arcsinh(1), 0)
    res = Amat(V)

    B = np.fliplr(np.diag([1 / np.sqrt(2)] * 2))
    O = np.zeros_like(B)
    ex = np.block([[B, O], [O, B]])
    assert np.allclose(res, ex)


def test_Amat_TMS_using_Q():
    """test Amat returns correct result for a two-mode squeezed state"""
    q = np.fliplr(np.diag([2.0] * 4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)
    res = Amat(Q, cov_is_qmat=True)

    B = np.fliplr(np.diag([1 / np.sqrt(2)] * 2))
    O = np.zeros_like(B)
    ex = np.block([[B, O], [O, B]])
    assert np.allclose(res, ex)


def test_beta():
    """test the correct beta is returned"""
    mu = np.arange(4)
    res = Beta(mu)

    alpha = (mu[:2] + 1j * mu[2:]) / np.sqrt(2 * 2)
    ex = np.concatenate([alpha, alpha.conj()])
    assert np.allclose(res, ex)


def test_Means():
    """test the correct beta is returned"""
    res = np.arange(4)
    mu = Beta(res)
    ex = Means(mu)
    assert np.allclose(res, ex)


def test_prefactor_vacuum():
    """test the correct prefactor of 0.5 is calculated for a vacuum state"""
    Q = np.identity(2)
    beta = np.zeros([2])

    res = prefactor(Means(beta), Covmat(Q))
    ex = 1
    assert np.allclose(res, ex)


def test_prefactor_TMS():
    """test the correct prefactor of 0.5 is calculated for a TMS state"""
    q = np.fliplr(np.diag([2.0] * 4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)

    beta = np.zeros([4])

    res = prefactor(Means(beta), Covmat(Q))
    ex = 0.5
    assert np.allclose(res, ex)


def test_prefactor_with_displacement():
    """test the correct prefactor of 0.5 is calculated for a TMS state"""
    q = np.fliplr(np.diag([2.0] * 4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)
    Qinv = np.linalg.inv(Q)

    vect = 1.2 * np.ones([2]) + 1j * np.ones(2)
    beta = np.concatenate([vect, vect.conj()])

    res = prefactor(Means(beta), Covmat(Q), hbar=2)
    ex = np.exp(-0.5 * beta @ Qinv @ beta.conj()) / np.sqrt(np.linalg.det(Q))
    assert np.allclose(res, ex)


def test_density_matrix_element_vacuum():
    """Test density matrix elements for the vacuum"""
    Q = np.identity(2)
    beta = np.zeros([2])

    el = [[0], [0]]
    ex = 1
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])
    assert np.allclose(ex, res)

    el = [[1], [1]]
    #    res = density_matrix_element(beta, A, Q, el[0], el[1])
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])

    assert np.allclose(0, res)

    el = [[1], [0]]
    #    res = density_matrix_element(beta, A, Q, el[0], el[1])
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])

    assert np.allclose(0, res)


# density matrix element
t0 = [[0, 0, 0], [0, 0, 0]], 0.7304280085350833
t1 = [[1, 0, 1], [0, 0, 0]], -0.009290003060522444 + 0.002061369459502776j
t2 = [[1, 0, 1], [0, 0, 1]], -0.004088994220552936 - 0.0009589367814578206j
t3 = [[0, 2, 0], [0, 0, 0]], 0.003384487265468196 - 0.03127114305387707j
t4 = [[0, 2, 0], [0, 2, 3]], -8.581668587044574e-05 - 6.134980446713632e-05j


V = np.array(
    [
        [0.6964938, 0.06016962, -0.01970064, 0.03794393, 0.07913992, -0.08890985],
        [0.06016962, 0.85435861, -0.01648842, 0.10493462, 0.01223525, 0.12484726],
        [-0.01970064, -0.01648842, 0.89450003, -0.13182502, 0.13529134, -0.10621978],
        [0.03794393, 0.10493462, -0.13182502, 1.47820656, -0.11611807, 0.05634905],
        [0.07913992, 0.01223525, 0.13529134, -0.11611807, 1.20819301, -0.0061647],
        [-0.08890985, 0.12484726, -0.10621978, 0.05634905, -0.0061647, 1.1636695],
    ]
)


mu = np.array([0.04948628, -0.55738964, 0.71298259, 0.17728629, -0.14381673, 0.33340778])


@pytest.mark.parametrize("t", [t0, t1, t2, t3, t4])
def test_density_matrix_element_disp(t):
    """Test density matrix elements for a state with displacement"""
    beta = Beta(mu)
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])
    assert np.allclose(ex, res)


# density matrix element
t0 = [[0, 0, 0], [0, 0, 0]], 0.9645169885669383
t1 = [[1, 0, 1], [0, 0, 0]], -0.016156769991363732 + 0.05039373212461916j
t2 = [[1, 0, 1], [0, 0, 1]], 0
t3 = [[0, 2, 0], [0, 0, 0]], -0.05911275266690908 - 0.0049431163436861j
t4 = [[0, 2, 0], [0, 2, 3]], 0


@pytest.mark.parametrize("t", [t0, t1, t2, t3, t4])
def test_density_matrix_element_no_disp(t):
    """Test density matrix elements for a state with no displacement"""
    beta = Beta(np.zeros([6]))
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])
    assert np.allclose(ex, res)


def test_density_matrix_vacuum():
    """Test density matrix for a vacuum state"""
    mu = np.zeros([2])
    V = np.identity(2)

    res = density_matrix(mu, V)

    expected = np.zeros([5, 5])
    expected[0, 0] = 1

    assert np.allclose(res, expected)


def test_density_matrix_squeezed():
    """Test density matrix for a squeezed state"""
    r = 0.43

    mu = np.zeros([2])
    V = np.diag(np.array(np.exp([-2 * r, 2 * r])))

    res = density_matrix(mu, V)

    expected = np.array(
        [
            [0.91417429, 0, -0.26200733, 0, 0.09196943],
            [0, 0, 0, 0, 0],
            [-0.26200733, 0, 0.07509273, 0, -0.02635894],
            [0, 0, 0, 0, 0],
            [0.09196943, 0, -0.02635894, 0, 0.00925248],
        ]
    )

    assert np.allclose(res, expected)


def test_coherent_squeezed():
    """Test density matrix for a squeezed displaced state"""
    r = 0.43

    mu = np.array([0.24, -0.2])
    V = np.diag(np.array(np.exp([-2 * r, 2 * r])))

    res = density_matrix(mu, V)

    # fmt: off
    expected = np.array(
        [[0.89054874, 0.15018085 + 0.05295904j, -0.23955467 + 0.01263025j, -0.0734589 - 0.02452154j, 0.07862323 - 0.00868528j],
         [0.15018085 - 0.05295904j, 0.02847564, -0.03964706 + 0.01637575j, -0.01384625 + 0.00023317j, 0.01274241 - 0.00614023j],
         [-0.23955467 - 0.01263025j, -0.03964706 - 0.01637575j, 0.06461854, 0.01941242 + 0.00763805j, -0.02127257 + 0.00122123j],
         [-0.0734589 + 0.02452154j, -0.01384625 - 0.00023317j, 0.01941242 - 0.00763805j, 0.00673463, -0.00624626 + 0.00288134j],
         [0.07862323 + 0.00868528j, 0.01274241 + 0.00614023j, -0.02127257 - 0.00122123j, -0.00624626 - 0.00288134j, 0.00702606]]
    )
    # fmt:on
    assert np.allclose(res, expected)


def test_density_matrix_squeezed_postselect():
    """Test density matrix for a squeezed state with postselection"""
    r = 0.43

    mu = np.zeros([4])
    V = np.diag(np.array(np.exp([1, -2 * r, 1, 2 * r])))

    res = density_matrix(mu, V, post_select={0: 0}, cutoff=15, normalize=True)[:5, :5]

    expected = np.array(
        [
            [0.91417429, 0, -0.26200733, 0, 0.09196943],
            [0, 0, 0, 0, 0],
            [-0.26200733, 0, 0.07509273, 0, -0.02635894],
            [0, 0, 0, 0, 0],
            [0.09196943, 0, -0.02635894, 0, 0.00925248],
        ]
    )

    assert np.allclose(res, expected)


def test_density_matrix_displaced_squeezed_postselect():
    """Test density matrix for a squeezed state with postselection"""
    r = 0.43

    mu = np.array([0, 0.24, 0, -0.2])
    V = np.diag(np.array(np.exp([1, -2 * r, 1, 2 * r])))

    res = density_matrix(mu, V, post_select={0: 0}, cutoff=20, normalize=True)[:5, :5]

    # fmt:off
    expected = np.array([[0.89054874, 0.15018085+0.05295904j, -0.23955467+0.01263025j, -0.0734589 -0.02452154j, 0.07862323-0.00868528j],
       [0.15018085-0.05295904j, 0.02847564, -0.03964706+0.01637575j, -0.01384625+0.00023317j, 0.01274241-0.00614023j],
       [-0.23955467-0.01263025j, -0.03964706-0.01637575j, 0.06461854, 0.01941242+0.00763805j, -0.02127257+0.00122123j],
       [-0.0734589 +0.02452154j, -0.01384625-0.00023317j, 0.01941242-0.00763805j, 0.00673463, -0.00624626+0.00288134j],
       [0.07862323+0.00868528j, 0.01274241+0.00614023j, -0.02127257-0.00122123j, -0.00624626-0.00288134j, 0.00702606]])
    # fmt:on

    assert np.allclose(res, expected)


def test_find_scaling_adjacency_matrix():
    """Test the find_scaling_adjacency matrix for a the one mode case"""
    r = 0.75 + 0.9j
    rabs = np.abs(r)
    n_mean = 10.0
    A = r * np.identity(1)
    sc_exact = np.sqrt(n_mean / (1.0 + n_mean)) / rabs
    sc_num = find_scaling_adjacency_matrix(A, n_mean)
    assert np.allclose(sc_exact, sc_num)


def test_Covmat():
    """ Test the Covmat function by checking that its inverse function is Qmat """
    n = 1
    B = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    B = B + B.T
    sc = find_scaling_adjacency_matrix(B, 1)
    idm = np.identity(2 * n)
    X = Xmat(n)
    Bsc = sc * B
    A = np.block([[Bsc, 0 * Bsc], [0 * Bsc, Bsc.conj()]])
    Q = np.linalg.inv(idm - X @ A)
    cov = Covmat(Q)
    Qrec = Qmat(cov)
    assert np.allclose(Q, Qrec)


def test_gen_Qmat_from_graph():
    """ Test the gen_Qmat_from_graph for the analytically solvable case of a single mode"""
    A = np.array([[10.0]])
    n_mean = 1.0
    cov = Covmat(gen_Qmat_from_graph(A, n_mean))
    r = np.arcsinh(np.sqrt(n_mean))
    cov_e = np.diag([(np.exp(2 * r)), (np.exp(-2 * r))])
    assert np.allclose(cov, cov_e)


def test_is_valid_cov():
    """ Test if is_valid_cov for a valid covariance matrix """
    hbar = 2
    val = is_valid_cov(V, hbar=hbar)
    assert val


def test_is_valid_cov_non_square():
    """ Test False if matrix is not square"""
    hbar = 2
    V = np.ones([3, 4])
    val = is_valid_cov(V, hbar=hbar)
    assert not val


def test_is_valid_cov_non_symmetric():
    """ Test False if matrix is not symmetric"""
    hbar = 2
    V = np.zeros([4, 4])
    V[0, 1] = 1
    val = is_valid_cov(V, hbar=hbar)
    assert not val


def test_is_valid_cov_not_even_dimension():
    """ Test False if matrix does not have even dimensions"""
    hbar = 2
    V = np.zeros([5, 5])
    val = is_valid_cov(V, hbar=hbar)
    assert not val


def test_is_valid_cov_too_certain():
    """ Test False if matrix does not satisfy the Heisenberg
    uncertainty relation"""
    hbar = 2
    V = np.random.random([4, 4])
    V += V.T
    val = is_valid_cov(V, hbar=hbar)
    assert not val


def test_is_pure_cov():
    """ Test if is_pure_cov for a pure state"""
    hbar = 2
    val = is_pure_cov(V, hbar=hbar)
    assert val


@pytest.mark.parametrize("nbar", [0, 1, 2])
def test_is_valid_cov_thermal(nbar):
    """ Test if is_valid_cov for a mixed state"""
    hbar = 2
    dim = 10
    cov = (2 * nbar + 1) * np.identity(dim)
    val = is_valid_cov(cov, hbar=hbar)
    assert val


@pytest.mark.parametrize("nbar", [0, 1, 2])
def test_is_pure_cov_thermal(nbar):
    """ Test if is_pure_cov for vacuum and thermal states"""
    hbar = 2
    dim = 10
    cov = (2 * nbar + 1) * np.identity(dim)
    val = is_pure_cov(cov, hbar=hbar)
    if nbar == 0:
        assert val
    else:
        assert not val


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
def test_pure_state_amplitude_two_mode_squeezed(i, j):
    """ Tests pure state amplitude for a two mode squeezed vacuum state """
    nbar = 1.0
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    if i != j:
        exact = 0.0
    else:
        exact = np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar)
    num = pure_state_amplitude(mu, cov, [i, j])

    assert np.allclose(exact, num)


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
def test_pure_state_amplitude_coherent(i):
    """ Tests pure state amplitude for a coherent state """
    cov = np.identity(2)
    mu = np.array([1.0, 2.0])
    beta = Beta(mu)
    alpha = beta[0]
    exact = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** i / np.sqrt(np.math.factorial(i))
    num = pure_state_amplitude(mu, cov, [i])
    assert np.allclose(exact, num)


def test_pure_state_amplitude_squeezed_coherent():
    """Test density matrix for a squeezed coherent state"""
    r = 0.43

    mu = np.array([0.24, -0.2])
    V = np.diag(np.array(np.exp([-2 * r, 2 * r])))

    amps = np.array([pure_state_amplitude(mu, V, [i]) for i in range(5)])
    numerical = np.outer(amps, amps.conj())
    # fmt: off
    expected = np.array(
        [[0.89054874, 0.15018085 + 0.05295904j, -0.23955467 + 0.01263025j, -0.0734589 - 0.02452154j, 0.07862323 - 0.00868528j],
         [0.15018085 - 0.05295904j, 0.02847564, -0.03964706 + 0.01637575j, -0.01384625 + 0.00023317j, 0.01274241 - 0.00614023j],
         [-0.23955467 - 0.01263025j, -0.03964706 - 0.01637575j, 0.06461854, 0.01941242 + 0.00763805j, -0.02127257 + 0.00122123j],
         [-0.0734589 + 0.02452154j, -0.01384625 - 0.00023317j, 0.01941242 - 0.00763805j, 0.00673463, -0.00624626 + 0.00288134j],
         [0.07862323 + 0.00868528j, 0.01274241 + 0.00614023j, -0.02127257 - 0.00122123j, -0.00624626 - 0.00288134j, 0.00702606]]
    )
    # fmt:on
    assert np.allclose(expected, numerical)


def test_state_vector_two_mode_squeezed():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.array(
        [
            (np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar))
            for i in range(cutoff)
        ]
    )
    psi = state_vector(mu, cov, cutoff=cutoff)
    expected = np.diag(exact)
    assert np.allclose(psi, expected)


def test_state_vector_two_mode_squeezed_post():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.diag(
        np.array(
            [
                (np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar))
                for i in range(cutoff)
            ]
        )
    )
    val = 2
    post_select = {0: val}
    psi = state_vector(mu, cov, cutoff=cutoff, post_select=post_select)
    expected = exact[val]
    assert np.allclose(psi, expected)


def test_state_vector_coherent():
    """ Tests state vector for a coherent state """
    cutoff = 5
    cov = np.identity(2)
    mu = np.array([1.0, 2.0])
    beta = Beta(mu)
    alpha = beta[0]
    exact = np.array(
        [
            (np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** i / np.sqrt(np.math.factorial(i)))
            for i in range(cutoff)
        ]
    )
    num = state_vector(mu, cov, cutoff=cutoff)
    assert np.allclose(exact, num)


def test_state_vector_two_mode_squeezed_post_normalize():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.diag(
        np.array(
            [
                (np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar))
                for i in range(cutoff)
            ]
        )
    )
    val = 2
    post_select = {0: val}
    psi = state_vector(mu, cov, cutoff=cutoff, post_select=post_select, normalize=True)
    expected = exact[val]
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(psi, expected)


def test_is_classical_cov_squeezed():
    """ Tests that a squeezed state is not classical"""
    nbar = 1.0
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    assert not is_classical_cov(cov)


@pytest.mark.parametrize("nbar", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_is_classical_cov_thermal(nbar):
    """ Tests that a thermal state is classical"""
    cov = (2 * nbar + 1) * np.identity(2)
    assert is_classical_cov(cov)

@pytest.mark.parametrize("cutoff", [50, 51, 52, 53])
def test_total_photon_num_dist_pure_state(cutoff):
    """ Test the correct photon number distribution is obtained for n modes
    with nmean number of photons up to Fock cutoff nmax"""
    n = 3
    nmean = 1.0
    cutoff = 50
    rs = np.arcsinh(np.sqrt(nmean)) * np.ones([n])
    cov = np.diag(np.concatenate([np.exp(2 * rs), np.exp(-2 * rs)]))
    p1 = total_photon_num_dist_pure_state(cov, cutoff=cutoff)
    p2 = gen_single_mode_dist(np.arcsinh(np.sqrt(nmean)), N=n, cutoff=cutoff)
    assert np.allclose(p1, p2)




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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if sum(p) != sum(q): #Check that there are the same total number of photons in the bra and the ket
                r = tuple(list(p) + list(q))
                assert np.allclose(T[r], 0.0, atol=tol, rtol=0)


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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
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
    T = fock_tensor(S, alphas, cutoff, r=r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if p[0] - q[0] != p[1] - q[1]:
                t = tuple(list(p) + list(q))
                assert np.allclose(T[t], 0, atol=tol, rtol=0)
