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
from scipy.stats import poisson

from thewalrus.symplectic import rotation, squeezing, interferometer, two_mode_squeezing, beam_splitter, loss

from thewalrus.random import random_covariance, random_interferometer

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
    find_scaling_adjacency_matrix_torontonian,
    mean_number_of_clicks,
    Covmat,
    gen_Qmat_from_graph,
    Means,
    photon_number_covmat,
    is_valid_cov,
    is_pure_cov,
    pure_state_amplitude,
    state_vector,
    is_classical_cov,
    total_photon_num_dist_pure_state,
    gen_single_mode_dist,
    fock_tensor,
    photon_number_mean_vector,
    photon_number_mean,
    probabilities,
    update_probabilities_with_loss,
    update_probabilities_with_noise,
    loss_mat,
    fidelity,
)


# make tests deterministic
np.random.seed(137)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_reduced_gaussian(n):
    """test that reduced gaussian returns the correct result"""
    m = 5
    N = 2 * m
    mu = np.arange(N)
    cov = np.arange(N ** 2).reshape(N, N)
    res = reduced_gaussian(mu, cov, n)
    assert np.all(res[0] == np.array([n, n + m]))
    assert np.all(res[1] == np.array([[(N + 1) * n, (N + 1) * n + m], [(N + 1) * n + N * m, (N + 1) * n + N * m + m]]))


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


def test_Qmat_vacuum():
    """test Qmat returns correct result for a vacuum state"""
    V = np.identity(2)
    res = Qmat(V)
    ex = np.identity(2)
    assert np.allclose(res, ex)


def test_Qmat_TMS():
    """test Qmat returns correct result for a two-mode squeezed state"""
    V = two_mode_squeezing(2 * np.arcsinh(1), 0)
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
    V = two_mode_squeezing(2 * np.arcsinh(1), 0)
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

    expected = np.array([[0.91417429, 0, -0.26200733, 0, 0.09196943], [0, 0, 0, 0, 0], [-0.26200733, 0, 0.07509273, 0, -0.02635894], [0, 0, 0, 0, 0], [0.09196943, 0, -0.02635894, 0, 0.00925248],])

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

    expected = np.array([[0.91417429, 0, -0.26200733, 0, 0.09196943], [0, 0, 0, 0, 0], [-0.26200733, 0, 0.07509273, 0, -0.02635894], [0, 0, 0, 0, 0], [0.09196943, 0, -0.02635894, 0, 0.00925248],])

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


def test_find_scaling_adjacency_matrix_torontonian():
    """Test the find_scaling_adjacency_matrix_torontonian for a multimode problem"""
    n = 10
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    A += A.T
    nc = 3.0
    x = find_scaling_adjacency_matrix_torontonian(A, nc)
    assert np.allclose(mean_number_of_clicks(x * A), nc)

def test_mean_number_of_clicks():
    """Test that a two mode squeezed vacuum with parameter r has mean number of clicks equal to 2 tanh(r)"""
    r = 3.0
    tr = np.tanh(r)
    A = np.array([[0, tr], [tr, 0]])
    value = mean_number_of_clicks(A)
    expected = 2 * tr**2
    assert np.allclose(expected, value)

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
    cov = two_mode_squeezing(2 * r, phase)
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
    cov = two_mode_squeezing(2 * r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.array([(np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar)) for i in range(cutoff)])
    psi = state_vector(mu, cov, cutoff=cutoff)
    expected = np.diag(exact)
    assert np.allclose(psi, expected)


def test_state_vector_two_mode_squeezed_post():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = two_mode_squeezing(2 * r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.diag(np.array([(np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar)) for i in range(cutoff)]))
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
    exact = np.array([(np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** i / np.sqrt(np.math.factorial(i))) for i in range(cutoff)])
    num = state_vector(mu, cov, cutoff=cutoff)
    assert np.allclose(exact, num)


def test_state_vector_two_mode_squeezed_post_normalize():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = two_mode_squeezing(2 * r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.diag(np.array([(np.exp(1j * i * phase) * (nbar / (1.0 + nbar)) ** (i / 2) / np.sqrt(1.0 + nbar)) for i in range(cutoff)]))
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
    cov = two_mode_squeezing(2 * r, phase)
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
    rs = np.arcsinh(np.sqrt(nmean)) * np.ones([n])
    cov = np.diag(np.concatenate([np.exp(2 * rs), np.exp(-2 * rs)]))
    p1 = total_photon_num_dist_pure_state(cov, cutoff=cutoff)
    p2 = gen_single_mode_dist(np.arcsinh(np.sqrt(nmean)), N=n, cutoff=cutoff)
    assert np.allclose(p1, p2)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_identity(choi_r, tol):
    """Tests the correct construction of the single mode identity operation"""
    nmodes = 1
    cutoff = 7
    S = np.identity(2 * nmodes)
    alphas = np.zeros([nmodes])
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    expected = np.identity(cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_rotation(choi_r, tol):
    """Tests the correct construction of the single mode rotation operation"""
    nmodes = 1
    cutoff = 7
    theta = 2 * np.pi * np.random.rand()
    S = rotation(theta)
    alphas = np.zeros([nmodes])
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    expected = np.diag(np.exp(1j * theta * np.arange(cutoff)))
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_displacement(choi_r, tol):
    """Tests the correct construction of the single mode displacement operation"""
    nmodes = 1
    cutoff = 5
    alphas = (0.3 + 0.5 * 1j) * np.ones([nmodes])
    S = np.identity(2 * nmodes)
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.84366482 + 0.00000000e00j, -0.25309944 + 4.21832408e-01j, -0.09544978 - 1.78968334e-01j, 0.06819609 + 3.44424719e-03j, -0.01109048 + 1.65323865e-02j,],
            [0.25309944 + 4.21832408e-01j, 0.55681878 + 0.00000000e00j, -0.29708743 + 4.95145724e-01j, -0.14658716 - 2.74850926e-01j, 0.12479885 + 6.30297236e-03j,],
            [-0.09544978 + 1.78968334e-01j, 0.29708743 + 4.95145724e-01j, 0.31873657 + 0.00000000e00j, -0.29777767 + 4.96296112e-01j, -0.18306015 - 3.43237787e-01j,],
            [-0.06819609 + 3.44424719e-03j, -0.14658716 + 2.74850926e-01j, 0.29777767 + 4.96296112e-01j, 0.12389162 + 1.10385981e-17j, -0.27646677 + 4.60777945e-01j,],
            [-0.01109048 - 1.65323865e-02j, -0.12479885 + 6.30297236e-03j, -0.18306015 + 3.43237787e-01j, 0.27646677 + 4.60777945e-01j, -0.03277289 + 1.88440656e-17j,],
        ]
    )
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_squeezing(choi_r, tol):
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
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_single_mode_displacement_squeezing(choi_r, tol):
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
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    assert np.allclose(T, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("nmodes", [2, 3, 4])
@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_interferometer_selection_rules(choi_r, nmodes, tol):
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
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if sum(p) != sum(q):  # Check that there are the same total number of photons in the bra and the ket
                r = tuple(list(p) + list(q))
                assert np.allclose(T[r], 0.0, atol=tol, rtol=0)


@pytest.mark.parametrize("nmodes", [2, 3, 4])
@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_interferometer_single_excitation(choi_r, nmodes, tol):
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
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    # Construct a list with all the indices corresponding to |i \rangle
    vec_list = np.identity(nmodes, dtype=int).tolist()
    # Calculate the matrix \langle i | U | j \rangle = T[i+j]
    U_rec = np.empty([nmodes, nmodes], dtype=complex)
    for i, vec_i in enumerate(vec_list):
        for j, vec_j in enumerate(vec_list):
            U_rec[i, j] = T[tuple(vec_i + vec_j)]
    assert np.allclose(U_rec, U, atol=tol, rtol=0)


@pytest.mark.parametrize("phi", list(np.arange(0, np.pi, 0.1)))
@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_hong_ou_mandel_interference(choi_r, phi, tol):
    r"""Tests Hong-Ou-Mandel interference for a 50:50 beamsplitter.
    If one writes :math:`U` for the Fock representation of a 50-50 beamsplitter
    then it must hold that :math:`\langle 1,1|U|1,1 \rangle = 0`.
    """
    S = beam_splitter(np.pi / 4, phi)  # a 50-50 beamsplitter with phase phi
    cutoff = 2
    nmodes = 2
    alphas = np.zeros([nmodes])
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    assert np.allclose(T[1, 1, 1, 1], 0.0, atol=tol, rtol=0)


@pytest.mark.parametrize("choi_r", [0.5, np.arcsinh(1.0), 2])
def test_two_mode_squeezing(choi_r, tol):
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
    T = fock_tensor(S, alphas, cutoff, choi_r=choi_r)
    for p in product(list(range(cutoff)), repeat=nmodes):
        for q in product(list(range(cutoff)), repeat=nmodes):
            if p[0] - q[0] != p[1] - q[1]:
                t = tuple(list(p) + list(q))
                assert np.allclose(T[t], 0, atol=tol, rtol=0)


def test_sf_ordering_in_fock_tensor(tol):
    """Test that the reordering works when using sf_order=True"""
    cutoff = 5
    nmodes = 2
    s = np.arcsinh(1.0)
    phi = np.pi / 6
    alphas = np.zeros([nmodes])
    S = two_mode_squeezing(s, phi)
    T = fock_tensor(S, alphas, cutoff)
    Tsf = fock_tensor(S, alphas, cutoff, sf_order=True)
    assert np.allclose(T.transpose([0, 2, 1, 3]), Tsf, atol=tol, rtol=0)


LIST_FUNCS = [np.zeros, np.ones, np.arange]

@pytest.mark.parametrize("list_func", LIST_FUNCS)
@pytest.mark.parametrize("N", [1, 2, 4])
@pytest.mark.parametrize("hbar", [1, 2])
def test_pnd_coherent_state(tol, list_func, N, hbar):
    r"""Test the photon number mean and covariance of coherent states"""
    cov = np.eye(2 * N) * hbar / 2
    mu = list_func(2 * N)

    pnd_cov = photon_number_covmat(mu, cov, hbar=hbar)
    alpha = (mu[:N] ** 2 + mu[N:] ** 2) / (2 * hbar)
    pnd_mean = photon_number_mean_vector(mu, cov, hbar=hbar)
    assert np.allclose(pnd_cov, np.diag(alpha), atol=tol, rtol=0)
    assert np.allclose(pnd_mean, alpha, atol=tol, rtol=0)



@pytest.mark.parametrize("r", np.linspace(0, 2, 4))
@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 4))
@pytest.mark.parametrize("hbar", [1, 2])
def test_pnd_two_mode_squeeze_vacuum(tol, r, phi, hbar):
    """Test the photon number mean and covariance of the two-mode squeezed vacuum"""
    S = two_mode_squeezing(r, phi)
    mu = np.zeros(4)

    cov = hbar / 2 * (S @ S.T)
    pnd_cov = photon_number_covmat(mu, cov, hbar=hbar)
    n = np.sinh(r) ** 2
    pnd_mean = photon_number_mean_vector(mu, cov, hbar=hbar)
    assert np.allclose(pnd_cov, np.full((2, 2), n ** 2 + n), atol=tol, rtol=0)
    assert np.allclose(pnd_mean, np.array([n, n]), atol=tol, rtol=0)

@pytest.mark.parametrize("n", np.linspace(0, 10, 4))
@pytest.mark.parametrize("N", [1, 2, 4])
@pytest.mark.parametrize("hbar", [1, 2])
def test_pnd_thermal(tol, n, N, hbar):
    """Test the photon number mean and covariance of thermal states"""
    cov = (2 * n + 1) * np.eye(2 * N) * hbar / 2
    mu = np.zeros(2 * N)
    pnd_cov = photon_number_covmat(mu, cov, hbar=hbar)
    pnd_mean = photon_number_mean_vector(mu, cov, hbar=hbar)
    assert np.allclose(pnd_cov, np.diag([n ** 2 + n] * N), atol=tol, rtol=0)
    mean_expected = n * np.ones([N])
    assert np.allclose(pnd_mean, mean_expected, atol=tol, rtol=0)


@pytest.mark.parametrize("r", np.linspace(0, 2, 4))
@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 4))
@pytest.mark.parametrize("alpha", [0, 1.0, 1j, 1.0 + 1j])
@pytest.mark.parametrize("hbar", [1, 2])
def test_pnd_squeeze_displace(tol, r, phi, alpha, hbar):
    """Test the photon number number mean and covariance of the squeezed displaced state

    Eq. (17) in 'Benchmarking of Gaussian boson sampling using two-point correlators',
    Phillips et al. (https://ris.utwente.nl/ws/files/122721825/PhysRevA.99.023836.pdf).
    """
    S = squeezing(r, phi)
    mu = [np.sqrt(2 * hbar) * np.real(alpha), np.sqrt(2 * hbar) * np.imag(alpha)]

    cov = hbar / 2 * (S @ S.T)
    pnd_cov = photon_number_covmat(mu, cov, hbar=hbar)

    pnd_cov_analytic = np.sinh(r) ** 2 * np.cosh(r) ** 2 + np.sinh(r) ** 4 \
        + np.sinh(r) ** 2 + np.abs(alpha) ** 2 * (1 + 2 * np.sinh(r) ** 2) \
        - 2 * np.real(alpha ** 2 * np.exp(-1j * phi) * np.sinh(r) * np.cosh(r))

    mean_analytic = np.abs(alpha) ** 2 + np.sinh(r) ** 2
    assert np.isclose(float(pnd_cov), pnd_cov_analytic, atol=tol, rtol=0)
    assert np.isclose(photon_number_mean(mu, cov, 0, hbar=hbar), mean_analytic, atol=tol, rtol=0)


@pytest.mark.parametrize("hbar", [0.1, 1, 2])
@pytest.mark.parametrize("etas", [0.1, 0.4, 0.9, 1.0])
@pytest.mark.parametrize("etai", [0.1, 0.4, 0.9, 1.0])
@pytest.mark.parametrize("parallel", [True, False])
def test_update_with_loss_two_mode_squeezed(etas, etai, parallel, hbar, monkeypatch):
    """Test the probabilities are updated correctly for a lossy two mode squeezed vacuum state"""

    if parallel: # set single-thread use in OpenMP
        monkeypatch.setenv("OMP_NUM_THREADS", "1")

    cov2 = two_mode_squeezing(np.arcsinh(1.0), 0.0)
    cov2 = hbar * cov2 @ cov2.T / 2.0
    mean2 = np.zeros([4])
    eta2 = [etas, etai]
    cov2l = np.copy(cov2)

    for i, eta in enumerate(eta2):
        mean2, cov2l = loss(mean2, cov2l, eta, i, hbar=hbar)

    cutoff = 6
    probs = probabilities(mean2, cov2l, cutoff, parallel=parallel, hbar=hbar)
    probs_lossless = probabilities(mean2, cov2, 3 * cutoff, parallel=parallel, hbar=hbar)
    probs_updated = update_probabilities_with_loss(eta2, probs_lossless)

    assert np.allclose(probs, probs_updated[:cutoff, :cutoff], atol=1.0e-5)


@pytest.mark.parametrize("hbar", [0.1, 1, 2])
@pytest.mark.parametrize("etas", [0.1, 0.4, 0.9, 1.0])
@pytest.mark.parametrize("etai", [0.1, 0.4, 0.9, 1.0])
@pytest.mark.parametrize("parallel", [True, False])
def test_update_with_loss_coherent_states(etas, etai, parallel, hbar, monkeypatch):
    """Checks probabilities are updated correctly for coherent states"""

    if parallel: # set single-thread use in OpenMP
        monkeypatch.setenv("OMP_NUM_THREADS", "1")

    n_modes = 2
    cov = hbar * np.identity(2 * n_modes) / 2
    eta_vals = [etas, etai]
    means = 2 * np.random.rand(2 * n_modes)
    means_lossy = np.sqrt(np.array(eta_vals + eta_vals)) * means
    cutoff = 6
    probs_lossless = probabilities(means, cov, 10 * cutoff, parallel=parallel, hbar=hbar)

    probs = probabilities(means_lossy, cov, cutoff, parallel=parallel, hbar=hbar)
    probs_updated = update_probabilities_with_loss(eta_vals, probs_lossless)

    assert np.allclose(probs, probs_updated[:cutoff, :cutoff], atol=1.0e-5)


@pytest.mark.parametrize("eta", [0.1, 0.5, 1.0])
def test_loss_is_stochastic_matrix(eta):
    """Test the loss matrix is an stochastic matrix, implying that the sum
    of the entries a long the rows is 1"""
    n = 50
    M = loss_mat(eta, n)
    assert np.allclose(np.sum(M, axis=1), np.ones([n]))


@pytest.mark.parametrize("eta", [0.1, 0.5, 1.0])
def test_loss_is_nonnegative_matrix(eta):
    """Test the loss matrix is a nonnegative matrix"""
    n = 50
    M = loss_mat(eta, n)
    assert np.alltrue(M >= 0.0)


@pytest.mark.parametrize("eta", [-1.0, 2.0])
def test_loss_value_error(eta):
    """Tests the correct error is raised"""
    n = 50
    with pytest.raises(
        ValueError, match="The transmission parameter eta should be a number between 0 and 1."
    ):
        loss_mat(eta, n)


@pytest.mark.parametrize("num_modes", [1, 2, 3])
@pytest.mark.parametrize("parallel", [True, False])
def test_update_with_noise_coherent(num_modes, parallel, monkeypatch):
    """ Test that adding noise on coherent states gives the same probabilities at some other coherent states"""

    if parallel: # set single-thread use in OpenMP
        monkeypatch.setenv("OMP_NUM_THREADS", "1")

    cutoff = 15
    nbar_vals = np.random.rand(num_modes)
    noise_dists = np.array([poisson.pmf(np.arange(cutoff), nbar) for nbar in nbar_vals])
    hbar = 2
    beta = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
    means = Means(np.concatenate((beta, beta.conj())), hbar=hbar)
    cov = hbar * np.identity(2 * num_modes) / 2
    cutoff = 10

    probs = probabilities(means, cov, cutoff, parallel=parallel, hbar=2)
    updated_probs = update_probabilities_with_noise(noise_dists, probs)
    beta_expected = np.sqrt(nbar_vals + np.abs(beta) ** 2)
    means_expected = Means(
        np.concatenate((beta_expected, beta_expected.conj())), hbar=hbar
    )
    expected = probabilities(means_expected, cov, cutoff, parallel=parallel, hbar=2)
    assert np.allclose(updated_probs, expected)


def test_update_with_noise_coherent_value_error():
    """Tests the correct error is raised"""
    cutoff = 15
    num_modes = 3
    nbar_vals = np.random.rand(num_modes - 1)
    noise_dists = np.array([poisson.pmf(np.arange(cutoff), nbar) for nbar in nbar_vals])
    hbar = 2
    beta = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
    means = Means(np.concatenate((beta, beta.conj())), hbar=hbar)
    cov = hbar * np.identity(2 * num_modes) / 2
    cutoff = 10
    probs = probabilities(means, cov, cutoff, hbar=2)
    with pytest.raises(
        ValueError,
        match="The list of probability distributions probs_noise and the tensor of probabilities probs have incompatible dimensions.",
    ):
        update_probabilities_with_noise(noise_dists, probs)


@pytest.mark.parametrize("hbar", [1 / 2, 1, 2, 1.6])
@pytest.mark.parametrize("num_modes", np.arange(5, 10))
@pytest.mark.parametrize("pure", [True, False])
@pytest.mark.parametrize("block_diag", [True, False])
def test_fidelity_with_self(num_modes, hbar, pure, block_diag):
    """Test that the fidelity of two identical quantum states is 1"""
    cov = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
    means = np.random.rand(2 * num_modes)
    assert np.allclose(fidelity(means, cov, means, cov, hbar=hbar), 1, atol=1e-4)


@pytest.mark.parametrize("hbar", [1 / 2, 1, 2, 1.6])
@pytest.mark.parametrize("num_modes", np.arange(5, 10))
@pytest.mark.parametrize("pure", [True, False])
@pytest.mark.parametrize("block_diag", [True, False])
def test_fidelity_is_symmetric(num_modes, hbar, pure, block_diag):
    """Test that the fidelity is symmetric and between 0 and 1"""
    cov1 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
    means1 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
    cov2 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
    means2 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
    f12 = fidelity(means1, cov1, means2, cov2, hbar=hbar)
    f21 = fidelity(means2, cov2, means1, cov1, hbar=hbar)
    assert np.allclose(f12, f21)
    assert 0 <= np.real_if_close(f12) < 1.0


@pytest.mark.parametrize("num_modes", np.arange(5, 10))
@pytest.mark.parametrize("hbar", [0.5, 1, 2, 1.6])
def test_fidelity_coherent_state(num_modes, hbar):
    """Test the fidelity of two multimode coherent states"""
    beta1 = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
    beta2 = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
    means1 = Means(np.concatenate([beta1, beta1.conj()]), hbar=hbar)
    means2 = Means(np.concatenate([beta2, beta2.conj()]), hbar=hbar)
    cov1 = hbar * np.identity(2 * num_modes) / 2
    cov2 = hbar * np.identity(2 * num_modes) / 2
    fid = fidelity(means1, cov1, means2, cov2, hbar=hbar)
    expected = np.exp(-np.linalg.norm(beta1 - beta2) ** 2)
    assert np.allclose(expected, fid)


@pytest.mark.parametrize("hbar", [0.5, 1, 2, 1.6])
@pytest.mark.parametrize("r", [-2, 0, 2])
@pytest.mark.parametrize("alpha", np.random.rand(10) + 1j * np.random.rand(10))
def test_fidelity_vac_to_displaced_squeezed(r, alpha, hbar):
    """Calculates the fidelity between a coherent squeezed state and vacuum"""
    cov1 = np.diag([np.exp(2 * r), np.exp(-2 * r)]) * hbar / 2
    means1 = Means(np.array([alpha, np.conj(alpha)]), hbar=hbar)
    means2 = np.zeros([2])
    cov2 = np.identity(2) * hbar / 2
    expected = (
        np.exp(-np.abs(alpha) ** 2)
        * np.abs(np.exp(np.tanh(r) * np.conj(alpha) ** 2))
        / np.cosh(r)
    )
    assert np.allclose(expected, fidelity(means1, cov1, means2, cov2, hbar=hbar))


@pytest.mark.parametrize("hbar", [0.5, 1, 2, 1.6])
@pytest.mark.parametrize("r1", np.random.rand(3))
@pytest.mark.parametrize("r2", np.random.rand(3))
def test_fidelity_squeezed_vacuum(r1, r2, hbar):
    """Tests fidelity between two squeezed states"""
    cov1 = np.diag([np.exp(2 * r1), np.exp(-2 * r1)]) * hbar / 2
    cov2 = np.diag([np.exp(2 * r2), np.exp(-2 * r2)]) * hbar / 2
    mu = np.zeros([2])
    assert np.allclose(1 / np.cosh(r1 - r2), fidelity(mu, cov1, mu, cov2, hbar=hbar))


def test_fidelity_wrong_shape():
    """Tests the correct error is raised"""
    cov1 = np.identity(2)
    cov2 = np.identity(4)
    mu = np.zeros(2)
    with pytest.raises(
        ValueError, match="The inputs have incompatible shapes"
    ):
        fidelity(mu, cov1, mu, cov2)
