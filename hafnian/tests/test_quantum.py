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
"""Tests for the hafnian quantum functions"""
import pytest

import numpy as np
from hafnian.quantum import (
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
)


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
            [
                [(N + 1) * n, (N + 1) * n + m],
                [(N + 1) * n + N * m, (N + 1) * n + N * m + m],
            ]
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

    with pytest.raises(
        ValueError, match="Provided mode is larger than the number of subsystems."
    ):
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
    A = np.zeros([2, 2])
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


mu = np.array(
    [0.04948628, -0.55738964, 0.71298259, 0.17728629, -0.14381673, 0.33340778]
)


@pytest.mark.parametrize("t", [t0, t1, t2, t3, t4])
def test_density_matrix_element_disp(t):
    """Test density matrix elements for a state with displacement"""
    beta = Beta(mu)
    A = Amat(V)
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])
    #    res = density_matrix_element(beta, A, Q, el[0], el[1])
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
    A = Amat(V)
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(Means(beta), Covmat(Q), el[0], el[1])
    #    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(ex, res)


def test_density_matrix_vacuum():
    """Test density matrix for a squeezed state"""
    r = 0.43

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


def test_density_matrix_displaced_squeezed():
    """Test density matrix for a squeezed state"""
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
    expected = np.array([[ 0.89054874,  0.15018085+0.05295904j, -0.23955467+0.01263025j, -0.0734589 -0.02452154j, 0.07862323-0.00868528j],
       [ 0.15018085-0.05295904j,  0.02847564, -0.03964706+0.01637575j, -0.01384625+0.00023317j, 0.01274241-0.00614023j],
       [-0.23955467-0.01263025j, -0.03964706-0.01637575j, 0.06461854,  0.01941242+0.00763805j, -0.02127257+0.00122123j],
       [-0.0734589 +0.02452154j, -0.01384625-0.00023317j, 0.01941242-0.00763805j, 0.00673463, -0.00624626+0.00288134j],
       [ 0.07862323+0.00868528j,  0.01274241+0.00614023j, -0.02127257-0.00122123j, -0.00624626-0.00288134j, 0.00702606]])
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
    val = is_valid_cov(V, hbar=hbar)
    assert val


@pytest.mark.parametrize("nbar", [0, 1, 2])
def test_is_pure_cov_thermal(nbar):
    """ Test if is_pure_cov for vacuum and thermal states"""
    hbar = 2
    dim = 10
    cov = (2 * nbar + 1) * np.identity(dim)
    val = is_pure_cov(cov, hbar=hbar)
    if nbar == 0:
        assert val == True
    else:
        assert val == False


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
def test_pure_state_amplitude_two_mode_squezed(i, j):
    """ Tests pure state amplitude for a two mode squeezed vacuum state """
    nbar = 1.0
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    if i != j:
        exact = 0.0
    else:
        exact = (
            np.exp(-1j * i * phase)
            * (nbar / (1.0 + nbar)) ** (i / 2)
            / np.sqrt(1.0 + nbar)
        )
    num = pure_state_amplitude(mu, cov, [i, j])
    assert np.allclose(exact, num)


@pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
def test_pure_state_amplitude_coherent(i):
    """ Tests pure state amplitude for a coherent state """
    cov = np.identity(2)
    mu = np.array([1.0, 2.0])
    beta = Beta(mu)
    alpha = beta[0]
    exact = (
        np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** i / np.sqrt(np.math.factorial(i))
    )
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
    exact = np.array([(
            np.exp(-1j * i * phase)
            * (nbar / (1.0 + nbar)) ** (i / 2)
            / np.sqrt(1.0 + nbar)
        ) for i in range(cutoff)])
    psi = state_vector(mu, cov, cutoff = cutoff)
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
    exact = np.diag(np.array([(
            np.exp(-1j * i * phase)
            * (nbar / (1.0 + nbar)) ** (i / 2)
            / np.sqrt(1.0 + nbar)
        ) for i in range(cutoff)]))
    val = 2
    post_select={0: val}
    psi = state_vector(mu, cov, cutoff = cutoff, post_select=post_select)
    expected = exact[val]
    assert np.allclose(psi, expected)




def test_state_vector_coherent():
    """ Tests state vector for a coherent state """
    cutoff = 5
    cov = np.identity(2)
    mu = np.array([1.0, 2.0])
    beta = Beta(mu)
    alpha = beta[0]
    exact = np.array([(
        np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** i / np.sqrt(np.math.factorial(i))
    ) for i in range(cutoff)])
    num = state_vector(mu, cov, cutoff = cutoff)
    assert np.allclose(exact, num)

def test_state_vector_two_mode_squeezed_post_normalize():
    """ Tests state_vector for a two mode squeezed vacuum state """
    nbar = 1.0
    cutoff = 5
    phase = np.pi / 8
    r = np.arcsinh(np.sqrt(nbar))
    cov = TMS_cov(r, phase)
    mu = np.zeros([4], dtype=np.complex)
    exact = np.diag(np.array([(
            np.exp(-1j * i * phase)
            * (nbar / (1.0 + nbar)) ** (i / 2)
            / np.sqrt(1.0 + nbar)
        ) for i in range(cutoff)]))
    val = 2
    post_select={0: val}
    psi = state_vector(mu, cov, cutoff = cutoff, post_select=post_select, normalize = True)
    expected = exact[val]
    expected = expected/np.linalg.norm(expected)
    assert np.allclose(psi, expected)