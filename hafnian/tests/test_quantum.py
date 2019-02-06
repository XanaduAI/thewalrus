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
from hafnian.quantum import (reduced_gaussian, Xmat, Qmat, Amat, Beta,
                             prefactor, density_matrix_element, density_matrix)


def TMS_cov(r, phi):
    """returns the covariance matrix of a TMS state"""
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array([[ch, cp*sh, 0, sp*sh],
                  [cp*sh, ch, sp*sh, 0],
                  [0, sp*sh, ch, -cp*sh],
                  [sp*sh, 0, -cp*sh, ch]])

    return S @ S.T


@pytest.mark.parametrize("n", [0, 1, 2])
def test_reduced_gaussian(n):
    """test that reduced gaussian returns the correct result"""
    m = 5
    N = 2*m
    mu = np.arange(N)
    cov = np.arange(N**2).reshape(N, N)
    res = reduced_gaussian(mu, cov, n)
    assert np.all(res[0] == np.array([n, n+m]))
    assert np.all(res[1] == np.array([[(N+1)*n, (N+1)*n+m], [(N+1)*n+N*m, (N+1)*n+N*m+m]]))


def test_reduced_gaussian_two_mode():
    """test that reduced gaussian returns the correct result"""
    m = 5
    N = 2*m
    mu = np.arange(N)
    cov = np.arange(N**2).reshape(N, N)
    res = reduced_gaussian(mu, cov, [0, 2])
    assert np.all(res[0] == np.array([0, 2, m, m+2]))


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
    V = TMS_cov(np.arcsinh(1), 0)
    res = Qmat(V)

    q = np.fliplr(np.diag([2.]*4))
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

    B = np.fliplr(np.diag([1/np.sqrt(2)]*2))
    O = np.zeros_like(B)
    ex = np.block([[B, O], [O, B]])
    assert np.allclose(res, ex)


def test_Amat_TMS_using_Q():
    """test Amat returns correct result for a two-mode squeezed state"""
    q = np.fliplr(np.diag([2.]*4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)
    res = Amat(Q, cov_is_qmat=True)

    B = np.fliplr(np.diag([1/np.sqrt(2)]*2))
    O = np.zeros_like(B)
    ex = np.block([[B, O], [O, B]])
    assert np.allclose(res, ex)


def test_beta():
    """test the correct beta is returned"""
    mu = np.arange(4)
    res = Beta(mu)

    alpha = (mu[:2] + 1j*mu[2:])/np.sqrt(2*2)
    ex = np.concatenate([alpha, alpha.conj()])
    assert np.allclose(res, ex)


def test_prefactor_vacuum():
    """test the correct prefactor of 0.5 is calculated for a vacuum state"""
    Q = np.identity(2)
    A = np.zeros([2, 2])
    beta = np.zeros([2])

    res = prefactor(beta, A, Q)
    ex = 1
    assert np.allclose(res, ex)


def test_prefactor_TMS():
    """test the correct prefactor of 0.5 is calculated for a TMS state"""
    q = np.fliplr(np.diag([2.]*4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)

    B = np.fliplr(np.diag([1/np.sqrt(2)]*2))
    O = np.zeros_like(B)
    A = np.block([[B, O], [O, B]])

    beta = np.zeros([4])

    res = prefactor(beta, A, Q)
    ex = 0.5
    assert np.allclose(res, ex)


def test_prefactor_with_displacement():
    """test the correct prefactor of 0.5 is calculated for a TMS state"""
    q = np.fliplr(np.diag([2.]*4))
    np.fill_diagonal(q, np.sqrt(2))
    Q = np.fliplr(q)
    Qinv = np.linalg.inv(Q)

    B = np.fliplr(np.diag([1/np.sqrt(2)]*2))
    O = np.zeros_like(B)
    A = np.block([[B, O], [O, B]])

    beta = np.zeros([4])

    res = prefactor(beta, A, Q)
    ex =  np.exp(-0.5*beta @ Qinv @ beta.conj())/np.sqrt(np.linalg.det(Q))
    assert np.allclose(res, ex)


def test_density_matrix_element_vacuum():
    Q = np.identity(2)
    A = np.zeros([2, 2])
    beta = np.zeros([2])

    el = [[0], [0]]
    ex = 1
    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(ex, res)

    el = [[1], [1]]
    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(0, res)

    el = [[1], [0]]
    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(0, res)


# density matrix element
t0 = [[0,0,0],[0,0,0]], 0.7304280085350833
t1 = [[1,0,1],[0,0,0]], -0.009290003060522444+0.002061369459502776j
t2 = [[1,0,1],[0,0,1]], -0.004088994220552936-0.0009589367814578206j
t3 = [[0,2,0],[0,0,0]], 0.003384487265468196-0.03127114305387707j
t4 = [[0,2,0],[0,2,3]], -8.581668587044574e-05-6.134980446713632e-05j

V = np.array([[0.6964938, 0.06016962, -0.01970064, 0.03794393, 0.07913992, -0.08890985],
             [0.06016962, 0.85435861, -0.01648842, 0.10493462, 0.01223525, 0.12484726],
             [-0.01970064, -0.01648842, 0.89450003, -0.13182502, 0.13529134, -0.10621978],
             [0.03794393, 0.10493462, -0.13182502, 1.47820656, -0.11611807, 0.05634905],
             [0.07913992, 0.01223525, 0.13529134, -0.11611807, 1.20819301, -0.0061647 ],
             [-0.08890985, 0.12484726, -0.10621978, 0.05634905, -0.0061647, 1.1636695 ]])

mu = np.array([0.04948628, -0.55738964, 0.71298259, 0.17728629, -0.14381673, 0.33340778])

@pytest.mark.parametrize("t", [t0, t1, t2, t3, t4])
def test_density_matrix_element_disp(t):
    beta = Beta(mu)
    A = Amat(V)
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(ex, res)


# density matrix element
t0 = [[0,0,0],[0,0,0]], 0.9645169885669383
t1 = [[1,0,1],[0,0,0]], -0.016156769991363732+0.05039373212461916j
t2 = [[1,0,1],[0,0,1]], 0
t3 = [[0,2,0],[0,0,0]], -0.05911275266690908-0.0049431163436861j
t4 = [[0,2,0],[0,2,3]], 0


@pytest.mark.parametrize("t", [t0, t1, t2, t3, t4])
def test_density_matrix_element_no_disp(t):
    beta = Beta(np.zeros([6]))
    A = Amat(V)
    Q = Qmat(V)

    el = t[0]
    ex = t[1]
    res = density_matrix_element(beta, A, Q, el[0], el[1])
    assert np.allclose(ex, res)
