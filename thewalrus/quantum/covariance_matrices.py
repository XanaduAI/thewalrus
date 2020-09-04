# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions for transforming one type of covariance-matrix-like object into
another as well as various property tests for covariance matrices.
"""

import numpy as np

from ..symplectic import sympmat
from .._hafnian import hafnian, reduction


################################################################################
# Construct the reduced means and cov of a Gaussian state
################################################################################


def reduced_gaussian(mu, cov, modes):
    r""" Returns the vector of means and the covariance matrix of the specified modes.

    Args:
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        modes (int of Sequence[int]): indices of the requested modes

    Returns:
        tuple (means, cov): where means is an array containing the vector of means,
        and cov is a square array containing the covariance matrix.
    """
    N = len(mu) // 2

    # reduce rho down to specified subsystems
    if isinstance(modes, int):
        modes = [modes]

    if np.any(np.array(modes) > N):
        raise ValueError("Provided mode is larger than the number of subsystems.")

    if len(modes) == N:
        # reduced state is full state
        return mu, cov

    ind = np.concatenate([np.array(modes), np.array(modes) + N])
    rows = ind.reshape(-1, 1)
    cols = ind.reshape(1, -1)

    return mu[ind], cov[rows, cols]


################################################################################
# Transform one type of covariance-matrix-like object into another
################################################################################


def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    return X


def Qmat(cov, hbar=2):
    r"""Returns the :math:`Q` Husimi matrix of the Gaussian state.

    Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the :math:`Q` matrix.
    """
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q


def Covmat(Q, hbar=2):
    r"""Returns the Wigner covariance matrix in the :math:`xp`-ordering of the Gaussian state.
    This is the inverse function of Qmat.

    Args:
        Q (array): :math:`2N\times 2N` Husimi Q matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the :math:`xp`-ordered covariance matrix in the xp-ordering.
    """
    # number of modes
    n = len(Q) // 2
    I = np.identity(n)
    N = Q[0:n, 0:n] - I
    M = Q[n : 2 * n, 0:n]
    mm11a = 2 * (N.real + M.real) + np.identity(n)
    mm22a = 2 * (N.real - M.real) + np.identity(n)
    mm12a = 2 * (M.imag + N.imag)
    cov = np.block([[mm11a, mm12a], [mm12a.T, mm22a]])

    return (hbar / 2) * cov


def Amat(cov, hbar=2, cov_is_qmat=False):
    r"""Returns the :math:`A` matrix of the Gaussian state whose hafnian gives the photon number probabilities.

    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cov_is_qmat (bool): if ``True``, it is assumed that ``cov`` is in fact the Q matrix.

    Returns:
        array: the :math:`A` matrix.
    """
    # number of modes
    N = len(cov) // 2
    X = Xmat(N)

    # inverse Q matrix
    if cov_is_qmat:
        Q = cov
    else:
        Q = Qmat(cov, hbar=hbar)

    Qinv = np.linalg.inv(Q)

    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    return A


def normal_ordered_expectation(mu, cov, rpt, hbar=2):
    r"""Calculates the expectation value of the normal ordered product
    :math:`\prod_{i=0}^{N-1} a_i^{\dagger n_i} \prod_{j=0}^{N-1} a_j^{m_j}` with respect to an N-mode Gaussian state,
    where :math:`\text{rpt}=(n_0, n_1, \ldots, n_{N-1}, m_0, m_1, \ldots, m_{N-1})`.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        rpt (list): integers specifying the terms to calculate.
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        (float): expectation value of the normal ordered product of operators
    """
    alpha = Beta(mu, hbar=hbar)
    n = len(cov)
    V = (Qmat(cov, hbar=hbar) - np.identity(n)) @ Xmat(n // 2)
    A = reduction(V, rpt)
    if np.allclose(mu, 0):
        res = np.conj(hafnian(A))
    else:
        np.fill_diagonal(A, reduction(np.conj(alpha), rpt))
        res = np.conj(hafnian(A, loop=True))
    return np.conj(res)


def Beta(mu, hbar=2):
    r"""Returns the vector of complex displacements and conjugate displacements.

    Args:
        mu (array): length-:math:`2N` means vector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the expectation values
        :math:`[\langle a_1\rangle, \langle a_2\rangle,\dots,\langle a_N\rangle, \langle a^\dagger_1\rangle, \dots, \langle a^\dagger_N\rangle]`
    """
    N = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:N] + 1j * mu[N:]) / np.sqrt(2 * hbar)
    # the expectation values (<a_1>, <a_2>,...,<a_N>, <a^\dagger_1>, ..., <a^\dagger_N>)
    return np.concatenate([alpha, alpha.conj()])


def Means(beta, hbar=2):
    r"""Returns the vector of real quadrature displacements.

    Args:
        beta (array): length-:math:`2N` means bivector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the quadrature expectation values
        :math:`[\langle q_1\rangle, \langle q_2\rangle,\dots,\langle q_N\rangle, \langle p_1\rangle, \dots, \langle p_N\rangle]`
    """

    N = len(beta) // 2
    alpha = beta[0:N]
    return np.sqrt(2 * hbar) * np.concatenate([alpha.real, alpha.imag])

################################################################################
# Test properties of covariance matrices
################################################################################


def is_valid_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
    r""" Checks if the covariance matrix is a valid quantum covariance matrix.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (bool): whether the given covariance matrix is a valid covariance matrix
    """
    (n, m) = cov.shape
    if n != m:
        # raise ValueError("The input matrix must be square")
        return False
    if not np.allclose(cov, np.transpose(cov), rtol=rtol, atol=atol):
        # raise ValueError("The input matrix is not symmetric")
        return False
    if n % 2 != 0:
        # raise ValueError("The input matrix is of even dimension")
        return False

    nmodes = n // 2
    vals = np.linalg.eigvalsh(cov + 0.5j * hbar * sympmat(nmodes))
    vals[np.abs(vals) < atol] = 0.0
    if np.all(vals >= 0):
        # raise ValueError("The input matrix violates the uncertainty relation")
        return True

    return False


def is_pure_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
    r""" Checks if the covariance matrix is a valid quantum covariance matrix
    that corresponds to a quantum pure state

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (bool): whether the given covariance matrix corresponds to a pure state
    """
    if is_valid_cov(cov, hbar=hbar, rtol=rtol, atol=atol):
        purity = 1 / np.sqrt(np.linalg.det(2 * cov / hbar))
        if np.allclose(purity, 1.0, rtol=rtol, atol=atol):
            return True

    return False


def is_classical_cov(cov, hbar=2, atol=1e-08):
    r""" Checks if the covariance matrix can be efficiently sampled.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (bool): whether the given covariance matrix corresponds to a classical state
    """

    if is_valid_cov(cov, hbar=hbar, atol=atol):
        (n, _) = cov.shape
        vals = np.linalg.eigvalsh(cov - 0.5 * hbar * np.identity(n))
        vals[np.abs(vals) < atol] = 0.0

        if np.all(vals >= 0):
            return True
    return False
