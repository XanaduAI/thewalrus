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
"""
Torontonian Python interface
"""
import numpy as np
import numba
from thewalrus.quantum.conversions import Qmat, reduced_gaussian
from ._hafnian import reduction, find_kept_edges, nb_ix


def tor_input_checks(A, loops=None):
    """Checks the correcteness of the inputs for the torontonian/montrealer.
    Args:
        A (array): an NxN array of even dimensions.
        loops (array): optinal argument, an N-length vector of even dimensions.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if matshape[0] % 2 != 0:
        raise ValueError("matrix dimension must be even")

    if loops is not None:
        if not isinstance(loops, np.ndarray):
            raise TypeError("Input matrix must be a NumPy array.")
        if matshape[0] != len(loops):
            raise ValueError("gamma must be a vector matching the dimension of A")


def tor(A, recursive=True):
    """Returns the Torontonian of a matrix.

    Args:
        A (array): a square array of even dimensions.
        recursive: use the faster recursive implementation.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    tor_input_checks(A)
    return rec_torontonian(A) if recursive else numba_tor(A)


def ltor(A, gamma, recursive=True):
    """Returns the loop Torontonian of an NxN matrix and an N-length vector.

    Args:
        A (array): an NxN array of even dimensions.
        gamma (array): an N-length vector of even dimensions
        recursive: use the faster recursive implementation

    Returns:
        np.float64 or np.complex128: the loop torontonian of matrix A, vector gamma
    """
    tor_input_checks(A, gamma)

    return rec_ltorontonian(A, gamma) if recursive else numba_ltor(A, gamma)


def threshold_detection_prob(
    mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10
):  # pylint: disable=too-many-arguments
    r"""Threshold detection probabilities for Gaussian states.
    Formula from Jake Bulmer, Nicolas Quesada and Stefano Paesani.
    When state is displaced, `threshold_detection_prob_displacement` is called.
    Otherwise, `tor` is called.


    Args:
        mu (1d array) : means of xp Gaussian Wigner function
        cov (2d array) : : xp Wigner covariance matrix
        det_pattern (1d array) : array of {0,1} to describe the threshold detection outcome
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        np.float64 : probability of detection pattern
    """

    n = cov.shape[0] // 2

    if np.allclose(mu, 0, atol=atol, rtol=rtol):
        # no displacement
        Q = Qmat(cov, hbar)
        O = np.eye(2 * n) - np.linalg.inv(Q)
        rpt2 = np.concatenate((det_pattern, det_pattern))
        Os = reduction(O, rpt2)
        return tor(Os) / np.sqrt(np.linalg.det(Q))

    x = mu[:n]
    p = mu[n:]

    alpha = np.concatenate((x + 1j * p, x - 1j * p)) / np.sqrt(2 * hbar)

    sigma = Qmat(cov, hbar=hbar).conj()
    I = np.eye(2 * n)
    inv_sigma = np.linalg.inv(sigma)
    O = I - inv_sigma
    gamma = (inv_sigma @ alpha).conj()

    gamma_red, O_red = reduced_gaussian(gamma, O, np.where(np.array(det_pattern) == 1)[0])
    return numba_vac_prob(alpha, sigma) * numba_ltor(O_red, gamma_red).real


@numba.jit(nopython=True)
def numba_tor(O):  # pragma: no cover
    r"""Returns the Torontonian of a matrix using numba.

    Args:
        O (array): a square, symmetric array of even dimensions.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    N = O.shape[0] // 2
    N_odd = N % 2

    steps = 2**N
    ones = np.ones(N, dtype=np.int8)

    total = 0.0
    for j in numba.prange(steps):
        X_modes = find_kept_edges(j, ones)
        lenX = X_modes.sum()
        I = np.eye(2 * lenX, dtype=O.dtype)
        plusminus = (-1) ** ((N_odd - lenX % 2) % 2)

        kept_modes = np.where(X_modes != 0)[0]
        kept_rows = np.concatenate((kept_modes, kept_modes + N))
        O_XX = nb_ix(O, kept_rows, kept_rows)

        bottom = np.sqrt(O.dtype.type(np.real(np.linalg.det(I - O_XX))))

        total += plusminus / bottom

    return total


@numba.jit(nopython=True)
def quad_cholesky(L, Z, idx, mat):  # pragma: no cover
    """Returns the Cholesky factorization of a matrix using sub-matrix of prior

    Cholesky based on the new matrix and lower right quadrant.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): previous Cholesky
        Z (array): new sub-matrix indices
        idx: index of starting row/column of lower right quadrant
        mat (array): new matrix

    Returns:
        np.float64 or np.complex128: the Cholesky of matrix ``mat``
    """
    Ls = nb_ix(L, Z, Z)
    for i in range(idx, len(mat)):
        for j in range(idx, i):
            z = 0.0
            for k in range(j):
                z += Ls[i, k] * Ls[j, k].conjugate()
            Ls[i, j] = (mat[i][j] - z) / Ls[j, j]
        z = 0.0
        for k in range(i):
            z += Ls[i, k] * Ls[i, k].conjugate()
        Ls[i, i] = L.dtype.type(np.real(np.sqrt(mat[i, i] - z)))
    return Ls


@numba.jit(nopython=True)
def recursiveTor(L, modes, A, n):  # pragma: no cover
    """Returns the recursive Torontonian sub-computation of a matrix
    using numba.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): current Cholesky
        modes (array): optical mode
        A (array): a square, symmetric array of even dimensions
        n: size of the original matrix

    Returns:
        np.float64 or np.complex128: the recursive torontonian
        sub-computation of matrix ``A``
    """
    tot = 0.0
    if len(modes) == 0:
        start = 0
    else:
        start = modes[-1] + 1

    for i in range(start, n):
        nextModes = np.append(modes, i)
        nm, idx = len(A) >> 1, (i - len(modes)) * 2
        Z = np.concatenate((np.arange(idx), np.arange(idx + 2, nm * 2)), axis=0)
        nm -= 1

        Az = nb_ix(A, Z, Z)
        Ls = quad_cholesky(L, Z, idx, np.eye(2 * nm) - Az)
        det = np.square(np.prod(np.diag(Ls)))
        tot += ((-1) ** len(nextModes)) / np.sqrt(det) + recursiveTor(Ls, nextModes, Az, n)

    return tot


@numba.jit(nopython=True)
def rec_torontonian(A):  # pragma: no cover
    """Returns the Torontonian of a matrix using numba.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        A (array): a square, symmetric array of even dimensions

    Returns:
        np.float64 or np.complex128: the torontonian of matrix ``A``
    """
    n = A.shape[0] >> 1
    Z = np.empty((2 * n,), dtype=np.int_)
    Z[0::2] = np.arange(0, n)
    Z[1::2] = np.arange(n, 2 * n)
    A = nb_ix(A, Z, Z)
    L = np.linalg.cholesky(np.eye(2 * n) - A)
    det = np.square(np.prod(np.diag(L)))
    return 1 / np.sqrt(det) + recursiveTor(L, np.empty(0, dtype=np.int_), A, n)


@numba.jit(nopython=True)
def solve_triangular(L, y):  # pragma: no cover
    """Returns the solution to the inverse of a lower non-unit
    triangular matrix times a vector like the dtrsv function of
    LAPACK/BLAS or scipy solve_triangular.

    Args:
        L (array): invertible triangular matrix
        y (array): vector

    Returns:
        np.float64 or np.complex128: solution of L^(-1)y
    """
    n = len(y)
    x = np.copy(y)
    for j in range(0, n):
        if x[j] == 0:
            continue
        x[j] = x[j] / L[j, j]
        temp = x[j]
        for i in range(j + 1, n):
            x[i] -= temp * L[i, j]
    return x


@numba.jit(nopython=True)
def recursiveLTor(L, modes, A, n, gammaL):  # pragma: no cover
    """Returns the recursive loop Torontonian sub-computation of a matrix
    using numba.

    Combines algorithm from papers:
    https://arxiv.org/pdf/2109.04528.pdf
    https://arxiv.org/pdf/2202.04600.pdf

    Args:
        L (array): current Cholesky
        modes (array): optical mode
        A (array): a square, symmetric array of even dimensions
        n: size of the original matrix
        gammaL (array): a vector of even dimension

    Returns:
        np.float64 or np.complex128: the recursive loop torontonian
        sub-computation of matrix ``A`` and vector ``gammaL``
    """
    tot = 0.0
    if len(modes) == 0:
        start = 0
    else:
        start = modes[-1] + 1
    for i in range(start, n):
        nextModes = np.append(modes, i)
        nm, idx = len(A) >> 1, (i - len(modes)) * 2
        Z = np.concatenate((np.arange(idx), np.arange(idx + 2, nm * 2)), axis=0)
        nm -= 1
        Az = nb_ix(A, Z, Z)
        Ls = quad_cholesky(L, Z, idx, np.eye(2 * nm) - Az)
        det = np.square(np.prod(np.diag(Ls)))
        gammaX = gammaL[Z]
        Lsinv = solve_triangular(Ls, gammaX.conj())
        lc = Lsinv.conj().T @ Lsinv
        tot += ((-1) ** len(nextModes)) * np.exp(0.5 * lc) / np.sqrt(det) + recursiveLTor(
            Ls, nextModes, Az, n, gammaX
        )
    return tot


@numba.jit(nopython=True)
def rec_ltorontonian(A, gamma):  # pragma: no cover
    """Returns the loop Torontonian of a matrix using numba.

    Combines algorithm from papers:
    https://arxiv.org/pdf/2109.04528.pdf
    https://arxiv.org/pdf/2202.04600.pdf

    Args:
        A (array): a square, symmetric array of even dimensions
        gamma (array): a vector of even dimension

    Returns:
        np.float64 or np.complex128: the torontonian of matrix ``A``
        and vector ``gamma``
    """
    n = A.shape[0] >> 1
    Z = np.empty((2 * n,), dtype=np.int_)
    Z[0::2] = np.arange(0, n)
    Z[1::2] = np.arange(n, 2 * n)
    A = nb_ix(A.astype(np.complex128), Z, Z)
    gamma = gamma[Z].astype(np.complex128)
    L = np.linalg.cholesky(np.eye(2 * n) - A)
    det = np.square(np.prod(np.diag(L)))
    Ls = solve_triangular(L, gamma.conj())
    lc = Ls.conj().T @ Ls
    return np.exp(0.5 * lc) / np.sqrt(det) + recursiveLTor(
        L, np.empty(0, dtype=np.int_), A, n, gamma
    )


@numba.jit(nopython=True)
def numba_vac_prob(alpha, sigma):  # pragma: no cover
    r"""
    Return the vacuum probability of a Gaussian state with Q function `sigma`
    and displacement vector, `alpha`.


    Args:
        alpha (array): a 2M-length vector describing the complex displacement
        sigma (array): a 2Mx2M matrix describing the Q-function covariance matrix
    Returns:
        float: vacuum probability of Gaussian state
    """
    alpha = alpha.astype(np.complex128)
    sigma = sigma.astype(np.complex128)
    return (
        np.exp(-0.5 * alpha.conj() @ np.linalg.inv(sigma) @ alpha).real
        / np.sqrt(np.linalg.det(sigma))
    ).real


@numba.jit(nopython=True, parallel=True)
def numba_ltor(O, gamma):  # pragma: no cover
    r"""Returns the loop Torontonian of a matrix using numba.

    Args:
        O (array): a square, symmetric array of even dimensions.
        gamma (array): a vector of even dimension

    Returns:
        np.complex128: the loop torontonian of matrix O, vector gamma
    """
    N = O.shape[0] // 2
    N_odd = N % 2

    steps = 2**N
    ones = np.ones(N, dtype=np.int8)

    gamma = gamma.astype(np.complex128)
    O = O.astype(np.complex128)

    total = 0.0
    for j in numba.prange(steps):
        X_modes = find_kept_edges(j, ones)
        lenX = X_modes.sum()
        I = np.eye(2 * lenX, dtype=O.dtype)
        plusminus = (-1) ** ((N_odd - lenX % 2) % 2)

        kept_modes = np.where(X_modes != 0)[0]
        kept_rows = np.concatenate((kept_modes, kept_modes + N))
        O_XX = nb_ix(O, kept_rows, kept_rows)

        I_m_O_XX = I - O_XX
        I_m_O_XX_inv = np.linalg.inv(I_m_O_XX)

        gamma_X = gamma[kept_rows]

        top = np.exp(0.5 * gamma_X @ I_m_O_XX_inv @ gamma_X.conj())

        bottom_complex = np.linalg.det(I_m_O_XX)
        bottom = np.sqrt(O.dtype.type(bottom_complex.real))

        total += plusminus * top / bottom

    return total
