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
from thewalrus.quantum.conversions import Xmat, Amat
from ._hafnian import reduction


def tor(A, recursive=True):
    """Returns the Torontonian of a matrix.

    Args:
        A (array): a square array of even dimensions.

        recursive: use the faster recursive implementation.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    return rec_torontonian(A) if recursive else numba_tor(A)


@numba.jit(nopython=True)
def combinations(pool, r):  # pragma: no cover
    """Numba implementation of `itertools.combinations`.
    As itertools.combinations not callable from numba decorated functions.

    Taken from: https://stackoverflow.com/a/61393666

    Args:
        pool (Sequence): sequence to take combinations from

        r (int) : number of elements of combination
    Yields:
        results (list) : length r combination from pool
    """
    n = len(pool)
    indices = list(range(r))
    empty = not (n and (0 < r <= n))

    if not empty:
        result = [pool[i] for i in indices]
        yield result

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1

            result = [pool[i] for i in indices]
            yield result


@numba.jit(nopython=True)
def powerset(parent_set):  # pragma: no cover
    """Generates the powerset, the set of all the subsets, of its input. Does not include the empty set.

    Args:
        parent_set (Sequence) : sequence to take powerset from
    Return:
        subset (tuple) : subset of parent_set
    """
    n = len(parent_set)
    for i in range(n + 1):
        for subset in combinations(parent_set, i):
            yield subset


@numba.jit(nopython=True)
def nb_block(X):  # pragma: no cover
    """Numba implementation of `np.block`.
    Only suitable for 2x2 blocks.

    Taken from: https://stackoverflow.com/a/57562911

    Args:
        X (array) : arrays for blocks of matrix
    Return:
        array : the block matrix from X
    """
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.jit(nopython=True)
def numba_ix(arr, rows, cols):  # pragma: no cover
    """Numba implementation of `np.ix_`.
    Required due to numba lacking support for advanced numpy indexing.

    Args:
        arr (array) : matrix to take submatrix of
        rows (array) : rows to be selected in submatrix
        cols (array) : columns to be selected in submatrix

    Return:
        array: selected submatrix of arr, of shape `(len(rows), len(cols))`

    """
    return arr[rows][:, cols]


@numba.jit(nopython=True)
def Qmat_numba(cov, hbar=2):  # pragma: no cover
    r"""Numba compatible version of `thewalrus.quantum.Qmat`

    Returns the :math:`Q` Husimi matrix of the Gaussian state.
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

    x = cov[:N, :N] * (2.0 / hbar)
    xp = cov[:N, N:] * (2.0 / hbar)
    p = cov[N:, N:] * (2.0 / hbar)
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    Q = nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
    return Q


@numba.jit(nopython=True)
def threshold_detection_prob_displacement(mu, cov, det_pattern, hbar=2):  # pragma: no cover
    r"""Threshold detection probabilities for Gaussian states with displacement.
    Formula from Jake Bulmer and Stefano Paesani.


    Args:
        mu (1d array) : means of xp Gaussian Wigner function
        cov (2d array) : : xp Wigner covariance matrix
        det_pattern (1d numpy array) : array of {0,1} to describe the threshold detection outcome
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.

    Returns:
        np.float64 : probability of detection pattern
    """
    det_pattern = np.asarray(det_pattern).astype(np.int8)

    m = len(cov)
    assert cov.shape == (m, m)
    assert m % 2 == 0
    n = m // 2

    means_x = mu[:n]
    means_p = mu[n:]
    avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(
        2 * hbar
    )

    Q = Qmat_numba(cov, hbar=hbar)

    if max(det_pattern) > 1:
        raise ValueError(
            "When using threshold detectors, the detection pattern can contain only 1s or 0s."
        )

    nonzero_idxs = np.where(det_pattern == 1)[0]
    zero_idxs = np.where(det_pattern == 0)[0]

    ii1 = np.concatenate((nonzero_idxs, nonzero_idxs + n), axis=0)
    ii0 = np.concatenate((zero_idxs, zero_idxs + n), axis=0)

    Qaa = numba_ix(Q, ii0, ii0)
    Qab = numba_ix(Q, ii0, ii1)
    Qba = numba_ix(Q, ii1, ii0)
    Qbb = numba_ix(Q, ii1, ii1)

    Qaa_inv = np.linalg.inv(Qaa)
    Qcond = Qbb - Qba @ Qaa_inv @ Qab

    avec_a = avec[ii0]
    avec_b = avec[ii1]
    avec_cond = avec_b - Qba @ Qaa_inv @ avec_a

    p0a_fact_exp = np.exp(avec_a @ Qaa_inv @ avec_a.conj() * (-0.5)).real
    p0a_fact_det = np.sqrt(np.linalg.det(Qaa).real)
    p0a = p0a_fact_exp / p0a_fact_det

    n_det = len(nonzero_idxs)
    p_sum = 1.0  # empty set is not included in the powerset function so we start at 1
    for z in powerset(np.arange(n_det)):
        Z = np.asarray(z)
        ZZ = np.concatenate((Z, Z + n_det), axis=0)

        avec0 = avec_cond[ZZ]
        Q0 = numba_ix(Qcond, ZZ, ZZ)
        Q0inv = np.linalg.inv(Q0)

        fact_exp = np.exp(avec0 @ Q0inv @ avec0.conj() * (-0.5)).real
        fact_det = np.sqrt(np.linalg.det(Q0).real)

        p_sum += ((-1) ** len(Z)) * fact_exp / fact_det

    return p0a * p_sum


def threshold_detection_prob(
    mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10
):  # pylint: disable=too-many-arguments
    r"""Threshold detection probabilities for Gaussian states.
    Formula from Jake Bulmer and Stefano Paesani.
    When state is displaced, threshold_detection_prob_displacement is called.
    Otherwise, tor is called.

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

    if np.allclose(mu, 0, atol=atol, rtol=rtol):
        # no displacement
        n_modes = cov.shape[0] // 2
        Q = Qmat_numba(cov, hbar)
        O = Xmat(n_modes) @ Amat(cov, hbar=hbar)
        rpt2 = np.concatenate((det_pattern, det_pattern))
        Os = reduction(O, rpt2)
        return tor(Os) / np.sqrt(np.linalg.det(Q))
    det_pattern = np.asarray(det_pattern).astype(np.int8)
    return threshold_detection_prob_displacement(mu, cov, det_pattern, hbar)


@numba.jit(nopython=True)
def numba_tor(A):  # pragma: no cover
    """Returns the Torontonian of a matrix using numba.

    For more direct control, you may wish to call :func:`tor_real` or
    :func:`tor_complex` directly.

    The input matrix is cast to quadruple precision
    internally for a quadruple precision torontonian computation.

    Args:
        A (array): a square, symmetric array of even dimensions.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    n_det = A.shape[0] // 2
    p_sum = 1.0  # empty set is not included in the powerset function so we start at 1
    for z in powerset(np.arange(n_det)):
        Z = np.asarray(z)
        ZZ = np.concatenate((Z, Z + n_det), axis=0)
        A_ZZ = numba_ix(A, ZZ, ZZ)
        n = len(Z)
        p_sum += ((-1) ** n) / np.sqrt(np.linalg.det(np.eye(2 * n) - A_ZZ))

    return p_sum * (-1) ** (n_det)


@numba.jit(nopython=True)
def quad_cholesky(L, Z, idx, mat):
    """Returns the Cholesky factorization of a matrix using sub-matrix of prior

    Cholesky based on the new matrix and lower right quadrant.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): previous Cholesky.

        Z (array): new sub-matrix indices.

        idx: index of starting row/column of lower right quadrant.

        mat (array): new matrix.

    Returns:
        np.float64 or np.complex128: the Cholesky of matrix mat.
    """
    Ls = numba_ix(L, Z, Z)
    for i in range(idx, len(mat)):
        for j in range(idx, i):
            z = 0.0
            for k in range(j):
                z += Ls[i, k] * Ls[j, k].conjugate()
            Ls[i, j] = (mat[i][j] - z) / Ls[j, j]
        z = 0.0
        for k in range(i):
            z += Ls[i, k] * Ls[i, k].conjugate()
        Ls[i, i] = np.sqrt(mat[i, i] - z)
    return Ls


@numba.jit(nopython=True)
def recursiveTor(L, modes, A, n):
    """Returns the recursive Torontonian sub-computation of a matrix
    using numba.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        L (array): current Cholesky.

        modes (array): optical mode

        A (array): a square, symmetric array of even dimensions.

        n: size of the original matrix.

    Returns:
        np.float64 or np.complex128: the recursive torontonian
        sub-computation of matrix A.
    """
    tot, start = 0.0, 0 if len(modes) == 0 else modes[-1] + 1
    for i in range(start, n):
        nextModes = np.append(modes, i)
        nm, idx = len(A) >> 1, (i - len(modes)) * 2
        Z = np.concatenate((np.arange(idx), np.arange(idx + 2, nm * 2)), axis=0)
        nm -= 1
        Az = numba_ix(A, Z, Z)
        Ls = quad_cholesky(L, Z, idx, np.eye(2 * nm) - Az)
        det = np.square(np.prod(np.diag(Ls)))
        tot += ((-1) ** len(nextModes)) / np.sqrt(det) + recursiveTor(
            Ls, nextModes, Az, n
        )
    return tot


@numba.jit(nopython=True)
def rec_torontonian(A):
    """Returns the Torontonian of a matrix using numba.

    Algorithm from paper:
    https://arxiv.org/pdf/2109.04528.pdf

    Args:
        A (array): a square, symmetric array of even dimensions.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    n = A.shape[0] >> 1
    Z = np.empty((2 * n,), dtype=np.int_)
    Z[0::2] = np.arange(0, n)
    Z[1::2] = np.arange(n, 2 * n)
    A = numba_ix(A, Z, Z)
    L = np.linalg.cholesky(np.eye(2 * n) - A)
    det = np.square(np.prod(np.diag(L)))
    return 1 / np.sqrt(det) + recursiveTor(L, np.empty(0, dtype=np.int_), A, n)
