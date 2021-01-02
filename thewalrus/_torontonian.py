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
from .libwalrus import torontonian_complex as tor_complex
from .libwalrus import torontonian_real as tor_real


def tor(A, fsum=False):
    """Returns the Torontonian of a matrix.

    For more direct control, you may wish to call :func:`tor_real` or
    :func:`tor_complex` directly.

    The input matrix is cast to quadruple precision
    internally for a quadruple precision torontonian computation.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        fsum (bool): if ``True``, the `Shewchuck algorithm <https://github.com/achan001/fsum>`_
            for more accurate summation is performed. This can significantly increase
            the `accuracy of the computation <https://link.springer.com/article/10.1007%2FPL00009321>`_,
            but no casting to quadruple precision takes place, as the Shewchuck algorithm
            only supports double precision.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return tor_complex(A, fsum=fsum)
        return tor_real(np.float64(A.real), fsum=fsum)

    return tor_real(A, fsum=fsum)


@numba.jit(nopython=True)
def combinations(pool, r):
    """
    numba implementation of itertools.combinations
    taken from: https://stackoverflow.com/a/61393666

    Args:
        pool (iterable/array) : array/iterable to take combinations from
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
def powerset(S):
    """
    generates the powerset of S

    does not include the empty set 

    Args:
        S (array/iterable) : set to take powerset from
    Yields:
        s (list) : subset of S
    """
    n = len(S)
    for i in range(n + 1):
        for s in combinations(S, i):
            yield s


@numba.jit(nopython=True)
def nb_block(X):
    """ 
    numba implementation of np.block
    taken from: https://stackoverflow.com/a/57562911

    Args:
        X (tuple of arrays) : arrays for blocks of matrix
    Return:
        array : the block matrix from X
    """
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.jit(nopython=True)
def numba_ix(arr, rows, cols):
    """
    numba implementation of np.ix_

    Args:
        arr (2d array) : matrix to take submatrix of
        rows (array) : rows to be selected in submatrix
        cols (array) : columns to be selected in submatrix

    Return: 
        len(rows) * len(cols) array : selected submatrix of arr
    """
    return arr[rows][:, cols]


@numba.jit(nopython=True)
def Qmat_numba(cov, hbar=2):
    r"""
    numba compatible version of thewalrus.quantum Qmat

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

    x = cov[:N, :N] * (2. / hbar)
    xp = cov[:N, N:] * (2. / hbar)
    p = cov[N:, N:] * (2. / hbar)
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
    return Q


@numba.jit(nopython=True)
def threshold_detection_prob(mu, cov, det_pattern, hbar=2):
    r"""
    thershold detection probabilities for Gaussian states with displacement

    formula from Jake Bulmer and Stefano Paesani 

    Args:
        mu (1d array) : means of xp Gaussian Wigner function 
        cov (2d array) : : xp Wigner covariance matrix
        det_pattern (1d array) : array of {0,1} to describe the threshold detection outcome
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        np.float64 : probability of detection pattern
    """
    det_pattern = np.asarray(det_pattern)

    m = len(cov)
    assert cov.shape == (m, m)
    assert m % 2 == 0
    n = m // 2

    means_x = mu[:n]
    means_p = mu[n:]
    avec = np.concatenate((means_x + 1j * means_p, means_x - 1j * means_p), axis=0) / np.sqrt(2 * hbar)

    Q = Qmat_numba(cov, hbar=hbar)

    if max(det_pattern) > 1:
        raise ValueError(
            "When using threshold detectors, the detection pattern can contain only 1s or 0s.")

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
    p_sum = 1.  # empty set is not included in the powerset function so we start at 1
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
