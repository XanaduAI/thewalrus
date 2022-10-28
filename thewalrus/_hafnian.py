# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Hafnian Python interface
"""
import warnings
from functools import lru_cache
from collections import Counter
from itertools import chain, combinations
import numba
import numpy as np
from thewalrus import charpoly


@numba.jit(nopython=True, cache=True)
def nb_binom(n, k):  # pragma: no cover
    """Numba version of binomial coefficient function.

    Args:
        n (int): how many options
        k (int): how many are chosen

    Returns:
        int: how many ways of choosing
    """
    if k < 0 or k > n:
        return 0
    if k == (0, n):
        return 1
    binom = 1
    for i in range(min(k, n - k)):
        binom *= n - i
        binom //= i + 1
    return binom


@numba.jit(nopython=True, cache=True)
def precompute_binoms(max_binom):  # pragma: no cover
    """Precompute binomial coefficients, return as a 2d array.

    Args:
        max_binom (int): max value of n in the binomial

    Returns:
        array: ``max_binom + 1 * max_binom + 1`` array of binomial coefficients
    """
    binoms = np.zeros((max_binom + 1, max_binom + 1), dtype=type(max_binom))
    for i in range(max_binom + 1):
        for j in range(max_binom + 1):
            binoms[i, j] = nb_binom(i, j)
    return binoms


@numba.jit(nopython=True, cache=True)
def nb_ix(arr, rows, cols):  # pragma: no cover
    """Numba implementation of ``np.ix_``.

    Args:
        arr (2d array): matrix to take submatrix of
        rows (array): rows to be selected in submatrix
        cols (array): columns to be selected in submatrix

    Return:
        array: selected submatrix of ``arr`` with dimension ``len(rows) * len(cols)``
    """
    return arr[rows][:, cols]


def matched_reps(reps):  # pylint: disable = too-many-branches
    """Takes the repeated rows and find a way to pair them up to create a perfect
    matching with many repeated edges.

    Args:
        reps (list): list of repeated rows/cols

    Returns:
        tuple[array, array, int]: tuple with vertex pairs (length ``2N`` for ``N`` edges; index
        ``i`` is matched with ``i + N``), length ``N`` array for how many times each edge is
        repeated, and index of odd mode (``None`` if even number of vertices)
    """
    n = len(reps)

    if sum(reps) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), None

    # need to pair off the indices with high numbers of repetitions...
    x = range(n)  # the starting set of indices
    edgesA = []  # contains part A of each pair
    edgesB = []  # part B of each pair
    edgereps = []  # number of repetitions of a pair
    reps, x = zip(
        *sorted(zip(reps, x), reverse=True)
    )  # sort according to reps, in descending order
    reps = list(reps)
    x = list(x)

    # remove zeros
    nonzero_reps = []
    nonzero_x = []
    for i, r in zip(x, reps):
        if r > 0:
            nonzero_reps.append(r)
            nonzero_x.append(i)
    reps = nonzero_reps
    x = nonzero_x

    while len(reps) > 1 or (len(reps) == 1 and reps[0] > 1):
        reps, x = zip(*sorted(zip(reps, x), reverse=True))  # sort
        reps = list(reps)
        x = list(x)
        if len(reps) == 1 or reps[0] > reps[1] * 2:
            # if largest number of reps is more than double the 2nd largest, pair it with itself
            edgesA += [x[0]]
            edgesB += [x[0]]
            edgereps += [reps[0] // 2]
            if reps[0] % 2 == 0:
                x = x[1:]
                reps = reps[1:]
            else:
                reps[0] = 1
        else:
            # otherwise, form pairs between largest reps and 2nd largest reps
            edgesA += [x[0]]
            edgesB += [x[1]]
            edgereps += [reps[1]]
            if reps[0] > reps[1]:
                if len(x) > 2:
                    x = [x[0]] + x[2:]
                    reps = [reps[0] - reps[1]] + reps[2:]
                else:
                    x = [x[0]]
                    reps = [reps[0] - reps[1]]
            else:
                x = x[2:]
                reps = reps[2:]

    if len(x) == 1:
        oddmode = x[0]  # if there is an unpaired mode, store it
    else:
        oddmode = None

    # the adjacency matrix of red edges connects 1 to N/2+1, 2 to N/2+2, etc.
    # Reorder the indices (from x2 back to x) so that the paired indices get
    # connected by red edges
    x = np.asarray(edgesA + edgesB, dtype=np.int64)  # reordered list of indices
    edgereps = np.asarray(edgereps, dtype=np.int64)

    return x, edgereps, oddmode


@numba.jit(nopython=True, cache=True)
def find_kept_edges(j, reps):  # pragma: no cover
    """Write ``j`` as a string where the ith digit is in base ``reps[i]+1``
    decides which edges are included given index of the inclusion/exclusion sum.

    Args:
        j (int): index of sum
        reps (list): number of repetitions of each edge

    Returns:
        array: number of repetitions kept for the current inclusion/exclusion step
    """
    num = j
    output = []
    bases = np.asarray(reps) + 1
    for base in bases[::-1]:
        output.append(num % base)
        num //= base
    return np.array(output[::-1], dtype=reps.dtype)


@numba.jit(nopython=True, cache=True)
def f(A, n):  # pragma: no cover
    """Evaluate the polynomial coefficients of the function in the eigenvalue-trace formula.

    Args:
        A (array): a two-dimensional matrix
        n (int): number of polynomial coefficients to compute

    Returns:
        array: polynomial coefficients
    """
    # Compute combinations in O(n^2log n) time
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    powtrace = charpoly.powertrace(A, n // 2 + 1)
    for i in range(1, n // 2 + 1):
        factor = powtrace[i] / (2 * i)
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor
    return comb[count, :]


@numba.jit(nopython=True, cache=True)
def f_loop(AX, AX_S, XD_S, D_S, n):  # pragma: no cover
    """Evaluate the polynomial coefficients of the function in the eigenvalue-trace formula.

    Args:
        AX (array): two-dimensional matrix
        AX_S (array): ``AX_S`` with weights given by repetitions and excluded rows removed
        XD_S (array): diagonal multiplied by ``X``
        D_S (array): diagonal
        n (int): number of polynomial coefficients to compute

    Returns:
        array: polynomial coefficients
    """
    # Compute combinations in O(n^2log n) time
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    powtrace = charpoly.powertrace(AX, n // 2 + 1)
    for i in range(1, n // 2 + 1):
        factor = powtrace[i] / (2 * i) + (XD_S @ D_S) / 2
        XD_S = XD_S @ AX_S
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor
    return comb[count, :]


# pylint: disable = too-many-arguments
@numba.jit(nopython=True, cache=True)
def f_loop_odd(AX, AX_S, XD_S, D_S, n, oddloop, oddVX_S):  # pragma: no cover
    """Evaluate the polynomial coefficients of the function in the eigenvalue-trace formula
    when there is a self-edge in the fixed perfect matching.

    Args:
        AX (array): two-dimensional matrix
        AX_S (array): ``AX_S`` with weights given by repetitions and excluded rows removed
        XD_S (array): diagonal multiplied by ``X``
        D_S (array): diagonal
        n (int): number of polynomial coefficients to compute
        oddloop (float): weight of self-edge
        oddVX_S (array): vector corresponding to matrix at the index of the self-edge

    Returns:
        array: polynomial coefficients
    """

    count = 0
    comb = np.zeros((2, n + 1), dtype=np.complex128)
    comb[0, 0] = 1
    powtrace = charpoly.powertrace(AX, n + 1)
    for i in range(1, n + 1):
        if i == 1:
            factor = oddloop
        elif i % 2 == 0:
            factor = powtrace[i // 2] / i + (XD_S @ D_S) / 2
        else:
            factor = oddVX_S @ D_S
            D_S = AX_S @ D_S

        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // i + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, :]


@numba.jit(nopython=True, cache=True)
def f_from_powertrace(powertraces, n):
    """Evaluate the polynomial coefficients of the function in the eigenvalue-trace formula, using the powertraces.

    Args:
        pow_traces (array): power traces of some matrix
        n (int): number of polynomial coefficients to compute

    Returns:
        array: polynomial coefficients
    """
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = powertraces[i] / (2 * i)
        powfactor = 1.0
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor = powfactor * factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, n // 2]


@numba.jit(nopython=True, cache=True)
def f_all_from_powertrace(powertraces, n):
    """Evaluate the polynomial coefficients of the function in the eigenvalue-trace formula, using the powertraces.

    Args:
        pow_traces (array): power traces of some matrix
        n (int): number of polynomial coefficients to compute

    Returns:
        array: polynomial coefficients
    """
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = powertraces[i] / (2 * i)
        powfactor = 1.0
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor = powfactor * factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, :]


@numba.jit(nopython=True, cache=True)
def get_AX_S(kept_edges, A):  # pragma: no cover
    """Given the kept edges, return the appropriate scaled submatrices to compute ``f``.

    Args:
        kept_edges (array): number of repetitions of each edge
        A (array): matrix before repetitions applied

    Returns:
        array: scaled ``A @ X``, where ``X = ((0, I), (I, 0))``
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:, :n_nonzero_edges] = kept_edges_nonzero * A_nonzero[:, n_nonzero_edges:]
    AX_nonzero[:, n_nonzero_edges:] = kept_edges_nonzero * A_nonzero[:, :n_nonzero_edges]

    return AX_nonzero


@numba.jit(nopython=True, cache=True)
def get_submatrices(kept_edges, A, D, oddV):  # pragma: no cover
    """Given the kept edges, return the appropriate scaled submatrices to compute ``f``.

    Args:
        kept_edges (array): number of repetitions of each edge
        A (array): matrix before repetitions applied
        D (array): diagonal before repetitions applied
        oddV (array): Row of matrix at index of self-edge. ``None`` is no self-edge.

    Returns:
        tuple[array, array, array, array]: scaled ``A @ X `` (where ``X = ((0, I), (I, 0))``),
        scaled ``X @ D``, scaled ``D``, and scaled ``oddV @ X``
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:, :n_nonzero_edges] = kept_edges_nonzero * A_nonzero[:, n_nonzero_edges:]
    AX_nonzero[:, n_nonzero_edges:] = kept_edges_nonzero * A_nonzero[:, :n_nonzero_edges]

    D_nonzero = D[nonzero_rows]

    XD_nonzero = np.empty_like(D_nonzero, dtype=np.complex128)
    XD_nonzero[:n_nonzero_edges] = kept_edges_nonzero * D_nonzero[n_nonzero_edges:]
    XD_nonzero[n_nonzero_edges:] = kept_edges_nonzero * D_nonzero[:n_nonzero_edges]

    if oddV is not None:
        oddV_nonzero = oddV[nonzero_rows]
        oddVX_nonzero = np.empty_like(oddV_nonzero, dtype=np.complex128)
        oddVX_nonzero[:n_nonzero_edges] = kept_edges_nonzero * oddV_nonzero[n_nonzero_edges:]
        oddVX_nonzero[n_nonzero_edges:] = kept_edges_nonzero * oddV_nonzero[:n_nonzero_edges]
    else:
        oddVX_nonzero = None

    return AX_nonzero, XD_nonzero, D_nonzero, oddVX_nonzero


@numba.jit(nopython=True, cache=True)
def get_submatrix_batch_odd0(kept_edges, oddV0):  # pragma: no cover
    """Find ``oddVX_nonzero0`` for batching (sometimes different vertices are
    identified as self edges).

    Args:
        kept_edges (array): number of repetitions of each edge
        oddV0 (array): Row of matrix at index of self-edge. ``None`` is no self-edge.
    Returns:
        array: scaled ``oddV0 @ X``
    """
    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]
    oddV_nonzero0 = oddV0[nonzero_rows]
    oddVX_nonzero0 = np.empty_like(oddV_nonzero0, dtype=np.complex128)
    oddVX_nonzero0[:n_nonzero_edges] = kept_edges_nonzero * oddV_nonzero0[n_nonzero_edges:]
    oddVX_nonzero0[n_nonzero_edges:] = kept_edges_nonzero * oddV_nonzero0[:n_nonzero_edges]

    return oddVX_nonzero0


@numba.jit(nopython=True, cache=True)
def get_Dsubmatrices(kept_edges, D):  # pragma: no cover
    """Find submatrices for batch gamma functions."""

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    D_nonzero = D[nonzero_rows]

    XD_nonzero = np.empty_like(D_nonzero, dtype=np.complex128)
    XD_nonzero[:n_nonzero_edges] = kept_edges_nonzero * D_nonzero[n_nonzero_edges:]
    XD_nonzero[n_nonzero_edges:] = kept_edges_nonzero * D_nonzero[:n_nonzero_edges]

    return XD_nonzero, D_nonzero


@numba.jit(nopython=True, cache=True)
def eigvals(M):  # pragma: no cover
    """Computes the eigenvalues of a matrix.

    Args:
        M (array): square matrix

    Returns:
        array: eigenvalues of the matrix ``M``
    """
    return np.linalg.eigvals(M)


# pylint: disable=W0612, E1133
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_hafnian(A, edge_reps, glynn=True):  # pragma: no cover
    r"""Compute hafnian, using inputs as prepared by frontend hafnian function compiled with Numba.

    Args:
        A (array): matrix ordered according to the chosen perfect matching
        edge_reps (array): how many times each edge in the perfect matching is repeated
        glynn (bool): whether to use finite difference sieve

    Returns:
        complex: value of hafnian
    """

    n = A.shape[0]
    N = 2 * edge_reps.sum()  # number of photons

    if glynn:
        steps = ((edge_reps[0] + 2) // 2) * np.prod(edge_reps[1:] + 1)
    else:
        steps = np.prod(edge_reps + 1)

    # precompute binomial coefficients
    max_binom = edge_reps.max() + 1
    binoms = precompute_binoms(max_binom)

    H = np.complex128(0)  # start running total for the hafnian

    for j in numba.prange(steps):

        kept_edges = find_kept_edges(j, edge_reps)
        edge_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        if glynn:
            kept_edges = 2 * kept_edges - edge_reps

        AX_S = get_AX_S(kept_edges, A)

        prefac = (-1.0) ** (N // 2 - edge_sum) * binom_prod

        if glynn and kept_edges[0] == 0:
            prefac *= 0.5
        Hnew = prefac * f(AX_S, N)[N // 2]

        H += Hnew

    if glynn:
        H = H * 0.5 ** (N // 2 - 1)

    return H


def _haf(A, reps=None, glynn=True):
    r"""Calculate hafnian with (optional) repeated rows and columns.
    Code contributed by `Jake F.F. Bulmer <https://github.com/jakeffbulmer/gbs>`_ based on
    `arXiv:2108.01622 <https://arxiv.org/abs/2108.01622>`_.

    Args:
        A (array): N x N matrix.
        reps (list): Length-N list of repetitions of each row/col (optional). If not provided,
            each row/column assumed to be repeated once.
        glynn (bool): If ``True``, use Glynn-style finite difference sieve formula. If ``False``,
            use Ryser style inclusion/exclusion principle.

    Returns
        complex: result of hafnian calculation
    """

    n = A.shape[0]

    if reps is None:
        reps = [1] * n

    N = sum(reps)

    if N == 0:
        return 1.0

    if N % 2 == 1:
        return 0.0

    assert n == len(reps)

    x, edge_reps, oddmode = matched_reps(reps)

    # make new A matrix using the ordering from above

    Ax = A[np.ix_(x, x)].astype(np.complex128)

    H = _calc_hafnian(Ax, edge_reps, glynn)
    return H


# pylint: disable=too-many-arguments, redefined-outer-name, not-an-iterable
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_loop_hafnian(A, D, edge_reps, oddloop=None, oddV=None, glynn=True):  # pragma: no cover
    """Compute loop hafnian, using inputs as prepared by frontend loop_hafnian function
    compiled with Numba.
    Code contributed by `Jake F.F. Bulmer <https://github.com/jakeffbulmer/gbs>`_ based on
    `arXiv:2108.01622 <https://arxiv.org/abs/2108.01622>`_.

    Args:
        A (array): matrix ordered according to the chosen perfect matching.
        D (array): diagonals ordered according to the chosen perfect matchin
        edge_reps (array): how many times each edge in the perfect matching is repeated
        oddloop (float): weight of self-loop in perfect matching, None if no self-loops
        oddV (array): row of matrix corresponding to the odd loop in the perfect matching
        glynn (bool): whether to use finite difference sieve

    Returns:
        complex: value of loop hafnian
    """

    n = A.shape[0]
    N = 2 * edge_reps.sum()  # Number of photons
    if oddloop is not None:
        N += 1
    if glynn and (oddloop is None):
        steps = ((edge_reps[0] + 2) // 2) * np.prod(edge_reps[1:] + 1)
    else:
        steps = np.prod(edge_reps + 1)

    # Precompute binomial coefficients
    max_binom = edge_reps.max() + 1
    binoms = precompute_binoms(max_binom)

    H = np.complex128(0)  # Start running total for the hafnian

    for j in numba.prange(steps):

        kept_edges = find_kept_edges(j, edge_reps)
        edge_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        if glynn:
            kept_edges = 2 * kept_edges - edge_reps

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(kept_edges, A, D, oddV)
        AX = AX_S.copy()

        prefac = (-1.0) ** (N // 2 - edge_sum) * binom_prod

        if oddloop is not None:
            Hnew = prefac * f_loop_odd(AX, AX_S, XD_S, D_S, N, oddloop, oddVX_S)[N]
        else:
            if glynn and kept_edges[0] == 0:
                prefac *= 0.5
            Hnew = prefac * f_loop(AX, AX_S, XD_S, D_S, N)[N // 2]

        H += Hnew

    if glynn:
        if oddloop is None:
            H = H * 0.5 ** (N // 2 - 1)
        else:
            H = H * 0.5 ** (N // 2)

    return H


# pylint: disable=redefined-outer-name
def loop_hafnian(A, D=None, reps=None, glynn=True):
    """Calculate loop hafnian with (optional) repeated rows and columns.
    Code contributed by `Jake F.F. Bulmer <https://github.com/jakeffbulmer/gbs>`_ based on
    `arXiv:2108.01622 <https://arxiv.org/abs/2108.01622>`_.

    Args:
        A (array): N x N matrix.
        D (array): Diagonal entries of matrix (optional). If not provided, ``D`` is the diagonal of ``A``.
            If repetitions are provided, ``D`` should be provided explicitly.
        reps (list): Length-N list of repetitions of each row/col (optional), if not provided, each
            row/column assumed to be repeated once.
        glynn (bool): If ``True``, use Glynn-style finite difference sieve formula, if ``False``,
            use Ryser style inclusion/exclusion principle.

    Returns
        complex: result of loop hafnian calculation
    """
    n = A.shape[0]

    if reps is None:
        reps = [1] * n
    if D is None:
        D = A.diagonal()

    N = sum(reps)

    if N == 0:
        return 1.0

    if N == 1:
        return D[np.where(np.array(reps) == 1)[0][0]]

    assert n == len(reps)

    assert D.shape[0] == n

    x, edge_reps, oddmode = matched_reps(reps)

    # Make new A matrix and D vector using the ordering from above
    if oddmode is not None:
        oddloop = D[oddmode].astype(np.complex128)
        oddV = A[oddmode, x].astype(np.complex128)
    else:
        oddloop = None
        oddV = None

    Ax = A[np.ix_(x, x)].astype(np.complex128)
    Dx = D[x].astype(np.complex128)

    H = _calc_loop_hafnian(Ax, Dx, edge_reps, oddloop, oddV, glynn)
    return H


def input_validation(A, rtol=1e-05, atol=1e-08):
    """Checks that the matrix A satisfies the requirements for Hafnian calculation.
    These include:
    * That the ``A`` is a NumPy array
    * That ``A`` is square
    * That ``A`` does not contain any NaNs
    * That ``A`` is symmetric

    Args:
        A (array): a NumPy array.
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``

    Returns:
        bool: returns ``True`` if the matrix satisfies all requirements
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    n = A.shape

    if n[0] != n[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if not np.allclose(A, A.T, rtol=rtol, atol=atol):
        raise ValueError("Input matrix must be symmetric.")

    return True


def bandwidth(A):
    """Calculates the upper bandwidth of the matrix A.

    Args:
        A (array): input matrix

    Returns:
        int: bandwidth of matrix
    """
    n, _ = A.shape
    for i in range(n - 1, 0, -1):
        vali = np.diag(A, i)
        if not np.allclose(vali, 0):
            return i
    return 0


def powerset(iterable):
    """Calculates the powerset of a list.

    Args:
        iterable (iterable): input list

    Returns:
        chain: chain of all subsets of input list
    """
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1))


def reduction(A, rpt):
    r"""Calculates the reduction of an array by a vector of indices.
    This is equivalent to repeating the ith row/column of :math:`A`, :math:`rpt_i` times.

    Args:
        A (array): matrix of size ``[N, N]``
        rpt (Sequence): sequence of N positive integers indicating the corresponding rows/columns
            of ``A`` to be repeated.

    Returns:
        array: the reduction of ``A`` by the index vector ``rpt``
    """
    rows = [i for sublist in [[idx] * j for idx, j in enumerate(rpt)] for i in sublist]

    if A.ndim == 1:
        return A[rows]

    return A[:, rows][rows]


# pylint: disable=too-many-arguments
def hafnian(
    A,
    loop=False,
    rtol=1e-05,
    atol=1e-08,
    approx=False,
    num_samples=1000,
    method="glynn",
):  # pylint: disable=too-many-arguments
    """Returns the hafnian of a matrix.
    Code contributed by `Jake F.F. Bulmer <https://github.com/jakeffbulmer/gbs>`_ based on
    `arXiv:2108.01622 <https://arxiv.org/abs/2108.01622>`_.

    Args:
        A (array): a square, symmetric array of even dimensions
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        method (string): Set this to ``"glynn"`` to use the
            glynn formula,
            or ``"inclexcl"`` to use the inclusion exclusion principle,
            or ``"recursive"`` to use a recursive algorithm.
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``
        approx (bool): If ``True``, an approximation algorithm is used to estimate the hafnian. Note
            that the approximation algorithm can only be applied to matrices ``A`` that only have
            non-negative entries.
        num_samples (int): If ``approx=True``, the approximation algorithm performs ``num_samples``
            iterations for estimation of the hafnian of the non-negative matrix ``A``

    Returns:
        int or float or complex: the hafnian of matrix ``A``
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, rtol=rtol, atol=atol)

    matshape = A.shape

    if method == "glynn":
        glynn = True

    if method == "inclexcl":
        glynn = False

    if matshape == (0, 0):
        return 1

    if matshape[0] % 2 != 0 and not loop:
        return 0.0

    if np.allclose(np.diag(np.diag(A)), A, rtol=rtol, atol=atol):
        if loop:
            return np.prod(np.diag(A))
        return 0

    matshape = A.shape

    if matshape[0] == 2:
        if loop:
            return A[0, 1] + A[0, 0] * A[1, 1]
        return A[0][1]

    if matshape[0] == 3 and loop:
        return (
            A[0, 0] * A[1, 2] + A[1, 1] * A[0, 2] + A[2, 2] * A[0, 1] + A[0, 0] * A[1, 1] * A[2, 2]
        )

    if matshape[0] == 4:
        if loop:
            result = (
                A[0, 1] * A[2, 3]
                + A[0, 2] * A[1, 3]
                + A[0, 3] * A[1, 2]
                + A[0, 0] * A[1, 1] * A[2, 3]
                + A[0, 1] * A[2, 2] * A[3, 3]
                + A[0, 2] * A[1, 1] * A[3, 3]
                + A[0, 0] * A[2, 2] * A[1, 3]
                + A[0, 0] * A[3, 3] * A[1, 2]
                + A[0, 3] * A[1, 1] * A[2, 2]
                + A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3]
            )
            return result

        return A[0, 1] * A[2, 3] + A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    if approx:
        if np.any(np.iscomplex(A)):
            raise ValueError("Input matrix must be real")

        if np.any(A < 0):
            raise ValueError("Input matrix must not have negative entries")

        return hafnian_approx(A, num_samples=num_samples)

    if loop:
        if method == "recursive":
            warnings.warn("Recursive algorithm does not support the loop hafnian")
        return loop_hafnian(A, D=None, reps=None, glynn=True)

    if method == "recursive":
        return recursive_hafnian(A)

    return _haf(A, reps=None, glynn=glynn)


def hafnian_sparse(A, D=None, loop=False):
    r"""Returns the hafnian of a sparse symmetric matrix.
    This pure python implementation is very slow on full matrices, but faster the sparser a matrix is.
    As a rule of thumb, the crossover in runtime with respect to :func:`~.hafnian` happens around 50% sparsity.

    Args:
        A (array): the symmetric matrix of which we want to compute the hafnian
        D (set): Set of indices that identify a submatrix. If ``None`` (default) it computes
            the hafnian of the whole matrix.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.

    Returns:
        float: hafnian of ``A`` or of the submatrix of ``A`` defined by the set of indices ``D``
    """
    if D is None:
        D = frozenset(range(len(A)))
    else:
        D = frozenset(D)

    if not loop:
        A = A - np.diag(np.diag(A))

    if np.allclose(A, 0):
        return 0.0

    r, _ = np.nonzero(A)
    m = max(Counter(r).values())  # max nonzero values per row/column

    @lru_cache(maxsize=2**m)
    def indices(d, k):
        return d.intersection(set(np.nonzero(A[k])[0]))

    @lru_cache(maxsize=2**m)
    def lhaf(d: frozenset) -> float:
        if not d:
            return 1
        d_ = set(d)
        k = d_.pop()
        return sum(A[i, k] * lhaf(frozenset(d_).difference({i})) for i in indices(d, k))

    return lhaf(D)


def hafnian_repeated(A, rpt, mu=None, loop=False, rtol=1e-05, atol=1e-08, glynn=True):
    r"""Returns the hafnian of matrix with repeated rows/columns.

    Code contributed by `Jake F.F. Bulmer <https://github.com/jakeffbulmer/gbs>`_ based on
    `arXiv:2108.01622 <https://arxiv.org/abs/2108.01622>`_.

    The :func:`reduction` function may be used to show the resulting matrix
    with repeated rows and columns as per ``rpt``.

    As a result, the following are identical:

    .. code:

        >>> hafnian_repeated(A, rpt)
        >>> hafnian(reduction(A, rpt))

    However, using ``hafnian_repeated`` in the case where there are a large number
    of repeated rows and columns (:math:`\sum_{i}rpt_i \gg N`) can be
    significantly faster.

    .. note::

        If :math:`rpt=(1, 1, \dots, 1)`, then

        >>> hafnian_repeated(A, rpt) == hafnian(A)

    Args:
        A (array): a square, symmetric :math:`N\times N` array
        rpt (Sequence): a length-:math:`N` positive integer sequence, corresponding
            to the number of times each row/column of matrix :math:`A` is repeated
        mu (array): A vector of length :math:`N` representing the vector of means/displacement.
            If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this
            only affects the loop hafnian.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``
        glynn (bool): whether to use finite difference sieve

    Returns:
        int or float or complex: the hafnian of matrix A
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, atol=atol, rtol=rtol)

    if len(rpt) != len(A):
        raise ValueError("the rpt argument must be 1-dimensional sequence of length len(A).")

    nud = np.array(rpt, dtype=np.int32)

    if not np.all(np.mod(rpt, 1) == 0) or np.any(nud < 0):
        raise ValueError("the rpt argument must contain non-negative integers.")

    if np.all(nud == 0):
        return 1.0

    if np.sum(nud) % 2 != 0 and not loop:
        return 0.0

    if mu is None:
        mu = A.diagonal().copy()

    if np.allclose(A, 0, rtol=rtol, atol=atol):
        if loop:
            return np.prod(mu**rpt)
        return 0

    if len(mu) != len(A):
        raise ValueError("Length of means vector must be the same length as the matrix A.")

    if loop:
        return loop_hafnian(A, D=mu, reps=rpt, glynn=glynn)

    return _haf(A, reps=rpt, glynn=glynn)


def hafnian_banded(A, loop=False, rtol=1e-05, atol=1e-08):
    """Returns the loop hafnian of a banded matrix.
    For the derivation see Section V of `'Efficient sampling from shallow Gaussian quantum-optical
    circuits with local interactions', Qi et al. <https://arxiv.org/abs/2009.11824>`_.

    Args:
        A (array): a square, symmetric array of even dimensions

    Returns:
        int or float or complex: the loop hafnian of matrix ``A``
    """
    input_validation(A, atol=atol, rtol=rtol)
    (n, _) = A.shape
    w = bandwidth(A)
    if not loop:
        A = A - np.diag(np.diag(A))
    loop_haf = {(): 1, (1,): A[0, 0]}
    for t in range(1, n + 1):
        if t - 2 * w - 1 > 0:
            lower_end = set(range(1, t - 2 * w))
        else:
            lower_end = set()
        upper_end = set(range(1, t + 1))
        diff = [item for item in upper_end if item not in lower_end]
        # Makes sure set ordering is preserved when the difference of two sets is taken
        # This is also used in the if statement below
        ps = powerset(diff)
        lower_end = tuple(lower_end)
        for D in ps:
            if lower_end + D not in loop_haf:
                loop_haf[lower_end + D] = sum(
                    [
                        A[i - 1, t - 1]
                        * loop_haf[tuple(item for item in lower_end + D if item not in set((i, t)))]
                        for i in D
                    ]
                )

    return loop_haf[tuple(range(1, n + 1))]


@numba.jit(nopython=True)
def recursive_hafnian(A):  # pragma: no cover
    r"""Computes the hafnian of the matrix with the recursive algorithm. It is an implementation of
    algorithm 2 in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
    This code is a modified version of the code found here:
    `Recursive hafnian
    <https://codegolf.stackexchange.com/questions/157049/calculate-the-hafnian-as-quickly-as-possible>`_.

    Args:
        A (array): the input matrix

    Returns:
        float: the hafnian of the input matrix
    """
    nb_lines, nb_columns = A.shape
    if nb_lines != nb_columns:
        raise ValueError("Matrix must be square")

    if nb_lines % 2 != 0:
        raise ValueError("Matrix size must be even")

    n = len(A) // 2
    z = np.zeros((n * (2 * n - 1), n + 1), dtype=A.dtype)
    for j in range(1, 2 * n):
        ind = j * (j - 1) // 2
        for k in range(j):
            z[ind + k][0] = A[j][k]
    g = np.zeros(n + 1, dtype=A.dtype)
    g[0] = 1
    return solve(z, 2 * n, 1, g, n)


@numba.jit(nopython=True)
def solve(b, s, w, g, n):  # pragma: no cover
    r"""Implements the recursive algorithm.

    Args:
        b (array): matrix that is transformed recursively
        s (int): size of the original matrix that changes at every recursion
        k (int): a variable of the recursive algorithm
        g (int): matrix that is transformed recursively
        n (int): size of the original matrix divided by 2

    Returns:
        float: the hafnian of the input matrix
    """
    if s == 0:
        return w * g[n]
    c = np.zeros(((s - 2) * (s - 3) // 2, n + 1), dtype=g.dtype)
    i = 0
    for j in range(1, s - 2):
        for k in range(j):
            c[i] = b[(j + 1) * (j + 2) // 2 + k + 2]
            i += 1
    h = solve(c, s - 2, -w, g, n)
    e = g.copy()
    for u in range(n):
        for v in range(n - u):
            e[u + v + 1] += g[u] * b[0][v]
    for j in range(1, s - 2):
        for k in range(j):
            for u in range(n):
                for v in range(n - u):
                    c[j * (j - 1) // 2 + k][u + v + 1] += (
                        b[(j + 1) * (j + 2) // 2][u] * b[(k + 1) * (k + 2) // 2 + 1][v]
                        + b[(k + 1) * (k + 2) // 2][u] * b[(j + 1) * (j + 2) // 2 + 1][v]
                    )
    return h + solve(c, s - 2, w, e, n)


@numba.jit(nopython=True)
def _one_det(B):  # pragma: no cover
    """Calculates the determinant of an antisymmetric matrix with entries distributed
    according to a normal distribution, with scale equal to the entries of the symmetric matrix
    given as input.

    Args:
        B (array[float]): symmetric matrix

    Returns:
        float: determinant of the samples antisymmetric matrix
    """
    mat = np.empty_like(B, dtype=np.float64)
    n, m = B.shape
    for i in range(n):
        for j in range(m):
            mat[i, j] = B[i, j] * np.random.normal()
            mat[j, i] = -mat[i, j]
    return np.linalg.det(mat)


@numba.jit(nopython=True)
def hafnian_approx(A, num_samples=1000):  # pragma: no cover
    """Returns the approximation to the hafnian of a matrix with non-negative entries.

    The approximation follows the stochastic Barvinok's approximation allowing the
    hafnian can be approximated as the sum of determinants of matrices.
    The accuracy of the approximation increases with increasing number of iterations.

    Args:
        B (array[float]): a symmetric matrix

    Returns:
        float: approximate hafnian of the input
    """

    sqrtA = np.sqrt(A)
    return np.array([_one_det(sqrtA) for _ in range(num_samples)]).mean()
