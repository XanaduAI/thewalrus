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
Bulmer, Jacob F. F., et al. “The Boundary for Quantum Advantage in
Gaussian Boson Sampling.” ArXiv:2108.01622 [Quant-Ph], Aug. 2021.
arXiv.org, http://arxiv.org/abs/2108.01622
"""
import numba
import numpy as np

# pylint: disable= too-many-arguments, too-many-branches, consider-using-in


@numba.jit(nopython=True, cache=True)
def nb_binom(n, k):  # pragma: no cover
    """
    Numba version of binomial coefficient function.

    Args:
        n (int): How many options
        k (int): How many are chosen

    Returns:
        int: How many ways of choosing
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    binom = 1
    for i in range(min(k, n - k)):
        binom *= n - i
        binom //= i + 1
    return binom


@numba.jit(nopython=True, cache=True)
def precompute_binoms(max_binom):  # pragma: no cover
    """
    Precompute binomial coefficients, return as a 2d array.

    Args:
        max_binom (int): Max value of n in the binomial

    Returns:
        array: max_binom+1 x max_binom+1 array of binomial coefficients
    """
    binoms = np.zeros((max_binom + 1, max_binom + 1), dtype=type(max_binom))
    for i in range(max_binom + 1):
        for j in range(max_binom + 1):
            binoms[i, j] = nb_binom(i, j)
    return binoms


@numba.jit(nopython=True, cache=True)
def nb_ix(arr, rows, cols):  # pragma: no cover
    """
    Numba implementation of np.ix_ .

    Args:
        arr (2d array) : Matrix to take submatrix of
        rows (array) : Rows to be selected in submatrix
        cols (array) : Columns to be selected in submatrix

    Return:
        len(rows) * len(cols) array : Selected submatrix of arr
    """
    return arr[rows][:, cols]


def matched_reps(reps):
    """
    Takes the repeated rows and find a way to pair them up
    to create a perfect matching with many repeated edges.

    Args:
        reps (list): List of repeated rows/cols

    Returns:
        x (array): Vertex pairs, length 2N for N edges. Index i is matched with i+N
        edgereps (array): length N array for how many times each edge is repeated
        oddmode (int or None): index of odd mode, None if even number of vertices
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
    """
    Write j as a string where the ith digit is in base (reps[i]+1)
    decides which edges are included given index of the inclusion/exclusion sum.

    Args:
        j (int): index of sum
        reps (list): number of repetitions of each edge

    Returns:
        array : number of repetitions kept for the current inclusion/exclusion step
    """
    num = j
    output = []
    bases = np.asarray(reps) + 1
    for base in bases[::-1]:
        output.append(num % base)
        num //= base
    return np.array(output[::-1], dtype=reps.dtype)


@numba.jit(nopython=True, cache=True)
def f(E, n):  # pragma: no cover
    """
    Evaluate the polyonial coefficients of the function in the eigevalue-trace formula.

    Args:
        E (array): eigevavlues of AX
        n (int): number of polynomial coefficients to compute

    Returns:
        array: polynomial coefficients
    """
    E_k = E.copy()
    # Compute combinations in O(n^2log n) time
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = E_k.sum() / (2 * i)
        E_k *= E
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor
    return comb[count, :]


@numba.jit(nopython=True, cache=True)
def f_loop(E, AX_S, XD_S, D_S, n):  # pragma : no cover
    """
    Evaluate the polyonial coefficients of the function in the eigenvalue-trace formula.

    Args:
        E (array): Eigenvalues of AX
        AX_S (array): AX_S with weights given by repetitions and exluded rows removed
        XD_S (array): Diagonal multiplied by X
        D_S (array): Diagonal
        n (int): Number of polynomial coefficients to compute

    Returns:
        array: Polynomial coefficients
    """
    E_k = E.copy()
    # Compute combinations in O(n^2log n) time
    # code translated from thewalrus matlab script
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = E_k.sum() / (2 * i) + (XD_S @ D_S) / 2
        E_k *= E
        XD_S = XD_S @ AX_S
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor
    return comb[count, :]


@numba.jit(nopython=True, cache=True)
def f_loop_odd(E, AX_S, XD_S, D_S, n, oddloop, oddVX_S):  # pragma: no cover
    """
    Evaluate the polyonial coefficients of the function in the eigevalue-trace formula
    when there is a self-edge in the fixed perfect matching.

    Args:
        E (array): Eigenvalues of AX
        AX_S (array): AX_S with weights given by repetitions and exluded rows removed
        XD_S (array): Diagonal multiplied by X
        D_S (array): Diagonal
        n (int): Number of polynomial coefficients to compute
        oddloop (float): Weight of self-edge
        oddVX_S (array): Vector corresponding to matrix at the index of the self-edge

    Returns:
        array: Polynomial coefficients
    """
    E_k = E.copy()

    count = 0
    comb = np.zeros((2, n + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n + 1):
        if i == 1:
            factor = oddloop
        elif i % 2 == 0:
            factor = E_k.sum() / i + (XD_S @ D_S) / 2
            E_k *= E
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
def get_AX_S(kept_edges, A):  # pragma: no cover
    """
    Given the kept edges, return the appropriate scaled submatrices to compute f.

    Args:
        kept_edges (array): Number of repetitions of each edge
        A (array): Matrix before repetitions applied

    Returns:
        AX_nonzero (array): Scaled A @ X (where X = ((0, I), (I, 0))
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:, :n_nonzero_edges] = (
        kept_edges_nonzero * A_nonzero[:, n_nonzero_edges:]
    )
    AX_nonzero[:, n_nonzero_edges:] = (
        kept_edges_nonzero * A_nonzero[:, :n_nonzero_edges]
    )

    return AX_nonzero


@numba.jit(nopython=True, cache=True)
def get_submatrices(kept_edges, A, D, oddV):  # pragma: no cover
    """
    Given the kept edges, return the appropriate scaled submatrices to compute f.

    Args:
        kept_edges (array): Number of repetitions of each edge
        A (array): Matrix before repetitions applied
        D (array): Diagonal before repetitions applied
        oddV (array): Row of matrix at index of self-edge, None is no self-edge

    Returns:
        AX_nonzero (array): Scaled A @ X (where X = ((0, I), (I, 0))
        XD_nonzero (array): Scaled X @ D
        D_nonzero (array): Scaled D
        oddVX_nonzero (array): Scaled oddV @ X
    """

    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]

    A_nonzero = nb_ix(A, nonzero_rows, nonzero_rows)

    AX_nonzero = np.empty_like(A_nonzero, dtype=np.complex128)
    AX_nonzero[:, :n_nonzero_edges] = (
        kept_edges_nonzero * A_nonzero[:, n_nonzero_edges:]
    )
    AX_nonzero[:, n_nonzero_edges:] = (
        kept_edges_nonzero * A_nonzero[:, :n_nonzero_edges]
    )

    D_nonzero = D[nonzero_rows]

    XD_nonzero = np.empty_like(D_nonzero, dtype=np.complex128)
    XD_nonzero[:n_nonzero_edges] = kept_edges_nonzero * D_nonzero[n_nonzero_edges:]
    XD_nonzero[n_nonzero_edges:] = kept_edges_nonzero * D_nonzero[:n_nonzero_edges]

    if oddV is not None:
        oddV_nonzero = oddV[nonzero_rows]
        oddVX_nonzero = np.empty_like(oddV_nonzero, dtype=np.complex128)
        oddVX_nonzero[:n_nonzero_edges] = (
            kept_edges_nonzero * oddV_nonzero[n_nonzero_edges:]
        )
        oddVX_nonzero[n_nonzero_edges:] = (
            kept_edges_nonzero * oddV_nonzero[:n_nonzero_edges]
        )
    else:
        oddVX_nonzero = None

    return AX_nonzero, XD_nonzero, D_nonzero, oddVX_nonzero


@numba.jit(nopython=True, cache=True)
def get_submatrix_batch_odd0(kept_edges, oddV0):  # pragma: no cover
    """
    Find oddVX_nonzero0 for batching (sometimes different vertices are identified as self edges).

    Args:
        kept_edges (array): Number of repetitions of each edge
        oddV0 (array): Row of matrix at index of self-edge, None is no self-edge

    Returns:
        array: Scaled oddV0 @ X for
    """
    z = np.concatenate((kept_edges, kept_edges))
    nonzero_rows = np.where(z != 0)[0]
    n_nonzero_edges = len(nonzero_rows) // 2

    kept_edges_nonzero = kept_edges[np.where(kept_edges != 0)]
    oddV_nonzero0 = oddV0[nonzero_rows]
    oddVX_nonzero0 = np.empty_like(oddV_nonzero0, dtype=np.complex128)
    oddVX_nonzero0[:n_nonzero_edges] = (
        kept_edges_nonzero * oddV_nonzero0[n_nonzero_edges:]
    )
    oddVX_nonzero0[n_nonzero_edges:] = (
        kept_edges_nonzero * oddV_nonzero0[:n_nonzero_edges]
    )

    return oddVX_nonzero0


@numba.jit(nopython=True, cache=True)
def get_Dsubmatrices(kept_edges, D):  # pragma: no cover
    """
    Find submatrices for batch gamma functions.
    """

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
    """
    Computes the eigenvalues of a matrix.

    Args:
        M (array): Square matrix

    Returns:
        array: Eigenvalues of the matrix M
    """
    return np.linalg.eigvals(M)


# @numba.jit(nopython=True, cache=True)
# pylint: disable = missing-function-docstring
def calc_approx_steps(fixed_reps, N_cutoff):
    steps = int(np.prod(np.sqrt(fixed_reps)) + 1) * N_cutoff // 2
    return steps
