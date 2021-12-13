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
Loop-Hafnian Python interface

Bulmer, Jacob F. F., et al. “The Boundary for Quantum Advantage in
Gaussian Boson Sampling.” ArXiv:2108.01622 [Quant-Ph], Aug. 2021.
arXiv.org, http://arxiv.org/abs/2108.01622
"""

import numpy as np
import numba
from _j_loop_hafnian_subroutines import (
    precompute_binoms,
    matched_reps,
    find_kept_edges,
    f_loop,
    f_loop_odd,
    get_submatrices,
    eigvals,
)

# pylint: disable=too-many-arguments, redefined-outer-name, not-an-iterable
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_loop_hafnian(A, D, edge_reps, oddloop=None, oddV=None, glynn=True):  # pragma: no cover
    """
    Compute loop hafnian, using inputs as prepared by frontend loop_hafnian function
    compiled with Numba.

    Args:
        A (array): Matrix ordered according to the chosen perfect matching.
        D (array): Diagonals ordered according to the chosen perfect matching.
        edge_reps (array): How many times each edge in the perfect matching is repeated.
        oddloop (float): Weight of self-loop in perfect matching, None if no self-loops.
        oddV (array): Row of matrix corresponding to the odd loop in the perfect matching.
        glynn (bool): Whether to use finite difference sieve.

    Returns:
        complex128: Value of loop hafnian.
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

        E = eigvals(AX_S)  # O(n^3) step

        prefac = (-1.0) ** (N // 2 - edge_sum) * binom_prod

        if oddloop is not None:
            Hnew = prefac * f_loop_odd(E, AX_S, XD_S, D_S, N, oddloop, oddVX_S)[N]
        else:
            if glynn and kept_edges[0] == 0:
                prefac *= 0.5
            Hnew = prefac * f_loop(E, AX_S, XD_S, D_S, N)[N // 2]

        H += Hnew

    if glynn:
        if oddloop is None:
            H = H * 0.5 ** (N // 2 - 1)
        else:
            H = H * 0.5 ** (N // 2)

    return H


# pylint: disable=redefined-outer-name
def loop_hafnian(A, D=None, reps=None, glynn=True):
    """
    Calculate loop hafnian with (optional) repeated rows and columns.

    Args:
        A (array): N x N matrix.
        D (array): Diagonal entries of matrix (optional). If not provided, D is the diagonal of A.
                    If repetitions are provided, D should be provided explicitly.
        reps (list): Length-N list of repetitions of each row/col (optional), if not provided, each row/column
                    assumed to be repeated once.
        glynn (bool): If True, use Glynn-style finite difference sieve formula, if False, use Ryser style inclusion/exclusion principle.

    Returns
        np.complex128: Result of loop hafnian calculation.
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
        return D[0]

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


### Compile code on some small instances ###
A = np.ones((3, 3))
assert np.allclose(loop_hafnian(A), 4)
A = np.ones((4, 4))
assert np.allclose(loop_hafnian(A), 10)
############################################
