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
Computation of the hafnian.
"""
import numpy as np
import numba
from j_loop_hafnian_subroutines import (
    precompute_binoms,
    matched_reps,
    find_kept_edges,
    f,
    get_AX_S,
    eigvals,
)

# pylint: disable=W0612, E1133
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_hafnian(A, edge_reps, glynn=True):  # pragma: no cover

    r"""
    Compute hafnian, using inputs as prepared by frontend hafnian function
    compiled with Numba.


    Args:
        A (array): matrix ordered according to the chosen perfect matching.
        edge_reps (array): how many times each edge in the perfect matching is repeated.
        glynn (bool): whether to use finite difference sieve.

    Returns:
        complex128: value of loop hafnian.
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

        E = eigvals(AX_S)  # O(n^3) step

        prefac = (-1.0) ** (N // 2 - edge_sum) * binom_prod

        if glynn and kept_edges[0] == 0:
            prefac *= 0.5
        Hnew = prefac * f(E, N)[N // 2]

        H += Hnew

    if glynn:
        H = H * 0.5 ** (N // 2 - 1)

    return H


def hafnian(A, reps=None, glynn=True):
    r"""
    Calculate hafnian with (optional) repeated rows and columns.


    Args:
        A (array): N x N matrix.
        reps (list): length-N list of repetitions of each row/col (optional), if not provided, each row/column
                    assumed to be repeated once.
        glynn (bool): If True, use Glynn-style finite difference sieve formula, if False, use Ryser style inclusion/exclusion principle.

    Returns
        np.complex128: result of loop hafnian calculation.
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
