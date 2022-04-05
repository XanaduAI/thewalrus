# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Sampling algorithms
===================

**Module name:** :mod:`thewalrus.loop_hafnian_batch`

.. currentmodule:: thewalrus.loop_hafnian_batch

Tools to evaluate the loop hafnian batch to do gaussian boson sampling.

Summary
-------

.. autosummary::
    _calc_loop_hafnian_batch_even
    _calc_loop_hafnian_batch_odd
    add_batch_edges_even
    add_batch_edges_odd
    loop_hafnian_batch

Code details
------------
"""
import numpy as np
import numba
from thewalrus._hafnian import (
    precompute_binoms,
    matched_reps,
    find_kept_edges,
    f_loop,
    f_loop_odd,
    get_submatrices,
    get_submatrix_batch_odd0,
)

# pylint: disable = too-many-arguments, not-an-iterable
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_loop_hafnian_batch_even(
    A, D, fixed_edge_reps, batch_max, odd_cutoff, glynn=True
):  # pragma: no cover
    r"""Calculate the loop hafnian batch for even modes.

    Args:
        A (array): input matrix.
        D (array): diagonal.
        fixed_edge_reps (array): fixed number of edge repetition.
        batch_max (array): maximum batch.
        odd_cutoff (array): cutoff for odd modes.
        glynn (boolean): determines the method used to evaluate the loop hafnian batch.

    Returns:
        H_batch (array): matrix result.
    """
    oddloop = D[0]
    oddV = A[0, :]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum()  # number of photons

    N_max = N_fixed + 2 * batch_max + odd_cutoff

    edge_reps = np.concatenate((np.array([batch_max]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)
    # precompute binomial coefficients
    max_binom = edge_reps.max() + odd_cutoff
    binoms = precompute_binoms(max_binom)

    H_batch = np.zeros(2 * batch_max + odd_cutoff + 1, dtype=np.complex128)
    # prange to range
    for j in numba.prange(steps):

        Hnew = np.zeros(2 * batch_max + odd_cutoff + 1, dtype=np.complex128)

        kept_edges = find_kept_edges(j, edge_reps)
        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(1, n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        if glynn:
            delta = 2 * kept_edges - edge_reps
        else:
            delta = kept_edges

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D, oddV)

        AX_S_copy = AX_S.copy()

        f_even = f_loop(AX_S_copy, AX_S, XD_S, D_S, N_max)
        f_odd = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_max, oddloop, oddVX_S)

        for N_det in range(2 * kept_edges[0], 2 * batch_max + odd_cutoff + 1):
            N = N_fixed + N_det
            plus_minus = (-1.0) ** (N // 2 - edges_sum)

            n_det_binom_prod = binoms[N_det // 2, kept_edges[0]] * binom_prod

            if N_det % 2 == 0:
                Hnew[N_det] += n_det_binom_prod * plus_minus * f_even[N // 2]
            else:
                Hnew[N_det] += n_det_binom_prod * plus_minus * f_odd[N]

        H_batch += Hnew

    if glynn:
        for j in range(H_batch.shape[0]):
            x = N_fixed + j
            H_batch[j] *= 0.5 ** (x // 2)

    return H_batch


# pylint: disable = too-many-arguments, not-an-iterable
@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_loop_hafnian_batch_odd(
    A, D, fixed_edge_reps, batch_max, even_cutoff, glynn=True
):  # pragma: no cover
    r"""Calculate the loop hafnian batch for odd modes.

    Args:
        A (array): input matrix.
        D (array): diagonal.
        fixed_edge_reps (array): fixed number of edge repetition.
        batch_max (array): maximum batch.
        even_cutoff (array): cutoff for even modes.
        glynn (boolean): determines the method used to evaluate the loop hafnian batch.

    Returns:
        H_batch (array): matrix result.
    """
    oddloop = D[0]
    oddV = A[0, :]

    # when I added the extra edges, I place the edge which goes from the oddmode to
    # to the current mode in the index 1 position of the array
    oddloop0 = D[1]
    oddV0 = A[1, :]

    n = A.shape[0]
    N_fixed = 2 * fixed_edge_reps.sum() + 1
    N_max = N_fixed + 2 * batch_max + even_cutoff + 1

    edge_reps = np.concatenate((np.array([batch_max, 1]), fixed_edge_reps))
    steps = np.prod(edge_reps + 1)
    # precompute binomial coefficients
    max_binom = edge_reps.max() + even_cutoff
    binoms = precompute_binoms(max_binom)

    H_batch = np.zeros(2 * batch_max + even_cutoff + 2, dtype=np.complex128)
    for j in numba.prange(steps):

        Hnew = np.zeros(2 * batch_max + even_cutoff + 2, dtype=np.complex128)

        kept_edges = find_kept_edges(j, edge_reps)
        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(1, n // 2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]

        if glynn:
            delta = 2 * kept_edges - edge_reps
        else:
            delta = kept_edges

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(delta, A, D, oddV)

        AX_S_copy = AX_S.copy()

        if kept_edges[0] == 0 and kept_edges[1] == 0:
            oddVX_S0 = get_submatrix_batch_odd0(delta, oddV0)
            plus_minus = (-1) ** (N_fixed // 2 - edges_sum)
            f = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_fixed, oddloop0, oddVX_S0)[N_fixed]
            H_batch[0] += binom_prod * plus_minus * f

        f_even = f_loop(AX_S_copy, AX_S, XD_S, D_S, N_max)
        f_odd = f_loop_odd(AX_S_copy, AX_S, XD_S, D_S, N_max, oddloop, oddVX_S)

        for N_det in range(2 * kept_edges[0] + 1, 2 * batch_max + even_cutoff + 2):
            N = N_fixed + N_det
            plus_minus = (-1) ** (N // 2 - edges_sum)

            n_det_binom_prod = binoms[(N_det - 1) // 2, kept_edges[0]] * binom_prod

            if N % 2 == 0:
                Hnew[N_det] += n_det_binom_prod * plus_minus * f_even[N // 2]
            else:
                Hnew[N_det] += n_det_binom_prod * plus_minus * f_odd[N]

        H_batch += Hnew

    if glynn:
        for j in range(H_batch.shape[0]):
            x = N_fixed + j
            H_batch[j] *= 0.5 ** (x // 2)

    return H_batch


def add_batch_edges_even(fixed_edges):
    r"""Add batch even.

    Args:
        fixed_edges (array): fixed number of edge repetition.

    Returns:
        edges (array): edges for even modes.
    """
    if len(fixed_edges) == 0:
        return np.array([0, 0], dtype=int)
    n_edges = fixed_edges.shape[0]
    edges = np.zeros(n_edges + 2, dtype=int)
    new_edge = max(fixed_edges) + 1
    edges[0] = new_edge
    edges[1 : n_edges // 2 + 1] = fixed_edges[: n_edges // 2]
    edges[n_edges // 2 + 1] = new_edge
    edges[n_edges // 2 + 2 :] = fixed_edges[n_edges // 2 :]
    return edges


def add_batch_edges_odd(fixed_edges, oddmode):
    r""".

    Args:
        fixed_edges (array): fixed number of edge repetition.
        oddmode (int): number of odd modes.

    Returns:
        edges (array): edges for odd modes.
    """
    if len(fixed_edges) == 0:
        return np.array([1, oddmode, 1, 1], dtype=int)
    n_edges = fixed_edges.shape[0]
    edges = np.zeros(n_edges + 4, dtype=int)
    new_edge = max(max(fixed_edges), oddmode) + 1
    edges[0] = new_edge
    edges[1] = oddmode
    edges[2 : n_edges // 2 + 2] = fixed_edges[: n_edges // 2]
    edges[n_edges // 2 + 2] = new_edge
    edges[n_edges // 2 + 3] = new_edge
    edges[n_edges // 2 + 4 :] = fixed_edges[n_edges // 2 :]
    return edges


def loop_hafnian_batch(A, D, fixed_reps, N_cutoff, glynn=True):
    r"""Calculate the loop hafnian batch.

    Args:
        A (array): input matrix.
        D (array): diagonal.
        fixed_edge_reps (array): fixed number of edge repetition.
        N_cutoff (int):
        glynn (boolean): determines the method used to evaluate the loop hafnian batch.

    Returns:
        loop hafnian batch (array):
    """
    # checks
    n = A.shape[0]
    assert A.shape[1] == n
    assert D.shape == (n,)
    assert len(fixed_reps) == n - 1

    nz = np.nonzero(list(fixed_reps) + [1])[0]
    Anz = A[np.ix_(nz, nz)]
    Dnz = D[nz]

    fixed_reps = np.asarray(fixed_reps)
    fixed_reps_nz = fixed_reps[nz[:-1]]

    fixed_edges, fixed_m_reps, oddmode = matched_reps(fixed_reps_nz)

    if oddmode is None:
        batch_max = N_cutoff // 2
        odd_cutoff = N_cutoff % 2
        edges = add_batch_edges_even(fixed_edges)
        Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
        Dx = Dnz[edges].astype(np.complex128)
        return _calc_loop_hafnian_batch_even(
            Ax, Dx, fixed_m_reps, batch_max, odd_cutoff, glynn=glynn
        )
    edges = add_batch_edges_odd(fixed_edges, oddmode)
    Ax = Anz[np.ix_(edges, edges)].astype(np.complex128)
    Dx = Dnz[edges].astype(np.complex128)
    batch_max = (N_cutoff - 1) // 2
    even_cutoff = 1 - (N_cutoff % 2)
    return _calc_loop_hafnian_batch_odd(Ax, Dx, fixed_m_reps, batch_max, even_cutoff, glynn=glynn)
