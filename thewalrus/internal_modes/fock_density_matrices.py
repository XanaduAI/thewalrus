# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

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
Set of functions for calculating Fock basis density matrices for heralded states created by PNR measurements on Gaussian states with multiple internal modes
"""

import numpy as np 
import numba

from .._hafnian import nb_binom, nb_ix, find_kept_edges
from .useful_tools import (
    nb_block, 
    nb_Qmat, 
    f_all_charpoly,
    spatial_reps_to_schmidt_reps,
    fact
    )

@numba.jit(nopython=True, parallel=True, cache=True)
def _density_matrix_single_mode(cov, pattern, cutoff=13):
    """
    numba function (use the wrapper function: density_matrix_multimode)

    density matrix of first mode when heralded by pattern on a zero-displaced Gaussian state,
    and allows for herald modes to contain multiple internal modes per PNR detector.

    Args:
        cov (array): 2M x 2M covariance matrix
        pattern (list): M-1 length list of the heralding pattern
        cutoff (int): photon number cutoff. Should be odd. Even numbers will be rounded up to an odd number
    """

    M = len(pattern) + 1

    K = cov.shape[0] // (2 * M)

    #filter out all unwanted schmidt modes in heralded spatial mode
    
    #create passive transformation of filter
    T = np.zeros((M*K, M*K), dtype=np.float64)
    T[0,0] = 1 
    T[K:,K:] = np.eye((M-1)*K, dtype=np.float64)

    #apply channel of filter
    P = nb_block(((T, np.zeros_like(T)), (np.zeros_like(T), T)))
    L = np.eye(P.shape[0]) - P @ P.T  
    cov = P @ cov @ P.T + L

    Q = nb_Qmat(cov)
    O = np.identity(2 * M * K) - np.linalg.inv(Q)
    A = np.empty_like(O, dtype=np.complex128)
    A[:M*K,:] = O[M*K:,:].conj()
    A[M*K:,:] = O[:M*K,:].conj()

    if cutoff % 2 == 0:
        cutoff += 1

    half_c = cutoff // 2
    x = [0] + [M*K] + [0] + list(range(K, M*K)) + [0] + [M*K] + [M*K] + list(range((M+1)*K, 2*M*K))
    x = np.array(x)
    Ax = nb_ix(A, x, x) #A[np.ix_(x, x)]

    edge_reps = np.array([half_c, half_c, 1] + list(pattern))
    n_edges = 3 + K * len(pattern)

    assert n_edges == Ax.shape[0] // 2 == 3 + K * (M - 1)

    N_max = 2 * edge_reps.sum()
    N_fixed = 2 * np.sum(pattern)

    steps = np.prod(edge_reps + 1)

    haf_arr = np.zeros((cutoff+1, cutoff+1), dtype=np.complex128)
    for j in numba.prange(steps):
        haf_arr_new = np.zeros_like(haf_arr)
        kept_edges = find_kept_edges(j, edge_reps)

        edges_sum = kept_edges.sum()

        binom_prod = 1.
        for i in range(3, M+2):
            binom_prod *= nb_binom(edge_reps[i], kept_edges[i])

        glynn_edges = 2 * kept_edges - edge_reps

        glynn_edges_heralding = spatial_reps_to_schmidt_reps(glynn_edges[3:], K) - (K - 1)
        edge_weights = np.concatenate((glynn_edges[:3], glynn_edges_heralding))
        assert len(edge_weights) == n_edges

        AX_S = np.empty_like(Ax)
        AX_S[:, :n_edges] = edge_weights * Ax[:, n_edges:]
        AX_S[:, n_edges:] = edge_weights * Ax[:, :n_edges]

        f_j = f_all_charpoly(AX_S, N_max)

        for i_n in range(kept_edges[0], edge_reps[0]+1):
            for i_m in range(kept_edges[1], edge_reps[1]+1):
                for i_oddodd in range(kept_edges[2], edge_reps[2]+1):
                    n = 2 * i_n + i_oddodd
                    m = 2 * i_m + i_oddodd
                    binom_prod_i = (
                                binom_prod * 
                                nb_binom(i_n, kept_edges[0]) * 
                                nb_binom(i_m, kept_edges[1]) *
                                nb_binom(i_oddodd, kept_edges[2])
                                )
                    N = N_fixed + n + m
                    plusminus = (-1) ** ((edges_sum + (n + m) // 2) % 2)
                    haf_arr_new[m, n] = plusminus * binom_prod_i * f_j[N//2]

        haf_arr += haf_arr_new

    rho = (-1) ** pattern.sum() * haf_arr / (np.sqrt(np.linalg.det(Q).real) * np.prod(fact[pattern]))

    for n in range(cutoff+1):
        for m in range(cutoff+1):
            rho[n,m] /= np.sqrt(fact[n] * fact[m]) * (2 ** ((N_fixed + n + m) // 2))

    return rho

def density_matrix_single_mode(cov, pattern, cutoff=13):
    """
    calculates density matrix of first mode when heralded by pattern on a zero-displaced, M-mode Gaussian state
    where each mode contains K internal modes.

    Args:
        cov (array): 2MK x 2MK covariance matrix
        pattern (list): M-1 length list of the heralding pattern
        cutoff (int): photon number cutoff. Should be odd. Even numbers will be rounded up to an odd number
    Returns:
        array[complex]: (cutoff+1, cutoff+1) dimension density matrix
    """

    cov = np.array(cov).astype(np.float64)
    pattern = np.array(pattern)
    return _density_matrix_single_mode(cov, pattern, cutoff)
