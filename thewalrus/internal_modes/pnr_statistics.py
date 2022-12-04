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
Set of functions for calculating photon number resolved measurement probabilities on Gaussian states with multiple internal modes
"""

import numpy as np

import numba

from scipy.special import factorial as fac

from ..quantum import Qmat
from .._hafnian import find_kept_edges, nb_binom, f_from_powertrace, nb_ix
from ..charpoly import powertrace

from .utils import spatial_reps_to_schmidt_reps, spatial_modes_to_schmidt_modes


@numba.jit(nopython=True, parallel=True, cache=True)
def hafkd(As, edge_reps, K=1):
    r"""
    generalised version of hafnian to include multiple internal modes

    Args:
        As (array[:,:,:]): :math:`D\times2MK\times2MK` array of D covariance matrices, each being a 2MK covariance matrix
        edge_reps (array[int]): length-:math:M array. Gives how many times edges are repeated, or how many photons in each mode
        K (int): How many internal modes are in each detected mode
    """

    M = As[0].shape[0] // (2 * K)

    N = 2 * edge_reps.sum()

    steps = np.prod(edge_reps + 1)

    H = 0.0

    for j in numba.prange(steps):
        kept_edges = find_kept_edges(j, edge_reps)
        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(M):
            binom_prod *= nb_binom(edge_reps[i], kept_edges[i])

        glynn_edges = 2 * kept_edges - edge_reps

        z = np.concatenate((glynn_edges, glynn_edges))
        nonzero_rows = np.where(z != 0)[0]
        n_nonzero_edges = K * len(nonzero_rows) // 2

        glynn_edges_nonzero = glynn_edges[np.where(glynn_edges != 0)]
        glynn_schmidt_edges_nonzero = spatial_reps_to_schmidt_reps(glynn_edges_nonzero, K)

        nonzero_schmidt_modes = spatial_modes_to_schmidt_modes(nonzero_rows, K)
        prefac = (-1.0) ** (N // 2 - edges_sum) * binom_prod

        powertraces = np.zeros(N // 2 + 1, dtype=np.complex128)
        for A in As:
            A_nonzero = nb_ix(A, nonzero_schmidt_modes, nonzero_schmidt_modes)
            AX_S = np.empty_like(A_nonzero, dtype=np.complex128)
            AX_S[:, :n_nonzero_edges] = glynn_schmidt_edges_nonzero * A_nonzero[:, n_nonzero_edges:]
            AX_S[:, n_nonzero_edges:] = glynn_schmidt_edges_nonzero * A_nonzero[:, :n_nonzero_edges]
            powertraces += powertrace(AX_S, N // 2 + 1)[: N // 2 + 1]

        f_j = f_from_powertrace(powertraces, N)[N // 2]

        Hnew = prefac * f_j
        H += Hnew

    return H / (2 ** (N // 2))


def pnr_prob(covs, i, hbar=2):
    r"""
    Returns the probability of measuring a pnr outcome given by i on a Gaussian state
    which may have multiple internal modes.

    Args:
        covs (list[array[:,:]]): each array is an :math:`2MK \times 2MK` covariance matrix, for M modes, each with K internal modes and corresponds to a seperable set of internal modes
        i (array[int]): length-:math:`M` photon number outcome
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    """

    # if given a single covariance matrix, nest into a list
    if covs[0].ndim == 1:
        covs = [covs]

    M = len(i)
    K = covs[0].shape[0] // (2 * M)

    vac_prob = 1.0
    As = []
    for cov in covs:
        Q = Qmat(cov, hbar=hbar)
        I = np.identity(2 * M * K)
        O = I - np.linalg.inv(Q)
        A = np.empty_like(O, dtype=np.complex128)
        A[: M * K, :] = O[M * K :, :].conj()
        A[M * K :, :] = O[: M * K, :].conj()
        As.append(A)

        vac_prob /= np.sqrt(np.linalg.det(Q).real)

    fac_prod = np.prod(fac(i), dtype=np.float64)
    haf = hafkd(np.array(As), np.array(i), K)
    prob = haf.real * vac_prob / fac_prod

    return prob
