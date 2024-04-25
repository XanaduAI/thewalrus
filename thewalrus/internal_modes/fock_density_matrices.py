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

import warnings
import numpy as np
import numba
from scipy.special import factorial

from .._hafnian import nb_binom, nb_ix, find_kept_edges, f_from_matrix
from .utils import (
    nb_Qmat,
    spatial_reps_to_schmidt_reps,
    fact,
    project_onto_local_oscillator,
)
from .pnr_statistics import haf_blocked
from ..quantum import Qmat, Amat


# pylint: disable=too-many-arguments, too-many-statements
@numba.jit(nopython=True, parallel=True, cache=True)
def _density_matrix_single_mode(
    cov, pattern, LO_overlap=None, cutoff=13, hbar=2
):  # pragma: no cover
    """
    numba function (use the wrapper function: density_matrix_multimode)

    density matrix of first mode when heralded by pattern on a zero-displaced Gaussian state,
    and allows for herald modes to contain multiple internal modes per PNR detector.

    Args:
        cov (array): 2MK x 2MK covariance matrix
        pattern (array): M-1 length array of the heralding pattern
        LO_overlap (array): overlap between internal modes and local oscillator
        cutoff (int): photon number cutoff. Should be odd. Even numbers will be rounded up to an odd number
        hbar (float): the value of hbar (default 2)
    """

    M = len(pattern) + 1

    K = cov.shape[0] // (2 * M)

    # filter out all unwanted Schmidt modes in heralded spatial mode

    cov = project_onto_local_oscillator(cov, M, LO_overlap=LO_overlap, hbar=hbar)

    Q = nb_Qmat(cov, hbar=hbar)
    O = np.eye(2 * M * K) - np.linalg.inv(Q)
    A = np.empty_like(O, dtype=np.complex128)
    A[: M * K, :] = O[M * K :, :].conj()
    A[M * K :, :] = O[: M * K, :].conj()

    half_c = (cutoff - 1) // 2

    x = (
        [0]
        + [M * K]
        + [0]
        + list(range(K, M * K))
        + [0]
        + [M * K]
        + [M * K]
        + list(range((M + 1) * K, 2 * M * K))
    )
    x = np.array(x)
    Ax = nb_ix(A, x, x)  # A[np.ix_(x, x)]

    edge_reps = np.array((half_c, half_c, 1) + pattern)
    n_edges = 3 + K * len(pattern)

    assert n_edges == Ax.shape[0] // 2 == 3 + K * (M - 1)

    N_max = 2 * edge_reps.sum()
    N_fixed = 2 * sum(pattern)

    steps = np.prod(edge_reps + 1)

    haf_arr = np.zeros((cutoff + 1, cutoff + 1), dtype=np.complex128)
    for j in numba.prange(steps):
        haf_arr_new = np.zeros_like(haf_arr)
        kept_edges = find_kept_edges(j, edge_reps)

        edges_sum = kept_edges.sum()

        binom_prod = 1.0
        for i in range(3, M + 2):
            binom_prod *= nb_binom(edge_reps[i], kept_edges[i])

        glynn_edges = 2 * kept_edges - edge_reps

        glynn_edges_heralding = spatial_reps_to_schmidt_reps(glynn_edges[3:], K) - (K - 1)
        edge_weights = np.concatenate((glynn_edges[:3], glynn_edges_heralding))
        assert len(edge_weights) == n_edges

        AX_S = np.empty_like(Ax)
        AX_S[:, :n_edges] = edge_weights * Ax[:, n_edges:]
        AX_S[:, n_edges:] = edge_weights * Ax[:, :n_edges]

        f_j = f_from_matrix(AX_S, N_max)

        for i_n in range(kept_edges[0], edge_reps[0] + 1):
            for i_m in range(kept_edges[1], edge_reps[1] + 1):
                for i_oddodd in range(kept_edges[2], edge_reps[2] + 1):
                    n = 2 * i_n + i_oddodd
                    m = 2 * i_m + i_oddodd
                    binom_prod_i = (
                        binom_prod
                        * nb_binom(i_n, kept_edges[0])
                        * nb_binom(i_m, kept_edges[1])
                        * nb_binom(i_oddodd, kept_edges[2])
                    )
                    N = N_fixed + n + m
                    plusminus = (-1) ** ((edges_sum + (n + m) // 2) % 2)
                    haf_arr_new[m, n] = plusminus * binom_prod_i * f_j[N // 2]

        haf_arr += haf_arr_new

    rho = (
        (-1) ** sum(pattern)
        * haf_arr
        / (np.sqrt(np.linalg.det(Q).real) * np.prod(fact[np.array(list(pattern))]))
    )

    for n in range(cutoff):
        for m in range(cutoff):
            rho[n, m] /= np.sqrt(fact[n] * fact[m]) * (2 ** ((N_fixed + n + m) // 2))

    rho = rho[:cutoff, :cutoff]

    return rho


def check_probabilities(probs, atol=1e-08):
    """
    Convenience function for checking that the input is close enough to a probability distribution.

    Args:
        probs (array): probabilities to be tested.
        atol (float): absolute tolerance relative to the normalization.

    Returns:
        (boolean): whether the test passed or not.
    """
    real_probs = probs.real
    imag_probs = probs.imag
    pos_probs = real_probs[real_probs > 0]
    neg_probs = real_probs[real_probs < 0]
    net_prob = sum(pos_probs)
    if np.any(np.abs(imag_probs) > atol * net_prob):
        return False
    if np.any(np.abs(neg_probs) > atol * net_prob):
        return False
    return True


def density_matrix_single_mode(
    cov,
    pattern,
    normalize=False,
    LO_overlap=None,
    cutoff=13,
    hbar=2,
    method="recursive",
    atol=1e-08,
):
    """
    Calculates density matrix of first mode when heralded by pattern on a zero-displaced, M-mode Gaussian state
    where each mode contains K internal modes.

    Args:
        cov (array): 2MK x 2MK covariance matrix
        pattern (dict): heralding pattern total photon number in the spatial modes (int), indexed by spatial mode
        normalize (bool): whether to normalise the output density matrix
        LO_overlap (array): overlap between internal modes and local oscillator
        cutoff (int): photon number cutoff. Should be odd. Even numbers will be rounded up to an odd number
        hbar (float): the value of hbar (default 2)
        method (str): which method to use, "recursive", "non-recursive" or "diagonals"
        atol (float): value for raising warning when testing for valid probabilities
    Returns:
        array[complex]: (cutoff+1, cutoff+1) dimension density matrix
    """

    cov = np.array(cov.real).astype(np.float64)
    M = len(pattern) + 1
    K = cov.shape[0] // (2 * M)
    if not set(list(pattern.keys())).issubset(set(list(np.arange(M)))):
        raise ValueError("Keys of pattern must correspond to all but one spatial mode")
    N_nums = tuple(pattern.values())
    HM = list(set(list(np.arange(M))).difference(list(pattern.keys())))[0]
    if LO_overlap is not None:
        if not K == LO_overlap.shape[0]:
            raise ValueError("Number of overlaps with LO must match number of internal modes")
        if not (np.linalg.norm(LO_overlap) < 1 or np.allclose(np.linalg.norm(LO_overlap), 1)):
            raise ValueError("Norm of overlaps must not be greater than 1")

    # swapping the spatial modes around such that we are heralding in spatial mode 0
    if HM != 0:
        swapV = list(range(M))
        (swapV[0], swapV[HM]) = (swapV[HM], swapV[0])
        perm = (np.arange(M * K).reshape(M, K))[swapV].flatten()
        double_perm = np.concatenate([perm, perm + M * K])
        cov = cov[:, double_perm][double_perm]

    if method == "recursive":
        vals = _density_matrix_single_mode(cov, N_nums, LO_overlap, cutoff, hbar)
        if check_probabilities(np.diag(vals), atol=atol) is False:
            warnings.warn(
                "Some of the diagonal elements of the density matrix are significantly negative or have significant imaginary parts. Try using the `non-recursive` method instead.",
                UserWarning,
            )
        if normalize:
            vals = vals / np.trace(vals).real
        return vals
    if method in ["non-recursive", "diagonals"]:
        cov = project_onto_local_oscillator(cov, M, LO_overlap=LO_overlap, hbar=hbar)
        A = Amat(cov)
        Q = Qmat(cov)
        pref = 1 / np.sqrt(np.linalg.det(Q).real)
        blocks = np.arange(K * M).reshape([M, K])
        dm = np.zeros([cutoff, cutoff], dtype=np.complex128)
        if method == "non-recursive":
            num_modes = M * K
            block_size = K
            for i in range(cutoff):
                for j in range(i + 1):
                    if (i - j) % 2 == 0:
                        patt_long = (j,) + N_nums + ((i - j) // 2,)
                        new_blocks = np.concatenate((blocks, np.array([K + blocks[-1]])), axis=0)
                        perm = (
                            list(range(num_modes))
                            + list(range(block_size))
                            + list(range(num_modes, 2 * num_modes))
                            + list(range(block_size))
                        )
                        Aperm = A[:, perm][perm]
                        dm[j, i] = (
                            pref
                            * haf_blocked(Aperm, blocks=new_blocks, repeats=patt_long)
                            / (
                                np.prod(factorial(patt_long[1:-1]))
                                * np.sqrt(factorial(i) * factorial(j))
                            )
                        )
                        dm[i, j] = np.conj(dm[j, i])
                    else:
                        dm[i, j] = 0
                        dm[j, i] = 0
        else:
            for i in range(cutoff):
                patt_long = (i,) + N_nums
                dm[i, i] = (
                    pref
                    * haf_blocked(A, blocks=blocks, repeats=patt_long)
                    / np.prod(factorial(patt_long))
                )
        if check_probabilities(np.diag(dm)) is False:
            warnings.warn(
                "Some of the diagonal elements of the density matrix are significantly negative or have significant imaginary parts.",
                UserWarning,
            )
        if normalize:
            dm = dm / np.trace(dm)
        return dm

    raise ValueError("Unknown method for density_matrix_single_mode")
