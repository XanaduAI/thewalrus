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
Function for calculating probability of distinguishable squeezing Gaussian Boson sampling experiments
"""
import numpy as np
import numba

from ..symplectic import passive_transformation, squeezing
from ..quantum import Qmat
from .._hafnian import nb_binom
from .utils import nb_Qmat, fact


@numba.jit(nopython=True, cache=True)
def f_all(E, n):
    """
    evaluate the polyonial coefficients of the function in the eigevalue-trace formula
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


@numba.jit(nopython=True)
def guan_code(n):
    """
    generator for a Guan code sequence

    Args:
        n (array[int]): maximum value for each digit in the Guan code
    Yields:
        j_new (int): index of the changed digit
        j_pm (int): +1 / -1 depending on whether the index increases or decreases
    """
    n = n[::-1]  # maximum for each digit
    K = len(n)
    g = np.zeros(K + 1, dtype=np.uint8)  # the guan code
    u = np.ones(K + 1, dtype=np.int8)
    while g[K] == 0:
        # enumerate next gray code
        i = 0
        k = g[0] + u[0]
        while k > n[i] or k < 0:
            u[i] = -u[i]
            i += 1
            k = g[i] + u[i]
            if i >= K:
                break
        g[i] = k

        if i < K:
            yield K - i - 1, u[i], g[-2::-1]


@numba.jit(nopython=True, cache=True)
def _dist_prob_gray(pattern, covs, M, hbar=2):
    r"""
    probability for distinguishable squeezing GBS.

    Uses some Gray/Guan code tricks to calculate eigenvalues at each step in O(N) time for each squeezer

    algorithm created by Jake Bulmer and Nicolas Quesada (currently unpublished)

    Args:
        pattern (array[int]): photon number outcome
        covs (array[:,:,:]): array of covariance matrices, one for each squeezer
        M (int): number of modes
        hbar (float): the value of hbar (default 2)
    """
    # pylint: disable=too-many-statements

    N = 2 * pattern.sum()

    D = len(covs)

    vac_prob = np.float64(1.0)

    AXs = []
    abs2_AXs = []
    assert len(pattern) == M
    for cov in covs:
        assert cov.shape[0] == 2 * M
        Q = nb_Qmat(cov, hbar=hbar)
        O = np.identity(2 * M) - np.linalg.inv(Q)
        A = np.empty_like(O)
        A[:M, :] = O[M:, :].conj()
        A[M:, :] = O[:M, :].conj()
        AX = np.empty_like(O)
        AX[:, :M] = A[:, M:]
        AX[:, M:] = A[:, :M]
        # As.append(A)
        AXs.append(AX)

        abs2_AX = np.real(AX * AX.conj()).astype(np.float64)
        abs2_AXs.append(abs2_AX)

        vac_prob /= np.sqrt(np.linalg.det(Q).real)

    fac_prod = np.float64(1.0)
    for p in pattern:
        fac_prod = fac_prod * np.float64(fact[p])

    traces = np.zeros(D, dtype=np.float64)
    abs2_AX_sums = np.zeros(D, dtype=np.float64)

    prev_nonzero_rows = [np.int64(i) for i in range(0)]
    nonzero_rows = [np.int64(i) for i in range(0)]

    edges_sum = np.int64(0)

    H = np.float64(0.0)
    for j_new, j_pm, kept_edges in guan_code(pattern):
        edges_sum += j_pm

        if j_pm == 1:
            nonzero_rows.append(j_new)
            nonzero_rows.append(j_new + M)
        else:
            nonzero_rows.remove(j_new)
            nonzero_rows.remove(j_new + M)

        # this could be done a tiny bit faster using gray code tricks
        binom_prod = 1.0
        for i in range(M):
            binom_prod *= nb_binom(pattern[i], kept_edges[i])

        prefac = (-1) ** ((N // 2 - edges_sum) % 2) * binom_prod

        E = []
        for i, (AX, abs2_AX) in enumerate(zip(AXs, abs2_AXs)):

            if j_pm == 1:
                traces[i] += AX[j_new, j_new].real + AX[j_new + M, j_new + M].real

                # add new off-diagonal elements of submatrix
                sum_i = abs2_AX[j_new, j_new + M]
                for r in prev_nonzero_rows:
                    sum_i += abs2_AX[j_new, r] + abs2_AX[j_new + M, r]
                sum_i *= 2  # abs2_O is ortogonal so we add half the new elements then x2
                # add new diagonal elements of submatrix
                sum_i += abs2_AX[j_new, j_new] + abs2_AX[j_new + M, j_new + M]
                abs2_AX_sums[i] += sum_i
            else:
                traces[i] -= AX[j_new, j_new].real + AX[j_new + M, j_new + M].real

                sum_i = abs2_AX[j_new, j_new + M]
                for r in nonzero_rows:
                    sum_i += abs2_AX[j_new, r] + abs2_AX[j_new + M, r]
                sum_i *= 2
                sum_i += abs2_AX[j_new, j_new] + abs2_AX[j_new + M, j_new + M]
                abs2_AX_sums[i] -= sum_i

            # avoid errors by not allowing negative square root
            frob_norm = np.sqrt(max(abs2_AX_sums[i], 0))
            disc = np.sqrt(max(2 * frob_norm**2 - traces[i] ** 2, 0))
            x = 0.5 * (traces[i] + disc)
            y = 0.5 * (traces[i] - disc)

            E.extend([x, y])

        E = np.array(E)
        Hnew = prefac * f_all(E, N)[N // 2].real

        prev_nonzero_rows = nonzero_rows.copy()

        H += Hnew

    prob = H * vac_prob / fac_prod

    return prob


def _vac_prob(covs, M, hbar=2):
    """
    vacuum probability

    Args
        covs (list[array[:,:]]): each array is 2MKx2MK length array of means, for each seperable schmidt mode
        M (int): number of spatial modes
        hbar (float): the value of hbar (default 2)
    Returns:
        float: vacuum probability
    """

    vac_prob = 1.0

    for cov in covs:
        assert cov.shape[0] == 2 * M
        Q = Qmat(cov, hbar=hbar)

        vac_prob /= np.sqrt(np.linalg.det(Q).real)

    return vac_prob


def distinguishable_pnr_prob(pattern, rs, T):
    """
    probabilities for completely distinguishable GBS

    Args:
        pattern (array[int]): photon number outcome for each of the M modes
        rs (sequence[float]): squeezing parameter in each of the M modes
        T (array[complex:complex]): MxM linear optical transmission matrix (unitary if no loss)

    Returns:
        float: probability of pattern
    """

    M = T.shape[0]
    mu = np.zeros(2 * M)
    covs = []
    for i, r in enumerate(rs):
        if abs(r) > 0:
            rs_i = np.zeros(M)
            rs_i[i] = r
            cov = squeezing(2 * rs_i)
            mu, cov = passive_transformation(mu, cov, T)
            covs.append(cov)

    if sum(pattern) == 0:
        return _vac_prob(np.array(covs), M)

    return _dist_prob_gray(np.array(pattern), np.array(covs), M)
