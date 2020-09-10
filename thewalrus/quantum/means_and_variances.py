# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Functions for constructing/calculating the means, variances and covariances of
Gaussian states.
"""

from itertools import product

import numpy as np

from .._hafnian import hafnian, reduction

from .conversions import (
    reduced_gaussian,
    Qmat,
    Xmat,
    complex_to_real_displacements
)


def photon_number_mean(mu, cov, j, hbar=2):
    r""" Calculate the mean photon number of mode j of a Gaussian state.

    Args:
        mu (array): vector of means of the Gaussian state using the ordering
            :math:`[q_1, q_2, \dots, q_n, p_1, p_2, \dots, p_n]`
        cov (array): the covariance matrix of the Gaussian state
        j (int): the j :sup:`th` mode
        hbar (float): the ``hbar`` convention used in the commutation
            relation :math:`[q, p]=i\hbar`

    Returns:
        float: the mean photon number in mode :math:`j`.
    """
    num_modes = len(mu) // 2
    return (
        mu[j] ** 2
        + mu[j + num_modes] ** 2
        + cov[j, j]
        + cov[j + num_modes, j + num_modes]
        - hbar
    ) / (2 * hbar)


def photon_number_mean_vector(mu, cov, hbar=2):
    r""" Calculate the mean photon number of each of the modes in a Gaussian state

    Args:
        mu (array): vector of means of the Gaussian state using the ordering
            :math:`[q_1, q_2, \dots, q_n, p_1, p_2, \dots, p_n]`
        cov (array): the covariance matrix of the Gaussian state
        hbar (float): the ``hbar`` convention used in the commutation
            relation :math:`[q, p]=i\hbar`

    Returns:
        array: the vector of means of the photon number distribution
    """

    N = len(mu) // 2
    return np.array([photon_number_mean(mu, cov, j, hbar=hbar) for j in range(N)])


def photon_number_covar(mu, cov, j, k, hbar=2):
    r""" Calculate the variance/covariance of the photon number distribution
    of a Gaussian state.

    Implements the covariance matrix of the photon number distribution of a
    Gaussian state according to the Last two eq. of Part II. in
    `'Multidimensional Hermite polynomials and photon distribution for polymode
    mixed light', Dodonov et al. <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.813>`_

    .. math::
        \sigma_{n_j n_j} &= \frac{1}{2}\left(T_j^2 - 2d_j - \frac{1}{2}\right)
        + \left<\mathbf{Q}_j\right>\mathcal{M}_j\left<\mathbf{Q}_j\right>, \\
        \sigma_{n_j n_k} &= \frac{1}{2}\mathrm{Tr}\left(\Lambda_j \mathbf{M} \Lambda_k \mathbf{M}\right)
        + \frac{1}{2}\left<\mathbf{Q}\right>\Lambda_j \mathbf{M} \Lambda_k\left<\mathbf{Q}\right>,

    where :math:`T_j` and :math:`d_j` are the trace and the determinant of
    :math:`2 \times 2` matrix :math:`\mathcal{M}_j` whose elements coincide
    with the nonzero elements of matrix :math:`\mathbf{M}_j = \Lambda_j \mathbf{M} \Lambda_k`
    while the two-vector :math:`\mathbf{Q}_j` has the components :math:`(q_j, p_j)`.
    :math:`2N \times 2N` projector matrix :math:`\Lambda_j` has only two nonzero
    elements: :math:`\left(\Lambda_j\right)_{jj} = \left(\Lambda_j\right)_{j+N,j+N} = 1`.
    Note that the convention for ``mu`` used here differs from the one used in Dodonov et al.,
    They both provide the same results in this particular case.

    Args:
        mu (array): vector of means of the Gaussian state using the ordering
            :math:`[q_1, q_2, \dots, q_n, p_1, p_2, \dots, p_n]`
        cov (array): the covariance matrix of the Gaussian state
        j (int): the j :sup:`th` mode
        k (int): the k :sup:`th` mode
        hbar (float): the ``hbar`` convention used in the commutation
            relation :math:`[q, p]=i\hbar`

    Returns:
        float: the covariance for the photon numbers at modes :math:`j` and  :math:`k`.
    """
    # renormalise the covariance matrix
    cov = cov / hbar

    N = len(mu) // 2
    mu = np.array(mu) / np.sqrt(hbar)

    lambda_1 = np.zeros((2 * N, 2 * N))
    lambda_1[j, j] = lambda_1[j + N, j + N] = 1

    lambda_2 = np.zeros((2 * N, 2 * N))
    lambda_2[k, k] = lambda_2[k + N, k + N] = 1

    if j == k:
        idxs = ((j, j, j + N, j + N), (j, j + N, j, j + N))
        M = (lambda_1 @ cov @ lambda_2)[idxs].reshape(2, 2)

        term_1 = (np.trace(M) ** 2 - 2 * np.linalg.det(M) - 0.5) / 2
        term_2 = mu[[j, j + N]] @ M @ mu[[j, j + N]]
    else:
        term_1 = np.trace(lambda_1 @ cov @ lambda_2 @ cov) / 2
        term_2 = (mu @ lambda_1 @ cov @ lambda_2 @ mu) / 2

    return term_1 + term_2


def photon_number_covmat(mu, cov, hbar=2):
    r""" Calculate the covariance matrix of the photon number distribution of a
    Gaussian state.

    Args:
        mu (array): vector of means of the Gaussian state using the ordering
            :math:`[q_1, q_2, \dots, q_n, p_1, p_2, \dots, p_n]`
        cov (array): the covariance matrix of the Gaussian state
        hbar (float): the ``hbar`` convention used in the commutation
            relation :math:`[q, p]=i\hbar`

    Returns:
        array: the covariance matrix of the photon number distribution
    """
    N = len(mu) // 2
    pnd_cov = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1):
            pnd_cov[i][j] = photon_number_covar(mu, cov, i, j, hbar=hbar)
            pnd_cov[j][i] = pnd_cov[i][j]
    return pnd_cov


def photon_number_expectation(mu, cov, modes, hbar=2):
    r"""Calculates the expectation value of the product of the number operator of the modes in a Gaussian state.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        modes (list): list of modes
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        (float): expectation value of the product of the number operators of the modes.
    """
    n, _ = cov.shape
    n_modes = n // 2
    rpt = np.zeros([n], dtype=int)
    for i in modes:
        rpt[i] = 1
        rpt[i + n_modes] = 1

    return normal_ordered_expectation(mu, cov, rpt, hbar=hbar)


def photon_number_squared_expectation(mu, cov, modes, hbar=2):
    r"""Calculates the expectation value of the square of the product of the number operator of the modes in
    a Gaussian state.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        modes (list): list of modes
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        (float): expectation value of the square of the product of the number operator of the modes.
    """
    n_modes = len(modes)

    mu_red, cov_red = reduced_gaussian(mu, cov, modes)
    result = 0
    for item in product([1, 2], repeat=n_modes):
        rpt = item + item
        term = normal_ordered_expectation(mu_red, cov_red, rpt, hbar=hbar)
        result += term
    return result


def normal_ordered_expectation(mu, cov, rpt, hbar=2):
    r"""Calculates the expectation value of the normal ordered product
    :math:`\prod_{i=0}^{N-1} a_i^{\dagger n_i} \prod_{j=0}^{N-1} a_j^{m_j}` with respect to an N-mode Gaussian state,
    where :math:`\text{rpt}=(n_0, n_1, \ldots, n_{N-1}, m_0, m_1, \ldots, m_{N-1})`.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        rpt (list): integers specifying the terms to calculate.
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        (float): expectation value of the normal ordered product of operators
    """
    alpha = complex_to_real_displacements(mu, hbar=hbar)
    n = len(cov)
    V = (Qmat(cov, hbar=hbar) - np.identity(n)) @ Xmat(n // 2)
    A = reduction(V, rpt)
    if np.allclose(mu, 0):
        res = np.conj(hafnian(A))
    else:
        np.fill_diagonal(A, reduction(np.conj(alpha), rpt))
        res = np.conj(hafnian(A, loop=True))
    return np.conj(res)


def mean_clicks(cov, hbar=2):
    r""" Calculates the total mean number of clicks when a zero-mean gaussian state
    is measured using threshold detectors.

    Args
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`

    Returns
        float: mean number of clicks
    """
    n, _ = cov.shape
    nmodes = n // 2
    Q = Qmat(cov, hbar=hbar)
    meanc = 1.0 * nmodes

    for i in range(nmodes):
        det_val = np.real(Q[i, i] * Q[i + nmodes, i + nmodes] - Q[i + nmodes, i] * Q[i, i + nmodes])
        meanc -= 1.0 / np.sqrt(det_val)
    return meanc


def variance_clicks(cov, hbar=2):
    r""" Calculates the variance of the total number of clicks when a zero-mean gaussian state
    is measured using threshold detectors.

    Args
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`

    Returns
        float: variance in the total number of clicks
    """
    n, _ = cov.shape
    means = np.zeros([n])
    nmodes = n // 2
    Q = Qmat(cov, hbar=hbar)
    vac_probs = np.array(
        [
            np.real(Q[i, i] * Q[i + nmodes, i + nmodes] - Q[i + nmodes, i] * Q[i, i + nmodes])
            for i in range(nmodes)
        ]
    )
    vac_probs = np.sqrt(vac_probs)
    vac_probs = 1 / vac_probs
    term1 = np.sum(vac_probs * (1 - vac_probs))
    term2 = 0
    for i in range(nmodes):
        for j in range(i):
            _, Qij = reduced_gaussian(means, Q, [i, j])
            prob_vac_ij = np.linalg.det(Qij).real
            prob_vac_ij = 1.0 / np.sqrt(prob_vac_ij)
            term2 += prob_vac_ij - vac_probs[i] * vac_probs[j]

    return term1 + 2 * term2
