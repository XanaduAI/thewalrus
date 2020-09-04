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
Gaussian states and photon number distributions of Gaussian states.
"""

from itertools import product

import numpy as np

from .covariance_matrices import normal_ordered_expectation

################################################################################
# Construct the reduced means and cov of a Gaussian state
################################################################################


def reduced_gaussian(mu, cov, modes):
    r""" Returns the vector of means and the covariance matrix of the specified modes.

    Args:
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        modes (int of Sequence[int]): indices of the requested modes

    Returns:
        tuple (means, cov): where means is an array containing the vector of means,
        and cov is a square array containing the covariance matrix.
    """
    N = len(mu) // 2

    # reduce rho down to specified subsystems
    if isinstance(modes, int):
        modes = [modes]

    if np.any(np.array(modes) > N):
        raise ValueError("Provided mode is larger than the number of subsystems.")

    if len(modes) == N:
        # reduced state is full state
        return mu, cov

    ind = np.concatenate([np.array(modes), np.array(modes) + N])
    rows = ind.reshape(-1, 1)
    cols = ind.reshape(1, -1)

    return mu[ind], cov[rows, cols]


################################################################################
# Calculate means or variances of photon number
################################################################################

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
