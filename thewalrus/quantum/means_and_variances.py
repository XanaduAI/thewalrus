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

from scipy.special import factorial

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

    if j == k:
        mu, cov = reduced_gaussian(mu, cov, [j])
        term_1 = 0.5 * np.trace(cov) ** 2 - np.linalg.det(cov)
        term_2 = mu @ cov @ mu
        return ((term_1 + term_2) / hbar ** 2) - 0.25

    mu, cov = reduced_gaussian(mu, cov, [j, k])
    term_1 = cov[0, 1] ** 2 + cov[0, 3] ** 2 + cov[2, 1] ** 2 + cov[2, 3] ** 2
    term_2 = (
        cov[0, 1] * mu[0] * mu[1]
        + cov[2, 1] * mu[1] * mu[2]
        + cov[0, 3] * mu[0] * mu[3]
        + cov[2, 3] * mu[2] * mu[3]
    )

    return (term_1 + 2 * term_2) / (2 * hbar ** 2)



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
    return s_ordered_expectation(mu, cov, rpt, hbar, s=1)

def s_ordered_expectation(mu, cov, rpt, hbar=2, s=0):
    r"""Calculates the expectation value of the s-ordered product
    obtained by taking deirvatives of the characteristic function of a Gaussian states,
    Here, :math:`\text{rpt}=(n_0, n_1, \ldots, n_{N-1}, m_0, m_1, \ldots, m_{N-1})`.
    indicates how many derivatives are taken with respect to the complex argument and its
    conjugate.
    The values :math:`s=\{1,0,-1\}` correspond respectively to normal, symmetric and antinormal order.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        rpt (list): integers specifying the terms to calculate.
        hbar (float): value of hbar in the uncertainty relation.
        s (float): value setting the ordering it must be between -1 and 1.

    Returns:
        (float): expectation value of the normal ordered product of operators
    """

    # The following seven lines are written so that we remove from the calculation the
    # modes k that we don't care about. These modes have rpt[k] = rpt[k+M] = 0
    if np.allclose(rpt, 0):
        return 1.0
    M = len(cov) // 2
    modes = np.where(np.array(rpt[0:M]) + np.array(rpt[M : 2 * M]) != 0)[0]
    mu, cov = reduced_gaussian(mu, cov, list(modes))
    ind = list(modes) + list(modes + M)
    rpt = list(np.array(rpt)[np.array(ind)])

    alpha = complex_to_real_displacements(mu, hbar=hbar)
    n = len(cov)
    V = (Qmat(cov, hbar=hbar) - 0.5 * (s + 1) * np.identity(n)) @ Xmat(n // 2)
    A = reduction(V, rpt)
    if np.allclose(mu, 0):
        return hafnian(A)

    np.fill_diagonal(A, reduction(np.conj(alpha), rpt))
    return hafnian(A, loop=True)




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

def _coeff_normal_ordered(m, k):
    r"""Returns the coefficients giving the expansion of a photon number power in terms of normal ordered power of creation
    and annihilation operators.

    Args:
        m (int): power of the photon number operator, :math:`(a^\dagger a)^m `.
        k (int): power of the normal ordered term, :math:`a^{\dagger i} a^i`.

    Returns:
        (float): expansion coefficient
    """

    return sum(
        [
            (1 / (factorial(mu) * factorial(k - mu)))
            * ((-1) ** (k - mu) * (mu ** m))
            for mu in range(0, k + 1)
        ]
    )


def photon_number_moment(mu, cov, indices, hbar=2):
    r"""Calculates the expectation value of product of powers of photon number operators of a Gaussian state.
    The powers are specified by a dictionary with modes as keys and powers as values.

    The calculation is performed by first writing any power of the photon number as

    :math:`(a^\dagger a)^m = \sum_{k=1}^m c_k a^{\dagger k} a^k`

    where the coefficients :math:`c_i` are provided by the function `_coeff_normal_ordered`.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        indices (dictionary): specification of the different modes and their power of their photon number
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        float: the expectation value of the photon number powers.

    """
    N = len(cov) // 2
    list_indices = [indices[key] for key in indices]
    modes = list(indices)
    # Find the expansion coefficients of all the different powers
    expansion_coeff = [
        [_coeff_normal_ordered(indices[key], i) for i in range(1, 1 + indices[key])]
        for key in indices
    ]
    values = [list(range(i)) for i in list_indices]
    net_sum = 0.0
    # Construct the product of each possible term appearing in the normal ordered expansion
    for item in product(*values):
        rpt = [0] * N
        for i, key in enumerate(modes):
            rpt[key] = item[i] + 1
        rpt = rpt + rpt
        prod_coeff = np.prod([expansion_coeff[i][coeff] for i, coeff in enumerate(item)])
        net_sum += prod_coeff * s_ordered_expectation(mu, cov, rpt, s=1, hbar=hbar)
    return np.real_if_close(net_sum)

def partition(collection):
    """Generate all set partitions of a collection.

    Taken from: https://stackoverflow.com/a/30134039

    Args:
        collection (sequence): set to find partitions of

    Yields:
        list[list]: set partition of collection
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller

def _list_to_freq_dict(words):
    """Convert between a list which of "words" and a dictionary
    which shows how many times each word appears in word

    Args:
        words (list): list of words
    Returns:
        dict : how many times a word appears. key is word, value is multiplicity
    """
    return {i : words.count(i) for i in set(words)}

def photon_number_cumulant(mu, cov, modes, hbar=2):
    r"""Calculates the cumulant of the modes in the Gaussian state.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering.
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering.
        modes (list or array): list of modes. Note that it can have repetitions.
        hbar (float): value of hbar in the uncertainty relation.

    Returns:
        (float): the cumulant
    """

    modes = list(modes) # turns modes from array to list if passed in as array

    kappa = 0
    for pi in partition(modes):
        size = len(pi)
        term = factorial(size - 1) * (-1) ** (size - 1)
        for B in pi:
            indices = _list_to_freq_dict(B)
            term *= photon_number_moment(mu, cov, indices, hbar=hbar)
        kappa += term

    return kappa
