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
Functions for calculating the photon number distributions of various states.
"""

import numpy as np
from scipy.stats import nbinom
from scipy.special import binom, hyp2f1

from .conversions import Amat
from .gaussian_checks import is_pure_cov


def pure_state_distribution(cov, cutoff=50, hbar=2, padding_factor=2):
    r"""Calculates the total photon number distribution of a pure state
    with zero mean.

    Args:
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering
        cutoff (int): Fock cutoff
        tol (float): tolerance for determining if displacement is negligible
        hbar (float): the value of :math:`\hbar` in the commutation
        padding_factor (int): expanded size of the photon distribution to avoid accumulation of errors

    Returns:
        (array): Total photon number distribution
    """
    if is_pure_cov(cov):
        A = Amat(cov, hbar=hbar)
        (n, _) = A.shape
        N = n // 2
        B = A[0:N, 0:N]
        rs = np.arctanh(np.linalg.svd(B, compute_uv=False))
        return _convolve_squeezed_state_distribution(
            rs, cutoff=cutoff, padding_factor=padding_factor
        )[0:cutoff]
    raise ValueError("The Gaussian state is not pure")


def _squeezed_state_distribution(s, cutoff=50, N=1):
    """Generate the photon number distribution of :math:`N` identical single mode squeezed states.

    Args:
        s (float): squeezing parameter
        cutoff (int): Fock cutoff
        N (float): number of squeezed states

    Returns:
        (array): Photon number distribution
    """
    r = 0.5 * N
    q = 1.0 - np.tanh(s) ** 2
    N = cutoff // 2
    ps_tot = np.zeros(cutoff)
    if cutoff % 2 == 0:
        ps = nbinom.pmf(np.arange(N), p=q, n=r)
        ps_tot[0::2] = ps
    else:
        ps = nbinom.pmf(np.arange(N + 1), p=q, n=r)
        ps_tot[0:-1][0::2] = ps[0:-1]
        ps_tot[-1] = ps[-1]

    return ps_tot


def _convolve_squeezed_state_distribution(s, cutoff=50, padding_factor=2):
    """Generates the total photon number distribution of single mode squeezed states with different squeezing values.

    Args:
        s (array): array of squeezing parameters
        cutoff (int): Fock cutoff
        padding_factor (int): expanded size of the photon distribution to avoid accumulation of errors
    Returns:
        (array[int]): total photon number distribution
    """
    scale = padding_factor
    cutoff_sc = scale * cutoff
    ps = np.zeros(cutoff_sc)
    ps[0] = 1.0
    for s_val in s:
        ps = np.convolve(ps, _squeezed_state_distribution(s_val, cutoff_sc))[0:cutoff_sc]
    return ps


def total_photon_number_distribution(n, k, s, eta, pref=1.0):
    r"""Probability of observing a total of :math:`n` photons when :math:`k` identical
    single-mode squeezed vacua with squeezing parameter :math:`s` undergo loss by transmission :math:`\eta`.

    For the derivation see Appendix E of `'Quantum Computational Supremacy via High-Dimensional Gaussian Boson Sampling',
    Deshpande et al. <https://arxiv.org/abs/2102.12474>`_.


    Args:
        n (int): number of photons
        k (int): number of squeezed modes
        s (float): squeezing parameter
        eta (float): transmission parameter, between 0 and 1 inclusive
        pref (float): use to return the probability times ``pref**n``
    Returns:
        (float): probability of observing a total of ``n`` photons or the probability times ``pref ** n``.
    """
    if n % 2 == 0:
        peven = (
            binom(-1 + k / 2.0 + n / 2.0, n / 2.0)
            * hyp2f1(0.5 + n / 2.0, k / 2.0 + n / 2.0, 0.5, (1 - eta) ** 2 * np.tanh(s) ** 2)
            * (1 / np.cosh(s)) ** k
            * (pref * eta * np.tanh(s)) ** n
        )
        return peven

    podd = (
        (1 + n)
        * (1 - eta)
        * binom((-1 + k + n) / 2.0, (1 + n) / 2.0)
        * hyp2f1((2 + n) / 2.0, (1 + k + n) / 2.0, 1.5, (1 - eta) ** 2 * np.tanh(s) ** 2)
        * (1 / np.cosh(s)) ** k
        * np.tanh(s)
        * (pref * eta * np.tanh(s)) ** n
    )
    return podd


def characteristic_function(
    k, s, eta, mu, max_iter=10000, delta=1e-14, poly_corr=None
):  # pylint: disable=too-many-arguments
    r"""Calculates the expectation value of the characteristic function
    :math:`\langle n^m \exp(mu n) \rangle` where :math:`n` is the total photon number of :math:`k` identical
    single-mode squeezed vacua with squeezing parameter :math:`s` undergoing loss by
    transmission :math:`\eta`.

    Args:
        k (int): number of squeezed modes
        s (float): squeezing parameter
        eta (float): transmission parameter, between 0 and 1 inclusive
        mu (float): value at which to evaluate the characteristic function
        max_iter (int): maximum number of terms allowed in the sum
        delta (float): fractional change in the sum after which the sum is stopped
        poly_corr (int): give the value of the exponent :math:`m` of the polynomial correction

    Returns:
        (float): the expected value of the moment generation function
    """

    if poly_corr is None or poly_corr == 0:
        f = lambda x: 1
    else:
        f = lambda x: x**poly_corr

    if s == 0 or eta == 0:
        return f(0)

    pref = np.exp(mu)
    tot_sum = f(0) * total_photon_number_distribution(0, k, s, eta, pref=pref)
    converged = False

    i = 1
    prev_addend = tot_sum
    while converged is False and i < max_iter:
        old_tot_sum = tot_sum
        addend = f(i) * total_photon_number_distribution(i, k, s, eta, pref=pref)
        tot_sum += addend
        i += 1
        # Note that we check that the sum of the last *two* values does not change the net
        # sum much, this is because for eta=0 the distrobution does not have support over
        # the odd integers.
        ratio = (addend + prev_addend) / old_tot_sum
        if ratio < delta:
            converged = True
        prev_addend = addend
    return tot_sum
