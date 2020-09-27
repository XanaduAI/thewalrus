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

from .conversions import Amat
from .gaussian_checks import is_pure_cov


def pure_state_distribution(cov, cutoff=50, hbar=2, padding_factor=2):
    r""" Calculates the total photon number distribution of a pure state
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
        return _convolve_squeezed_state_distribution(rs, cutoff=cutoff, padding_factor=padding_factor)[0:cutoff]
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
