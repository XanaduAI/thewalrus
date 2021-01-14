# Copyright 2020 Xanadu Quantum Technologies Inc.

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
New Sampling algorithms
=======================

.. currentmodule:: thewalrus.new_samples

This submodule provides access to algorithms to sample the photon
number distribution of Gaussian states


Hafnian sampling
----------------
"""
import numpy as np
from thewalrus.quantum import pure_state_amplitude, reduced_gaussian
from strawberryfields.decompositions import williamson


def invert_perm(p):
    s = np.empty(p.size, dtype=np.int32)
    for i in np.arange(p.size):
        s[p[i]] = i
    return s


def generate_hafnian_sample(
    cov1, mean=None, hbar=2, cutoff=6, max_photons=30
):  # pylint: disable=too-many-branches
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        mean (array): a :math:`2N`` ``np.float64`` vector of means representing the Gaussian
            state.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation.
        max_photons (int): specifies the maximum number of photons that can be counted.
        approx (bool): if ``True``, the approximate hafnian algorithm is used.
            Note that this can only be used for real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.

    Returns:
        np.array[int]: a photon number sample from the Gaussian states.
    """
    ### TODO: the code below assumed hbar=2, mean=0
    ### The total photon number cutoff provided by max_photons is not implemented
    n, m = cov1.shape
    assert n == m
    assert n % 2 == 0
    nmodes = n // 2
    # The next two lines permute randomly the ordering of the modes
    # in the covariance matrix
    perm = np.random.permutation(nmodes)
    eperm = np.concatenate([perm, nmodes + perm])
    cov = cov1[eperm][:, eperm]
    mu = np.zeros([n])
    sample = []
    for mode in range(nmodes):
        modes = list(range(mode + 1))
        mured, covred = reduced_gaussian(mu, cov, modes)
        d, S = williamson(covred)
        vnoise = S @ (d - np.identity(len(d))) @ S.T
        vpure = S @ S.T
        rand_dist = np.random.multivariate_normal(mured, vnoise)
        probs = np.zeros([cutoff])
        for i in range(cutoff):
            local_sample = sample
            pat = local_sample + [i]
            probs[i] = (
                np.abs(
                    pure_state_amplitude(
                        rand_dist, vpure, pat, check_purity=False, include_prefactor=False
                    )
                )
                ** 2
            )
        probs /= np.sum(probs)
        var = np.random.choice(np.arange(cutoff), p=probs)
        sample = sample + [var]
    # Since the covariance matrix had its mode permuted
    # we unpermute the sample with the inverse permutation.
    iperm = invert_perm(perm)
    return np.array(sample)[iperm]
