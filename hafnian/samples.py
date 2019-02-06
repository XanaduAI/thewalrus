# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Hafnian and torontonian sampling
"""
# pylint: disable=too-many-arguments
import random

import numpy as np

from .quantum import Amat, Beta, Qmat, prefactor, density_matrix_element, reduced_gaussian
from .lib.libtor import torontonian_samples as libtor


def generate_hafnian_sample(cov, mu, hbar=2, cutoff=None, tol=1e-2):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation (optional). This overwrites
            ``tol`` if used.
        tol (float): determines dynamically the cutoff to use, by determining
            :math:`D` such that :math:`\sum_{i=0}^D\mathfrak{P}_i > 1-\epsilon`.

    Returns:
        np.array[int]: samples from the Hafnian of the Gaussian state.
    """
    N = len(cov)//2
    result = []
    result_prob = 1

    for k in [list(range(k)) for k in range(1, N+1)]:
        mu_red, V_red = reduced_gaussian(mu, cov, k)

        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)
        beta = Beta(mu_red, hbar=hbar)
        prefac = prefactor(beta, A, Q)/result_prob
        # print("k = ", k, "A = ", np.round(A, 4))

        if cutoff is not None:
            probs = np.zeros([cutoff+1], dtype=np.float64)

            for i in range(cutoff):
                event = result + [i]
                # print("Photon event:", event)
                res = density_matrix_element(beta, A, Q, event, event, include_prefactor=False)
                probs[i] = np.real_if_close(res)

            probs *= np.real_if_close(prefac)

        else:
            probs = []

            i = 0
            while True:
                event = result + [i]
                res = density_matrix_element(beta, A, Q, event, event, include_prefactor=False)
                probs.append(np.real_if_close(res*prefac))
                i += 1

                if sum(probs) > 1-tol:
                    break

            probs.append(0)
            probs = np.array(probs, dtype=np.float64)

        if np.any(probs > 1):
            probs /= np.max(probs)
        else:
            probs[-1] = 1-np.sum(probs)

        probs = np.round(probs, 10)
        # print("probs", probs)

        result.append(np.random.choice(a=range(len(probs)), p=probs))
        result_prob = probs[result[-1]]

    return result


def hafnian_sample(cov, mu=None, samples=1, hbar=2, cutoff=None, tol=1e-4):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        samples (int): the number of samples to return.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation (optional). This overwrites
            ``tol`` if used.
        tol (float): determines dynamically the cutoff to use, by determining
            :math:`D` such that :math:`\sum_{i=0}^D\mathfrak{P}_i > 1-\epsilon`.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape
    N = matshape[0]

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")

    mean = mu
    if mu is None:
        mean = np.zeros([N], dtype=np.float64)

    if len(mean) != N:
        raise ValueError("Means vector must be the same length as the covariance matrix.")

    samples_array = []
    for _ in range(samples):
        samples_array.append(generate_hafnian_sample(cov, mean, hbar=hbar, cutoff=cutoff, tol=tol))

    return np.vstack(samples_array)


def torontonian_sample(cov, mu=None, samples=1):
    r"""Returns samples from the Torontonian of a Gaussian state using the
    Fortran torontonian sampling library.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        mu (array): a length-:math:`2N` ``np.float64`` vector of means.
        samples (int): the number of samples to return.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape
    N = matshape[0]

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")

    if len(mu) != N:
        raise ValueError("Means vector must be the same length as the covariance matrix.")

    mean = mu
    if mu is None:
        mean = np.zeros([N], dtype=np.float64)

    samples_array = []
    for _ in range(samples):
        seed = random.randint(0, 10**6)
        samples_array.append(libtor.generatesample(covmat=cov, mean=mean, seed=seed, n_sample=N//2))

    return np.vstack(samples_array)
