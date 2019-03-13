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
================================

.. currentmodule:: hafnian.samples

This submodule provides access to algorithms to sample from the
hafnian or the torontonian of Gaussian quantum states.


Summary
-------

.. autosummary::
    generate_hafnian_sample
    hafnian_sample
    generate_torontonian_sample
    torontonian_sample


Code details
------------
"""
# pylint: disable=too-many-arguments
import numpy as np
from scipy.special import factorial as fac

from ._hafnian import hafnian, hafnian_approx, kron_reduced
from ._torontonian import tor
from .quantum import Amat, Qmat, Xmat, reduced_gaussian


def generate_hafnian_sample(cov, hbar=2, cutoff=6, approx=False, approx_samples=1e5):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation.
        approx (bool): if ``True``, the :func:`~.hafnian_approx` function is used
            to approximate the hafnian. Note that this can only be used for
            real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.

    Returns:
        np.array[int]: samples from the Hafnian of the Gaussian state.
    """
    N = len(cov)//2
    result = []
    prev_prob = 1.0
    nmodes = N
    mu = np.zeros(2*N)

    for k in range(nmodes):
        probs1 = np.zeros([cutoff+1], dtype=np.float64)
        kk = np.arange(k+1)
        _, V_red = reduced_gaussian(mu, cov, kk)
        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)

        for i in range(cutoff):
            indices = result+[i]
            ind2 = indices+indices

            factpref = np.prod(fac(indices))
            mat = kron_reduced(A, ind2)

            if approx:
                probs1[i] = hafnian_approx(mat.real, num_samples=approx_samples)/factpref
            else:
                probs1[i] = hafnian(mat).real/factpref

        probs1a = probs1/np.sqrt(np.linalg.det(Q).real)
        probs2 = probs1a/prev_prob
        probs3 = np.maximum(probs2, np.zeros_like(probs2)) # pylint: disable=assignment-from-no-return
        ssum = np.sum(probs3)

        if ssum < 1.0:
            probs3[-1] = 1.0-ssum

        # The following normalization of probabilities is needed when approx=True
        if approx:
            if ssum > 1.0:
                probs3 = probs3/ssum

        result.append(np.random.choice(a=range(len(probs3)), p=probs3))
        if result[-1] == cutoff:
            return -1

        prev_prob = probs1a[result[-1]]

    return result


def hafnian_sample(cov, samples=1, hbar=2, cutoff=5, approx=False, approx_samples=1e5):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        samples (int): the number of samples to return.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation.
        approx (bool): if ``True``, the :func:`~.hafnian_approx` function is used
            to approximate the hafnian. Note that this can only be used for
            real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")

    samples_array = []
    j = 0

    while j < samples:
        result = generate_hafnian_sample(cov, hbar=hbar, cutoff=cutoff, approx=approx, approx_samples=approx_samples)
        if result != -1:
            # if result == -1, then you never get anything beyond cutoff
            samples_array.append(result)
            j = j+1
    return np.vstack(samples_array)


def torontonian_sample(cov, samples=1):
    r"""Returns samples from the Torontonian of a Gaussian state

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.

    Returns:
        np.array[int]: samples from the Torontonian of the covariance matrix.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError("Covariance matrix must be a NumPy array.")

    matshape = cov.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Covariance matrix must be square.")

    if np.isnan(cov).any():
        raise ValueError("Covariance matrix must not contain NaNs.")

    samples_array = []
    for _ in range(samples):
        samples_array.append(generate_torontonian_sample(cov))

    return np.vstack(samples_array)


def generate_torontonian_sample(cov, hbar=2):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        np.array[int]: samples from the Hafnian of the Gaussian state.
    """
    result = []
    n1, n2 = cov.shape

    if n1 != n2:
        raise ValueError("Covariance matrix must be square.")

    nmodes = n1//2
    prev_prob = 1.0
    mu = np.zeros(n1)

    for k in range(nmodes):
        probs1 = np.zeros([2], dtype=np.float64)
        kk = np.arange(k+1)
        _, V_red = reduced_gaussian(mu, cov, kk)

        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)
        O = Xmat(k+1) @ A

        indices = result+[0]
        ind2 = indices+indices

        probs1[0] = tor(np.complex128(kron_reduced(O, ind2))).real

        indices = result+[1]
        ind2 = indices+indices
        probs1[1] = tor(np.complex128(kron_reduced(O, ind2))).real

        probs1a = probs1/np.sqrt(np.linalg.det(Q).real)
        probs2 = probs1a/prev_prob
        probs3 = np.maximum(probs2, np.zeros_like(probs2)) # pylint: disable=assignment-from-no-return

        probs3 /= np.sum(probs3)
        result.append(np.random.choice(a=range(len(probs3)), p=probs3))

        prev_prob = probs1a[result[-1]]

        if np.sum(result) >= 30:
            break

    return result
