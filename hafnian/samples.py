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
Sampling algorithms
===================

.. currentmodule:: hafnian.samples

This submodule provides access to algorithms to sample from the
hafnian or the torontonian of Gaussian quantum states.


Hafnian sampling
----------------

.. autosummary::
    generate_hafnian_sample
    hafnian_sample_state
    hafnian_sample_graph
    hafnian_sample_classical_state


Torontonian sampling
--------------------

.. autosummary::
    generate_torontonian_sample
    torontonian_sample_state
    torontonian_sample_graph
    torontonian_sample_classical_state

Code details
------------
"""
# pylint: disable=too-many-arguments
import multiprocessing
from multiprocessing import Pool

import numpy as np
from scipy.special import factorial as fac

from ._hafnian import hafnian, reduction
from ._torontonian import tor
from .quantum import (
    Amat,
    Covmat,
    Qmat,
    Xmat,
    gen_Qmat_from_graph,
    is_classical_cov,
    reduced_gaussian,
    density_matrix_element,
)

# ===============================================================================================
# Hafnian sampling
# ===============================================================================================


def generate_hafnian_sample(
    cov, mu=None, hbar=2, cutoff=6, max_photons=30, approx=False, approx_samples=1e5
):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
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
    N = len(cov) // 2
    result = []
    prev_prob = 1.0
    nmodes = N
    if mu is None:
        local_mu = np.zeros(2 * N)
    else:
        local_mu = mu
    A = Amat(Qmat(cov), hbar=hbar)

    for k in range(nmodes):
        probs1 = np.zeros([cutoff + 1], dtype=np.float64)
        kk = np.arange(k + 1)
        mu_red, V_red = reduced_gaussian(local_mu, cov, kk)

        if approx:
            Q = Qmat(V_red, hbar=hbar)
            A = Amat(Q, hbar=hbar, cov_is_qmat=True)

        for i in range(cutoff):
            indices = result + [i]
            ind2 = indices + indices
            if approx:
                factpref = np.prod(fac(indices))
                mat = reduction(A, ind2)
                probs1[i] = (
                    hafnian(np.abs(mat.real), approx=True, num_samples=approx_samples) / factpref
                )
            else:
                probs1[i] = density_matrix_element(mu_red, V_red, indices, indices, include_prefactor=True, hbar=hbar).real

        if approx:
            probs1 = probs1/np.sqrt(np.linalg.det(Q).real)

        #probs1a = probs1
        probs2 = probs1 / prev_prob
        probs3 = np.maximum(
            probs2, np.zeros_like(probs2)
        )  # pylint: disable=assignment-from-no-return
        ssum = np.sum(probs3)
        if ssum < 1.0:
            probs3[-1] = 1.0 - ssum

        # The following normalization of probabilities is needed when approx=True
        if approx:
            if ssum > 1.0:
                probs3 = probs3 / ssum

        result.append(np.random.choice(a=range(len(probs3)), p=probs3))
        if result[-1] == cutoff:
            return -1
        if sum(result) > max_photons:
            return -1
        prev_prob = probs1[result[-1]]

    return result


def _hafnian_sample(args):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Note: this is a wrapper function, instead of using this function
    directly, please use either :func:`torontonian_sample_state` or
    :func:`torontonian_sample_graph`.

    Args:
        args (list): a list containing the following parameters:

            cov (array)
                a :math:`2N\times 2N` ``np.float64`` covariance matrix
                representing an :math:`N` mode quantum state. This can be obtained
                via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.

            samples (int)
                the number of samples to return.

            hbar (float)
                the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.

            cutoff (int)
                the Fock basis truncation.

            max_photons (int)
                specifies the maximum number of photons that can be counted.

            approx (bool)
                if ``True``, the approximate hafnian algorithm is used.
                Note that this can only be used for real, non-negative matrices.

            approx_samples (int)
                the number of samples used to approximate the hafnian if ``approx=True``.

    Returns:
        np.array[int]: photon number samples from the Gaussian state
    """
    cov, samples, hbar, cutoff, max_photons, approx, approx_samples = args

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
        result = generate_hafnian_sample(
            cov,
            hbar=hbar,
            cutoff=cutoff,
            max_photons=max_photons,
            approx=approx,
            approx_samples=approx_samples,
        )

        if result != -1:
            # if result == -1, then you never get anything beyond cutoff
            samples_array.append(result)
            j = j + 1

    return np.vstack(samples_array)


def hafnian_sample_state(
    cov, samples, hbar=2, cutoff=5, max_photons=30, approx=False, approx_samples=1e5, pool=False
):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): the number of samples to return.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation.
        max_photons (int): specifies the maximum number of photons that can be counted.
        approx (bool): if ``True``, the :func:`~.hafnian_approx` function is used
            to approximate the hafnian. Note that this can only be used for
            real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.
        pool (bool): if ``True``, uses ``multiprocessor.Pool`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Gaussian state
    """
    if not pool:
        params = [cov, samples, hbar, cutoff, max_photons, approx, approx_samples]
        return _hafnian_sample(params)

    pool = Pool()
    nprocs = multiprocessing.cpu_count()
    localsamps = samples // nprocs

    params = [[cov, localsamps, hbar, cutoff, max_photons, approx, approx_samples]] * (nprocs - 1)
    params.append(
        [
            cov,
            samples - localsamps * (nprocs - 1),
            hbar,
            cutoff,
            max_photons,
            approx,
            approx_samples,
        ]
    )

    result = np.vstack(pool.map(_hafnian_sample, params))
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks

    return result


def hafnian_sample_graph(
    A, n_mean, samples=1, cutoff=5, max_photons=30, approx=False, approx_samples=1e5, pool=False
):
    r"""Returns samples from the Gaussian state specified by the adjacency matrix :math:`A`
    and with total mean photon number :math:`n_{mean}`

    Args:
        A (array): a :math:`N\times N` ``np.float64`` (symmetric) adjacency matrix matrix
        n_mean (float): mean photon number of the Gaussian state
        samples (int): the number of samples to return.
        cutoff (int): the Fock basis truncation.
        max_photons (int): specifies the maximum number of photons that can be counted.
        approx (bool): if ``True``, the approximate hafnian algorithm is used.
            Note that this can only be used for real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.
        pool (bool): if ``True``, uses ``multiprocessor.Pool`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Gaussian state
    """
    Q = gen_Qmat_from_graph(A, n_mean)
    cov = Covmat(Q)
    return hafnian_sample_state(
        cov,
        samples,
        hbar=2,
        cutoff=cutoff,
        max_photons=max_photons,
        approx=approx,
        approx_samples=approx_samples,
        pool=pool,
    )


# ===============================================================================================
# Torontonian sampling
# ===============================================================================================


def generate_torontonian_sample(cov, hbar=2, max_photons=30):
    r"""Returns a single sample from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        max_photons (int): specifies the maximum number of clicks that can be counted.

    Returns:
        np.array[int]: a threshold sample from the Gaussian state.
    """
    result = []
    n1, n2 = cov.shape

    if n1 != n2:
        raise ValueError("Covariance matrix must be square.")

    nmodes = n1 // 2
    prev_prob = 1.0
    mu = np.zeros(n1)

    for k in range(nmodes):
        probs1 = np.zeros([2], dtype=np.float64)
        kk = np.arange(k + 1)
        _, V_red = reduced_gaussian(mu, cov, kk)

        Q = Qmat(V_red, hbar=hbar)
        A = Amat(Q, hbar=hbar, cov_is_qmat=True)
        O = Xmat(k + 1) @ A

        indices = result + [0]
        ind2 = indices + indices

        probs1[0] = tor(np.complex128(reduction(O, ind2))).real

        indices = result + [1]
        ind2 = indices + indices
        pref = np.sqrt(np.linalg.det(Q).real)
        probs1a = probs1 / pref

        probs2 = probs1a / prev_prob
        probs2[1] = 1.0 - probs2[0]
        probs1a[1] = probs2[1] * prev_prob
        probs3 = np.maximum(
            probs2, np.zeros_like(probs2)
        )  # pylint: disable=assignment-from-no-return
        probs3 /= np.sum(probs3)
        result.append(np.random.choice(a=range(len(probs3)), p=probs3))

        prev_prob = probs1a[result[-1]]

        if np.sum(result) >= max_photons:
            return -1

    return result


def _torontonian_sample(args):
    r"""Returns samples from the Torontonian of a Gaussian state.

    Note: this is a wrapper function, instead of using this function
    directly, please use either :func:`torontonian_sample_state` or
    :func:`torontonian_sample_graph`.

    Args:
        args (list): a list containing the following parameters:

            cov (array)
                a :math:`2N\times 2N` ``np.float64`` covariance matrix
                representing an :math:`N` mode quantum state. This can be obtained
                via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.

            samples (int)
                number of samples to generate

            hbar (float)
                the value of :math:`\hbar` in the commutation
                relation :math:`[\x,\p]=i\hbar`.

            max_photons (int)
                specifies the maximum number of clicks that can be counted.

    Returns:
        np.array[int]:  threshold samples from the Gaussian state.
    """
    cov, samples, hbar, max_photons = args

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
        result = generate_torontonian_sample(cov, hbar=hbar, max_photons=max_photons)
        if result != -1:
            samples_array.append(result)
            j = j + 1

    return np.vstack(samples_array)


def torontonian_sample_state(cov, samples, hbar=2, max_photons=30, pool=False):
    r"""Returns samples from the Torontonian of a Gaussian state

    Args:
        cov(array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): number of samples to generate
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        max_photons (int): specifies the maximum number of clicks that can be counted.
        pool (boolean): if ``True``, uses ``multiprocessor.Pool`` for parallelization of samples

    Returns:
        np.array[int]:  threshold samples from the Gaussian state.
    """
    if not pool:
        params = [cov, samples, hbar, max_photons]
        return _torontonian_sample(params)

    pool = Pool()
    nprocs = multiprocessing.cpu_count()
    localsamps = samples // nprocs

    params = [[cov, localsamps, hbar, max_photons]] * (nprocs - 1)
    params.append([cov, samples - localsamps * (nprocs - 1), hbar, max_photons])

    result = np.vstack(pool.map(_torontonian_sample, params))
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks

    return result


def torontonian_sample_graph(A, n_mean, samples=1, max_photons=30, pool=False):
    r"""Returns samples from the Torontonian of a Gaussian state specified by the adjacency matrix :math:`A`
    and with total mean photon number :math:`n_{mean}`

    Args:
        A (array): a :math:`N\times N` ``np.float64`` (symmetric) adjacency matrix matrix
        n_mean (float): mean photon number of the Gaussian state
        samples (int): the number of samples to return.
        max_photons (int): specifies the maximum number of photons that can be counted.
        pool (boolean): if ``True``, uses ``multiprocessor.Pool`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Torontonian of the Gaussian state
    """
    Q = gen_Qmat_from_graph(A, n_mean)
    cov = Covmat(Q)
    return torontonian_sample_state(cov, samples, hbar=2, max_photons=max_photons, pool=pool)


def hafnian_sample_classical_state(cov, samples, mean=None, hbar=2, atol=1e-08):
    r"""Returns samples from a Gaussian state that has a positive :math:`P` function.

    Args:
        cov(array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): number of samples to generate
        mean (array): vector of means of the gaussian state
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        sigdigits (integer): precision to check that the covariance matrix is a true covariance matrix of a gaussian state.

    Returns:
        np.array[int]: photon number samples from the Gaussian state with covariance cov and vector means mean.
    """
    if not is_classical_cov(cov, hbar=hbar, atol=atol):
        raise ValueError("Not a classical covariance matrix")

    (n, _) = cov.shape
    if mean is None:
        mean = np.zeros([n])
    else:
        if mean.shape != (n,):
            raise ValueError("mean and cov do not have compatible shapes")

    R = np.random.multivariate_normal(mean, cov - 0.5 * hbar * np.identity(n), samples)
    N = n // 2
    alpha = (1.0 / np.sqrt(2 * hbar)) * (R[:, 0:N] + 1j * R[:, N : 2 * N])
    samples = np.random.poisson(np.abs(alpha) ** 2)
    return samples


def torontonian_sample_classical_state(cov, samples, mean=None, hbar=2, atol=1e-08):
    r""" Returns threshold samples from a Gaussian state that has a positive P function.

    Args:
        cov(array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): number of samples to generate
        mean (array): vector of means of the Gaussian state
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        sigdigits (integer): precision to check that the covariance matrix is a true covariance matrix of a gaussian state.

    Returns:
        np.array[int]: threshold samples from the Gaussian state with covariance cov and vector means mean.
    """
    return np.where(
        hafnian_sample_classical_state(cov, samples, mean=mean, hbar=hbar, atol=atol) > 0, 1, 0
    )
