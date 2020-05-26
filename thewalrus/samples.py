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

.. currentmodule:: thewalrus.samples

This submodule provides access to algorithms to sample from the
hafnian or the torontonian of Gaussian quantum states.


Hafnian sampling
----------------

.. autosummary::
    generate_hafnian_sample
    hafnian_sample_state
    hafnian_sample_graph
    hafnian_sample_classical_state
    hafnian_sample_graph_rank_one

Torontonian sampling
--------------------

.. autosummary::
    generate_torontonian_sample
    torontonian_sample_state
    torontonian_sample_graph
    torontonian_sample_classical_state

Brute force sampling
--------------------

.. autosummary::
    photon_number_sampler

Code details
------------
"""
# pylint: disable=too-many-arguments
import dask
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

# pylint: disable=too-many-branches
def generate_hafnian_sample(
    cov, mean=None, hbar=2, cutoff=6, max_photons=30, approx=False, approx_samples=1e5
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
    N = len(cov) // 2
    result = []
    prev_prob = 1.0
    nmodes = N
    if mean is None:
        local_mu = np.zeros(2 * N)
    else:
        local_mu = mean
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
                probs1[i] = density_matrix_element(
                    mu_red, V_red, indices, indices, include_prefactor=True, hbar=hbar
                ).real

        if approx:
            probs1 = probs1 / np.sqrt(np.linalg.det(Q).real)

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
        if np.sum(result) > max_photons:
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

            mean (array): a :math:`2N`` ``np.float64`` vector of means representing the Gaussian
                state.

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
    cov, samples, mean, hbar, cutoff, max_photons, approx, approx_samples = args

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
            mean=mean,
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
    cov,
    samples,
    mean=None,
    hbar=2,
    cutoff=5,
    max_photons=30,
    approx=False,
    approx_samples=1e5,
    parallel=False,
):
    r"""Returns samples from the Hafnian of a Gaussian state.

    Args:
        cov (array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): the number of samples to return.
        mean (array): a :math:`2N`` ``np.float64`` vector of means representing the Gaussian
                state.
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cutoff (int): the Fock basis truncation.
        max_photons (int): specifies the maximum number of photons that can be counted.
        approx (bool): if ``True``, the :func:`~.hafnian_approx` function is used
            to approximate the hafnian. Note that this can only be used for
            real, non-negative matrices.
        approx_samples: the number of samples used to approximate the hafnian if ``approx=True``.
        parallel (bool): if ``True``, uses ``dask`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Gaussian state
    """
    if parallel:
        params = [[cov, 1, mean, hbar, cutoff, max_photons, approx, approx_samples]] * samples
        compute_list = []
        for p in params:
            compute_list.append(dask.delayed(_hafnian_sample)(p))

        results = dask.compute(*compute_list, scheduler="threads")

        return np.vstack(results)

    params = [cov, samples, mean, hbar, cutoff, max_photons, approx, approx_samples]
    return _hafnian_sample(params)



def hafnian_sample_graph(
    A, n_mean, samples=1, cutoff=5, max_photons=30, approx=False, approx_samples=1e5, parallel=False
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
        parallel (bool): if ``True``, uses ``dask`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Gaussian state
    """
    Q = gen_Qmat_from_graph(A, n_mean)
    cov = Covmat(Q, hbar=2)
    return hafnian_sample_state(
        cov,
        samples,
        mean=None,
        hbar=2,
        cutoff=cutoff,
        max_photons=max_photons,
        approx=approx,
        approx_samples=approx_samples,
        parallel=parallel,
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

        if np.sum(result) > max_photons:
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


def torontonian_sample_state(cov, samples, hbar=2, max_photons=30, parallel=False):
    r"""Returns samples from the Torontonian of a Gaussian state

    Args:
        cov(array): a :math:`2N\times 2N` ``np.float64`` covariance matrix
            representing an :math:`N` mode quantum state. This can be obtained
            via the ``scovmavxp`` method of the Gaussian backend of Strawberry Fields.
        samples (int): number of samples to generate
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        max_photons (int): specifies the maximum number of clicks that can be counted.
        parallel (bool): if ``True``, uses ``dask`` for parallelization of samples

    Returns:
        np.array[int]:  threshold samples from the Gaussian state.
    """
    if parallel:
        params = [[cov, 1, hbar, max_photons]] * samples
        compute_list = []
        for p in params:
            compute_list.append(dask.delayed(_torontonian_sample)(p))

        results = dask.compute(*compute_list, scheduler="threads")

        return np.vstack(results)

    params = [cov, samples, hbar, max_photons]
    return _torontonian_sample(params)



def torontonian_sample_graph(A, n_mean, samples=1, max_photons=30, parallel=False):
    r"""Returns samples from the Torontonian of a Gaussian state specified by the adjacency matrix :math:`A`
    and with total mean photon number :math:`n_{mean}`

    Args:
        A (array): a :math:`N\times N` ``np.float64`` (symmetric) adjacency matrix matrix
        n_mean (float): mean photon number of the Gaussian state
        samples (int): the number of samples to return.
        max_photons (int): specifies the maximum number of photons that can be counted.
        parallel (bool): if ``True``, uses ``dask`` for parallelization of samples

    Returns:
        np.array[int]: photon number samples from the Torontonian of the Gaussian state
    """
    Q = gen_Qmat_from_graph(A, n_mean)
    cov = Covmat(Q, hbar=2)
    return torontonian_sample_state(cov, samples, hbar=2, max_photons=max_photons, parallel=parallel)


# pylint: disable=unused-argument
def hafnian_sample_classical_state(
    cov, samples, mean=None, hbar=2, atol=1e-08, cutoff=None
):  # add cutoff for consistency pylint: disable=unused-argument
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


def photon_number_sampler(probabilities, num_samples, out_of_bounds=False):
    """Given a photon-number probability mass function(PMF) it returns samples according to said PMF.

    Args:
        probabilities (array): probability tensor of the modes, has shape ``[cutoff]*num_modes``
        num_samples (int): number of samples requested
        out_of_bounds (boolean): if ``False`` the probability distribution is renormalized. If not ``False``, the value of
            ``out_of_bounds`` is used as a placeholder for samples where more than the cutoff of probabilities are detected.

    Returns:
        (array): Samples, with shape [num_sample, num_modes]
    """
    num_modes = len(probabilities.shape)
    cutoff = probabilities.shape[0]
    sum_p = np.sum(probabilities)

    if out_of_bounds is False:
        probabilities = probabilities.flatten() / sum_p
        vals = np.arange(cutoff**num_modes, dtype=int)
        return [
            np.unravel_index(np.random.choice(vals, p=probabilities), [cutoff] * num_modes)
            for _ in range(num_samples)
            ]

    upper_limit = cutoff ** num_modes

    def sorter(index):
        if index == upper_limit:
            return out_of_bounds

        return np.unravel_index(index, [cutoff] * num_modes)

    vals = np.arange(1 + cutoff**num_modes, dtype=int)
    probabilities = np.append(probabilities.flatten(), 1.0 - sum_p)
    return [sorter(np.random.choice(vals, p=probabilities)) for _ in range(num_samples)]


def seed(seed_val=None):
    r""" Seeds the random number generator used in the sampling algorithms.

    This function is a wrapper around ``numpy.random.seed()``. By setting the seed
    to a specific integer, the sampling algorithms will exhibit deterministic behaviour.

    Args:
        seed_val (int): Seed for RandomState. Must be convertible to 32 bit unsigned integers.
    """
    np.random.seed(seed_val)



def _hafnian_sample_graph_rank_one(G, n_mean):
    r"""Returns a sample from a rank one adjacency matrix `\bm{A} = \bm{G} \bm{G}^T` where :math:`\bm{G}`
    is a row vector.

    Args:
        G (array): factorization of the rank-one matrix A = G @ G.T.
        nmean (float): Total mean photon number.

    Returns:
        (array): sample.
    """
    s = np.arcsinh(np.sqrt(n_mean))
    q = 1.0 - np.tanh(s) ** 2
    total_photon_num = 2 * np.random.negative_binomial(0.5, q, 1)[0]
    sample = np.zeros(len(G))
    single_ph_ps = np.abs(G) ** 2
    single_ph_ps /= np.sum(single_ph_ps)
    for _ in range(total_photon_num):
        detector = np.random.choice(len(G), p=single_ph_ps)
        sample[detector] += 1
    return sample


def hafnian_sample_graph_rank_one(G, n_mean, samples=1):
    r"""Returns samples from a rank one adjacency matrix `\bm{A} = \bm{G} \bm{G}^T` where :math:`\bm{G}`
    is a row vector.

    Args:
        G (array): factorization of the rank-one matrix A = G @ G.T.
        nmean (float): Total mean photon number.
        samples (int): the number of samples to return.

    Returns
        (array): samples.
    """
    return np.array([_hafnian_sample_graph_rank_one(G, n_mean) for _ in range(samples)])
