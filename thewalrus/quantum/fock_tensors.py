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
Set of functions for calculating various state representations, probabilities and
classical subsystems of Gaussian states.
"""
# pylint: disable=too-many-arguments

from itertools import count, product, chain

import numpy as np
import dask

from scipy.special import factorial as fac
from numba import jit

from ..symplectic import expand, is_symplectic, reduced_state
from ..libwalrus import interferometer, interferometer_real

from .._hafnian import hafnian, hafnian_repeated, reduction
from .._hermite_multidimensional import hermite_multidimensional, hafnian_batched

from .conversions import (
    Amat,
    Qmat,
    reduced_gaussian,
    complex_to_real_displacements,
)

from .gaussian_checks import (
    is_classical_cov,
    is_pure_cov,
    is_valid_cov
)


def pure_state_amplitude(mu, cov, i, include_prefactor=True, tol=1e-10, hbar=2, check_purity=True):
    r"""Returns the :math:`\langle i | \psi\rangle` element of the state ket
    of a Gaussian state defined by covariance matrix cov.


    Args:
        mu (array): length-:math:`2N` quadrature displacement vector
        cov (array): length-:math:`2N` covariance matrix
        i (list): list of amplitude elements
        include_prefactor (bool): if ``True``, the prefactor is automatically calculated
            used to scale the result.
        tol (float): tolerance for determining if displacement is negligible
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        check_purity (bool): if ``True``, the purity of the Gaussian state is checked
            before calculating the state vector.

    Returns:
        complex: the pure state amplitude
    """
    if check_purity:
        if not is_pure_cov(cov, hbar=hbar, rtol=1e-05, atol=1e-08):
            raise ValueError("The covariance matrix does not correspond to a pure state")

    rpt = i
    beta = complex_to_real_displacements(mu, hbar=hbar)
    Q = Qmat(cov, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    (n, _) = cov.shape
    N = n // 2
    B = A[0:N, 0:N].conj()
    alpha = beta[0:N]

    if np.linalg.norm(alpha) < tol:
        # no displacement
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            haf = hafnian(B_rpt)
        else:
            haf = hafnian_repeated(B, rpt)
    else:
        gamma = alpha - B @ np.conj(alpha)
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            np.fill_diagonal(B_rpt, reduction(gamma, rpt))
            haf = hafnian(B_rpt, loop=True)
        else:
            haf = hafnian_repeated(B, rpt, mu=gamma, loop=True)

    if include_prefactor:
        pref = np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - alpha @ B @ alpha))
        haf *= pref

    return haf / np.sqrt(np.prod(fac(rpt)) * np.sqrt(np.linalg.det(Q)))


def state_vector(
    mu, cov, post_select=None, normalize=False, cutoff=5, hbar=2, check_purity=True, **kwargs
):
    r"""Returns the state vector of a (PNR post-selected) Gaussian state.

    The resulting density matrix will have shape

    .. math:: \underbrace{D\times D \times \cdots \times D}_M

    where :math:`D` is the Fock space cutoff, and :math:`M` is the
    number of *non* post-selected modes, i.e. ``M = len(mu)//2 - len(post_select)``.

    If post_select is None then the density matrix elements are calculated using
    the multidimensional Hermite polynomials which provide a significantly faster
    evaluation.


    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering
        post_select (dict): dictionary containing the post-selected modes, of
            the form ``{mode: value}``.
        normalize (bool): If ``True``, a post-selected density matrix is re-normalized.
        cutoff (dim): the final length (i.e., Hilbert space dimension) of each
            mode in the density matrix.
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        check_purity (bool): if ``True``, the purity of the Gaussian state is checked
            before calculating the state vector.

    Keyword Args:
        choi_r (float or None): Value of the two-mode squeezing parameter used in Choi-Jamiolkoski
            trick in :func:`~.fock_tensor`. This keyword argument should only be used when ``state_vector``
            is called by :func:`~.fock_tensor`.

    Returns:
        np.array[complex]: the state vector of the Gaussian state
    """
    if check_purity:
        if not is_pure_cov(cov, hbar=hbar, rtol=1e-05, atol=1e-08):
            raise ValueError("The covariance matrix does not correspond to a pure state")

    beta = complex_to_real_displacements(mu, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    Q = Qmat(cov, hbar=hbar)

    (n, _) = cov.shape
    N = n // 2

    B = A[0:N, 0:N]
    alpha = beta[0:N]
    gamma = np.conj(alpha) - B @ alpha
    prefexp = -0.5 * (np.linalg.norm(alpha) ** 2 - alpha @ B @ alpha)
    pref = np.exp(prefexp.conj())
    if post_select is None:
        choi_r = kwargs.get("choi_r", None)
        if choi_r is None:
            denom = np.sqrt(np.sqrt(np.linalg.det(Q).real))
        else:
            rescaling = np.concatenate(
                [np.ones([N // 2]), (1.0 / np.tanh(choi_r)) * np.ones([N // 2])]
            )
            B = np.diag(rescaling) @ B @ np.diag(rescaling)
            gamma = rescaling * gamma
            denom = np.sqrt(np.sqrt(np.linalg.det(Q / np.cosh(choi_r)).real))

        psi = (
            pref
            * hafnian_batched(B.conj(), cutoff, mu=gamma.conj(), renorm=True)
            / denom
        )
    else:
        M = N - len(post_select)
        psi = np.zeros([cutoff] * (M), dtype=np.complex128)

        for idx in product(range(cutoff), repeat=M):
            el = []

            counter = count(0)
            modes = (np.arange(N)).tolist()
            el = [post_select[i] if i in post_select else idx[next(counter)] for i in modes]
            psi[idx] = pure_state_amplitude(
                mu, cov, el, check_purity=False, include_prefactor=False, hbar=hbar
            )

        psi = psi * pref

    if normalize:
        norm = np.sqrt(np.sum(np.abs(psi) ** 2))
        psi = psi / norm

    return psi


def density_matrix_element(mu, cov, i, j, include_prefactor=True, tol=1e-10, hbar=2):
    r"""Returns the :math:`\langle i | \rho | j \rangle` element of the density matrix
    of a Gaussian state defined by covariance matrix cov.

    Args:
        mu (array): length-:math:`2N` quadrature displacement vector
        cov (array): length-:math:`2N` covariance matrix
        i (list): list of density matrix rows
        j (list): list of density matrix columns
        include_prefactor (bool): if ``True``, the prefactor is automatically calculated
            used to scale the result.
        tol (float): tolerance for determining if displacement is negligible
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        complex: the density matrix element
    """
    rpt = i + j
    beta = complex_to_real_displacements(mu, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    if np.linalg.norm(beta) < tol:
        # no displacement
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            A_rpt = reduction(A, rpt)
            haf = hafnian(A_rpt)
        else:
            haf = hafnian_repeated(A, rpt)
    else:
        # replace the diagonal of A with gamma
        gamma = beta.conj() - A @ beta
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            A_rpt = reduction(A, rpt)
            np.fill_diagonal(A_rpt, reduction(gamma, rpt))
            haf = hafnian(A_rpt, loop=True)
        else:
            haf = hafnian_repeated(A, rpt, mu=gamma, loop=True)

    if include_prefactor:
        haf *= _prefactor(mu, cov, hbar=hbar)

    return haf / np.sqrt(np.prod(fac(rpt)))


def density_matrix(mu, cov, post_select=None, normalize=False, cutoff=5, hbar=2):
    r"""Returns the density matrix of a (PNR post-selected) Gaussian state.

    The resulting density matrix will have shape

    .. math:: \underbrace{D\times D \times \cdots \times D}_{2M}

    where :math:`D` is the Fock space cutoff, and :math:`M` is the
    number of *non* post-selected modes, i.e. ``M = len(mu)//2 - len(post_select)``.

    Note that we use the Strawberry Fields convention for indexing the density
    matrix; the first two dimensions correspond to subsystem 1, the second two
    dimensions correspond to subsystem 2, etc.
    If post_select is None then the density matrix elements are calculated using
    the multidimensional Hermite polynomials which provide a significantly faster
    evaluation.

    Args:
        mu (array): length-:math:`2N` means vector in xp-ordering
        cov (array): :math:`2N\times 2N` covariance matrix in xp-ordering
        post_select (dict): dictionary containing the post-selected modes, of
            the form ``{mode: value}``. If post_select is None the whole non post-selected density matrix
            is calculated directly using (multidimensional) Hermite polynomials, which is significantly faster
            than calculating one hafnian at a time.
        normalize (bool): If ``True``, a post-selected density matrix is re-normalized.
        cutoff (dim): the final length (i.e., Hilbert space dimension) of each
            mode in the density matrix.
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        np.array[complex]: the density matrix of the Gaussian state
    """
    N = len(mu) // 2
    pref = _prefactor(mu, cov, hbar=hbar)

    if post_select is None:
        A = Amat(cov, hbar=hbar).conj()
        sf_order = tuple(chain.from_iterable([[i, i + N] for i in range(N)]))

        if np.allclose(mu, np.zeros_like(mu)):
            tensor = pref * hermite_multidimensional(
                -A, cutoff, renorm=True, modified=True
            )
            return tensor.transpose(sf_order)
        beta = complex_to_real_displacements(mu, hbar=hbar)
        y = beta - A @ beta.conj()
        tensor = pref * hermite_multidimensional(
            -A, cutoff, y=y, renorm=True, modified=True
        )
        return tensor.transpose(sf_order)

    M = N - len(post_select)
    rho = np.zeros([cutoff] * (2 * M), dtype=np.complex128)

    for idx in product(range(cutoff), repeat=2 * M):
        el = []

        counter = count(0)
        modes = (np.arange(2 * N) % N).tolist()
        el = [post_select[i] if i in post_select else idx[next(counter)] for i in modes]

        el = np.array(el).reshape(2, -1)
        el0 = el[0].tolist()
        el1 = el[1].tolist()

        sf_idx = np.array(idx).reshape(2, -1)
        sf_el = tuple(sf_idx[::-1].T.flatten())

        rho[sf_el] = density_matrix_element(mu, cov, el0, el1, include_prefactor=False, hbar=hbar)

    rho *= pref

    if normalize:
        # construct the standard 2D density matrix, and take the trace
        new_ax = np.arange(2 * M).reshape([M, 2]).T.flatten()
        tr = np.trace(rho.transpose(new_ax).reshape([cutoff ** M, cutoff ** M])).real
        # renormalize
        rho /= tr

    return rho


def fock_tensor(
    S,
    alpha,
    cutoff,
    choi_r=np.arcsinh(1.0),
    check_symplectic=True,
    sf_order=False,
    rtol=1e-05,
    atol=1e-08,
):
    r"""
    Calculates the Fock representation of a Gaussian unitary parametrized by
    the symplectic matrix S and the displacements alpha up to cutoff in Fock space.

    Args:
        S (array): symplectic matrix
        alpha (array): complex vector of displacements
        cutoff (int): cutoff in Fock space
        choi_r (float): squeezing parameter used for the Choi expansion
        check_symplectic (boolean): checks whether the input matrix is symplectic
        sf_order (boolean): reshapes the tensor so that it follows the sf ordering of indices
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Return:
        (array): Tensor containing the Fock representation of the Gaussian unitary
    """
    # Check the matrix is symplectic
    if check_symplectic:
        if not is_symplectic(S, rtol=rtol, atol=atol):
            raise ValueError("The matrix S is not symplectic")

    # And that S and alpha have compatible dimensions
    m, _ = S.shape
    l = m // 2
    if l != len(alpha):
        raise ValueError(
            "The matrix S and the vector alpha do not have compatible dimensions"
        )
    # Check if S corresponds to an interferometer, if so use optimized routines
    if np.allclose(S @ S.T, np.identity(m), rtol=rtol, atol=atol) and np.allclose(
        alpha, 0, rtol=rtol, atol=atol
    ):
        reU = S[:l, :l]
        imU = S[:l, l:]
        if np.allclose(imU, 0, rtol=rtol, atol=atol):
            Ub = np.block([[0 * reU, -reU], [-reU.T, 0 * reU]])
            tensor = interferometer_real(Ub, cutoff)
        else:
            U = reU - 1j * imU
            Ub = np.block([[0 * U, -U], [-U.T, 0 * U]])
            tensor = interferometer(Ub, cutoff)
    else:
        # Construct the covariance matrix of l two-mode squeezed vacua pairing modes i and i+l
        ch = np.cosh(choi_r) * np.identity(l)
        sh = np.sinh(choi_r) * np.identity(l)
        zh = np.zeros([l, l])
        Schoi = np.block(
            [[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]]
        )
        # And then its Choi expanded symplectic
        S_exp = expand(S, list(range(l)), 2 * l) @ Schoi
        # And this is the corresponding covariance matrix
        cov = S_exp @ S_exp.T
        alphat = np.array(list(alpha) + ([0] * l))
        x = 2 * alphat.real
        p = 2 * alphat.imag
        mu = np.concatenate([x, p])

        tensor = state_vector(
            mu,
            cov,
            normalize=False,
            cutoff=cutoff,
            hbar=2,
            check_purity=False,
            choi_r=choi_r,
        )

    if sf_order:
        sf_indexing = tuple(chain.from_iterable([[i, i + l] for i in range(l)]))
        return tensor.transpose(sf_indexing)

    return tensor


def probabilities(mu, cov, cutoff, parallel=False, hbar=2.0, rtol=1e-05, atol=1e-08):
    r"""Generate the Fock space probabilities of a Gaussian state up to a Fock space cutoff.

    .. note::

        Individual density matrix elements are computed using multithreading by OpenMP.
        Setting ``parallel=True`` will further result in *multiple* density matrix elements
        being computed in parallel.

        When setting ``parallel=True``, OpenMP will need to be turned off by setting the
        environment variable ``OMP_NUM_THREADS=1`` (forcing single threaded use for individual
        matrix elements). Remove the environment variable or set it to ``OMP_NUM_THREADS=''``
        to again use multithreading with OpenMP.

    Args:
        mu (array): vector of means of length ``2*n_modes``
        cov (array): covariance matrix of shape ``[2*n_modes, 2*n_modes]``
        cutoff (int): cutoff in Fock space
        parallel (bool): if ``True``, uses ``dask`` for parallelization instead of OpenMP
        hbar (float): value of :math:`\hbar` in the commutation relation :math;`[\hat{x}, \hat{p}]=i\hbar`
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``

    Returns:
        (array): Fock space probabilities up to cutoff. The shape of this tensor is ``[cutoff]*num_modes``.
    """
    if is_pure_cov(cov, hbar=hbar, rtol=rtol, atol=atol):  # Check if the covariance matrix cov is pure
        return np.abs(state_vector(mu, cov, cutoff=cutoff, hbar=hbar, check_purity=False)) ** 2
    num_modes = len(mu) // 2

    if parallel:
        compute_list = []
        # create a list of parallelizable computations
        for i in product(range(cutoff), repeat=num_modes):
            compute_list.append(dask.delayed(density_matrix_element)(mu, cov, i, i, hbar=hbar))

        probs = np.maximum(
            0.0, np.real_if_close(dask.compute(*compute_list, scheduler="processes"))
        ).reshape([cutoff] * num_modes)
        # maximum is needed because sometimes a probability is very close to zero from below
    else:
        probs = np.zeros([cutoff] * num_modes)
        for i in product(range(cutoff), repeat=num_modes):
            probs[i] = np.maximum(
                0.0, np.real_if_close(density_matrix_element(mu, cov, i, i, hbar=hbar))
            )
            # maximum is needed because sometimes a probability is very close to zero from below
    return probs

@jit(nopython=True)
def loss_mat(eta, cutoff): # pragma: no cover
    r"""Constructs a binomial loss matrix with transmission eta up to n photons.

    Args:
        eta (float): Transmission coefficient. ``eta=0.0`` corresponds to complete loss and ``eta=1.0`` corresponds to no loss.
        cutoff (int): cutoff in Fock space.

    Returns:
        array: :math:`n\times n` matrix representing the loss.
    """
    # If full transmission return the identity

    if eta < 0.0 or eta > 1.0:
        raise ValueError("The transmission parameter eta should be a number between 0 and 1.")

    if eta == 1.0:
        return np.identity(cutoff)

    # Otherwise construct the matrix elements recursively
    lm = np.zeros((cutoff, cutoff))
    mu = 1.0 - eta
    lm[:, 0] = mu ** (np.arange(cutoff))
    for i in range(cutoff):
        for j in range(1, i + 1):
            lm[i, j] = lm[i, j - 1] * (eta / mu) * (i - j + 1) / (j)
    return lm

def update_probabilities_with_loss(etas, probs):
    """Given a list of transmissivities a tensor of probabilitites, calculate
    an updated tensor of probabilities after loss is applied.

    Args:
        etas (list): List of transmissitivities describing the loss in each of the modes
        probs (array): Array of probabilitites in the different modes

    Returns:
        array: List of loss-updated probabilities with the same shape as probs.
    """

    probs_shape = probs.shape
    if len(probs_shape) != len(etas):
        raise ValueError("The list of transmission etas and the tensor of probabilities probs have incompatible dimensions.")

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    cutoff = probs_shape[0]
    for i, eta in enumerate(etas):
        einstrings = "ij,{}i...->{}j...".format(alphabet[:i], alphabet[:i])

        qein = np.zeros_like(probs)
        qein = np.einsum(einstrings, loss_mat(eta, cutoff), probs)
        probs = np.copy(qein)
    return qein

@jit(nopython=True)
def _update_1d(probs, one_d, cutoff): # pragma: no cover
    """ Performs a convolution of the two arrays. The first one does not need to be one dimensional, which is why we do not use ``np.convolve``.

    Args:
        probs (array): (multidimensional) array
        one_d (array): one dimensional array
        cutoff (int): cutoff in Fock space for the first array

    Returns:
        (array): the convolution of the two arrays, with the same shape as ``probs``.
    """
    new_d = np.zeros_like(probs)
    for i in range(cutoff):
        for j in range(min(i + 1, len(one_d))):
            new_d[i] += probs[i - j] * one_d[j]
    return new_d

def update_probabilities_with_noise(probs_noise, probs):
    """Given a list of noise probability distributions for each of the modes and a tensor of
    probabilitites, calculate an updated tensor of probabilities after noise is applied.

    Args:
        probs_noise (list): List of probability distributions describing the noise in each of the modes
        probs (array): Array of probabilitites in the different modes

    Returns:
        array: List of noise-updated probabilities with the same shape as probs.
    """
    probs_shape = probs.shape
    num_modes = len(probs_shape)
    cutoff = probs_shape[0]
    if num_modes != len(probs_noise):
        raise ValueError(
            "The list of probability distributions probs_noise and the tensor of probabilities probs have incompatible dimensions."
        )

    for k in range(num_modes): #update one mode at a time
        perm = np.arange(num_modes)
        perm[0] = k
        perm[k] = 0
        one_d = probs_noise[k]
        probs_masked = np.transpose(probs, axes=perm)
        probs_masked = _update_1d(probs_masked, one_d, cutoff)
        probs = np.transpose(probs_masked, axes=perm)
    return probs


def find_classical_subsystem(cov, hbar=2, atol=1e-08):
    """Find the largest integer ``k`` so that subsystem in modes ``[0,1,...,k-1]`` is a classical state.


    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        atol (float): the absolute tolerance parameter used when determining if the state is classical

    Returns:
        int: the largest k so that modes ``[0,1,...,k-1]`` are in a classical state.
    """
    n, _ = cov.shape
    nmodes = n // 2
    if is_classical_cov(cov, hbar=hbar, atol=atol):
        return nmodes
    k = 0
    mu = np.zeros(n)
    is_classical = True
    while is_classical:
        _, Vk = reduced_gaussian(mu, cov, list(range(k + 1)))
        is_classical = is_classical_cov(Vk, hbar=hbar, atol=atol)
        k += 1
    return k - 1


def _prefactor(mu, cov, hbar=2):
    r"""Returns the prefactor.

    .. math:: prefactor = \frac{e^{-\beta Q^{-1}\beta^*/2}}{n_1!\cdots n_m! \sqrt{|Q|}}

    Args:
        mu (array): length-:math:`2N` vector of mean values :math:`[\alpha,\alpha^*]`
        cov (array): length-:math:`2N` `xp`-covariance matrix

    Returns:
        float: the prefactor
    """
    Q = Qmat(cov, hbar=hbar)
    beta = complex_to_real_displacements(mu, hbar=hbar)
    Qinv = np.linalg.inv(Q)
    return np.exp(-0.5 * beta @ Qinv @ beta.conj()) / np.sqrt(np.linalg.det(Q))

def tvd_cutoff_bounds(mu, cov, cutoff, hbar=2, check_is_valid_cov=True, rtol=1e-05, atol=1e-08):
    r""" Gives bounds of the total variation distance between the exact Gaussian Boson Sampling
    distribution extending to infinity in Fock space and the ones truncated by any value between 0
    and the user provided cutoff.

    For the derivation see Appendix B of `'Exact simulation of Gaussian boson sampling in polynomial space and exponential time',
    Quesada and Arrazola et al. <10.1103/PhysRevResearch.2.023005>`_.

    Args:
        mu (array): vector of means of the Gaussian state
        cov (array): covariance matrix of the Gaussian state
        cutoff (int): cutoff in Fock space
        check_is_valid_cov (bool): verify that the covariance matrix is physical
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (array): values of the bound for different local Fock space dimensions up to cutoff
    """
    if check_is_valid_cov:
        if not is_valid_cov(cov, hbar=hbar, rtol=rtol, atol=atol):
            raise ValueError("The input covariance matrix violates the uncertainty relation.")
    nmodes = cov.shape[0] // 2
    bounds = np.zeros([cutoff])
    for i in range(nmodes):
        mu_red, cov_red = reduced_state(mu, cov, [i])
        ps = np.real_if_close(np.diag(density_matrix(mu_red, cov_red, cutoff=cutoff, hbar=hbar)))
        bounds += 1 - np.cumsum(ps)
    return bounds
