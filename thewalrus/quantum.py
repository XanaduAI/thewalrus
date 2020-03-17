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
Quantum algorithms
==================

.. currentmodule:: thewalrus.quantum

This submodule provides access to various utility functions that act on Gaussian
quantum states.

For more details on how the hafnian relates to various properties of Gaussian quantum
states, see:

* Kruse, R., Hamilton, C. S., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "Detailed study of Gaussian boson sampling." `Physical Review A 100, 032326 (2019)
  <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.032326>`_

* Hamilton, C. S., Kruse, R., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "Gaussian boson sampling." `Physical Review Letters, 119(17), 170501 (2017)
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.170501>`_

* Quesada, N.
  "Franck-Condon factors by counting perfect matchings of graphs with loops."
  `Journal of Chemical Physics 150, 164113 (2019) <https://aip.scitation.org/doi/10.1063/1.5086387>`_

* Quesada, N., Helt, L. G., Izaac, J., Arrazola, J. M., Shahrokhshahi, R., Myers, C. R., & Sabapathy, K. K.
  "Simulating realistic non-Gaussian state preparation." `Physical Review A 100, 022341 (2019)
  <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.022341>`_


Fock states and tensors
-----------------------

.. autosummary::

    pure_state_amplitude
    state_vector
    density_matrix_element
    density_matrix
    fock_tensor
    generate_probabilities
    update_probabilities_with_loss

Details
^^^^^^^

.. autofunction::
    pure_state_amplitude

.. autofunction::
    state_vector

.. autofunction::
    density_matrix_element

.. autofunction::
    density_matrix

.. autofunction::
    fock_tensor

.. autofunction::
    generate_probabilities

.. autofunction::
    update_probabilities_with_loss

Utility functions
-----------------

.. autosummary::

    reduced_gaussian
    Xmat
    Qmat
    Covmat
    Amat
    Beta
    Means
    prefactor
    find_scaling_adjacency_matrix
    mean_number_of_clicks
    find_scaling_adjacency_matrix_torontonian
    gen_Qmat_from_graph
    photon_number_mean
    photon_number_mean_vector
    photon_number_covar
    photon_number_covmat
    is_valid_cov
    is_pure_cov
    is_classical_cov
    total_photon_num_dist_pure_state
    gen_single_mode_dist
    gen_multi_mode_dist


Details
^^^^^^^
"""
# pylint: disable=too-many-arguments
from itertools import count, product, chain

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import factorial as fac
from scipy.stats import nbinom
from numba import jit

from thewalrus.symplectic import expand, sympmat, is_symplectic

from ._hafnian import hafnian, hafnian_repeated, reduction
from ._hermite_multidimensional import hermite_multidimensional, hafnian_batched


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


def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    return X


def Qmat(cov, hbar=2):
    r"""Returns the :math:`Q` Husimi matrix of the Gaussian state.

    Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the :math:`Q` matrix.
    """
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q


def Covmat(Q, hbar=2):
    r"""Returns the Wigner covariance matrix in the :math:`xp`-ordering of the Gaussian state.
    This is the inverse function of Qmat.

    Args:
        Q (array): :math:`2N\times 2N` Husimi Q matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the :math:`xp`-ordered covariance matrix in the xp-ordering.
    """
    # number of modes
    n = len(Q) // 2
    I = np.identity(n)
    N = Q[0:n, 0:n] - I
    M = Q[n : 2 * n, 0:n]
    mm11a = 2 * (N.real + M.real) + np.identity(n)
    mm22a = 2 * (N.real - M.real) + np.identity(n)
    mm12a = 2 * (M.imag + N.imag)
    cov = np.block([[mm11a, mm12a], [mm12a.T, mm22a]])

    return (hbar / 2) * cov


def Amat(cov, hbar=2, cov_is_qmat=False):
    r"""Returns the :math:`A` matrix of the Gaussian state whose hafnian gives the photon number probabilities.

    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cov_is_qmat (bool): if ``True``, it is assumed that ``cov`` is in fact the Q matrix.

    Returns:
        array: the :math:`A` matrix.
    """
    # number of modes
    N = len(cov) // 2
    X = Xmat(N)

    # inverse Q matrix
    if cov_is_qmat:
        Q = cov
    else:
        Q = Qmat(cov, hbar=hbar)

    Qinv = np.linalg.inv(Q)

    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    return A


def Beta(mu, hbar=2):
    r"""Returns the vector of complex displacements and conjugate displacements.

    Args:
        mu (array): length-:math:`2N` means vector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the expectation values
        :math:`[\langle a_1\rangle, \langle a_2\rangle,\dots,\langle a_N\rangle, \langle a^\dagger_1\rangle, \dots, \langle a^\dagger_N\rangle]`
    """
    N = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:N] + 1j * mu[N:]) / np.sqrt(2 * hbar)
    # the expectation values (<a_1>, <a_2>,...,<a_N>, <a^\dagger_1>, ..., <a^\dagger_N>)
    return np.concatenate([alpha, alpha.conj()])


def Means(beta, hbar=2):
    r"""Returns the vector of real quadrature displacements.

    Args:
        beta (array): length-:math:`2N` means bivector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.

    Returns:
        array: the quadrature expectation values
        :math:`[\langle q_1\rangle, \langle q_2\rangle,\dots,\langle q_N\rangle, \langle p_1\rangle, \dots, \langle p_N\rangle]`
    """

    N = len(beta) // 2
    alpha = beta[0:N]
    return np.sqrt(2 * hbar) * np.concatenate([alpha.real, alpha.imag])


def prefactor(mu, cov, hbar=2):
    r"""Returns the prefactor.

    .. math:: prefactor = \frac{e^{-\beta Q^{-1}\beta^*/2}}{n_1!\cdots n_m! \sqrt{|Q|}}

    Args:
        mu (array): length-:math:`2N` vector of mean values :math:`[\alpha,\alpha^*]`
        cov (array): length-:math:`2N` `xp`-covariance matrix

    Returns:
        float: the prefactor
    """
    Q = Qmat(cov, hbar=hbar)
    beta = Beta(mu, hbar=hbar)
    Qinv = np.linalg.inv(Q)
    return np.exp(-0.5 * beta @ Qinv @ beta.conj()) / np.sqrt(np.linalg.det(Q))


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
    beta = Beta(mu, hbar=hbar)
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
        haf *= prefactor(mu, cov, hbar=hbar)

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
    pref = prefactor(mu, cov, hbar=hbar)

    if post_select is None:
        A = Amat(cov, hbar=hbar).conj()
        sf_order = tuple(chain.from_iterable([[i, i + N] for i in range(N)]))

        if np.allclose(mu, np.zeros_like(mu)):
            tensor = np.real_if_close(pref) * hermite_multidimensional(
                -A, cutoff, renorm=True, modified=True
            )
            return tensor.transpose(sf_order)
        beta = Beta(mu, hbar=hbar)
        y = beta - A @ beta.conj()
        tensor = np.real_if_close(pref) * hermite_multidimensional(
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
    beta = Beta(mu, hbar=hbar)
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

    beta = Beta(mu, hbar=hbar)
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
            np.real_if_close(pref)
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


def mean_number_of_clicks(A):
    r""" Given an adjacency matrix this function calculates the mean number of clicks.
    For this to make sense the user must provide a matrix with singular values
    less than or equal to one. See Appendix A.3 of <https://arxiv.org/abs/1902.00462>`_
    by Banchi et al.

    Args:
        A (array): rescaled adjacency matrix

    Returns:
        float: mean number of clicks
    """
    n, _ = A.shape
    idn = np.identity(n)
    X = np.block([[0 * idn, idn], [idn, 0 * idn]])
    B = np.block([[A, 0 * A], [0 * A, np.conj(A)]])
    Q = np.linalg.inv(np.identity(2 * n) - X @ B)
    meanc = 1.0 * n

    for i in range(n):
        det_val = np.real(Q[i, i]*Q[i+n, i+n] - Q[i+n, i]*Q[i, i+n])
        meanc -= 1.0 / np.sqrt(det_val)
    return meanc


def find_scaling_adjacency_matrix_torontonian(A, c_mean):
    r""" Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that encodes it has
    give a mean number of clicks equal to ``c_mean`` when measured with
    threshold detectors.

    Args:
        A (array): adjacency matrix
        c_mean (float): mean photon number of the Gaussian state

    Returns:
        float: scaling parameter
    """
    n, _ = A.shape
    if c_mean < 0 or c_mean > n:
        raise ValueError("The mean number of clicks should be smaller than the number of modes")

    vals = np.linalg.svd(A, compute_uv=False)
    localA = A / vals[0]  # rescale the matrix so that the singular values are between 0 and 1.

    def cost(x):
        r""" Cost function giving the difference between the wanted number of clicks and the number
        of clicks at a given scaling value. It assumes that the adjacency matrix has been rescaled
        so that it has singular values between 0 and 1.

        Args:
            x (float): scaling value

        Return:
            float: difference between desired and obtained mean number of clicks
        """
        if x >= 1.0:
            return c_mean - n
        if x <= 0:
            return c_mean
        return c_mean - mean_number_of_clicks(x * localA)

    res = root_scalar(cost, x0=0.5, bracket=(0.0, 1.0))  # Do the optimization

    if not res.converged:
        raise ValueError("The search for a scaling value failed")
    return res.root / vals[0]


def find_scaling_adjacency_matrix(A, n_mean):
    r""" Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that endodes it has
    a total mean photon number n_mean.

    Args:
        A (array): Adjacency matrix
        n_mean (float): Mean photon number of the Gaussian state

    Returns:
        float: Scaling parameter
    """
    eps = 1e-10
    ls = np.linalg.svd(A, compute_uv=False)
    max_sv = ls[0]
    a_lim = 0.0
    b_lim = 1.0 / (eps + max_sv)
    x_init = 0.5 * b_lim

    if 1000 * eps >= max_sv:
        raise ValueError("The singular values of the matrix A are too small.")

    def mean_photon_number(x, vals):
        r""" Returns the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A where vals are the singular values of A

        Args:
            x (float): Scaling parameter
            vals (array): Singular values of the matrix A

        Returns:
            n_mean: Mean photon number in the Gaussian state
        """
        vals2 = (x * vals) ** 2
        n = np.sum(vals2 / (1.0 - vals2))
        return n

    # The following function is implicitly tested in test_find_scaling_adjacency_matrix
    def grad_mean_photon_number(x, vals):  # pragma: no cover
        r""" Returns the gradient od the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A with respect to x.
        vals are the singular values of A

        Args:
            x (float): Scaling parameter
            vals (array): Singular values of the matrix A

        Returns:
            d_n_mean: Derivative of the mean photon number in the Gaussian state
                with respect to x
        """
        vals1 = vals * x
        dn = (2.0 / x) * np.sum((vals1 / (1 - vals1 ** 2)) ** 2)
        return dn

    f = lambda x: mean_photon_number(x, ls) - n_mean
    df = lambda x: grad_mean_photon_number(x, ls)
    res = root_scalar(f, fprime=df, x0=x_init, bracket=(a_lim, b_lim))

    if not res.converged:
        raise ValueError("The search for a scaling value failed")

    return res.root


def gen_Qmat_from_graph(A, n_mean):
    r""" Returns the Qmat xp-covariance matrix associated to a graph with
    adjacency matrix :math:`A` and with mean photon number :math:`n_{mean}`.

    Args:
        A (array): a :math:`N\times N` ``np.float64`` (symmetric) adjacency matrix
        n_mean (float): mean photon number of the Gaussian state

    Returns:
        array: the :math:`2N\times 2N` Q matrix.
    """
    n, m = A.shape

    if n != m:
        raise ValueError("Matrix must be square.")

    sc = find_scaling_adjacency_matrix(A, n_mean)
    Asc = sc * A
    A = np.block([[Asc, 0 * Asc], [0 * Asc, Asc.conj()]])
    I = np.identity(2 * n)
    X = Xmat(n)
    Q = np.linalg.inv(I - X @ A)
    return Q

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

def is_valid_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
    r""" Checks if the covariance matrix is a valid quantum covariance matrix.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation

    Returns:
        (bool): whether the given covariance matrix is a valid covariance matrix
    """
    (n, m) = cov.shape
    if n != m:
        # raise ValueError("The input matrix must be square")
        return False
    if not np.allclose(cov, np.transpose(cov), rtol=rtol, atol=atol):
        # raise ValueError("The input matrix is not symmetric")
        return False
    if n % 2 != 0:
        # raise ValueError("The input matrix is of even dimension")
        return False

    nmodes = n // 2
    vals = np.linalg.eigvalsh(cov + 0.5j * hbar * sympmat(nmodes))
    vals[np.abs(vals) < atol] = 0.0
    if np.all(vals >= 0):
        # raise ValueError("The input matrix violates the uncertainty relation")
        return True

    return False


def is_pure_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
    r""" Checks if the covariance matrix is a valid quantum covariance matrix
    that corresponds to a quantum pure state

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation

    Returns:
        (bool): whether the given covariance matrix corresponds to a pure state
    """
    if is_valid_cov(cov, hbar=hbar, rtol=rtol, atol=atol):
        purity = 1 / np.sqrt(np.linalg.det(2 * cov / hbar))
        if np.allclose(purity, 1.0, rtol=rtol, atol=atol):
            return True

    return False


def is_classical_cov(cov, hbar=2, atol=1e-08):
    r""" Checks if the covariance matrix can be efficiently sampled.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation

    Returns:
        (bool): whether the given covariance matrix corresponds to a classical state
    """

    if is_valid_cov(cov, hbar=hbar, atol=atol):
        (n, _) = cov.shape
        vals = np.linalg.eigvalsh(cov - 0.5 * hbar * np.identity(n))
        vals[np.abs(vals) < atol] = 0.0

        if np.all(vals >= 0):
            return True
    return False


def gen_single_mode_dist(s, cutoff=50, N=1):
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


def gen_multi_mode_dist(s, cutoff=50, padding_factor=2):
    """Generates the total photon number distribution of single mode squeezed states with different squeezing values.

    Args:
        s (array): array of squeezing parameters
        cutoff (int): Fock cutoff

    Returns:
        (array[int]): total photon number distribution
    """
    scale = padding_factor
    cutoff_sc = scale * cutoff
    ps = np.zeros(cutoff_sc)
    ps[0] = 1.0
    for s_val in s:
        ps = np.convolve(ps, gen_single_mode_dist(s_val, cutoff_sc))[0:cutoff_sc]
    return ps


def total_photon_num_dist_pure_state(cov, cutoff=50, hbar=2, padding_factor=2):
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
        return gen_multi_mode_dist(rs, cutoff=cutoff, padding_factor=padding_factor)[0:cutoff]
    raise ValueError("The Gaussian state is not pure")


def fock_tensor(S, alpha, cutoff, choi_r=np.arcsinh(1.0), check_symplectic=True, sf_order=False):
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

    Return:
        (array): Tensor containing the Fock representation of the Gaussian unitary
    """
    # Check the matrix is symplectic
    if check_symplectic:
        if not is_symplectic(S):
            raise ValueError("The matrix S is not symplectic")

    # And that S and alpha have compatible dimensions
    m, _ = S.shape
    if m // 2 != len(alpha):
        raise ValueError("The matrix S and the vector alpha do not have compatible dimensions")

    # Construct the covariance matrix of l two-mode squeezed vacua pairing modes i and i+l
    l = m // 2
    ch = np.cosh(choi_r) * np.identity(l)
    sh = np.sinh(choi_r) * np.identity(l)
    zh = np.zeros([l, l])
    Schoi = np.block([[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]])
    # And then its Choi expanded symplectic
    S_exp = expand(S, list(range(l)), 2 * l) @ Schoi
    # And this is the corresponding covariance matrix
    cov = S_exp @ S_exp.T
    alphat = np.array(list(alpha) + ([0] * l))
    x = 2 * alphat.real
    p = 2 * alphat.imag
    mu = np.concatenate([x, p])

    tensor = state_vector(
        mu, cov, normalize=False, cutoff=cutoff, hbar=2, check_purity=False, choi_r=choi_r
    )

    if sf_order:
        sf_indexing = tuple(chain.from_iterable([[i, i + l] for i in range(l)]))
        return tensor.transpose(sf_indexing)

    return tensor


def generate_probabilities(mu, cov, cutoff, hbar=2.0):
    """ Generate the Fock space probabilities of Gaussian state with vector of mean
    mu and covariance matrix cov up to Fock space cutoff.

    Args:
        mu (array): vector of means of length 2*n_modes
        cov (array): covariance matrix of shape [2*n_modes, 2*n_modes]
        cutoff (int): Fock space cutoff
        hbar (float): value of hbar

    Returns:
        (array): Fock space probabilities up to cutoff. The shape of this tensor is [cutoff]*num_modes
    """
    if is_pure_cov(cov, hbar=hbar):  # Check if the covariance matrix cov is pure
        return np.abs(state_vector(mu, cov, cutoff=cutoff, hbar=hbar, check_purity=False)) ** 2
    num_modes = len(mu) // 2
    probabilities = np.zeros([cutoff] * num_modes)
    for i in product(range(cutoff), repeat=num_modes):
        probabilities[i] = np.maximum(
            0.0, np.real_if_close(density_matrix_element(mu, cov, i, i, hbar=hbar))
        )
        # The maximum is needed because every now and then a probability is very close to zero from below.
    return probabilities


@jit(nopython=True)
def loss_mat(eta, cutoff): # pragma: no cover
    """ Constructs a binomial loss matrix with transmission eta up to n photons.

    Args:
        eta (float): Transmission coefficient. eta=0.0 means complete loss and eta=1.0 means no loss.
        n (int): photon number cutoff.

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
    """ Given a list of transmissivities etas and a tensor of probabilitites probs it calculates
    an updated tensor of probabilities after loss is applied.

    Args:
        etas (list): List of transmission describing the loss in each of the modes
        probs (array): Array of probabilitites in the different modes

    Returns:
        array: List of loss-updated probabilities.

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
