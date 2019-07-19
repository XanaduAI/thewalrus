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

.. currentmodule:: hafnian.quantum

This submodule provides access to various utility functions that act on Gaussian
quantum states.

For more details on how the hafnian relates to various properties of Gaussian quantum
states, see:

* Kruse, R., Hamilton, C. S., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "A detailed study of Gaussian Boson Sampling." `arXiv:1801.07488. (2018).
  <https://arxiv.org/abs/1801.07488>`_

* Hamilton, C. S., Kruse, R., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "Gaussian boson sampling." `Physical review letters, 119(17), 170501. (2017).
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.170501>`_

* Quesada, N. 
  "Franck-Condon factors by counting perfect matchings of graphs with loops."
  `Journal of Chemical Physics 150, 164113 (2019). 10.1063/1.5086387>`_ 

* Quesada, N., Helt, L. G., Izaac, J., Arrazola, J. M., Shahrokhshahi, R., Myers, C. R., & Sabapathy, K. K. 
  "Simulating realistic non-Gaussian state preparation."
  `arXiv:1905.07011. (2019). <https://arxiv.org/abs/1905.07011>`


Fock states
-----------

.. autosummary::

    pure_state_amplitude
    state_vector
    density_matrix_element
    density_matrix

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


Utility functions
-----------------

.. autosummary::

    reduced_gaussian
    Xmat
    Sympmat
    Qmat
    Covmat
    Amat
    Beta
    Means
    prefactor
    find_scaling_adjacency_matrix
    gen_Qmat_from_graph
    is_valid_cov
    is_pure_cov
    is_classical_cov

Details
^^^^^^^
"""
# pylint: disable=too-many-arguments
from itertools import count, product

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import factorial as fac

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


def Sympmat(N):
    r"""Returns the matrix :math:`\Omega_n = \begin{bmatrix}0 & I_n\\ -I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    S = np.block([[O, I], [-I, O]])
    return S


def Qmat(cov, hbar=2):
    r"""Returns the :math:`Q` matrix of the Gaussian state.

    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
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
    r"""Returns the covariance matrix in the :math:`xp`-ordering of the Gaussian state.
    This is the inverse function of Qmat.

    Args:
        Q (array): :math:`2N\times 2N` Q matrix
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
        # gamma = X @ np.linalg.inv(Q).conj() @ beta
        gamma = beta.conj() - A @ beta
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            A_rpt = reduction(A, rpt)
            np.fill_diagonal(A_rpt, reduction(gamma, rpt))
            haf = hafnian(A_rpt, loop=True)
        else:
            haf = hafnian_repeated(A, rpt, mu=gamma, loop=True)

    if include_prefactor:
        haf *= prefactor(mu, cov, hbar=2)

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
        A = Amat(cov, hbar=hbar)
        if np.allclose(mu, np.zeros_like(mu)):
            return pref * hermite_multidimensional(-A, cutoff, renorm=True)
        try:
            y = np.linalg.inv(A) @ np.linalg.inv(Qmat(cov)) @ Beta(mu, hbar=hbar)
            return pref * hermite_multidimensional(-A, cutoff, y=-y, renorm=True)
        except np.linalg.LinAlgError:
            pass
        post_select = {}

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
        if not is_pure_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
            raise ValueError("The covariance matrix does not correspond to a pure state")

    rpt = i
    beta = Beta(mu, hbar=hbar)
    Q = Qmat(cov, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    (n, _) = cov.shape
    N = n // 2
    B = A[0:N, 0:N]
    alpha = beta[0:N]

    if np.linalg.norm(alpha) < tol:
        # no displacement
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            haf = hafnian(B_rpt)
        else:
            haf = hafnian_repeated(B, rpt)
    else:
        # replace the diagonal of A with gamma
        # gamma = X @ np.linalg.inv(Q).conj() @ beta
        zeta = alpha - B @ (alpha.conj())

        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            np.fill_diagonal(B_rpt, reduction(zeta, rpt))
            haf = hafnian(B_rpt, loop=True)
        else:
            haf = hafnian_repeated(B, rpt, mu=zeta, loop=True)

    if include_prefactor:
        pref = np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - alpha.conj() @ B @ alpha.conj()))
        haf *= pref

    return haf / np.sqrt(np.prod(fac(rpt)) * np.sqrt(np.linalg.det(Q)))


def state_vector(mu, cov, post_select=None, normalize=False, cutoff=5, hbar=2, check_purity=True):
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

    Returns:
        np.array[complex]: the state vector of the Gaussian state
    """
    if check_purity:
        if not is_pure_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
            raise ValueError("The covariance matrix does not correspond to a pure state")

    beta = Beta(mu, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    Q = Qmat(cov, hbar=hbar)

    (n, _) = cov.shape
    N = n // 2

    B = A[0:N, 0:N]
    alpha = beta[0:N]
    pref = np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - alpha.conj() @ B @ alpha.conj()))

    if post_select is None:
        psi = (
            pref
            * hafnian_batched(B, cutoff, mu=alpha, renorm=True)
            / np.sqrt(np.sqrt(np.linalg.det(Q).real))
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
                mu, cov, el, check_purity=False, include_prefactor=False
            )

        psi = psi * pref

    if normalize:
        norm = np.sqrt(np.sum(np.abs(psi) ** 2))
        psi = psi / norm

    return psi


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

    def grad_mean_photon_number(x, vals):
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
    assert res.converged
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
    vals = np.linalg.eigvalsh(cov + 0.5j * hbar * Sympmat(nmodes))
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
        (bool): whether the given covariance matrix corresponds to a pure state
    """

    if is_valid_cov(cov, hbar=hbar, atol=atol):
        (n, _) = cov.shape
        vals = np.linalg.eigvalsh(cov - 0.5 * hbar * np.identity(n))
        vals[np.abs(vals) < atol] = 0.0

        if np.all(vals >= 0):
            return True
    return False
