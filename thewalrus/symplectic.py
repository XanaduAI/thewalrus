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
Symplectic operations
=====================

**Module name:** :mod:`thewalrus.symplectic`

.. currentmodule:: thewalrus.symplectic

Contains some Gaussian operations and auxiliary functions.

Auxiliary functions
-------------------

.. autosummary::
    expand
    expand_vector
    expand_passive
    reduced_state
    is_symplectic
    sympmat
    xxpp_to_xpxp
    xpxp_to_xxpp

Gaussian states
---------------

.. autosummary::
    vacuum_state


Gates and operations
--------------------

.. autosummary::
    two_mode_squeezing
    squeezing
    interferometer
    loss
    mean_photon_number
    beam_splitter
    rotation

Code details
------------
"""
import warnings
import numpy as np
from scipy.sparse import (
    identity as sparse_identity,
    issparse,
    coo_array,
    dia_array,
    bsr_array,
    csr_array,
)


def expand(S, modes, N):
    r"""Expands a Symplectic matrix S to act on the entire subsystem.
    If the input is a single mode symplectic, then extends it to act
    on multiple modes.

    Supports scipy sparse matrices. Instances of ``coo_array``, ``dia_array``,
    ``bsr_array`` will be transformed into `csr_array``.

    Args:
        S (ndarray or spmatrix): a :math:`2M\times 2M` Symplectic matrix
        modes (Sequence[int]): the list of modes S acts on
        N (int): full size of the subsystem

    Returns:
        array: the resulting :math:`2N\times 2N` Symplectic matrix
    """
    M = S.shape[0] // 2
    S2 = np.identity(2 * N, dtype=S.dtype)

    if issparse(S):
        # cast to sparse matrix that supports slicing and indexing
        S2 = sparse_identity(2 * N, dtype=S.dtype, format="csr")
        if isinstance(S, (coo_array, dia_array, bsr_array)):
            warnings.warn(
                "Unsupported sparse matrix type, returning a Compressed Sparse Row (CSR) matrix."
            )
            S = csr_array(S)

    w = np.array([modes]) if isinstance(modes, int) else np.array(modes)

    # extend single mode symplectic to act on selected modes
    if M == 1:
        for m in w:
            S2[m, m], S2[m + N, m + N] = S[0, 0], S[1, 1]  # X, P
            S2[m, m + N], S2[m + N, m] = S[0, 1], S[1, 0]  # XP, PX
        return S2

    # make symplectic act on the selected subsystems
    S2[w.reshape(-1, 1), w.reshape(1, -1)] = S[:M, :M].copy()  # X
    S2[(w + N).reshape(-1, 1), (w + N).reshape(1, -1)] = S[M:, M:].copy()  # P
    S2[w.reshape(-1, 1), (w + N).reshape(1, -1)] = S[:M, M:].copy()  # XP
    S2[(w + N).reshape(-1, 1), w.reshape(1, -1)] = S[M:, :M].copy()  # PX
    return S2


def expand_vector(alpha, mode, N, hbar=2.0):
    """Returns the phase-space displacement vector associated to a displacement.

    Args:
        alpha (complex): complex displacement
        mode (int): mode index
        N (int): number of modes

    Returns:
        array: phase-space displacement vector of size 2*N
    """
    alpharealdtype = np.dtype(type(alpha))

    r = np.zeros(2 * N, dtype=alpharealdtype)
    r[mode] = np.sqrt(2 * hbar) * alpha.real
    r[N + mode] = np.sqrt(2 * hbar) * alpha.imag
    return r


def expand_passive(T, modes, N):
    r"""Returns the expanded linear optical transformation
    acting on specified modes, with identity acting on all other modes

    Args:
        T (array): square :math:`M \times M` matrix of linear optical transformation
        modes (array): the :math:`M` modes of the transformation
        N (int): number of modes in the new expanded transformation

    Returns:
        array: :math:`N \times N` array of expanded passive transformation
    """

    if T.shape[0] != T.shape[1]:
        raise ValueError("The input matrix is not square")

    if len(modes) != T.shape[0]:
        raise ValueError("length of modes must match the shape of T")

    T_expand = np.eye(N, dtype=T.dtype)
    T_expand[np.ix_(modes, modes)] = T

    return T_expand


def reduced_state(mu, cov, modes):
    r"""Returns the vector of means and the covariance matrix of the specified modes.

    Args:
        modes (int of Sequence[int]): indices of the requested modes

    Returns:
        tuple (means, cov): means is an array containing the vector of means,
        and cov is a square array containing the covariance matrix. Both use the :math:`xp`-ordering.
    """
    N = len(mu) // 2

    if modes == list(range(N)):
        # reduced state is full state
        return mu, cov

    # reduce state down to specified subsystems
    if isinstance(modes, int):
        modes = [modes]

    if np.any(np.array(modes) > N):
        raise ValueError("The specified modes cannot be larger than number of subsystems.")

    ind = np.concatenate([np.array(modes), np.array(modes) + N])
    rows = ind.reshape(-1, 1)
    cols = ind.reshape(1, -1)

    return mu.copy()[ind], cov.copy()[rows, cols]


def vacuum_state(modes, hbar=2.0, dtype=np.float64):
    r"""Returns the vacuum state.

    Args:
        modes (str): Returns the vector of means and the covariance matrix
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
        dtype (numpy.dtype): datatype to represent the covariance matrix and vector of means
    Returns:
        list[array]: the means vector and covariance matrix of the vacuum state
    """
    means = np.zeros((2 * modes), dtype=dtype)
    cov = np.identity(2 * modes, dtype=dtype) * hbar / 2
    state = [means, cov]
    return state


def squeezing(r, phi=None, dtype=np.float64):
    r"""Squeezing. In fock space this corresponds to:

    .. math::

         \exp(\tfrac{1}{2}r e^{i \phi} (a^2 - a^{\dagger 2}) ).


    By passing an array of squeezing parameters and phases, it applies a tensor product of squeezing operations.

    Args:
        r (Union[array, float]): squeezing magnitude
        phi (Union[array, float]): rotation parameter. If ``None``, then the function uses zeros of the same shape as ``r``.
        dtype (numpy.dtype): datatype to represent the Symplectic matrix.  Defaults to ``numpy.float64``.
    Returns:
        array: symplectic transformation matrix
    """
    # pylint: disable=assignment-from-no-return

    r = np.atleast_1d(r)

    if phi is None:
        phi = np.zeros_like(r)

    phi = np.atleast_1d(phi)

    M = len(r)
    S = np.identity(2 * M, dtype=dtype)

    for i, (r_i, phi_i) in enumerate(zip(r, phi)):
        S[i, i] = np.cosh(r_i) - np.sinh(r_i) * np.cos(phi_i)
        S[i, i + M] = -np.sinh(r_i) * np.sin(phi_i)
        S[i + M, i] = -np.sinh(r_i) * np.sin(phi_i)
        S[i + M, i + M] = np.cosh(r_i) + np.sinh(r_i) * np.cos(phi_i)

    return S


def two_mode_squeezing(r, phi, dtype=np.float64):
    """Two-mode squeezing.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: symplectic transformation matrix
    """
    # pylint: disable=assignment-from-no-return
    cp = np.cos(phi, dtype=dtype)
    sp = np.sin(phi, dtype=dtype)
    ch = np.cosh(r, dtype=dtype)
    sh = np.sinh(r, dtype=dtype)

    S = np.array(
        [
            [ch, cp * sh, 0, sp * sh],
            [cp * sh, ch, sp * sh, 0],
            [0, sp * sh, ch, -cp * sh],
            [sp * sh, 0, -cp * sh, ch],
        ]
    )

    return S


def interferometer(U):
    """Interferometer.

    Args:
        U (array): unitary matrix

    Returns:
        array: symplectic transformation matrix
    """
    X = U.real
    Y = U.imag
    S = np.block([[X, -Y], [Y, X]])

    return S


def passive_transformation(mu, cov, T, hbar=2):
    r"""Perform a covariance matrix transformation for an arbitrary linear optical channel
    on an :math:`N` modes state mapping it to a to an :math:`M` modes state.

    Args:
        mu (array): :math:`2N`-length means vector
        cov (array): :math:`2N \times 2N` covariance matrix
        T (array): :math:`M \times N` linear optical transformation

    Keyword Args:
        hbar (float)=2: the value to use for hbar

    Returns:
        array: :math:`2M`-length transformed means vector
        array :math:`2M \times 2M` tranformed covariance matrix
    """

    P = interferometer(T)
    L = (hbar / 2) * (np.eye(P.shape[0]) - P @ P.T)

    cov = P @ cov @ P.T + L
    mu = P @ mu

    return mu, cov


# pylint: disable=too-many-arguments
def loss(mu, cov, T, mode, nbar=0, hbar=2):
    r"""Loss channel acting on a Gaussian state.

    Args:
        mu (array): means vector
        cov (array): covariance matri
        T (float): transmission; 1 corresponds to no
            loss, 0 to full loss.
        mode (int): mode to act on
        nbar (float): thermal mean population (default 0)
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple[array]: the means vector and covariance matrix of the resulting state
    """
    N = len(cov) // 2

    cov_res = cov.copy()
    mu_res = mu.copy()

    for m in (mode, mode + N):
        mu_res[m] *= np.sqrt(T)
        cov_res[m, :] *= np.sqrt(T)
        cov_res[:, m] *= np.sqrt(T)
        cov_res[m, m] += (1 - T) * (2 * nbar + 1) * hbar / 2

    return mu_res, cov_res


### Comment: This function strongly overlaps with `quantum.photon_number_mean`
### Wonder if it is worth removing it.
def mean_photon_number(mu, cov, hbar=2):
    r"""Calculates the mean photon number for a given one-mode state.

    Args:
        mu (array): length-2 vector of means
        cov (array): :math:`2\times 2` covariance matrix
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple: the photon number expectation and variance
    """
    ex = (np.trace(cov) + mu.T @ mu) / (2 * hbar) - 1 / 2
    var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * hbar**2) - 1 / 4
    return ex, var


def beam_splitter(theta, phi, dtype=np.float64):
    """Beam-splitter.

    Args:
        theta (float): transmissivity parameter
        phi (float): phase parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: symplectic-orthogonal transformation matrix of an interferometer with angles theta and phi
    """
    ct = np.cos(theta, dtype=dtype)
    st = np.sin(theta, dtype=dtype)
    eip = np.cos(phi, dtype=dtype) + 1j * np.sin(phi, dtype=dtype)
    U = np.array(
        [
            [ct, -eip.conj() * st],
            [eip * st, ct],
        ]
    )
    return interferometer(U)


def rotation(theta, dtype=np.float64):
    """Rotation gate.

    Args:
        theta (float): rotation angle
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: rotation matrix by angle theta
    """
    V = np.identity(1) * (np.cos(theta, dtype=dtype) + 1j * np.sin(theta, dtype=dtype))
    return interferometer(V)


def sympmat(N, dtype=np.float64):
    r"""Returns the matrix :math:`\Omega_n = \begin{bmatrix}0 & I_n\\ -I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N, dtype=dtype)
    O = np.zeros_like(I, dtype=dtype)
    S = np.block([[O, I], [-I, O]])
    return S


def is_symplectic(S, rtol=1e-05, atol=1e-08):
    r"""Checks if matrix S is a symplectic matrix

    Args:
        S (array): a square matrix

    Returns:
        (bool): whether the given matrix is symplectic
    """
    n, m = S.shape
    if n != m:
        return False
    if n % 2 != 0:
        return False
    nmodes = n // 2
    Omega = sympmat(nmodes, dtype=S.dtype)

    return np.allclose(S.T @ Omega @ S, Omega, rtol=rtol, atol=atol)


def autonne(A, rtol=1e-05, atol=1e-08, svd_order=True):
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.

    Args:
        A (array): square, symmetric matrix
        rtol (float): the relative tolerance parameter between ``A`` and ``A.T``
        atol (float): the absolute tolerance parameter between ``A`` and ``A.T``
        svd_order (boolean): whether to return result by ordering the singular values of ``A`` in descending (``True``) or asceding (``False``) order.

    Returns:
        tuple[array, array]: (r, U), where r are the singular values,
        and U is the Autonne-Takagi unitary, such that :math:`A = U \diag(r) U^T`.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("The input matrix is not square")
    if not np.allclose(A, A.T, rtol=rtol, atol=atol):
        raise ValueError("The input matrix is not symmetric")
    Areal = A.real
    Aimag = A.imag

    B = np.empty((2 * n, 2 * n))
    B[:n, :n] = Areal
    B[n : 2 * n, :n] = Aimag
    B[:n, n : 2 * n] = Aimag
    B[n : 2 * n, n : 2 * n] = -Areal
    vals, vects = np.linalg.eigh(B)
    U = vects[:n, n : 2 * n] + 1j * vects[n : 2 * n, n : 2 * n]
    if svd_order:
        return (vals[n : 2 * n])[::-1], U[:, ::-1]
    return vals[n : 2 * n], U


def xxpp_to_xpxp(S):
    """Permutes the entries of the input from xxpp ordering to xpxp ordering.

    Args:
        S (array): input even dimensional square matrix or array

    Returns:
        (array): permuted matrix or array
    """
    shape = S.shape
    n = shape[0]

    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(2, -1).T.flatten()

    if len(shape) == 2:
        if shape[0] != shape[1]:
            raise ValueError("The input matrix is not square")
        return S[:, ind][ind]

    return S[ind]


def xpxp_to_xxpp(S):
    """Permutes the entries of the input from xpxp ordering to xxpp ordering.

    Args:
        S (array): input even dimensional square matrix or vector

    Returns:
        (array): permuted matrix or vector
    """
    shape = S.shape
    n = shape[0]

    if n % 2 != 0:
        raise ValueError("The input array is not even-dimensional")

    n = n // 2
    ind = np.arange(2 * n).reshape(-1, 2).T.flatten()

    if len(shape) == 2:
        if shape[0] != shape[1]:
            raise ValueError("The input matrix is not square")
        return S[:, ind][ind]

    return S[ind]
