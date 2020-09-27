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
Symplectic Operations
=====================

**Module name:** :mod:`thewalrus.symplectic`

.. currentmodule:: thewalrus.symplectic

Contains some Gaussian operations and auxiliary functions.

Auxiliary functions
-------------------

.. autosummary::
    expand
    expand_vector
    reduced_state
    is_symplectic
    sympmat

Gaussian states
---------------

.. autosummary::
    vacuum_state


Gates and operations
--------------------

.. autosummary::
    two_mode_squeezing
    interferometer
    loss
    mean_photon_number
    beam_splitter
    rotation

Code details
^^^^^^^^^^^^
"""
import numpy as np


def expand(S, modes, N):
    r"""Expands a Symplectic matrix S to act on the entire subsystem.

    Args:
        S (array): a :math:`2M\times 2M` Symplectic matrix
        modes (Sequence[int]): the list of modes S acts on
        N (int): full size of the subsystem

    Returns:
        array: the resulting :math:`2N\times 2N` Symplectic matrix
    """
    M = len(S) // 2
    S2 = np.identity(2 * N, dtype=S.dtype)
    w = np.array(modes)

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


def reduced_state(mu, cov, modes):
    r""" Returns the vector of means and the covariance matrix of the specified modes.

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


def squeezing(r, phi, dtype=np.float64):
    r"""Squeezing. In fock space this corresponds to \exp(\tfrac{1}{2}r e^{i \phi} (a^2 - a^{\dagger 2}) ).

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
    S = np.array([[ch - cp * sh, -sp * sh], [-sp * sh, ch + cp * sh]])

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
    var = (np.trace(cov @ cov) + 2 * mu.T @ cov @ mu) / (2 * hbar ** 2) - 1 / 4
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
    r""" Checks if matrix S is a symplectic matrix

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
