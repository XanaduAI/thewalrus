# Copyright 2021 Xanadu Quantum Technologies Inc.

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

**Module name:** :mod:`thewalrus.symplectic_tf`

.. currentmodule:: thewalrus.symplectic_tf

Contains some differentiable Gaussian operations and auxiliary functions.

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
import tensorflow as tf
from typing import Sequence
from functools import lru_cache

def expand(S, modes:Sequence[int], N:int):
    r"""Expands a Symplectic matrix S to act on the entire subsystem.

    Args:
        S (array): a :math:`2M\times 2M` Symplectic matrix
        modes (Sequence[int]): the list of modes S acts on
        N (int): full size of the subsystem

    Returns:
        array: the resulting :math:`2N\times 2N` Symplectic matrix
    """
    M = S.shape[-1] // 2
    modes = modes + [m+N for m in modes]
    idxS = iter(range(2*M))
    idxI = iter(range(2*M,2*N))
    Z = tf.zeros((2*M, 2*(N-M)), dtype=S.dtype)
    I = tf.eye(2*(N-M), dtype=S.dtype)
    S2 = tf.concat([tf.concat([S, Z], axis=1), tf.concat([tf.transpose(Z), I], axis=1)], axis=0)

    pick = [next(idxI) if m not in modes else next(idxS) for m in range(2*N)]

    return tf.gather(tf.gather(S2, pick, axis=0), pick, axis=1)


def expand_vector(alpha:complex, mode:int, N:int, hbar=2.0):
    """Returns the phase-space displacement vector associated to a displacement.

    Args:
        alpha (complex): complex displacement
        mode (int): mode index
        N (int): number of modes

    Returns:
        array: phase-space displacement vector of size 2*N
    """
    indices = [[mode], [mode+N]]
    updates = [np.sqrt(2 * hbar) * tf.math.real(alpha), np.sqrt(2 * hbar) * tf.math.imag(alpha)]
    Z = tf.zeros(2*N, dtype=alpha.dtype.real_dtype)
    return tf.tensor_scatter_nd_add(Z,indices, updates)



def reduced_state(mu, cov, modes:Sequence[int]):
    r""" Returns the vector of means and the covariance matrix of the specified modes.

    Args:
        mu
        cov
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

    ind = modes + [m+N for m in modes]

    return tf.gather(mu,ind), tf.gather(tf.gather(cov, ind, axis=0), ind, axis=1)


def vacuum_state(modes:int, hbar=2.0, dtype=tf.float64):
    r"""Returns the vacuum state.

    Args:
        modes (int): number of modes
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`
        dtype (numpy.dtype): datatype to represent the covariance matrix and vector of means
    Returns:
        list[array]: the means vector and covariance matrix of the vacuum state
    """
    means = tf.zeros((2 * modes), dtype=dtype)
    cov = tf.eye(2 * modes, dtype=dtype) * hbar / 2
    state = [means, cov]
    return state


def squeezing(r:float, phi:float, dtype=tf.float64):
    r"""Squeezing. In fock space this corresponds to \exp(\tfrac{1}{2}r e^{i \phi} (a^2 - a^{\dagger 2}) ).

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix
    Returns:
        array: symplectic transformation matrix
    """
    # pylint: disable=assignment-from-no-return
    cp = tf.cast(tf.math.cos(phi), dtype)
    sp = tf.cast(tf.math.sin(phi), dtype)
    ch = tf.cast(tf.math.cosh(r), dtype)
    sh = tf.cast(tf.math.sinh(r), dtype)
    S = tf.convert_to_tensor([[ch - cp * sh, -sp * sh], [-sp * sh, ch + cp * sh]])

    return S


def two_mode_squeezing(r:float, phi:float, dtype=tf.float64):
    """Two-mode squeezing.

    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: symplectic transformation matrix
    """
    # pylint: disable=assignment-from-no-return
    cp = tf.cast(tf.math.cos(phi), dtype)
    sp = tf.cast(tf.math.sin(phi), dtype)
    ch = tf.cast(tf.math.cosh(r), dtype)
    sh = tf.cast(tf.math.sinh(r), dtype)

    S = tf.convert_to_tensor(
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
    X = tf.math.real(U)
    Y = tf.math.imag(U)
    S = tf.concat([tf.concat([X, -Y], axis=1), tf.concat([Y, X], axis=1)], axis=0)

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
    sqrtT = tf.cast(tf.math.sqrt(T), dtype=mu.dtype)
    mu = tf.tensor_scatter_nd_update(mu, [[mode], [mode+N]], [mu[mode]*sqrtT, mu[mode+N]*sqrtT])
    cov = tf.tensor_scatter_nd_update(cov, [[mode], [mode+N]], [cov[mode]*sqrtT, cov[mode+N]*sqrtT])
    cov = tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(cov), [[mode], [mode+N]], [cov[:,mode]*sqrtT, cov[:,mode+N]*sqrtT]))
    cov = tf.tensor_scatter_nd_add(cov, [[mode,mode], [mode+N,mode+N]], [(1 - T) * (2 * nbar + 1) * hbar / 2, (1 - T) * (2 * nbar + 1) * hbar / 2])

    return mu, cov

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
    ex = (tf.linalg.trace(cov) + tf.transpose(mu) @ mu) / (2 * hbar) - 1 / 2
    var = (tf.linalg.trace(cov @ cov) + 2 * tf.transpose(mu) @ cov @ mu) / (2 * hbar ** 2) - 1 / 4
    return ex, var


def beam_splitter(theta:float, phi:float, dtype=tf.float64):
    """Beam-splitter.

    Args:
        theta (float): transmissivity parameter
        phi (float): phase parameter
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: symplectic-orthogonal transformation matrix of an interferometer with angles theta and phi
    """
    cdtype = {tf.float64:tf.complex128,
              np.float64:tf.complex128,
              tf.float32:tf.complex64,
              np.float32:tf.complex64,
              float:tf.complex64}
    dtype = cdtype[dtype]

    ct = tf.cast(tf.math.cos(theta), dtype)
    st = tf.cast(tf.math.sin(theta), dtype)
    eip = tf.complex(tf.cast(tf.math.cos(phi), dtype), tf.cast(tf.math.sin(phi), dtype))
    U = tf.convert_to_tensor(
        [
            [ct, -tf.math.conj(eip) * st],
            [eip * st, ct],
        ]
    )
    return interferometer(U)


def rotation(theta:float, dtype=tf.float64):
    """Rotation gate.

    Args:
        theta (float): rotation angle
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: rotation matrix by angle theta
    """
    cdtype = {tf.float32:tf.float64, tf.float64:tf.float128}
    x = tf.cast(tf.math.cos(theta), dtype)
    y = tf.cast(tf.math.sin(theta), dtype)
    V = tf.eye(1, dtype=cdtype[dtype]) * tf.complex(x, y)
    return interferometer(V)


def sympmat(N:int, dtype=tf.float64):
    r"""Returns the matrix :math:`\Omega_n = \begin{bmatrix}0 & I_n\\ -I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer
        dtype (numpy.dtype): datatype to represent the Symplectic matrix

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = tf.eye(N, dtype=dtype)
    O = tf.zeros_like(I)
    S = tf.concat([tf.concat([O, I], axis=1), tf.concat([-I, O],axis=1)], axis=0)
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

    return np.allclose(tf.transpose(S) @ Omega @ S, Omega, rtol=rtol, atol=atol)

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
    if not np.allclose(A, tf.transpose(A), rtol=rtol, atol=atol):
        raise ValueError("The input matrix is not symmetric")
    Areal = tf.math.real(A)
    Aimag = tf.math.imag(A)

    B = tf.concat([tf.concat([Areal, Aimag], axis = 1), tf.concat([Aimag, -Areal], axis=1)], axis=0)
    vals, vects = tf.linalg.eigh(B)
    vects = tf.cast(vects, dtype=tf.complex64)
    U = vects[:n, n : 2 * n] + 1j * vects[n : 2 * n, n : 2 * n]
    if svd_order:
        return (vals[n : 2 * n])[::-1], U[:, ::-1]
    return vals[n : 2 * n], U

@lru_cache()
def rotmat(l):
    """Rotation matrix from quadratures to complex amplitudes
    Args:
        l (int): size
    Returns:
        (array): rotation matrix
    """
    idl = np.identity(l)
    R = np.sqrt(0.5) * np.block([[idl, 1j * idl], [idl, -1j * idl]])
    return tf.convert_to_tensor(R)


@lru_cache()
def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`
    Args:
        N (int): positive integer
    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros((N,N))
    X = np.block([[O, I], [I, O]])
    return tf.convert_to_tensor(X+0.0j)


def Amat_states(cov, choi_r=np.arcsinh(1.0), hbar=2):
    """Generate the double adjacency matrix of a (in general) mixed Gaussian state
    Args:
        cov (array): covariance matrix
    Returns:
        (array): adjacency matrix of the Gaussian state
    """
    l, l = cov.shape
    R = rotmat(l)
    sigma = (1 / hbar) * R @ tf.cast(cov, dtype=R.dtype) @ tf.math.conj(tf.transpose(R))
    I = tf.eye(2*l, dtype=R.dtype)
    A = tf.matmul(Xmat(l), (I - tf.linalg.inv(sigma + 0.5 * I)))
    return A