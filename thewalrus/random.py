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
Random matrices
===============

.. currentmodule:: thewalrus.random

This submodule provides access to utility functions to generate random unitary, symplectic
and covariance matrices.

"""
import numpy as np
import scipy as sp

# ------------------------------------------------------------------------
# Random numbers and matrices                                           |
# ------------------------------------------------------------------------

def randnc(*arg):
    """Normally distributed array of random complex numbers."""
    return np.random.randn(*arg) + 1j * np.random.randn(*arg)


def random_covariance(N, hbar=2, pure=False, block_diag=False):
    r"""Random covariance matrix.

    Args:
        N (int): number of modes
        hbar (float): the value of :math:`\hbar` to use in the definition
            of the quadrature operators :math:`x` and :math:`p`
        pure (bool): If True, a random covariance matrix corresponding
            to a pure state is returned.
        block_diag (bool): If True, uses passive Gaussian transformations that are orthogonal
            instead of unitary. This implies that the positions :math:`x` do not mix with
            the momenta :math:`p` and thus the covariance matrix is block diagonal.

    Returns:
        array: random :math:`2N\times 2N` covariance matrix
    """
    S = random_symplectic(N, block_diag=block_diag)

    if pure:
        return (hbar / 2) * S @ S.T

    nbar = 2 * np.abs(np.random.random(N)) + 1
    Vth = (hbar / 2) * np.diag(np.concatenate([nbar, nbar]))

    return S @ Vth @ S.T


def random_symplectic(N, passive=False, block_diag=False, scale=1.0):
    r"""Random symplectic matrix representing a Gaussian transformation.

    The squeezing parameters :math:`r` for active transformations are randomly
    sampled from the standard normal distribution, while passive transformations
    are randomly sampled from the Haar measure. Note that for the Symplectic
    group there is no notion of Haar measure since this is group is not compact.

    Args:
        N (int): number of modes
        passive (bool): If True, returns a passive Gaussian transformation (i.e.,
            one that preserves photon number). If False, returns an active
            transformation.
        block_diag (bool): If True, uses passive Gaussian transformations that are orthogonal
            instead of unitary. This implies that the positions :math:`q` do not mix with
            the momenta :math:`p` and thus the symplectic operator is block diagonal
        scale (float): Sets the scale of the random values used as squeezing parameters.
            They will range from 0 to :math:`\sqrt{2}\texttt{scale}`

    Returns:
        array: random :math:`2N\times 2N` symplectic matrix
    """
    U = random_interferometer(N, real=block_diag)
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])

    if passive:
        return O

    U = random_interferometer(N, real=block_diag)
    P = np.block([[U.real, -U.imag], [U.imag, U.real]])

    r = scale * np.abs(randnc(N))
    Sq = np.diag(np.concatenate([np.exp(-r), np.exp(r)]))

    return O @ Sq @ P


def random_interferometer(N, real=False):
    r"""Random unitary matrix representing an interferometer.
    For more details, see :cite:`mezzadri2006`.

    Args:
        N (int): number of modes
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    if real:
        z = np.random.randn(N, N)
    else:
        z = randnc(N, N) / np.sqrt(2.0)
    q, r = sp.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    U = np.multiply(q, ph, q)
    return U


def random_block_interferometer(N, top_one=True, real=False):
    r"""Generates a random interferometer with blocks of at most size 2.

    Args:
        N (int): number of modes
        top_one (bool): if True places a `1\times1` interferometer in the top-left most block
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary with the specified block structure
    """
    if N % 2 == 0:
        if top_one:
            u2s = [random_interferometer(2, real=real) for i in range(((N // 2) - 1))]
            u0 = random_interferometer(1, real=real)
            u1 = random_interferometer(1, real=real)
            return sp.linalg.block_diag(u0, *u2s, u1)

        u2s = [random_interferometer(2, real=real) for i in range(N // 2)]
        return sp.linalg.block_diag(*u2s)

    u2s = [random_interferometer(2, real=real) for i in range((N - 1) // 2)]
    u0 = random_interferometer(1, real=real)
    if top_one:
        return sp.linalg.block_diag(u0, *u2s)
    return sp.linalg.block_diag(*u2s, u0)

def random_banded_interferometer(N, w, top_one_init=True, real=False):
    r"""Generates a banded unitary matrix.

    Args:
        N (int): number of modes
        w (int): bandwidth
        top_one_init (bool): if True places a `1\times1` interferometer in the top-left-most block of the first matrix in the product
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary with the specified block structure
    """
    if N < w + 1:
        raise ValueError("The bandwidth can be at most one minus the size of the matrix.")
    if N == w + 1:
        return random_interferometer(N, real=real)
    U = sp.linalg.block_diag(*[random_interferometer(1, real=real) for _ in range(N)])
    for _ in range(w):
        U = U @ random_block_interferometer(N, top_one=top_one_init, real=real)
        top_one_init = not top_one_init
    return U
