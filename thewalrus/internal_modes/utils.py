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
"""Useful tools for dealing with internal modes"""
import numpy as np
from scipy.special import factorial
import numba

fact = np.array([factorial(i) for i in range(110)], dtype=np.float64)


@numba.jit(nopython=True, cache=True)
def spatial_modes_to_schmidt_modes(spatial_modes, K):
    """
    returns index of schmidt modes corresponding to the give spatial modes.
    e.g. if there are K=3 schmidt modes and spatial_modes=[0,2]
    then schmidt_modes=[0,1,2,6,7,8]
    Args:
        spatial_modes (array): indices of spatial modes
        K (int): number of schmidt modes per spatial mode
    Returns:
        schmidt_modes (array): indices of schmidt modes
    """
    spatial_modes = np.asarray(spatial_modes)
    M = len(spatial_modes)
    schmidt_modes = np.empty(M * K, dtype=spatial_modes.dtype)

    for i in range(K):
        schmidt_modes[i::K] = K * spatial_modes + i

    return schmidt_modes


@numba.jit(nopython=True, cache=True)
def spatial_reps_to_schmidt_reps(spatial_reps, K):
    """
    returns reps of schmidt modes corresponding to the give spatial reps.
    e.g. if there are K=3 schmidt modes and spatial_reps=[1,2]
    then schmidt_reps=[1,1,1,2,2,2]
    Args:
        spatial_reps (array): number of spatial reps
        K (int): number of schmidt modes per spatial mode
    Returns:
        array: reps of schmidt modes
    """

    # M = len(spatial_reps)
    # schmidt_reps = np.empty(M * K, dtype=spatial_reps.dtype)
    # for i, r in enumerate(spatial_reps):
    #     schmidt_reps[i * K : (i + 1) * K] = r

    # return schmidt_reps
    return np.repeat(spatial_reps, K)


@numba.jit(nopython=True, cache=True)
def nb_Qmat(cov, hbar=2):  # pragma: no cover
    r"""Numba compatible version of `thewalrus.quantum.Qmat`
    Returns the :math:`Q` Husimi matrix of the Gaussian state.
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

    x = cov[:N, :N] * (2.0 / hbar)
    xp = cov[:N, N:] * (2.0 / hbar)
    p = cov[N:, N:] * (2.0 / hbar)
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    Q = nb_block(((aidaj, aiaj.conj()), (aiaj, aidaj.conj()))) + np.identity(2 * N)
    return Q


@numba.jit(nopython=True, cache=True)
def nb_block(X):  # pragma: no cover
    """Numba implementation of `np.block`.
    Only suitable for 2x2 blocks.
    Taken from: https://stackoverflow.com/a/57562911
    Args:
        X (array) : arrays for blocks of matrix
    Return:
        array : the block matrix from X
    """
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)


@numba.jit(nopython=True, parallel=True, cache=True)
def project_onto_local_oscillator(cov, M, LO_overlap=None, hbar=2):
    """Projects a given covariance matrix into the relevant internal mode in the first external mode.

    Args:
        cov (array): 2MK x 2MK covariance matrix
        LO_overlap (array): overlap between internal modes and local oscillator

    Returns:
        (array): projected covariance matrix
    """

    K = cov.shape[0] // (2 * M)

    # filter out all unwanted Schmidt modes in heralded spatial mode

    # create passive transformation of filter
    T = np.zeros((M * K, M * K), dtype=np.complex128)
    if LO_overlap is not None:
        T[0][:K] = LO_overlap
    else:
        T[0, 0] = 1
    T[K:, K:] = np.eye((M - 1) * K, dtype=np.complex128)

    # apply channel of filter
    P = nb_block(((T.real, -T.imag), (T.imag, T.real)))
    L = (hbar / 2) * (np.eye(P.shape[0]) - P @ P.T)
    cov = P @ cov @ P.T + L

    return cov
