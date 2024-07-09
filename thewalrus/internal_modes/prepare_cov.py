# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

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
Set of functions for forming a covariance matrix over multiple modes, based on overlaps of the squeezer schmidt modes
"""

from itertools import chain

import numpy as np

from ..decompositions import takagi
from ..quantum import fidelity
from ..symplectic import (
    interferometer,
    passive_transformation,
    reduced_state,
    squeezing,
    vacuum_state,
)


def indexconversion(j, k, njs):
    r"""
    Converts the double index (j, k) into a single index = k + \sum_{l=0}^{j-1} njs[l],
    with njs[0] defined as 0, njs[1] as the number of Schmidt modes in the first spatial mode, etc.

    Args:
        j (int): spatial mode index
        k (int): Schmidt mode index
        njs (array): number of Schmidt modes in each

    Returns:
        (int): converted index
    """
    return k + np.sum(njs[:j])


def swap_matrix(M, R):
    r"""
    Computes the matrix that swaps the ordering of modes from grouped by orthonormal modes to grouped by spatial modes.
    The inverse of the swap matrix is its transpose.

    Args:
        M (int): number of spatial modes
        R (int): number of orthonormal modes in each spatial mode

    Returns:
        (array): M*R x M*R swap matrix
    """
    P = np.zeros((M * R, M * R), dtype=int)
    in_modes = range(M * R)
    out_modes = list(chain.from_iterable(range(i, M * R, M) for i in range(M)))
    P[out_modes, in_modes] = 1
    return P


def O_matrix(F):
    r"""The overlap matrix of all of the Schmidt modes in each spatial mode.

    Args:
        F (list[list/array[array]]): List of arrays of the temporal modes of the Schmidt modes in each spatial mode, must be normalized and ordered by spatial mode.
                                     Sufficient to put the output temporal modes from yellowsubmaring in a list ordered by spatial mode.
    Returns:
        (array): the overlap matrix
    """
    Fs = [f for fj in F for f in fj]
    R = len(Fs)
    O = np.eye(R, dtype=np.complex128)
    for j in range(R):
        for k in range(R):
            O[j, k] = np.inner(Fs[j].conj() / np.linalg.norm(Fs[j]), Fs[k] / np.linalg.norm(Fs[k]))
    return O


# pylint: disable=too-many-branches
def orthonormal_basis(rjs, O=None, F=None, thr=1e-3):
    r"""
    Finds an orthonormal basis in operator space based on the total number of Schmidt modes in the whole system of multiple spatial modes.
    Using the overlap matrix and the squeezing parameters of each Schmidt mode in each waveguide it finds R orthonormal modes for each spatial mode,
    where R is less than or equal to the total combined number of Schmidt modes in the whole system of multiple spatial modes.

    Args:
        rjs (list[list]): list for each spatial mode of list of squeezing parameters in that spatial mode
        O (array): 2-dimensional matrix of the overlaps between each Schmidt mode in all spatial modes combined
        F (list[list/array[array]]): List of arrays of the temporal modes of the Schmidt modes in each spatial mode, must be normalized and ordered by spatial mode.
                                     Sufficient to put the output temporal modes from yellowsubmaring in a list ordered by spatial mode.
        thr (float): eigenvalue threshold under which orthonormal mode is discounted

    Returns:
        tuple (chis, eps, W): chis is a list of the temporal functions for the new orthonormal basis, only returned when F is given
                              eps is a list of arrays, in each array is the effective squeezing parameters for the orthonormal modes in each spatial mode
                              W is a list of arrays, each array is an R x R matrix for each spatial mode for the amplitude of each orthonormal mode.
    """
    rs = np.array([r for rj in rjs for r in rj])
    if F is not None:
        Fs = [f for fj in F for f in fj]
        Fs = [Fs[j] / np.linalg.norm(Fs[j]) for j in range(len(Fs))]
        if not np.allclose(len(Fs), rs.shape[0]):
            raise ValueError(
                "Length of F must equal the total number of Schmidt modes accross all spatial modes"
            )
        if O is not None:
            if not np.allclose(O, O_matrix(F)):
                raise ValueError("Both O and F were given but are not compatible")
        else:
            O = O_matrix(F)
    elif O is not None:
        if not np.allclose(O, O.conj().T):
            raise ValueError("O must be a Hermitian matrix")
        if not np.allclose(O.shape[0], rs.shape[0]):
            raise ValueError(
                "Length of O must equal the total number of Schmidt modes accross all spatial modes"
            )
    else:
        raise ValueError("Either F or O must be given")
    R = O.shape[0]
    M = len(rjs)
    njs = np.array([len(rj) for rj in rjs])
    lambd, V = np.linalg.eigh(np.outer(np.sqrt(rs).conj(), np.sqrt(rs)) * O)
    indices = np.argsort(-lambd)
    lambd, V = lambd[indices], V[:, indices]
    V = np.real_if_close(V / np.exp(1j * np.angle(V)[0]))
    inds = np.arange(lambd.shape[0])[lambd > thr]
    lambd = lambd[inds]
    R = lambd.shape[0]
    eps = []
    W = []
    for j in range(M):
        Rtemp = np.eye(R, dtype=np.complex128)
        for l in range(R):
            for m in range(R):
                Rtemp[l, m] = np.sum(
                    np.array(
                        [
                            V[indexconversion(j, k, njs), l].conj()
                            * V[indexconversion(j, k, njs), m].conj()
                            * np.sqrt(lambd[l])
                            * np.sqrt(lambd[m])
                            for k in range(njs[j])
                        ]
                    )
                )
        eps_temp, WT_temp = takagi(Rtemp)
        signs = np.sign(WT_temp.real)[0]
        for j, s in enumerate(signs):
            if np.allclose(s, 0):
                signs[j] = 1
        WT_temp /= signs
        eps.append(eps_temp)
        W.append(np.real_if_close(WT_temp).T)
    if F is not None:
        chis = []
        for k in range(R):
            chi = np.sum(
                np.array(
                    [np.sqrt(rs[j]) * Fs[j] * V[j, k] / np.sqrt(lambd[k]) for j in range(len(Fs))]
                ),
                axis=0,
            )
            chis.append(chi / np.linalg.norm(chi))
        return chis, eps, W
    return eps, W


def state_prep(eps, W, thresh=1e-4, hbar=2):
    r"""
    Computes the total covariance matrix (assuming zero displacement) of the initial state as determined by orthonormalization parameters.
    Modes are ordered first by orthonormalization modes, then spatial modes, i.e. for R orthonormalization modes the 1st R modes of the
    covariance matrix are for the first spatial mode, the next R lot of modes for the 2nd spatial mode, etc. If an orthonormal mode has
    a fidelity with vacuum of 1 - thresh or higher then it is traced over, unless every mode is like that then the whole system is returned.

    Args:
        eps (list[array]): list of arrays of squeezing parameters for each spatial mode as determined by the orthonormalization procedure
        W (list[array]): list of arrays of the orthonormalization amplitude matrix, one for each spatial mode

    Returns:
        (array): covariance matrix of the output state
    """
    if not np.allclose(len(eps), len(W)):
        raise ValueError("len of eps must be equal to len of W")
    epsBig = np.concatenate(eps, axis=0)

    M = len(eps)
    R = eps[0].shape[0]

    [_, covvac] = vacuum_state(M * R, hbar=hbar)
    S = squeezing(epsBig)
    covinit = S @ covvac @ S.T

    Wbig = np.eye(M * R, dtype=np.complex128)
    for j in range(M):
        Wbig[R * j : R * (j + 1), R * j : R * (j + 1)] = W[j]

    cov = interferometer(Wbig.T.conj()) @ covinit @ interferometer(Wbig.T.conj()).T

    # Getting rid of any system of orthonormal modes (i.e. in M spatial modes) that are very close to an M-mode vacuum state
    if R > 1:
        covswap = interferometer(swap_matrix(M, R)) @ cov @ interferometer(swap_matrix(M, R)).T
        [muVac, covVac] = vacuum_state(M, hbar=hbar)
        keep_modes = np.array([])
        for k in range(R):
            muTemp, covTemp = reduced_state(
                np.zeros(covswap.shape[0]), covswap, np.arange(M * k, M * (k + 1))
            )
            if 1 - fidelity(muVac, covVac, muTemp, covTemp) > thresh:
                keep_modes = np.append(keep_modes, np.arange(M * k, M * (k + 1)))

        keep_modes = keep_modes.astype(int)
        keep_modes = keep_modes.tolist()

        if len(keep_modes) > 0:
            R = len(keep_modes) // M
            _, covswap = reduced_state(np.zeros(covswap.shape[0]), covswap, keep_modes)

        cov = interferometer(swap_matrix(M, R).T) @ covswap @ interferometer(swap_matrix(M, R).T).T

    return cov


def LO_overlaps(chis, LO_shape):
    r"""
    Computes the overlap integral between the orthonormal moes and the local oscillator shape

    Args:
        chis (list[array]): list of the temporal functions for the new orthonormal basis
        LO_shape (array): temporal profile of local oscillator

    Returns:
        array: overlaps between temporal mode shapes and local oscillator
    """
    return np.array(
        [
            np.inner(LO_shape.conj() / np.linalg.norm(LO_shape), chi / np.linalg.norm(chi))
            for chi in chis
        ]
    )


# pylint: disable=too-many-arguments
def prepare_cov(rjs, T, O=None, F=None, thr=1e-3, thresh=1e-4, hbar=2):
    """
    prepare multimode covariance matrix using Lowdin orthonormalisation
    Lowdin modes which have a fidelity to vacuum of 1-thresh are traced over
    Args:
        rjs (list[list/array]): list for each spatial mode of list/array of squeezing parameters for each Schmidt mode in that spatial mode
        T (array): (unitary if lossless) matrix expressing the spatial mode interferometer
        O (array): 2-dimensional matrix of the overlaps between each Schmidt mode in all spatial modes combined
        F (list[list/array[array]]): List of arrays of the temporal modes of the Schmidt modes in each spatial mode, must be normalized and ordered by spatial mode.
                                     Sufficient to put the output temporal modes from yellowsubmaring in a list ordered by spatial mode.
        thr (float): eigenvalue threshold under which orthonormal mode is discounted
        thresh(float): threshold for ignoring states (default 1e-4)
        hbar (float): the value of hbar (default 2)
    Returns:
        tuple(array[:, :], list[array]): covariance matrix over all spatial modes and (Lowdin) internal modes
                                         if temporal modes are given, the orthonormal modes are returned

    """
    if not np.allclose(T.shape[0], len(rjs)):
        raise ValueError("Unitary is the wrong size, it must act on all spatial modes")
    s = np.linalg.svd(T, compute_uv=False)
    if not (max(s) <= 1 or np.allclose(max(s), 1)):
        raise ValueError("T must be have singular values <= 1")
    if O is not None:
        eps, W = orthonormal_basis(rjs, O=O, thr=thr)
    elif F is not None:
        chis, eps, W = orthonormal_basis(rjs, F=F, thr=thr)
    else:
        raise ValueError("Either O or F must be supplied")
    covinit = state_prep(eps, W, thresh=thresh, hbar=hbar)

    M = T.shape[0]
    K = covinit.shape[0] // (2 * M)
    T_K = np.zeros((M * K, M * K), dtype=np.complex128)
    for i in range(K):
        T_K[i::K, i::K] = T
    _, covu = passive_transformation(np.zeros(covinit.shape[0]), covinit, T_K, hbar=hbar)

    if F is not None:
        return covu, chis
    return covu
