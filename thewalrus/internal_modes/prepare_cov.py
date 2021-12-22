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

import numpy as np
from strawberryfields.decompositions import takagi
from ..quantum import fidelity
from ..symplectic import interferometer, reduced_state, squeezing, passive_transformation


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
    out_modes = list(chain.from_iterable(range(i, M*R, M) for i in range(M)))
    P[out_modes, in_modes] = 1
    return P

def vacuum_state(M, hbar=2):
    r"""
    Returns the displacement vector and covariance matrix for an M mode vacuum state.

    Args:
        M (int): number of modes (spatial, Schmidt, other etc.)
        hbar (float): (default 2) the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`

    Returns:
        tuple[array]: the displacement vector and covariance matrix of an M mode vacuum state
    """

    return np.zeros(2 * M), np.identity(2 * M) * hbar / 2

def orthonormal_basis(O, rjs, thr=1e-3):
    r"""
    Finds an orthonormal basis in operator space based on the total number of Schmidt modes in the whole system of multiple spatial modes.
    Using the overlap matrix and the squeezing parameters of each Schmidt mode in each waveguide it finds R orthonormal modes for each spatial mode,
    where R is less than or equal to the total combined number of Schmidt modes in the whole system of multiple spatial modes.

    Args:
        O (array): 2-dimensional matrix of the overlaps between each Schmidt mode in all spatial modes combined
        rjs (list[list]): list for each spatial mode of list of squeezing parameters in that spatial mode

    Returns:
        tuple (eps, W): eps is a list of arrays, in each array is the effective squeezing parameters for the orthonormal modes in each spatial mode
                        W is a list of arrays, each array is an R x R matrix for each spatial mode for the amplitude of each orthonormal mode.
    """

    if not np.allclose(O, O.conj().T):
        raise ValueError("O must be a Hermitian matrix")
    if not np.allclose(len(O), sum([len(listElem) for listElem in rjs])):
        raise ValueError(
            "Length of O must equal the total number of Schmidt modes accross all spatial modes"
        )
    R = len(O)
    M = len(rjs)
    njs = np.array([len(rjs[i]) for i in range(M)])
    lambd, V = np.linalg.eigh(O)
    X = np.fliplr(np.identity(len(lambd)))
    lambd, V = X @ lambd, V @ X
    inds = np.arange(len(lambd))[lambd > thr]
    lambd = lambd[inds]
    R = len(lambd)
    eps = []
    W = []
    for j in range(M):
        Rtemp = np.identity(R, dtype=np.complex128)
        for l in range(R):
            for m in range(R):
                Rtemp[l, m] = np.sum(
                    np.array(
                        [
                            rjs[j][k]
                            * V[indexconversion(j, k, njs), l].conj()
                            * V[indexconversion(j, k, njs), m].conj()
                            * np.sqrt(lambd[l])
                            * np.sqrt(lambd[m])
                            for k in range(njs[j])
                        ]
                    )
                )
        eps_temp, WT_temp = takagi(Rtemp)
        eps.append(eps_temp)
        W.append(WT_temp.T)
    return eps, W

def state_prep(eps, W, thresh=1e-2, hbar=2.0):
    r"""
    Computes the total covariance matrix (assuming zero displacement) of the initial state as determined by orthonormalization parameters.
    Modes are ordered first by orthonormalization modes, then spatial modes, i.e. for R orthonormalization modes the 1st R modes of the
    covariance matrix are for the first spatial mode, the next R lot of modes for the 2nd spatial mode, etc. If an orthonormal mode has
    a fidelity with vacuum of 1 - thresh or higher then it is traced over

    Args:
        eps (list[array]): list of arrays of squeezing parameters for each spatial mode as determined by the orthonormalization procedure
        W (list): list of arrays of the orthonormalization amplitude matrix, one for each spatial mode

    Returns:
        (array): covariance matrix of the output state
    """
    if not np.allclose(len(eps), len(W)):
        raise ValueError("len of eps must be equal to len of W")
    epsBig = np.array([])
    for i in eps:
        epsBig = np.append(epsBig, i)

    M = len(eps)
    R = len(eps[0])

    Qvac = (hbar / 2) * np.identity(2 * M * R)
    S = squeezing(epsBig)
    Qinit = S @ Qvac @ S.T

    Wbig = np.identity(M * R, dtype=np.complex128)
    for j in range(M):
        Wbig[R * j : R * (j + 1), R * j : R * (j + 1)] = W[j]

    Q = interferometer(Wbig.T.conj()) @ Qinit @ interferometer(Wbig.T.conj()).T
    Qswap = interferometer(swap_matrix(M, R)) @ Q @ interferometer(swap_matrix(M, R)).T
    muVac, covVac = vacuum_state(M, hbar=hbar)

    # Getting rid of any system of orthonormal modes (i.e. in M spatial modes) that are very close to an M-mode vacuum state
    if R > 1:  
        keep_modes = np.array([])
        for k in range(R):
            muTemp, QTemp = reduced_state(
                np.zeros(len(Qswap)), Qswap, np.arange(M * k, M * (k + 1))
            )
            if 1 - fidelity(muVac, covVac, muTemp, QTemp) > thresh:
                keep_modes = np.append(keep_modes, np.arange(M * k, M * (k + 1)))

        keep_modes = keep_modes.astype(int)
        keep_modes = keep_modes.tolist()

        R = int(len(keep_modes) / M)
        _, Qswap = reduced_state(np.zeros(len(Qswap)), Qswap, keep_modes)

    return interferometer(swap_matrix(M, R).T) @ Qswap @ interferometer(swap_matrix(M, R).T).T

def prepare_cov(rjs, O, T, thresh=1e-2, hbar=2.):
    """
    prepare multimode covariance matrix using Lowdin orthonormalisation

    Lowdin modes which have a fidelity to vacuum of 1-thresh are traced over

    Args:
        rjs (list[list/array]): list for each spatial mode of list/array of squeezing parameters for each Schmidt mode in that spatial mode
        O (array): 2-dimensional matrix of the overlaps between each Schmidt mode in all spatial modes combined
        T (array): (unitary if lossless) matrix expressing the spatial mode interferometer
        thresh(float): threshold for ignoring states (default 1e-2)
        hbar (float): the value of hbar (default 2.0)

    Returns:
        array[:,:]: covariance matrix over all spatial modes and (Lowdin) internal modes
    """
    # TODO: also return first Lowdin mode

    if not np.allclose(len(T), len(rjs)):
        raise ValueError("Unitary is the wrong size, it must act on all spatial modes")
    if not np.allclose(len(O), sum([len(listElem) for listElem in rjs])):
        raise ValueError(
            "Length of O must equal the total number of Schmidt modes accross all spatial modes"
        )
    s = np.linalg.svd(T, compute_uv=False)
    if max(s) > 1:
        raise ValueError("T must be have singular values <= 1")

    eps, W = orthonormal_basis(O, rjs)
    Qinit = state_prep(eps, W, thresh=thresh, hbar=hbar)

    mu = np.zeros(Qinit.shape[0])
    mu, Qu = passive_transformation(mu, Qinit, T, hbar=hbar)

    return Qu

