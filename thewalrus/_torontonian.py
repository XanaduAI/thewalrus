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
Torontonian Python interface
"""
import numpy as np
import numba
from thewalrus.quantum.conversions import Xmat, Qmat, Amat
from ._hafnian import reduction, find_kept_edges, nb_ix


def tor(A):
    """Returns the Torontonian of a matrix.

    Args:
        A (array): a square array of even dimensions.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if matshape[0] % 2 != 0:
        raise ValueError("matrix dimension must be even")

    return numba_tor(A)

def ltor(A, gamma):
    """Returns the loop Torontonian of an NxN matrix and an N-length vector.

    Args:
        A (array): an NxN array of even dimensions.
        gamma (array): an N-length vector of even dimensions

    Returns:
        np.float64 or np.complex128: the loop torontonian of matrix A, vector gamma
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    if not isinstance(gamma, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if matshape[0] != len(gamma):
        raise ValueError("gamma must be a vector matching the dimension of A")

    if matshape[0] % 2 != 0:
        raise ValueError("matrix dimension must be even")

    return numba_ltor(A, gamma)

def threshold_detection_prob(
    mu, cov, det_pattern, hbar=2, atol=1e-10, rtol=1e-10
):  # pylint: disable=too-many-arguments
    r"""Threshold detection probabilities for Gaussian states.
    Formula from Jake Bulmer, Nicolas Quesada and Stefano Paesani.
    When state is displaced, threshold_detection_prob_displacement is called.
    Otherwise, tor is called.

    Args:
        mu (1d array) : means of xp Gaussian Wigner function
        cov (2d array) : : xp Wigner covariance matrix
        det_pattern (1d array) : array of {0,1} to describe the threshold detection outcome
        hbar (float): the value of :math:`\hbar` in the commutation relation :math:`[\x,\p]=i\hbar`.
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        np.float64 : probability of detection pattern
    """

    n = cov.shape[0] // 2

    if np.allclose(mu, 0, atol=atol, rtol=rtol):
        # no displacement
        n_modes = cov.shape[0] // 2
        Q = Qmat(cov, hbar)
        O = np.eye(2*n) - np.linalg.inv(Q)
        rpt2 = np.concatenate((det_pattern, det_pattern))
        Os = reduction(O, rpt2)
        return tor(Os) / np.sqrt(np.linalg.det(Q))

    x = mu[:n]
    p = mu[n:]
    alpha = np.concatenate((x + 1j * p, x - 1j * p)) / np.sqrt(2 * hbar)
    
    sigma = Qmat(cov, hbar=hbar)
    I = np.eye(2 * n)
    inv_sigma = np.linalg.inv(sigma)
    O = I - inv_sigma
    gamma = inv_sigma @ alpha
    
    gamma_red, O_red = reduced_gaussian(gamma, O, np.where(np.array(det_pattern) == 1)[0])
    return vac_prob(alpha, sigma) * numba_ltor(O_red, gamma_red).real


@numba.jit(nopython=True)
def numba_tor(O):  # pragma: no cover
    r"""Returns the Torontonian of a matrix using numba.

    Args:
        O (array): a square, symmetric array of even dimensions.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    N = O.shape[0] // 2
    N_odd = N % 2
    
    steps = 2 ** N
    ones = np.ones(N, dtype=np.int8)
    
    tor = 0.
    for j in numba.prange(steps):
        X_modes = find_kept_edges(j, ones)
        lenX = X_modes.sum()
        I = np.eye(2 * lenX, dtype=O.dtype)
        plusminus = (-1) ** ((N_odd - lenX % 2) % 2)
        
        kept_modes = np.where(X_modes != 0)[0]
        kept_rows = np.concatenate((kept_modes, kept_modes + N))
        O_XX = nb_ix(O, kept_rows, kept_rows)
        
        bottom = np.sqrt(np.linalg.det(I_m_O_XX))
        
        tor += plusminus / bottom
    
    return tor

@numba.jit(nopython=True)
def vac_prob(alpha, sigma): # pragma: no cover
    r"""
    Return the vacuum probability of a Gaussian state with Q function sigma
    and displacement vector, alpha.

    Args:
        alpha (array): a 2M-length vector describing the complex displacement
        sigma (array): a 2Mx2M matrix describing the Q-function covariance matrix
    Returns:
        float: vacuum probability of Gaussian state
    """
    return (np.exp(-0.5 * alpha.conj() @ np.linalg.inv(sigma) @ alpha).real / np.sqrt(np.linalg.det(sigma))).real

@numba.jit(nopython=True, parallel=True)
def numba_ltor(O, gamma): # pragma: no cover
    r"""Returns the loop Torontonian of a matrix using numba.

    Args:
        O (array): a square, symmetric array of even dimensions.
        gamma (array): a vector of even dimension

    Returns:
        np.float64 or np.complex128: the loop torontonian of matrix O, vector gamma
    """
    N = O.shape[0] // 2
    N_odd = N % 2
    
    steps = 2 ** N
    ones = np.ones(N, dtype=np.int8)
    
    ltor = 0.
    for j in numba.prange(steps):
        X_modes = find_kept_edges(j, ones)
        lenX = X_modes.sum()
        I = np.eye(2 * lenX, dtype=O.dtype)
        plusminus = (-1) ** ((N_odd - lenX % 2) % 2)
        
        kept_modes = np.where(X_modes != 0)[0]
        kept_rows = np.concatenate((kept_modes, kept_modes + N))
        O_XX = nb_ix(O, kept_rows, kept_rows)
        
        I_m_O_XX = I - O_XX
        I_m_O_XX_inv = np.linalg.inv(I_m_O_XX)
        
        gamma_X = gamma[kept_rows]
        top = np.exp(0.5 * gamma_X.conj() @ I_m_O_XX_inv @ gamma_X)
        bottom = np.sqrt(np.linalg.det(I_m_O_XX))
        
        ltor += plusminus * top / bottom
    
    return ltor
