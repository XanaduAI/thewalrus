# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Tests for various properties of covariance matrices as well as fidelity
calculations for Gaussian states.
"""
# pylint: disable=too-many-arguments

import numpy as np
from scipy.linalg import sqrtm

from ..symplectic import sympmat


def is_valid_cov(cov, hbar=2, rtol=1e-05, atol=1e-08):
    r"""Checks if the covariance matrix is a valid quantum covariance matrix.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

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
    r"""Checks if the covariance matrix is a valid quantum covariance matrix
    that corresponds to a quantum pure state

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (bool): whether the given covariance matrix corresponds to a pure state
    """
    if is_valid_cov(cov, hbar=hbar, rtol=rtol, atol=atol):
        purity = 1 / np.sqrt(np.linalg.det(2 * cov / hbar))
        if np.allclose(purity, 1.0, rtol=rtol, atol=atol):
            return True

    return False


def is_classical_cov(cov, hbar=2, atol=1e-08):
    r"""Checks if the covariance matrix can be efficiently sampled.

    Args:
        cov (array): a covariance matrix
        hbar (float): value of hbar in the uncertainty relation
        atol (float): the absolute tolerance parameter used in `np.allclose`

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


def fidelity(mu1, cov1, mu2, cov2, hbar=2, rtol=1e-05, atol=1e-08):
    r"""Calculates the fidelity between two Gaussian quantum states.
    For two pure states :math:`|\psi_1 \rangle,  \  |\psi_2 \rangle`
    the fidelity is given by :math:`|\langle \psi_1|\psi_2 \rangle|^2`

    Note that if the covariance matrices correspond to pure states this
    function reduces to the modulus square of the overlap of their state vectors.
    For the derivation see  `'Quantum Fidelity for Arbitrary Gaussian States', Banchi et al. <10.1103/PhysRevLett.115.260501>`_.

    The actual implementation used here corresponds to the *square* of Eq. 96 of
    `'Gaussian states and operations - a quick reference', Brask <https://arxiv.org/abs/2102.05748>`_.

    Args:
        mu1 (array): vector of means of the first state
        cov1 (array): covariance matrix of the first state
        mu2 (array): vector of means of the second state
        cov2 (array): covariance matrix of the second state
        hbar (float): value of hbar in the uncertainty relation
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`

    Returns:
        (float): value of the fidelity between the two states
    """

    n0, n1 = cov1.shape
    m0, m1 = cov2.shape
    (l0,) = mu1.shape
    (l1,) = mu1.shape
    if not n0 == n1 == m0 == m1 == l0 == l1:
        raise ValueError("The inputs have incompatible shapes")

    # We first convert all the inputs to quantities where hbar = 1
    sigma1 = cov1 / hbar
    sigma2 = cov2 / hbar
    deltar = (mu1 - mu2) / np.sqrt(hbar)

    omega = sympmat(n0 // 2)  # The symplectic matrix

    sigma = sigma1 + sigma2
    sigma_inv = np.linalg.inv(sigma)
    vaux = omega.T @ sigma_inv @ (0.25 * omega + sigma2 @ omega @ sigma1)
    sqrtm_arg = np.identity(n0) + 0.25 * np.linalg.inv(vaux @ omega @ vaux @ omega)

    # The sqrtm function has issues with matrices that are close to zero, hence we branch
    if np.allclose(sqrtm_arg, 0, rtol=rtol, atol=atol):
        mat_sqrtm = np.zeros_like(sqrtm_arg)
    else:
        mat_sqrtm = sqrtm(sqrtm_arg)

    det_arg = 2 * (mat_sqrtm + np.identity(n0)) @ vaux
    f = np.sqrt(np.linalg.det(sigma_inv) * np.linalg.det(det_arg)) * np.exp(
        -0.5 * deltar @ sigma_inv @ deltar
    )
    # Note that we only take the square root and that we have a prefactor of 0.5
    # as opposed to 0.25 in Brask. This is because this function returns the square
    # of their fidelities.
    return f


def is_symplectic(S, rtol=1e-05, atol=1e-08):
    """Checks if the matrix is symplectic.

    Args:
        S (array): a matrix
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``

    Returns:
        bool: whether the given matrix is symplectic
    """
    (n, m) = S.shape
    if n != m:
        return False
    if n % 2 != 0:
        return False

    n = n // 2
    A = S[0:n, 0:n]
    B = S[0:n, n : 2 * n]
    C = S[n : 2 * n, 0:n]
    D = S[n : 2 * n, n : 2 * n]
    # The equations below are equivalent to S.T @ Omega @ S = Omega where Omega is the symplectic form
    if (
        np.allclose(A.T @ C, C.T @ A, rtol=rtol, atol=atol)
        and np.allclose(B.T @ D, D.T @ B, rtol=rtol, atol=atol)
        and np.allclose(A.T @ D - C.T @ B, np.eye(n), rtol=rtol, atol=atol)
    ):
        return True

    return False
