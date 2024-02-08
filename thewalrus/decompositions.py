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
Decompositions
==============

**Module name:** :mod:`thewalrus.decompositions`

.. currentmodule:: thewalrus.decompositions

This module implements common shared matrix decompositions that are
used to perform gate decompositions.

Summary
-------

.. autosummary::
    williamson
    symplectic_eigenvals
    blochmessiah
    takagi
    pre_iwazawa
    iwazawa

Code details
------------
"""
import numpy as np

from scipy.linalg import sqrtm, schur, polar, qr
from thewalrus.symplectic import sympmat
from thewalrus.quantum.gaussian_checks import is_symplectic


def williamson(V, rtol=1e-05, atol=1e-08):
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See `this thread <https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630>`_
    and the `Williamson decomposition documentation <https://strawberryfields.ai/photonics/conventions/decompositions.html#williamson-decomposition>`_

    Args:
        V (array[float]): positive definite symmetric (real) matrix
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S Db S^T`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    if not np.allclose(V, V.T, rtol=rtol, atol=atol):
        raise ValueError("The input matrix is not symmetric")

    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = sympmat(n)
    vals = np.linalg.eigvalsh(V)

    if not np.all(vals > 0):
        raise ValueError("Input matrix is not positive definite")

    M12 = np.real_if_close(sqrtm(V))
    Mm12 = np.linalg.inv(M12)
    Gamma = Mm12 @ omega @ Mm12
    a, O = schur(Gamma)
    # In what follows a permutation matrix perm is constructed so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus the permutation perm is updated
    # to go to the ordering x_1, ..., x_n, p_1, ... , p_n
    perm = np.arange(2 * n)
    for i in range(n):
        if a[2 * i, 2 * i + 1] <= 0:
            (perm[2 * i], perm[2 * i + 1]) = (perm[2 * i + 1], perm[2 * i])

    perm = np.array([perm[2 * i] for i in range(n)] + [perm[2 * i + 1] for i in range(n)])

    O = O[:, perm]
    phi = np.abs(np.diag(a, k=1)[::2])
    dd = np.concatenate([1 / phi, 1 / phi])
    ddsqrt = 1 / np.sqrt(dd)
    S = M12 @ O * ddsqrt
    return np.diag(dd), S


def symplectic_eigenvals(cov):
    r"""Returns the symplectic eigenvalues of a covariance matrix.

    Args:
        cov (array): a covariance matrix

    Returns:
        (array): symplectic eigenvalues
    """
    M = len(cov) // 2
    Omega = sympmat(M)
    return np.real_if_close(-1j * np.linalg.eigvals(Omega @ cov))[::2]


def blochmessiah(S):
    """Returns the Bloch-Messiah decomposition of a symplectic matrix S = O @ D @ Q
       where O and Q are orthogonal symplectic matrices and D is a positive-definite diagonal matrix
       of the form diag(d1,d2,...,dn,d1^-1, d2^-1,...,dn^-1),

    Args:
        S (array[float]): 2N x 2N real symplectic matrix

    Returns:
        tuple(array[float],  : orthogonal symplectic matrix O
              array[float],  : diagonal matrix D
              array[float])  : orthogonal symplectic matrix Q
    """

    N, _ = S.shape

    if not is_symplectic(S):
        raise ValueError("Input matrix is not symplectic.")
    N = N // 2
    V, P = polar(S, side="left")
    A = P[:N, :N]
    B = P[:N, N:]
    C = P[N:, N:]
    M = A - C + 1j * (B + B.T)
    Lam, W = takagi(M)
    Lam = 0.5 * Lam
    O = np.block([[W.real, -W.imag], [W.imag, W.real]])
    Q = O.T @ V
    sqrt1pLam2 = np.sqrt(1 + Lam**2)
    D = np.diag(np.concatenate([sqrt1pLam2 + Lam, sqrt1pLam2 - Lam]))
    return O, D, Q


def takagi(A, svd_order=True):
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.
    Note that the input matrix is internally symmetrized by taking its upper triangular part.
    If the input matrix is indeed symmetric this leaves it unchanged.
    See `Carl Caves note. <http://info.phys.unm.edu/~caves/courses/qinfo-s17/lectures/polarsingularAutonne.pdf>`_

    Args:
        A (array): square, symmetric matrix
        svd_order (boolean): whether to return result by ordering the singular values of ``A`` in descending (``True``) or ascending (``False``) order.

    Returns:
        tuple[array, array]: (r, U), where r are the singular values,
        and U is the Autonne-Takagi unitary, such that :math:`A = U \diag(r) U^T`.
    """

    n, m = A.shape
    if n != m:
        raise ValueError("The input matrix is not square")
    # Here we build a Symmetric matrix from the top right triangular part
    A = np.triu(A) + np.triu(A, k=1).T

    A = np.real_if_close(A)

    if np.allclose(A, 0):
        return np.zeros(n), np.eye(n)

    if np.isrealobj(A):
        # If the matrix A is real one can be more clever and use its eigendecomposition
        ls, U = np.linalg.eigh(A)
        vals = np.abs(ls)  # These are the Takagi eigenvalues
        signs = (-1) ** (1 + np.heaviside(ls, 1))
        phases = np.sqrt(np.complex128(signs))
        Uc = U * phases  # One needs to readjust the phases
        # Find the permutation to sort in decreasing order
        perm = np.argsort(vals)
        # if svd_order reverse it
        if svd_order:
            perm = perm[::-1]
        return vals[perm], Uc[:, perm]

    # Find the element with the largest absolute value
    pos = np.unravel_index(np.argmax(np.abs(A)), (n, n))
    # Use it to find whether the input is a global phase times a real matrix
    phi = np.angle(A[pos])
    Amr = np.real_if_close(np.exp(-1j * phi) * A)
    if np.isrealobj(Amr):
        vals, U = takagi(Amr, svd_order=svd_order)
        return vals, U * np.exp(1j * phi / 2)

    u, d, v = np.linalg.svd(A)
    U = u @ sqrtm((v @ np.conjugate(u)).T)
    if svd_order is False:
        return d[::-1], U[:, ::-1]
    return d, U


def pre_iwazawa(S):
    """Pre-Iwazawa decomposition of a symplectic matrix.
    See `Arvind et al. The Real Symplectic Groups in Quantum Mechanics and Optics <https://arxiv.org/pdf/quant-ph/9509002.pdf>`_


    Args:
        S (array): the symplectic matrix

    Returns:
        tuple[array, array, array]: (E,D,F) symplectic matrices such that E @ D @ F = S and,
        E = np.block([[np.eye(N), np.zeros(N,N)],[X, np.eye(N)]]) with X == X.T,
        D is block diagonal with the top left block being the inverse of the bottom right block,
        F is symplectic orthogonal.
    """

    if not is_symplectic(S):
        raise ValueError("Input matrix is not symplectic.")

    N, _ = S.shape
    N = N // 2
    zerom = np.zeros([N, N])
    idm = np.eye(N)
    A = S[:N, :N]
    B = S[:N, N:]
    C = S[N:, :N]
    D = S[N:, N:]
    A0 = sqrtm(A @ A.T + B @ B.T)
    A0inv = np.linalg.inv(A0)
    U = A0inv @ (A + 1j * B)
    X = U.real
    Y = U.imag
    C0 = (C @ A.T + D @ B.T) @ A0inv
    E = np.block([[idm, zerom], [C0 @ A0inv, idm]])
    D = np.block([[A0, zerom], [zerom, A0inv]])
    F = np.block([[X, Y], [-Y, X]])
    return E, D, F


def iwazawa(S):
    """Iwazawa decomposition of a symplectic matrix.
    See `Arvind et al. The Real Symplectic Groups in Quantum Mechanics and Optics <https://arxiv.org/pdf/quant-ph/9509002.pdf>`_


    Args:
        S (array): the symplectic matrix

    Returns:
        tuple[array, array, array]: (E,D,F) symplectic matrices such that E @ D @ F = S,
        EE = np.block([[AA, np.zeros(N,N)],[CC, np.linalg.inv(A.T)]]) with A.T @ C == C.T @ A, and AA upper trinagular with ones in the diagonal
        DD is diagonal and symplectic,
        FF is symplectic orthogonal.
    """

    E, D, F = pre_iwazawa(S)
    N, _ = S.shape
    N = N // 2
    DNN = D[:N, :N]
    Q, R = qr(DNN)
    R = R.T
    Q = Q.T
    dR = np.diag(R)
    dd = np.abs(dR)
    ds = np.sign(dR)
    R = R * (1 / dR)
    RinvT = np.linalg.inv(R).T
    DD = np.diag(np.concatenate([dd, 1 / dd]))
    zerom = np.zeros([N, N])
    OO = np.block([[R, zerom], [zerom, RinvT]])
    Q = ds[:, None] * Q
    AA = np.block([[Q, zerom], [zerom, Q]])
    EE = E @ OO
    FF = AA @ F
    return EE, DD, FF
