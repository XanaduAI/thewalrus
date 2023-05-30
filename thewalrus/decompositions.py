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
    autonne
    blochmessiah
    symplectic_eigenvals
    takagi
    williamson

Code details
------------
"""
from itertools import groupby
import numpy as np

from scipy.linalg import block_diag, sqrtm, schur
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
            and ``S`` is a symplectic matrix such that :math:`V = S^T Db S`
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

    for val in vals:
        if val <= 0:
            raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    seq = []

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1,p_1, ..., x_n,p_n  ordering thus I permute using perm
    # to go to the ordering x_1, ..., x_n, p_1, ... , p_n

    for i in range(n):
        if s1[2 * i, 2 * i + 1] > 0:
            seq.append(I)
        else:
            seq.append(X)
    perm = np.array([2 * i for i in range(n)] + [2 * i + 1 for i in range(n)])
    p = block_diag(*seq)
    Kt = K @ p
    Ktt = Kt[:, perm]
    s1t = p @ s1 @ p
    dd = [1 / s1t[2 * i, 2 * i + 1] for i in range(n)]
    Db = np.diag(dd + dd)
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T



def symplectic_eigenvals(cov):
    r"""Returns the symplectic eigenvalues of a covariance matrix.

    Args:
        cov (array): a covariance matrix

    Returns:
        (array): symplectic eigenvalues
    """
    M = int(len(cov) / 2)
    D, _ = williamson(cov)
    return np.diag(D)[:M]


def blochmessiah(S):
    """Returns the Bloch-Messiah decomposition of a symplectic matrix S = uff @ dff @ vff
       where uff and vff are orthogonal symplectic matrices and dff is a diagonal matrix
       of the form diag(d1,d2,...,dn,d1^-1, d2^-1,...,dn^-1),

    Args:
        S (array[float]): 2N x 2N real symplectic matrix

    Returns:
        tuple(array[float],  : orthogonal symplectic matrix uff
              array[float],  : diagonal matrix dff
              array[float])  : orthogonal symplectic matrix vff
    """

    N, _ = S.shape

    if not is_symplectic(S):
        raise ValueError("Input matrix is not symplectic.")

    # Changing Basis
    R = (1 / np.sqrt(2)) * np.block(
        [[np.eye(N // 2), 1j * np.eye(N // 2)], [np.eye(N // 2), -1j * np.eye(N // 2)]]
    )
    Sc = R @ S @ np.conjugate(R).T
    # Polar Decomposition
    u1, d1, v1 = np.linalg.svd(Sc)
    Sig = u1 @ np.diag(d1) @ np.conjugate(u1).T
    Unitary = u1 @ v1
    # Blocks of Unitary and Hermitian symplectics
    alpha = Unitary[0 : N // 2, 0 : N // 2]
    beta = Sig[0 : N // 2, N // 2 : N]
    # Bloch-Messiah in this Basis
    d2, takagibeta = takagi(beta)
    sval = np.arcsinh(d2)
    uf = block_diag(takagibeta, takagibeta.conj())
    blc = np.conjugate(takagibeta).T @ alpha
    vf = block_diag(blc, blc.conj())
    df = np.block(
        [
            [np.diag(np.cosh(sval)), np.diag(np.sinh(sval))],
            [np.diag(np.sinh(sval)), np.diag(np.cosh(sval))],
        ]
    )
    # Rotating Back to Original Basis
    uff = np.conjugate(R).T @ uf @ R
    vff = np.conjugate(R).T @ vf @ R
    dff = np.conjugate(R).T @ df @ R
    dff = np.real_if_close(dff)
    vff = np.real_if_close(vff)
    uff = np.real_if_close(uff)
    return uff, dff, vff


def takagi(A, svd_order=True):
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.
    Note that the input matrix is internally symmetrized. If the input matrix is indeed symmetric this leaves it unchanged.
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
    # Here we force symmetrize the matrix
    A = 0.5 * (A + A.T)

    A = np.real_if_close(A)

    if np.allclose(A, 0):
        return np.zeros(n), np.eye(n)

    if np.isrealobj(A):
        # If the matrix A is real one can be more clever and use its eigendecomposition
        ls, U = np.linalg.eigh(A)
        U = U / np.exp(1j * np.angle(U)[0])
        vals = np.abs(ls)  # These are the Takagi eigenvalues
        phases = -np.ones(vals.shape[0], dtype=np.complex128)
        for j, l in enumerate(ls):
            if np.allclose(l, 0) or l > 0:
                phases[j] = 1
        phases = np.sqrt(phases)
        Uc = U @ np.diag(phases)  # One needs to readjust the phases
        signs = np.sign(Uc.real)[0]
        for k, s in enumerate(signs):
            if np.allclose(s, 0):
                signs[k] = 1
        Uc = np.real_if_close(Uc / signs)
        list_vals = [(vals[i], i) for i in range(len(vals))]
        # And also rearrange the unitary and values so that they are decreasingly ordered
        list_vals.sort(reverse=svd_order)
        sorted_ls, permutation = zip(*list_vals)
        return np.array(sorted_ls), Uc[:, np.array(permutation)]

    phi = np.angle(A[0, 0])
    Amr = np.real_if_close(np.exp(-1j * phi) * A)
    if np.isrealobj(Amr):
        vals, U = takagi(Amr, svd_order=svd_order)
        return vals, U * np.exp(1j * phi / 2)

    u, d, v = np.linalg.svd(A)
    U = u @ sqrtm((v @ np.conjugate(u)).T)
    # The line above could be simplifed to the line below if the product v @ np.conjugate(u) is diagonal
    # Which it should be according to Caves http://info.phys.unm.edu/~caves/courses/qinfo-s17/lectures/polarsingularAutonne.pdf
    # U = u * np.sqrt(0j + np.diag(v @ np.conjugate(u)))
    # This however breaks test_degenerate
    if svd_order is False:
        return d[::-1], U[:, ::-1]
    return d, U

