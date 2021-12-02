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
This module implements the La Budde's algorithm to calculate the
characteristic polynomials of matrices.
"""
# pylint: disable=too-many-branches
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def get_reflection_vector(matrix, k):  # pragma: no cover
    r"""Compute reflection vector for householder transformation on
    general complex matrices. See Introduction to Numerical Analysis-Springer New York (2002)
    (3rd Edition) by J. Stoer and R. Bulirsch Section 6.5.1.

    Args:
        matrix (array): the matrix in the householder transformation
        k (int): offset for submatrix

    Returns:
        array: reflection vector
    """
    size = len(matrix)
    sizeH = size - k
    reflect_vector = np.zeros(sizeH, dtype=matrix.dtype)
    offset = k - 1

    matrix_column = np.zeros(sizeH, dtype=matrix.dtype)
    for i in range(0, sizeH):
        matrix_column[i] = matrix[k + i, offset]

    sigma = np.linalg.norm(matrix_column)
    if matrix_column[0] != 0:
        sigma *= matrix_column[0] / np.abs(matrix_column[0])

    for i in range(0, sizeH):
        reflect_vector[i] = matrix_column[i]

    reflect_vector[0] += sigma
    return reflect_vector


@jit(nopython=True, cache=True)
def apply_householder(A, v, k):
    r"""Apply householder transformation on a matrix A. See  Matrix Computations
    by Golub and Van Loan (4th Edition) Sections 5.1.4 and 7.4.2

    Args:
        A (array): A matrix to apply householder on
        v (array): reflection vector
        size_A (int): size of matrix A
        k (int): offset for submatrix
    """
    size_A = len(A)
    sizeH = len(v)
    norm_v_sqr = np.linalg.norm(v) ** 2
    if norm_v_sqr == 0:
        return

    vHA = np.zeros(size_A - k + 1, dtype=A.dtype)
    Av = np.zeros(size_A, dtype=A.dtype)

    for j in range(0, size_A - k + 1):
        for l in range(0, sizeH):
            vHA[j] += np.conj(v[l]) * A[k + l, k - 1 + j]

    for i in range(0, sizeH):
        for j in range(0, size_A - k + 1):
            A[k + i, k - 1 + j] -= 2 * v[i] * vHA[j] / norm_v_sqr

    for i in range(0, size_A):
        for l in range(0, sizeH):
            Av[i] += A[i, k + l] * v[l]

    for i in range(0, size_A):
        for j in range(0, sizeH):
            A[i, k + j] -= 2 * Av[i] * np.conj(v[j]) / norm_v_sqr


@jit(nopython=True, cache=True)
def reduce_matrix_to_hessenberg(matrix):
    r"""Reduce the matrix to upper hessenberg form without Lapack. This function
    only accepts Row-Order matrices.

    Args:
        matrix (array): the matrix to be reduced

    Returns:
        array: matrix in hessenberg form
    """
    for i in range(1, len(matrix) - 1):
        reflect_vector = get_reflection_vector(matrix, i)
        apply_householder(matrix, reflect_vector, i)


@jit(nopython=True, cache=True)
def beta(H, i):
    r"""Auxiliary function for Labudde algorithm. See pg 10 of
    [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for definition of beta.

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row

    Returns:
        float: element of the lower-diagonal of matrix H
    """
    return H[i - 1, i - 2]


@jit(nopython=True, cache=True)
def alpha(H, i):
    r"""Auxiliary function for La Budde's algorithm. See pg 10 of
    [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for definition of alpha.

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row

    Returns:
        float: element of the central-diagonal of matrix H
    """
    return H[i - 1, i - 1]


@jit(nopython=True, cache=True)
def hij(H, i, j):
    r"""Auxiliary function for La Budde's algorithm. See pg 10 of
    [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for definition of hij.

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        j (int): column

    Returns:
        float: element of the lower-diagonal of matrix H
    """
    return H[i - 1, j - 1]


@jit(nopython=True, cache=True)
def mlo(i, j):  # pragma: no cover
"""Auxiliary function for La Budde's algorithm.
    See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).

    .. note::

        The La Budde paper uses indices that start counting at 1 so this function
        lowers them to start counting at 0.

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        j (int): column

    Returns:
        int: linear matrix index lowered by 1
    """
    return tuple((i - 1, j - 1))


@jit(nopython=True, cache=True)
def _charpoly_from_labudde(H, k):  # pragma: no cover
    r"""Compute characteristic polynomial using La Budde's algorithm.
    See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).

    .. note::
        If the matrix is n by n but you only want coefficients ``k < n`` set
        ``k`` below ``n``. If you want all coefficients, set ``k = n``.
    Args:
        H (array): matrix in Hessenberg form (RowMajor)
        n (int): size of matrix
        k (int): compute coefficients up to k (k must be <= n)
    Returns:
        array: char-poly coeffs + auxiliary data (see comment in function)
    """
    n = len(H)
    c = np.zeros_like(H)
    c[mlo(1, 1)] = -alpha(H, 1)
    c[mlo(2, 1)] = c[mlo(1, 1)] - alpha(H, 2)
    c[mlo(2, 2)] = alpha(H, 1) * alpha(H, 2) - hij(H, 1, 2) * beta(H, 2)

    for i in range(3, k + 1):
        c[mlo(i, 1)] = c[mlo(i - 1, 1)] - alpha(H, i)
        for j in range(2, i):
            suma = 0
            beta_prod = 1
            for m in range(1, j - 1):
                beta_prod = 1
                for bm in range(i, i - m, -1):
                    beta_prod *= beta(H, bm)
                suma += hij(H, i - m, i) * beta_prod * c[mlo(i - m - 1, j - m - 1)]

            beta_prod = 1
            for bm in range(i, i - j + 1, -1):
                beta_prod *= beta(H, bm)

            c[mlo(i, j)] = (
                c[mlo(i - 1, j)]
                - alpha(H, i) * c[mlo(i - 1, j - 1)]
                - suma
                - hij(H, i - j + 1, i) * beta_prod
            )

        suma = 0
        beta_prod = 0

        for m in range(1, i - 1):
            beta_prod = 1
            for bm in range(i, i - m, -1):
                beta_prod *= beta(H, bm)
            suma += hij(H, i - m, i) * beta_prod * c[mlo(i - m - 1, i - m - 1)]

        beta_prod = 1
        for bm in range(i, 1, -1):
            beta_prod *= beta(H, bm)

        c[mlo(i, i)] = -alpha(H, i) * c[mlo(i - 1, i - 1)] - suma - hij(H, 1, i) * beta_prod

    for i in range(k + 1, n + 1):
        c[mlo(i, 1)] = c[mlo(i - 1, 1)] - alpha(H, i)

        if k >= 2:
            for j in range(2, k + 1):
                suma = 0.0
                beta_prod = 1
                for m in range(1, j - 1):
                    beta_prod = 1
                    for bm in range(i, i - m, -1):
                        beta_prod *= beta(H, bm)

                    suma += hij(H, i - m, i) * beta_prod * c[mlo(i - m - 1, j - m - 1)]

                beta_prod = 1
                for bm in range(i, i - j + 1, -1):
                    beta_prod *= beta(H, bm)

                c[mlo(i, j)] = (
                    c[mlo(i - 1, j)]
                    - alpha(H, i) * c[mlo(i - 1, j - 1)]
                    - suma
                    - hij(H, i - j + 1, i) * beta_prod
                )
    poly_list = [c[n - 1, i - 1] for i in range(1, n + 1)]

    return poly_list


@jit(nopython=True, cache=True)
def charpoly_from_labudde(H):
    """
    Calculates the characteristic polynomial of the matrix H.

    Args:
        H (array): square matrix
    Returns
        (array): list of power traces from 0 to n-1
    """
    n = len(H)
    reduce_matrix_to_hessenberg(H)
    coeff = _charpoly_from_labudde(H, n)
    return coeff


@jit(nopython=True, cache=True)
def power_trace_labudde(H, n):  # pragma: no cover
    """
    Calculates the powertraces of the matrix H up to power ``n-1``.

    Args:
        H (array): square matrix
        n (int): required order
    Returns
        (array): list of power traces from 0 to n-1
    """
    m = len(H)
    min_val = min(n, m)
    pow_traces = [m, np.trace(H)]
    A = H
    for _ in range(min_val - 2):
        A = A @ H
        pow_traces.append(np.trace(A))
    if n <= m:
        return np.array(pow_traces, dtype=H.dtype)
    char_pol = charpoly_from_labudde(H)
    for _ in range(min_val, n):
        ssum = 0
        for k in range(m):
            ssum -= char_pol[k] * pow_traces[-k - 1]
        pow_traces.append(ssum)
    return np.array(pow_traces, dtype=H.dtype)
