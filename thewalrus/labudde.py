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
This module implements the labudde algorithm to calculate the
characteristic polynomials of matrices.
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def get_reflection_vector(matrix, size, k):
    r"""Compute reflection vector for householder transformation on
    general complex matrices. See Introduction to Numerical Analysis-Springer New York (2002)
    (3rd Edition) by J. Stoer and R. Bulirsch Section 6.5.1.

    Args:
        matrix (array): the matrix in the householder transformation
        size (int): size of matrix
        k (int): offset for submatrix

    Returns:
        array: reflection vector
    """
    sizeH = size - k
    reflect_vector = np.zeros(sizeH, dtype=matrix.dtype)
    order = size - sizeH
    offset = order - 1

    matrix_column = np.zeros(sizeH, dtype=matrix.dtype)
    for i in range(0, sizeH):
        matrix_column[i] = matrix[(i + order) * size + offset]

    sigma = np.linalg.norm(matrix_column)
    if matrix_column[0] != 0:
        sigma *= matrix_column[0] / np.abs(matrix_column[0])

    for i in range(0, sizeH):
        reflect_vector[i] = matrix_column[i]

    reflect_vector[0] += sigma
    return reflect_vector


@jit(nopython=True, cache=True)
def apply_householder(A, v, size_A, k):
    r"""Apply householder transformation on a matrix A
    See  Matrix Computations by Golub and Van Loan
    (4th Edition) Sections 5.1.4 and 7.4.2

    Args:
        A (array): A matrix to apply householder on
        v (array): reflection vector
        size_A (int): size of matrix A
        k (int): offset for submatrix
    """
    sizeH = len(v)
    norm_v_sqr = np.linalg.norm(v) ** 2
    if norm_v_sqr == 0:
        return

    vHA = np.zeros(size_A - k + 1, dtype=A.dtype)
    Av = np.zeros(size_A, dtype=A.dtype)

    for j in range(0, size_A - k + 1):
        for l in range(0, sizeH):
            vHA[j] += np.conj(v[l]) * A[(k + l) * size_A + k - 1 + j]

    for i in range(0, sizeH):
        for j in range(0, size_A - k + 1):
            A[(k + i) * size_A + k - 1 + j] -= 2 * v[i] * vHA[j] / norm_v_sqr

    for i in range(0, size_A):
        for l in range(0, sizeH):
            Av[i] += A[(i) * size_A + k + l] * v[l]

    for i in range(0, size_A):
        for j in range(0, sizeH):
            A[(i) * size_A + k + j] -= 2 * Av[i] * np.conj(v[j]) / norm_v_sqr


@jit(nopython=True, cache=True)
def reduce_matrix_to_hessenberg(matrix, size):
    r"""Reduce the matrix to upper hessenberg form
         without Lapack. This function only accepts Row-Order
         matrices.

    Args:
        matrix (array): the matrix to be reduced
        size (int): size of matrix

    Returns:
        array: matrix in hessenberg form
    """
    for i in range(1, size - 1):
        reflect_vector = get_reflection_vector(matrix, size, i)
        apply_householder(matrix, reflect_vector, size, i)


@jit(nopython=True, cache=True)
def beta(H, i, size):
    r"""Auxiliary function for Labudde algorithm.
         See pg 10 of for definition of beta
         [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        size (int): size of the matrix

    Returns:
        float: element of the lower-diagonal of matrix H
    """
    return H[(i - 1) * size + i - 2]


@jit(nopython=True, cache=True)
def alpha(H, i, size):
    r"""Auxiliary function for Labudde algorithm.
         See pg 10 of for definition of alpha
         [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        size (int): size of the matrix

    Returns:
        float: element of the central-diagonal of matrix H
    """
    return H[(i - 1) * size + i - 1]


@jit(nopython=True, cache=True)
def hij(H, i, j, size):
    r"""Auxiliary function for Labudde algorithm.
         See pg 10 of for definition of hij
         [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).

    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        j (int): column
        size (int): size of the matrix

    Returns:
        float: element of the lower-diagonal of matrix H
    """
    return H[(i - 1) * size + j - 1]


@jit(nopython=True, cache=True)
def mlo(i, j, size):
    r"""Auxiliary function for Labudde algorithm.
         The labudde paper uses indices that start counting at 1
         so this function lowers them to start counting at 0.
         See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
    Args:
        matrix (array): upper-Hessenberg matrix
        i (int): row
        j (int): column
        size (int): size of the matrix

    Returns:
        int: linear matrix index lowered by 1
    """
    return (i - 1) * size + j - 1


@jit(nopython=True, cache=True)
def _charpoly_from_labudde(H, n, k):
    r"""Compute characteristic polynomial using the LaBudde algorithm.
         See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
         If the matrix is n by n but you only want coefficients k < n
         set k below n. If you want all coefficients, set k = n.
    Args:
        H (array): matrix in Hessenberg form (RowMajor)
        n (int): size of matrix
        k (int): compute coefficients up to k (k must be <= n)

    Returns:
        array: char-poly coeffs + auxiliary data (see comment in function)
    """

    c = np.zeros(n * n, dtype=H.dtype)
    c[mlo(1, 1, n)] = -alpha(H, 1, n)
    c[mlo(2, 1, n)] = c[mlo(1, 1, n)] - alpha(H, 2, n)
    c[mlo(2, 2, n)] = alpha(H, 1, n) * alpha(H, 2, n) - hij(H, 1, 2, n) * beta(H, 2, n)

    for i in range(3, k + 1):
        c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n)
        for j in range(2, i):
            suma = 0
            beta_prod = 1
            for m in range(1, j - 1):
                beta_prod = 1
                for bm in range(i, i - m, -1):
                    beta_prod *= beta(H, bm, n)
                suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)]

            beta_prod = 1
            for bm in range(i, i - j + 1, -1):
                beta_prod *= beta(H, bm, n)

            c[mlo(i, j, n)] = (
                c[mlo(i - 1, j, n)]
                - alpha(H, i, n) * c[mlo(i - 1, j - 1, n)]
                - suma
                - hij(H, i - j + 1, i, n) * beta_prod
            )

        suma = 0
        beta_prod = 0

        for m in range(1, i - 1):
            beta_prod = 1
            for bm in range(i, i - m, -1):
                beta_prod *= beta(H, bm, n)
            suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, i - m - 1, n)]

        beta_prod = 1
        for bm in range(i, 1, -1):
            beta_prod *= beta(H, bm, n)

        c[mlo(i, i, n)] = (
            -alpha(H, i, n) * c[mlo(i - 1, i - 1, n)] - suma - hij(H, 1, i, n) * beta_prod
        )

    for i in range(k + 1, n + 1):
        c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n)

        if k >= 2:
            for j in range(2, k + 1):
                suma = 0.0
                beta_prod = 1
                for m in range(1, j - 1):
                    beta_prod = 1
                    for bm in range(i, i - m, -1):
                        beta_prod *= beta(H, bm, n)

                    suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)]

                beta_prod = 1
                for bm in range(i, i - j + 1, -1):
                    beta_prod *= beta(H, bm, n)

                c[mlo(i, j, n)] = (
                    c[mlo(i - 1, j, n)]
                    - alpha(H, i, n) * c[mlo(i - 1, j - 1, n)]
                    - suma
                    - hij(H, i - j + 1, i, n) * beta_prod
                )
    poly_list = [c[(n - 1) * n + i - 1] for i in range(1, n + 1)]

    return poly_list


@jit(nopython=True, cache=True)
def charpoly_from_labudde(H, method="ravel"):
    """
    Calculates the characteristic polynomial of the matrix H
    Args:
        H (array): square matrix
        method (string): pre-processing operation on H
    Returns
        (array): list of power traces from 0 to n-1
    """
    if method == "ravel":
        Hflat = H.ravel()
    elif method == "flatten":
        Hflat = H.flatten()
    elif method == "reshape":
        Hflat = H.reshape(-1)
    n = len(H)
    reduce_matrix_to_hessenberg(Hflat, n)
    coeff = _charpoly_from_labudde(Hflat, n, n)
    return coeff


@jit(nopython=True, cache=True)
def power_trace_labudde(H, n):
    """
    Calculates the powertraces of the matrix H up to power n-1.
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
