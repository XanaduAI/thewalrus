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
Hafnian Python interface
"""
import numpy as np

from .lib.libhaf import haf_complex, haf_real, haf_int, haf_rpt_real, haf_rpt_complex


def kron_reduced(A, n):
    r"""Calculates the reduced Kronecker product :math:`A^{\oplus 2}\cancel{\otimes}J`.

    Args:
        A (array): matrix of size [N, N]
        n (Sequence): sequence of integers indicating the multi-mode photon detection event
    Returns:
        array: the reduced Kronecker product
    """
    rows = [i for sublist in [[idx]*j for idx, j in enumerate(n)] for i in sublist]
    return np.float64(A[:, rows][rows])


def hafnian(A, loop=False, recursive=True, tol=1e-12):
    """Returns the hafnian of matrix A via the C++ hafnian library.

    For more direct control, you may wish to call :func:`haf_real`,
    :func:`haf_complex`, or :func:`haf_int` directly.

    Args:
        A (array): a square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.
            If ``loop=True``, then this keyword argument is ignored.
        tol (float): the tolerance when checking that the matrix is
            symmetric. Default tolerance is 1e-12.

    Returns:
        np.int64 or np.float64 or np.complex128: the hafnian of matrix A.
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if np.linalg.norm(A-A.T) >= tol:
        raise ValueError("Input matrix must be symmetric.")

    if matshape[0] % 2 != 0 and not loop:
        return 0.0

    if matshape[0] %2 != 0 and loop:
        A = np.pad(A, pad_width=((0, 1), (0, 1)), mode='constant')
        A[-1, -1] = 1.0

    matshape = A.shape

    if matshape[0] == 2:
        if loop:
            return A[0, 1] + A[0, 0]*A[1, 1]
        return A[0][1]

    if matshape[0] == 4:
        if loop:
            result = A[0, 1]*A[2, 3] \
                + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2] \
                + A[0, 0]*A[1, 1]*A[2, 3] + A[0, 1]*A[2, 2]*A[3, 3] \
                + A[0, 2]*A[1, 1]*A[3, 3] + A[0, 0]*A[2, 2]*A[1, 3] \
                + A[0, 0]*A[3, 3]*A[1, 2] + A[0, 3]*A[1, 1]*A[2, 2] \
                + A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3]
            return result

        return A[0, 1]*A[2, 3] + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return haf_complex(A, loop=loop, recursive=recursive)
        return haf_real(np.float64(A.real), loop=loop, recursive=recursive)

    if np.all(np.mod(A, 1) == 0) and not loop:
        return haf_int(np.int64(A))

    return haf_real(A, loop=loop, recursive=recursive)


def hafnian_repeated(A, rpt, use_eigen=True, tol=1e-12):
    r"""Returns the hafnian of matrix A with repeated rows/columns via the C++ hafnian library.

    The :func:`kron_reduced` function may be used to show the resulting matrix
    with repeated rows and columns as per ``rpt``.

    As a result, the following are identical:

    >>> hafnian_repeated(A, rpt)
    >>> hafnian(kron_reduced(A, rpt))

    However, using ``hafnian_repeated`` in the case where there are a large number
    of repeated rows and columns (:math:`\sum_{i}rpt_i \gg N`) can be
    significantly faster.

    .. note::

        If :math:`rpt=(1, 1, \dots, 1)`, then

        >>> hafnian_repeated(A, rpt) == hafnian(A)

    For more direct control, you may wish to call :func:`haf_rpt_real` or
    :func:`haf_rpt_complex` directly.

    Args:
        A (array): a square, symmetric :math:`N\times N` array.
        rpt (Sequence): a length-:math:`N` positive integer sequence, corresponding
            to the number of times each row/column of matrix :math:`A` is repeated.
        use_eigen (bool): if True (default), the Eigen linear algebra library
            is used for matrix multiplication. If the hafnian library was compiled
            with BLAS/Lapack support, then BLAS will be used for matrix multiplication.
        tol (float): the tolerance when checking that the matrix is
            symmetric. Default tolerance is 1e-12.

    Returns:
        np.int64 or np.float64 or np.complex128: the hafnian of matrix A.
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if np.linalg.norm(A-A.T) >= tol:
        raise ValueError("Input matrix must be symmetric.")

    if len(rpt) != len(A):
        raise ValueError("the rpt argument must be 1-dimensional sequence of length len(A).")

    nud = np.array(rpt, dtype=np.int32)

    if not np.all(np.mod(nud, 1) == 0) or np.any(nud <= 0):
        raise ValueError("the rpt argument must contain positive integers.")

    if np.sum(nud) % 2 != 0:
        return 0.0

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return haf_rpt_complex(A, nud, use_eigen=use_eigen)
        return haf_rpt_real(np.float64(A.real), nud, use_eigen=use_eigen)

    if np.all(np.mod(A, 1) == 0):
        return np.int(haf_rpt_real(A, nud, use_eigen=use_eigen))

    return haf_rpt_real(A, nud, use_eigen=use_eigen)
