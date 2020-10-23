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

from .libwalrus import haf_complex, haf_int, haf_real, haf_rpt_complex, haf_rpt_real


def input_validation(A, rtol=1e-05, atol=1e-08):
    """Checks that the matrix A satisfies the requirements for Hafnian calculation.

    These include:

    * That the ``A`` is a NumPy array
    * That ``A`` is square
    * That ``A`` does not contain any NaNs
    * That ``A`` is symmetric

    Args:
        A (array): a NumPy array.
        rtol (float): the relative tolerance parameter used in ``np.allclose``.
        atol (float): the absolute tolerance parameter used in ``np.allclose``.

    Returns:
        bool: returns True if the matrix satisfies all requirements.
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    n = A.shape

    if n[0] != n[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if not np.allclose(A, A.T, rtol=rtol, atol=atol):
        raise ValueError("Input matrix must be symmetric.")

    return True


def reduction(A, rpt):
    r"""Calculates the reduction of an array by a vector of indices.

    This is equivalent to repeating the ith row/column of :math:`A`, :math:`rpt_i` times.

    Args:
        A (array): matrix of size [N, N]
        rpt (Sequence): sequence of N positive integers indicating the corresponding rows/columns
            of A to be repeated.
    Returns:
        array: the reduction of A by the index vector rpt
    """
    rows = [i for sublist in [[idx] * j for idx, j in enumerate(rpt)] for i in sublist]

    if A.ndim == 1:
        return A[rows]

    return A[:, rows][rows]

# pylint: disable=too-many-arguments
def hafnian(
    A, loop=False, recursive=True, rtol=1e-05, atol=1e-08, quad=True, approx=False, num_samples=1000
):  # pylint: disable=too-many-arguments
    """Returns the hafnian of a matrix.

    For more direct control, you may wish to call :func:`haf_real`,
    :func:`haf_complex`, or :func:`haf_int` directly.

    Args:
        A (array): a square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.
            If ``loop=True``, then this keyword argument is ignored.
        rtol (float): the relative tolerance parameter used in ``np.allclose``.
        atol (float): the absolute tolerance parameter used in ``np.allclose``.
        quad (bool): If ``True``, the hafnian algorithm is performed with quadruple precision.
        approx (bool): If ``True``, an approximation algorithm is used to estimate the hafnian. Note that
            the approximation algorithm can only be applied to matrices ``A`` that only have non-negative entries.
        num_samples (int): If ``approx=True``, the approximation algorithm performs ``num_samples`` iterations
            for estimation of the hafnian of the non-negative matrix ``A``.

    Returns:
        np.int64 or np.float64 or np.complex128: the hafnian of matrix A.
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, rtol=rtol, atol=atol)

    matshape = A.shape

    if matshape == (0, 0):
        return 1

    if matshape[0] % 2 != 0 and not loop:
        return 0.0

    if np.allclose(np.diag(np.diag(A)), A, rtol=rtol, atol=atol):
        if loop:
            return np.prod(np.diag(A))
        return 0

    if matshape[0] % 2 != 0 and loop:
        A = np.pad(A, pad_width=((0, 1), (0, 1)), mode="constant")
        A[-1, -1] = 1.0

    matshape = A.shape

    if matshape[0] == 2:
        if loop:
            return A[0, 1] + A[0, 0] * A[1, 1]
        return A[0][1]

    if matshape[0] == 4:
        if loop:
            result = (
                A[0, 1] * A[2, 3]
                + A[0, 2] * A[1, 3]
                + A[0, 3] * A[1, 2]
                + A[0, 0] * A[1, 1] * A[2, 3]
                + A[0, 1] * A[2, 2] * A[3, 3]
                + A[0, 2] * A[1, 1] * A[3, 3]
                + A[0, 0] * A[2, 2] * A[1, 3]
                + A[0, 0] * A[3, 3] * A[1, 2]
                + A[0, 3] * A[1, 1] * A[2, 2]
                + A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3]
            )
            return result

        return A[0, 1] * A[2, 3] + A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    if approx:
        if np.any(np.iscomplex(A)):
            raise ValueError("Input matrix must be real")

        if np.any(A < 0):
            raise ValueError("Input matrix must not have negative entries")

    if A.dtype == np.complex:
        # array data is complex type
        if np.any(np.iscomplex(A)):
            # array values contain non-zero imaginary parts
            return haf_complex(A, loop=loop, recursive=recursive, quad=quad)

        # all array values have zero imaginary parts
        return haf_real(np.float64(A.real), loop=loop, recursive=recursive, quad=quad)

    if np.issubdtype(A.dtype, np.integer) and not loop:
        # array data is an integer type, and the user is not
        # requesting the loop hafnian
        return haf_int(np.int64(A))

    if np.issubdtype(A.dtype, np.integer) and loop:
        # array data is an integer type, and the user is
        # requesting the loop hafnian. Currently no
        # integer function for loop hafnians, have to instead
        # convert to float and use haf_real
        A = np.float64(A)

    return haf_real(
        A, loop=loop, recursive=recursive, quad=quad, approx=approx, nsamples=num_samples
    )


def hafnian_repeated(A, rpt, mu=None, loop=False, rtol=1e-05, atol=1e-08):
    r"""Returns the hafnian of matrix with repeated rows/columns.

    The :func:`reduction` function may be used to show the resulting matrix
    with repeated rows and columns as per ``rpt``.

    As a result, the following are identical:

    >>> hafnian_repeated(A, rpt)
    >>> hafnian(reduction(A, rpt))

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
        mu (array): a vector of length :math:`N` representing the vector of means/displacement.
            If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this
            only affects the loop hafnian.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        use_eigen (bool): if True (default), the Eigen linear algebra library
            is used for matrix multiplication. If the hafnian library was compiled
            with BLAS/Lapack support, then BLAS will be used for matrix multiplication.
        rtol (float): the relative tolerance parameter used in ``np.allclose``.
        atol (float): the absolute tolerance parameter used in ``np.allclose``.

    Returns:
        np.int64 or np.float64 or np.complex128: the hafnian of matrix A.
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, atol=atol, rtol=rtol)

    if len(rpt) != len(A):
        raise ValueError("the rpt argument must be 1-dimensional sequence of length len(A).")

    nud = np.array(rpt, dtype=np.int32)

    if not np.all(np.mod(nud, 1) == 0) or np.any(nud < 0):
        raise ValueError("the rpt argument must contain non-negative integers.")

    if np.all(nud == 0):
        return 1.0

    if np.sum(nud) % 2 != 0 and not loop:
        return 0.0

    if mu is None:
        mu = A.diagonal().copy()

    if np.allclose(A, 0, rtol=rtol, atol=atol):
        if loop:
            return np.prod(mu ** rpt)
        return 0

    if len(mu) != len(A):
        raise ValueError("Length of means vector must be the same length as the matrix A.")

    if A.dtype == np.complex or mu.dtype == np.complex:
        return haf_rpt_complex(A, nud, mu=mu, loop=loop)

    return haf_rpt_real(A, nud, mu=mu, loop=loop)
