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
Permanent Python interface
"""
import numpy as np

from ._hafnian import hafnian_repeated
from .libwalrus import perm_complex, perm_real


def perm(A, quad=True, fsum=False):
    """Returns the permanent of a matrix via the
    `Ryser formula <https://en.wikipedia.org/wiki/Computing_the_permanent#Ryser_formula>`_.

    For more direct control, you may wish to call :func:`perm_real`
    or :func:`perm_complex` directly.

    Args:
        A (array): a square array.
        quad (bool): If ``True``, the input matrix is cast to a ``long double``
            matrix internally for a quadruple precision hafnian computation.
        fsum (bool): Whether to use the ``fsum`` method for higher accuracy summation.
            Note that if ``fsum`` is true, double precision will be used, and the
            ``quad`` keyword argument will be ignored.

    Returns:
        np.float64 or np.complex128: the permanent of matrix A.
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if matshape[0] == 2:
        return A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]

    if matshape[0] == 3:
        return (
            A[0, 2] * A[1, 1] * A[2, 0]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[0, 0] * A[1, 2] * A[2, 1]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2]
        )

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return perm_complex(A, quad=quad)
        return perm_real(np.float64(A.real), quad=quad, fsum=fsum)

    return perm_real(A, quad=quad, fsum=fsum)


def permanent_repeated(A, rpt):
    r"""Calculates the permanent of matrix :math:`A`, where the ith row/column
    of :math:`A` is repeated :math:`rpt_i` times.

    This function constructs the matrix

    .. math:: B = \begin{bmatrix} 0 & A\\ A^T & 0 \end{bmatrix},

    and then calculates :math:`perm(A)=haf(B)`, by calling

    >>> hafnian_repeated(B, rpt*2, loop=False)

    Args:
        A (array): matrix of size [N, N]
        rpt (Sequence): sequence of N positive integers indicating the corresponding rows/columns
            of A to be repeated.

    Returns:
        np.int64 or np.float64 or np.complex128: the permanent of matrix A.
    """
    n = A.shape[0]
    O = np.zeros([n, n])
    B = np.vstack([np.hstack([O, A]), np.hstack([A.T, O])])

    return hafnian_repeated(B, rpt * 2, loop=False)
