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
r"""
This submodule provides access to tools for finding the permanent of a matrix. The algorithms implemented
here was first derived in

* Ryser, Herbert John (1963).
  Combinatorial Mathematics, The Carus Mathematical Monographs, Vol. 14, Mathematical Association of America.
* Glynn, David G.
  (2010), "The permanent of a square matrix", European Journal of Combinatorics, 31 (7): 1887-1891.
  <doi:10.1016/j.ejc.2010.01.010>`_
"""
import numpy as np
from numba import jit
from ._hafnian import hafnian_repeated


def perm(A, method="bbfg"):
    """Returns the permanent of a matrix using various methods.


    Args:
        A (array[float or complex]): a square array.
        method (string): Set this to ``"ryser"`` to use the
            `Ryser formula
            <https://en.wikipedia.org/wiki/Computing_the_permanent#Ryser_formula>`_,
            or ``"bbfg"`` to use the
            `BBFG formula
            <https://en.wikipedia.org/wiki/Computing_the_permanent#Balasubramanian%E2%80%93Bax%E2%80%93Franklin%E2%80%93Glynn_formula>`_.

    Returns:
        float or complex: the permanent of matrix ``A``
    """

    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if matshape[0] == 0:
        return A.dtype.type(1.0)

    if matshape[0] == 1:
        return A[0, 0]

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

    isRyser = bool(method != "bbfg")

    return perm_ryser(A) if isRyser else perm_bbfg(A)


@jit(nopython=True)
def perm_ryser(M):  # pragma: no cover
    """
    Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    The code is an re-implementation from a Python 2 code found in
    `Permanent code golf
    <https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible>`_
    using Numba.

    Args:
        M (array) : a square array.

    Returns:
        float or complex: the permanent of matrix ``M``
    """
    n = len(M)
    if n == 0:
        return M.dtype.type(1.0)
    # row_comb keeps the sum of previous subsets.
    # Every iteration, it removes a term and/or adds a new term
    # to give the term to add for the next subset
    row_comb = np.zeros((n), dtype=M.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2**n
    for k in range(0, num_loops):
        bin_index = (k + 1) % num_loops
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict.index(grey_diff)
        new_vector = M[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_grey = new_grey
    return total


@jit(nopython=True)
def perm_bbfg(M):  # pragma: no cover
    """
    Returns the permanent of a matrix using the bbfg formula in Gray ordering

    The code is a re-implementation from a Python 2 code found in
    `Permanent code golf
    <https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible>`_
    using Numba.

    Args:
        M (array) : a square array.

    Returns:
        float or complex: the permanent of a matrix ``M``
    """

    n = len(M)
    if n == 0:
        return M.dtype.type(1.0)
    row_comb = np.sum(M, 0)
    total = 0
    old_gray = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2 ** (n - 1)
    for bin_index in range(1, num_loops + 1):
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_gray = bin_index ^ (bin_index // 2)
        gray_diff = old_gray ^ new_gray
        gray_diff_index = binary_power_dict.index(gray_diff)
        new_vector = M[gray_diff_index]
        direction = 2 * ((old_gray > new_gray) - (old_gray < new_gray))
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_gray = new_gray
    return total / num_loops


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
        int or float or complex: the permanent of matrix ``A``
    """
    n = A.shape[0]
    O = np.zeros([n, n])
    B = np.vstack([np.hstack([O, A]), np.hstack([A.T, O])])

    return hafnian_repeated(B, rpt * 2, loop=False)
