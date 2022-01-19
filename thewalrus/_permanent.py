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
from itertools import chain

import numpy as np
from numba import jit, prange

from scipy.special import factorial

from ._hafnian import hafnian_repeated, find_kept_edges


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
    # row_comb keeps the sum of previous subsets.
    # Every iteration, it removes a term and/or adds a new term
    # to give the term to add for the next subset
    row_comb = np.zeros((n), dtype=M.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2 ** i for i in range(n)]
    num_loops = 2 ** n
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
    row_comb = np.sum(M, 0)
    total = 0
    old_gray = 0
    sign = +1
    binary_power_dict = [2 ** i for i in range(n)]
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


@jit(nopython=True, parallel=True)
def brs(A, E):  # pragma: no cover
    r"""
    Calculates the Bristolian, a matrix function introduced for calculating the threshold detector
    statistics on measurements of Fock states interfering in linear optical interferometers.

    See the paper 'Threshold detector statistics of Bosonic states' for more detail (to be published soon)

    Args:
        A (array): matrix of size [m, n]
        E (array): matrix of size [r, n]

    Returns:
        int or float or complex: the Bristol of matrices A and E
    """
    m = A.shape[0]

    steps = 2 ** m
    ones = np.ones(m, dtype=np.int8)
    total = 0
    for j in prange(steps):
        kept_rows = np.where(find_kept_edges(j, ones) != 0)[0]
        Ay = A[kept_rows, :]
        plusminus = (-1) ** ((m - len(kept_rows)) % 2)
        total += plusminus * perm_bbfg(Ay.conj().T @ Ay + E)
    return total


@jit(nopython=True, parallel=True)
def ubrs(A):  # pragma: no cover
    r"""
    Calculates the Unitary Bristolian, a matrix function introduced for calculating the threshold detector
    statistics on measurements of Fock states interfering in lossless linear optical interferometers.

    See the paper 'Threshold detector statistics of Bosonic states' for more detail (to be published soon)

    Args:
        A (array): matrix of size [m, n]

    Returns:
        int or float or complex: the Unitary Bristol of matrix A
    """
    m = A.shape[0]
    steps = 2 ** m
    ones = np.ones(m, dtype=np.int8)
    total = 0
    for j in prange(1, steps):
        kept_rows = np.where(find_kept_edges(j, ones) != 0)[0]
        Az = A[kept_rows, :]
        plusminus = (-1) ** ((m - len(kept_rows)) % 2)
        total += plusminus * perm_bbfg(Az.conj().T @ Az)
    return total


def fock_prob(n, m, U):
    r"""
    Calculates the probability of a an input Fock state, n, scattering to an output Fock state, m, through
    an interferometer described by matrix U.
    The matrix U does not need to be unitary, but the total photon number at the input and the output must be equal.

    Args:
        n (sequence[int]): length-M list giving the input Fock state occupancy of each mode
        m (sequence[int]): length-M list giving the output Fock state occupancy of each mode
        U (array): M x M matrix describing the a linear optical transformation

    Returns:
        float: probability of Fock state, n, scattering to m, through an interferometer, U
    """
    if sum(n) != sum(m):
        raise ValueError("number of input photons must equal number of output photons")

    in_modes = np.array(list(chain(*[[i] * j for i, j in enumerate(n) if j > 0])))
    out_modes = np.array(list(chain(*[[i] * j for i, j in enumerate(m) if j > 0])))

    Umn = U[np.ix_(out_modes, in_modes)]

    n = np.array(n)
    m = np.array(m)

    return abs(perm(Umn)) ** 2 / (
        np.prod(factorial(n), dtype=np.float64) * np.prod(factorial(m), dtype=np.float64)
    )


def fock_threshold_prob(n, d, T):
    r"""
    Calculates the probability of a an M_in mode input Fock state, n, scattering through an interferometer described by
    T, being detected by M_out threshold detectors, with outcome given by M_out-length list, d.
    T is an M_out x M_in matrix. It does not need to be unitary but M_out <= M_in.

    Args:
        n (sequence[int]): length-M_in list giving the input Fock state occupancy of each mode
        d (sequence[int]): length-M_out list giving the outputs of threshold detectors
        T (array): M_out x M_in matrix describing the a linear optical transformation, M_out <= M_in

    Returns:
        float: probability of Fock state, n, scattering through an interferometer, T, to give threshold detector outcome, d
    """
    n = np.array(n)
    d = np.array(d)

    if len(n) != T.shape[1]:
        raise ValueError("length of n must matrix number of input modes of T")
    if len(d) != T.shape[0]:
        raise ValueError("length of d must match number of output modes of T")
    if T.shape[0] > T.shape[1]:
        raise ValueError("number of output modes cannot be larger than number of input modes")

    fac_prod = np.prod(factorial(n), dtype=np.float64)

    in_modes = np.array(list(chain(*[[i] * j for i, j in enumerate(n) if j > 0])))
    C = np.where(d > 0)[0]

    A = T[np.ix_(C, in_modes)]

    # if matrix is unitary, use the Unitary Bristolian
    E = np.eye(T.shape[1]) - T.conj().T @ T
    if np.allclose(E, np.zeros((T.shape[1], T.shape[1]))):
        U_dn = T[np.ix_(C, in_modes)]
        return ubrs(U_dn).real / fac_prod

    E_n = E[np.ix_(in_modes, in_modes)]

    return brs(A, E_n).real / fac_prod
