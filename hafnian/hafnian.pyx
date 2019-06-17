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
#cython: boundscheck=False, wraparound=False, embedsignature=True
# distutils: language=c++
cimport cython
from libcpp.vector cimport vector


cdef extern from "../src/hafnian.hpp" namespace "hafnian":
    T hafnian[T](vector[T] &mat)
    T hafnian_recursive[T](vector[T] &mat)
    T loop_hafnian[T](vector[T] &mat)

    T hafnian_rpt[T](vector[T] &mat, vector[int] &nud)
    T loop_hafnian_rpt[T](vector[T] &mat, vector[T] &mu, vector[int] &nud)

    double hafnian_recursive_quad(vector[double] &mat)
    double complex hafnian_recursive_quad(vector[double complex] &mat)

    double hafnian_rpt_quad(vector[double] &mat, vector[int] &nud)
    double complex hafnian_rpt_quad(vector[double complex] &mat, vector[int] &nu)

    double loop_hafnian_rpt_quad(vector[double] &mat, vector[double] &mu, vector[int] &nud)
    double complex loop_hafnian_rpt_quad(vector[double complex] &mat, vector[double complex] &mu, vector[int] &nud)

    double hafnian_approx(vector[double] &mat, int &nsamples)

    double torontonian_quad(vector[double] &mat)
    double complex torontonian_quad(vector[double complex] &mat)
    double torontonian_fsum[T](vector[T] &mat)


# ==============================================================================
# Torontonian


def torontonian_complex(double complex[:, :] A, fsum=False):
    """Returns the Torontonian of a complex matrix A via the C++ hafnian library.

    The input matrix is cast to a ``long double complex``
    matrix internally for a quadruple precision torontonian computation.

    However, if ``fsum=True``, no casting takes place, as the Shewchuk algorithm
    only support double precision.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        fsum (bool): if ``True``, the `Shewchuk algorithm <https://github.com/achan001/fsum>_
            for more accurate summation is performed. This can significantly increase
            the accuracy of the computation.

    Returns:
        np.complex128: the torontonian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double complex] mat
    cdef int m = n/2

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    if fsum:
        return torontonian_fsum(mat)

    return torontonian_quad(mat)


def torontonian_real(double[:, :] A, fsum=False):
    """Returns the Torontonian of a real matrix A via the C++ hafnian library.

    The input matrix is cast to a ``long double``
    matrix internally for a quadruple precision torontonian computation.

    However, if ``fsum=True``, no casting takes place, as the Shewchuk algorithm
    only support double precision.

    Args:
        A (array): a np.float64, square, symmetric array of even dimensions.
        fsum (bool): if ``True``, the `Shewchuk algorithm <https://github.com/achan001/fsum>_
            for more accurate summation is performed. This can significantly increase
            the accuracy of the computation.

    Returns:
        np.float64: the torontonian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double] mat
    cdef int m = n/2

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])


    if fsum:
        return torontonian_fsum(mat)

    return torontonian_quad(mat)


# ==============================================================================
# Hafnian repeated


def haf_rpt_real(double[:, :] A, int[:] rpt, double[:] mu=None, bint loop=False):
    r"""Returns the hafnian of a real matrix A via the C++ hafnian library
    using the rpt method. This method is more efficient for matrices with
    repeated rows and columns.

    Args:
        A (array): a np.float64, square, :math:`N\times N` array of even dimensions.
        rpt (array): a length :math:`N` array corresponding to the number of times
            each row/column of matrix A is repeated.
        mu (array): a vector of length :math:`N` representing the vector of means/displacement.
            If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this
            only affects the loop hafnian.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.float64: the hafnian
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[int] nud
    cdef vector[double] mat, d

    for i in range(n):
        nud.push_back(rpt[i])

        if mu is None:
            d.push_back(A[i, i])
        else:
            d.push_back(mu[i])

        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian_rpt_quad(mat, d, nud)

    return hafnian_rpt_quad(mat, nud)


def haf_rpt_complex(double complex[:, :] A, int[:] rpt, double complex[:] mu=None, bint loop=False):
    r"""Returns the hafnian of a complex matrix A via the C++ hafnian library
    using the rpt method. This method is more efficient for matrices with
    repeated rows and columns.

    Args:
        A (array): a np.complex128, square, :math:`N\times N` array of even dimensions.
        rpt (array): a length :math:`N` array corresponding to the number of times
            each row/column of matrix A is repeated.
        mu (array): a vector of length :math:`N` representing the vector of means/displacement.
            If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this
            only affects the loop hafnian.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.complex128: the hafnian
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[int] nud
    cdef vector[double complex] mat, d

    for i in range(n):
        nud.push_back(rpt[i])

        if mu is None:
            d.push_back(A[i, i])
        else:
            d.push_back(mu[i])

        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian_rpt_quad(mat, d, nud)

    return hafnian_rpt_quad(mat, nud)


# ==============================================================================
# Hafnian recursive


def haf_int(long long[:, :] A):
    """Returns the hafnian of an integer matrix A via the C++ hafnian library.
    Modified with permission from https://github.com/eklotek/Hafnian.

    .. note:: Currently does not support calculation of the loop hafnian.

    Args:
        A (array): a np.int64, square, symmetric array of even dimensions.

    Returns:
        np.int64: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[long long] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    return hafnian_recursive(mat)


# ==============================================================================
# Hafnian recursive and powtrace


def haf_complex(double complex[:, :] A, bint loop=False, bint recursive=True, quad=True):
    """Returns the hafnian of a complex matrix A via the C++ hafnian library.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.
        quad (bool): If ``True``, the input matrix is cast to a ``long double complex``
            matrix internally for a quadruple precision hafnian computation.

    Returns:
        np.complex128: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double complex] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian(mat)

    if recursive:
        if quad:
            return hafnian_recursive_quad(mat)
        return hafnian_recursive(mat)

    return hafnian(mat)


def haf_real(double[:, :] A, bint loop=False, bint recursive=True, quad=True, bint approx=False, nsamples=1000):
    """Returns the hafnian of a real matrix A via the C++ hafnian library.

    Args:
        A (array): a np.float64, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.
        quad (bool): If ``True``, the input matrix is cast to a ``long double``
            matrix internally for a quadruple precision hafnian computation.
        approx (bool): If ``True``, an approximation algorithm is used to estimate the hafnian. Note that
            the approximation algorithm can only be applied to matrices ``A`` that only have non-negative entries.
        num_samples (int): If ``approx=True``, the approximation algorithm performs ``num_samples`` iterations 
        	for estimation of the hafnian of the non-negative matrix ``A``.

    Returns:
        np.float64: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian(mat)

    if approx:
        return hafnian_approx(mat, nsamples)

    if recursive:
        if quad:
            return hafnian_recursive_quad(mat)
        return hafnian_recursive(mat)

    return hafnian(mat)
