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
    T hafnian_recursive[T](vector[T] &mat)
    T hafnian[T](vector[T] &mat)
    T loop_hafnian[T](vector[T] &mat)

    T hafnian_rpt[T](vector[T] &mat, vector[int] &nud, bint use_eigen)
    T hafnian_rpt[T](vector[T] &mat, vector[int] &nud)

    T loop_hafnian_rpt[T](vector[T] &mat, vector[int] &nud, bint use_eigen)
    T loop_hafnian_rpt[T](vector[T] &mat, vector[int] &nud)

    double hafnian_recursive_quad(vector[double] &mat)
    double complex hafnian_recursive_quad(vector[double complex] &mat)

    T torontonian[T](vector[T] &mat)
    double complex torontonian_quad(vector[double complex] &mat)


def torontonian_complex(double complex[:, :] A, quad=True):
    cdef int i, j, n = A.shape[0]
    cdef vector[double complex] mat
    cdef int m = n/2
    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    cdef int sign=1
    if m % 2 !=0:
        sign = -1

    if quad:
        return sign*torontonian_quad(mat)

    return sign*torontonian(mat)


def haf_rpt_real(double[:, :] A, int[:] rpt, bint loop=False, bint use_eigen=True):
    r"""Returns the hafnian of a real matrix A via the C++ hafnian library
    using the rpt method. This method is more efficient for matrices with
    repeated rows and columns.

    Args:
        A (array): a np.float64, square, :math:`N\times N` array of even dimensions.
        rpt (array): a length :math:`N` array corresponding to the number of times
            each row/column of matrix A is repeated.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        use_eigen (bool): if True (default), the Eigen linear algebra library
            is used for matrix multiplication.

    Returns:
        np.float64: the hafnian
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[int] nud
    cdef vector[double] mat

    for i in range(n):
        nud.push_back(rpt[i])
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian_rpt(mat, nud, use_eigen)

    return hafnian_rpt(mat, nud, use_eigen)


def haf_rpt_complex(double complex[:, :] A, int[:] rpt, bint loop=False, bint use_eigen=True):
    r"""Returns the hafnian of a complex matrix A via the C++ hafnian library
    using the rpt method. This method is more efficient for matrices with
    repeated rows and columns.

    Args:
        A (array): a np.complex128, square, :math:`N\times N` array of even dimensions.
        rpt (array): a length :math:`N` array corresponding to the number of times
            each row/column of matrix A is repeated.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        use_eigen (bool): if True (default), the Eigen linear algebra library
            is used for matrix multiplication.

    Returns:
        np.complex128: the hafnian
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[int] nud
    cdef vector[double complex] mat

    for i in range(n):
        nud.push_back(rpt[i])
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    if loop:
        return loop_hafnian_rpt(mat, nud, use_eigen)

    return hafnian_rpt(mat, nud, use_eigen)


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


def haf_complex(double complex[:, :] A, bint loop=False, bint recursive=False, quad=True):
    """Returns the hafnian of a complex matrix A via the C++ hafnian library.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.

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


def haf_real(double[:, :] A, bint loop=False, bint recursive=False, quad=True):
    """Returns the hafnian of a real matrix A via the C++ hafnian library.

    Args:
        A (array): a np.float64, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.
        recursive (bool): If ``True``, the recursive algorithm is used. Note:
            the recursive algorithm does not currently support the loop hafnian.

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

    if recursive:
        if quad:
            return hafnian_recursive_quad(mat)
        return hafnian_recursive(mat)

    return hafnian(mat)
