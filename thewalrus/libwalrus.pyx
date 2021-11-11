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
import cython
cimport numpy as np
import numpy as np
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
from libcpp.vector cimport vector

np.import_array()


cdef class ArrayWrapper:
    """Wrapper class to pass arrays of complex doubles transparently between
    C++ and python.
    Modified from:
    https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
    """
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_COMPLEX128, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        free(<void*>self.data_ptr)


cdef class ArrayWrapperFloat:
    """Wrapper class to pass arrays of doubles transparently between
    C++ and python.
    Modified from:
    https://stackoverflow.com/questions/45133276/passing-c-vector-to-numpy-through-cython-without-copying-and-taking-care-of-me
    """
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_FLOAT64, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        free(<void*>self.data_ptr)


cdef double_pointer_to_array(double *array, int length):
    """Converts a pointer of C doubles into a numpy array.

    Args:
        array (double*): pointer of double
        length (int): size of the array

    Returns:
        array[float64]: a numpy array with the contents of the input array
    """

    cdef np.ndarray ndarray
    array_wrapper = ArrayWrapperFloat()
    array_wrapper.set_data(length, <void*> array)
    ndarray = np.array(array_wrapper, copy=False)
    ndarray.base = <PyObject*> array_wrapper
    Py_INCREF(array_wrapper)
    return ndarray


cdef complex_pointer_to_array(double complex *array, int length):
    """Converts a pointer of C complex doubles into a numpy array.

    Args:
        array (double complex*): pointer of double
        length (int): size of the array

    Returns:
        array[complex128]: a numpy array with the contents of the input array
    """

    cdef np.ndarray ndarray
    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(length, <void*> array)
    ndarray = np.array(array_wrapper, copy=False)
    ndarray.base = <PyObject*> array_wrapper
    Py_INCREF(array_wrapper)
    return ndarray



cdef extern from "../include/libwalrus.hpp" namespace "libwalrus":
    T hafnian[T](vector[T] &mat)
    T hafnian_recursive[T](vector[T] &mat)
    T loop_hafnian[T](vector[T] &mat)
    T permanent[T](vector[T] &mat)
    T perm_BBFG[T](vector[T] &mat) 

    T hafnian_rpt[T](vector[T] &mat, vector[int] &nud)
    T loop_hafnian_rpt[T](vector[T] &mat, vector[T] &mu, vector[int] &nud)

    double permanent_quad(vector[double] &mat)
    double complex permanent_quad(vector[double complex] &mat)
    double perm_fsum[T](vector[T] &mat)
    double permanent_fsum(vector[double] &mat)
    double perm_BBFG_qp(vector[double] &mat)
    double complex perm_BBFG_cmplx(vector[double complex] &mat)

    double hafnian_recursive_quad(vector[double] &mat)
    double complex hafnian_recursive_quad(vector[double complex] &mat)

    double loop_hafnian_quad(vector[double] &mat)
    double complex loop_hafnian_quad(vector[double complex] &mat)

    double hafnian_rpt_quad(vector[double] &mat, vector[int] &nud)
    double complex hafnian_rpt_quad(vector[double complex] &mat, vector[int] &nu)

    double loop_hafnian_rpt_quad(vector[double] &mat, vector[double] &mu, vector[int] &nud)
    double complex loop_hafnian_rpt_quad(vector[double complex] &mat, vector[double complex] &mu, vector[int] &nud)


# ==============================================================================
# Hafnian repeated


def haf_rpt_real(double[:, :] A, int[:] rpt, double[:] mu=None, bint loop=False):
    r"""Returns the hafnian of a real matrix A via the C++ libwalrus library
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
    r"""Returns the hafnian of a complex matrix A via the C++ libwalrus library
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
    """Returns the hafnian of an integer matrix A via the C++ libwalrus library.
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
    """Returns the hafnian of a complex matrix A via the C++ libwalrus library.

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
        if quad:
            return loop_hafnian_quad(mat)
        return loop_hafnian(mat)

    if recursive:
        if quad:
            return hafnian_recursive_quad(mat)
        return hafnian_recursive(mat)

    return hafnian(mat)


def haf_real(double[:, :] A, bint loop=False, bint recursive=True, quad=True, bint approx=False, nsamples=1000):
    """Returns the hafnian of a real matrix A via the C++ libwalrus library.

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
        if quad:
            return loop_hafnian_quad(mat)
        return loop_hafnian(mat)

    if recursive:
        if quad:
            return hafnian_recursive_quad(mat)
        return hafnian_recursive(mat)

    return hafnian(mat)



# ==============================================================================
# Permanent

def perm_complex(double complex[:, :] A, quad=True):
    """Returns the hafnian of a complex matrix A via the C++ libwalrus library.

    Args:
        A (array): a np.float, square array
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
    if quad:
        return permanent_quad(mat)

    return permanent(mat)

def perm_real(double [:, :] A, quad=True, fsum=False):
    """Returns the hafnian of a real matrix A via the C++ libwalrus library.

    Args:
        A (array): a np.float64, square array
        quad (bool): If ``True``, the input matrix is cast to a ``long double``
            matrix internally for a quadruple precision hafnian computation.
        fsum (bool): If ``True``, ``fsum`` method is used for summation.


    Returns:
        np.float64: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    if fsum:
        return permanent_fsum(mat)

    # Exposes a c function to python
    if quad:
        return permanent_quad(mat)

    return permanent(mat)

def perm_BBFG_complex(double complex[:, :] A):
    """Returns the hafnian of a complex matrix A via the C++ libwalrus library
        using Balasubramanian-Bax-Franklin-Glynn formula.

    Args:
        A (array): a np.float, square array

    Returns:
        np.complex128: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double complex] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    return perm_BBFG_cmplx(mat)

def perm_BBFG_real(double [:, :] A):
    """Returns the hafnian of a real matrix A via the C++ libwalrus library
        using Balasubramanian-Bax-Franklin-Glynn formula.

    Args:
        A (array): a np.float64, square array

    Returns:
        np.float64: the hafnian of matrix A
    """
    cdef int i, j, n = A.shape[0]
    cdef vector[double] mat

    for i in range(n):
        for j in range(n):
            mat.push_back(A[i, j])

    # Exposes a c function to python
    return perm_BBFG_qp(mat)
