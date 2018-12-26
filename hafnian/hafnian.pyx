# Copyright 2018 Xanadu Quantum Technologies Inc.

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

cimport numpy as np


cdef extern from "../src/hafnian.hpp":
    long long hafnian_int(vector[long long] &mat)
    T hafnian[T](vector[T] &mat)
    T loop_hafnian[T](vector[T] &mat)


cdef vector[long long] i_arrayToVector(np.ndarray[np.int64_t, ndim=1] array):
    cdef long size = array.size
    cdef vector[long long] vec
    cdef long i

    for i in range(size):
        vec.push_back(array[i])

    return vec


cdef vector[double complex] c_arrayToVector(np.ndarray[np.complex128_t, ndim=1] array):
    cdef long size = array.size
    cdef vector[double complex] vec
    cdef long i

    for i in range(size):
        vec.push_back(array[i])

    return vec


cdef vector[double] d_arrayToVector(np.ndarray[np.double_t, ndim=1] array):
    cdef long size = array.size
    cdef vector[double] vec
    cdef long i

    for i in range(size):
        vec.push_back(array[i])

    return vec


def haf_int(np.ndarray[np.int64_t, ndim=2] A):
    """Returns the hafnian of an integer matrix A via the C hafnian library.
    Modified with permission from https://github.com/eklotek/Hafnian.

    Args:
        A (array): a np.int64, square, symmetric array of even dimensions.

    Returns:
        np.int64: the hafnian of matrix A
    """
    flat_vec = A.flatten()
    cdef vector[long long] mat = i_arrayToVector(flat_vec)
    # Exposes a c function to python
    return hafnian_int(mat)


def haf_complex(np.ndarray[np.complex128_t, ndim=2] A, bint loop=False):
    """Returns the hafnian of a complex matrix A via the C hafnian library.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.complex128: the hafnian of matrix A
    """
    flat_vec = A.flatten()
    cdef vector[double complex] mat = c_arrayToVector(flat_vec)
    # Exposes a c function to python
    if loop:
        return loop_hafnian(mat)

    return hafnian(mat)


def haf_real(np.ndarray[np.double_t, ndim=2] A, bint loop=False):
    """Returns the hafnian of a real matrix A via the C hafnian library.

    Args:
        A (array): a np.float128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.float128: the hafnian of matrix A
    """
    flat_vec = A.flatten()
    cdef vector[double] mat = d_arrayToVector(flat_vec)
    # Exposes a c function to python
    if loop:
        return loop_hafnian(mat)
    return hafnian(mat)
