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
cimport cython
from cython.parallel cimport prange
cimport openmp
from libc.stdlib import malloc, free

import numpy as np
cimport numpy as np

from scipy.linalg.cython_lapack cimport dgees


cdef extern from "../src/rlhafnian.h":
    double hafnian (double mat[], int n)
    double hafnian_loops(double *mat, int n)


cdef class Vector:
    cdef long long *data
    cdef public int n_ax0

    def __init__(Vector self, int n_ax0):
        self.data = <long long*> malloc (sizeof(long long) * n_ax0)
        self.n_ax0 = n_ax0

    def __dealloc__(Vector self):
        free(self.data)


cdef public void evals(double *z, double complex *vals, int n,
                       double *wr, double *wi, int lwork, double *work) nogil:
    """Provides a C interface to the SciPy dgees interface."""

    cdef int lda = n, ldvs = n, sdim = 0, info, i

    dgees('N', 'N', NULL, &n, &z[0], &lda, &sdim,
        &wr[0], &wi[0], NULL, &ldvs, &work[0], &lwork, NULL, &info)

    for i in range(n):
        vals[i] = wr[i] + 1j*wi[i]


def haf_real(double[:, :] A, bint loop=False):
    """Returns the hafnian of a real matrix A via the C hafnian library.

    Args:
        A (array): a np.float64, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.float64: the hafnian of matrix A
    """
    # Exposes a c function to python
    n = A.shape[0]
    if loop:
        return hafnian_loops(&A[0, 0], n)
    return hafnian(&A[0, 0], n)


cpdef public long long haf_int(long long[:, :] A):
    """Returns the hafnian of an integer matrix A via the C hafnian library.

    Modified with permission from https://github.com/eklotek/Hafnian.

    Args:
        A (array): a np.int64, square, symmetric array of even dimensions.

    Returns:
        np.int64: the hafnian of matrix A
    """
    cdef:
        int n = A.shape[0]//2, j, k
        long long[:, :] z = np.zeros([n*(2*n-1), n+1], dtype=np.int64)
        long long[:] g = np.zeros([n+1], dtype=np.int64)

    with nogil:
        g[0] = 1
        for j in prange(1, 2*n):
            for k in prange(j):
                z[j*(j-1)//2+k, 0] = A[j, k]

    return solve(z, 2*n, 1, g, n)


cdef long long solve(long long[:, :] b, int s, int w, long long[:] g, int n):
    """Recursive integer hafnian solver."""
    if s == 0:
        return w*g[n]

    cdef:
        long long[:, :] c = np.zeros([(s-2)*(s-3)/2, n+1], dtype=np.int64)
        long long[:] e = np.zeros([n+1], dtype=np.int64)
        long long h
        int u, v, j, k, i = 0

    for j in range(1, s-2):
        for k in range(j):
            for u in range(n+1):
                c[i, u] = b[(j+1)*(j+2)/2+k+2, u]
            i += 1

    h = solve(c, s-2, -w, g, n)

    e[:] = g
    for u in range(n):
        for v in range(n-u):
            e[u+v+1] += g[u]*b[0, v]

    with nogil:
        for j in prange(1, s-2):
            for k in range(j):
                for u in range(n):
                    for v in range(n-u):
                        c[j*(j-1)/2+k, u+v+1] += b[(j+1)*(j+2)/2, u]*b[(k+1)*(k+2)/2+1, v] \
                            + b[(k+1)*(k+2)/2, u]*b[(j+1)*(j+2)/2+1, v]

    return h + solve(c, s-2, w, e, n)
