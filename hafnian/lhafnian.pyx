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
cimport cython
import numpy as np
cimport numpy as np


cdef extern from "../src/lhafnian.h":
    double complex hafnian (double complex mat[], int n)
    double complex hafnian_loops(double complex *mat, int n)


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def haf_complex(double complex[:, :] A, bint loop=False):
    """Returns the hafnian of a complex matrix A via the C hafnian library.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.complex128: the hafnian of matrix A
    """
    # Exposes a c function to python
    n = A.shape[0]
    if loop:
        return hafnian_loops(&A[0,0], n)
    return hafnian(&A[0,0], n)
