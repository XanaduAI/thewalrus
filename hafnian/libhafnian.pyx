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

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "../src/lhafnian.h":
    double complex hafnian (double complex mat[], int n)
    double complex hafnian_loops(double complex *mat, int n)

cdef extern from "../src/rlhafnian.h":
    double complex hafnian (double mat[], int n)
    double complex hafnian_loops(double *mat, int n)


@cython.boundscheck(False)
def chafnian(double[:, :] A):
    # Exposes a c function to python
    n = A.shape[0]
    return hafnian(&A[0,0], n)
