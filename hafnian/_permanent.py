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
"""
Permanent Python interface
"""
import numpy as np

from .lib.libperm import perm as libperm

perm_real = libperm.re
perm_complex = libperm.comp


def perm(A):
    """Returns the permanent of matrix A via the Fortran permanent library.

    If the array is real valued (np.float), the result of
    :func:`perm_real` is returned.

    * If the array is complex (np.complex), this function queries
      whether the array A has non-zero imaginary part. If so, it
      calls the :func:`perm_complex` function.

    * Otherwise, if all elements are exactly real, the
      :func:`perm_real` function is called.

    For more direct control, you may wish to call :func:`perm_real`
    or :func:`perm_complex` directly.

    Args:
        A (array): a square array.

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
        return A[0, 0]*A[1, 1] + A[0, 1]*A[1, 0]

    if matshape[0] == 3:
        return A[0, 2]*A[1, 1]*A[2, 0] + A[0, 1]*A[1, 2]*A[2, 0] \
            + A[0, 2]*A[1, 0]*A[2, 1] + A[0, 0]*A[1, 2]*A[2, 1] \
            +  A[0, 1]*A[1, 0]*A[2, 2] + A[0, 0]*A[1, 1]*A[2, 2]

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return perm_complex(A)
        return perm_real(np.float64(A.real))

    return perm_real(A)
