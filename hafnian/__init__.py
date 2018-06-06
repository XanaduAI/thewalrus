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
Hafnian Python interface
========================

This is the top level module of the Hafnian python interface,
containing the function :func:`hafnian`. This function determines,
based on the input matrix, whether to use the complex C hafnian
library, or the slightly faster real C hafnian library.

For more advanced usage, the functions :func:`haf_real` and
:func:`haf_complex` are also provided.

Top-level functions
-------------------

.. autosummary::
   hafnian
   haf_complex
   haf_real
   version

Code details
-------------
"""

import numpy as np

from ._version import __version__
from ._lhafnian import hafnian as haf_complex
from ._rlhafnian import hafnian as haf_real


__all__ = ['hafnian', 'haf_complex', 'haf_real', 'version']


def hafnian(A, loop=False, tol=1e-12):
    """Returns the hafnian of matrix A via the C hafnian library.

    .. note::

        If the array is real valued (np.float), the result of
        :func:`haf_real` is returned.

        If the array is complex (np.complex), this function queries
        whether the array A has non-zero imaginary part. If so, it
        calls the :func:`haf_complex` function. Otherwise, if all elements
        are exactly real, the :func:`haf_real` function is called.

        For more direct control, you may wish to call :func:`haf_real`
        or :func:`haf_complex` directly.

    Args:
        A (array): a square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default is ``False``.
        toA (float): the tolerance when checking that the matrix is
            symmetric. Default tolerance is 1e-12.

    Returns:
        np.float64 or np.complex128: the hafnian of matrix l
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if matshape[0] % 2 != 0:
        raise ValueError("Input matrix must be of even dimensions.")

    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if np.linalg.norm(A-np.transpose(A)) >= tol:
        raise ValueError("Input matrix must be symmetric.")

    if matshape[0] == 2:
        if loop:
            return A[0, 1] + A[0, 0]*A[1, 1]
        return A[0][1]

    if matshape[0] == 4:
        if loop:
            result = A[0, 1]*A[2, 3] \
                + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2] \
                + A[0, 0]*A[1, 1]*A[2, 3] + A[0, 1]*A[2, 2]*A[3, 3] \
                + A[0, 2]*A[1, 1]*A[3, 3] + A[0, 0]*A[2, 2]*A[1, 3] \
                + A[0, 0]*A[3, 3]*A[1, 2] + A[0, 3]*A[1, 1]*A[2, 2] \
                + A[0, 0]*A[1, 1]*A[2, 2]*A[3, 3]
            return result

        return A[0, 1]*A[2, 3] + A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return haf_complex(A, loop=loop)

        return haf_real(np.float64(A.real), loop=loop)

    return haf_real(A, loop=loop)


def version():
    r"""
    Get version number of Hafnian

    Returns:
      str: The package version number
    """
    return __version__
