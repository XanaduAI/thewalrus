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
"""
Torontonian Python interface
"""
import numpy as np

from .lib.libhaf import torontonian_complex as tor_complex
from .lib.libhaf import torontonian_real as tor_real


def tor(A, fsum=False):
    """Returns the Torontonian of a matrix.

    For more direct control, you may wish to call :func:`tor_real` or
    :func:`tor_complex` directly.

    The input matrix is cast to quadruple precision
    internally for a quadruple precision torontonian computation.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        fsum (bool): if ``True``, the `Shewchuck algorithm <https://github.com/achan001/fsum>`_
            for more accurate summation is performed. This can significantly increase
            the `accuracy of the computation <https://link.springer.com/article/10.1007%2FPL00009321>`_,
            but no casting to quadruple precision takes place, as the Shewchuck algorithm
            only supports double precision.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return tor_complex(A, fsum=fsum)
        return tor_real(np.float64(A.real), fsum=fsum)

    return tor_real(A, fsum=fsum)
