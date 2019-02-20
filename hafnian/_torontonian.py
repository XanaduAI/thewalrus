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

from .lib.libtor import torontonian as libtor

tor_complex = libtor.tor
det_real = libtor.det_real


def tor(A):
    """Returns the Torontonian of matrix A via the Fortran torontonian library.

    For more direct control, you may wish to call :func:`tor_complex` directly.

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

    return tor_complex(A)


def det(A):

    return det_real(A)
