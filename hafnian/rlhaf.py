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
Real hafnian function
========================

**Module name:** :mod:`hafnian.rlhaf`

.. currentmodule:: hafnian.rlhaf

Summary
--------

.. autosummary::
    hafnian

Code details
-------------
"""
import ctypes
import os
import sys
import numpy as np


path = os.path.dirname(__file__)
sofile = os.path.join(path, "lib/rlhafnian.so")
cdll = ctypes.CDLL(sofile)


_calc_hafnian = cdll.dhaf
_calc_hafnian.restype = ctypes.c_double


def hafnian(l, tol=1e-12):
    """Returns the hafnian of real matrix l via the C hafnian library.

    Args:
        l (array): a real, square, symmetric array of even dimensions.
        tol (float): the tolerance when checking that the matrix is
            symmetric. Default tolerance is 1e-12.

    Returns:
        np.float64: the hafnian of matrix l
    """
    matshape = l.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if not isinstance(l, np.ndarray):
        raise ValueError("Input matrix must be a NumPy array.")

    if matshape[0] % 2 != 0:
        raise ValueError("Input matrix must be of even dimensions.")

    if np.linalg.norm(l-np.transpose(l)) >= tol:
        raise ValueError("Input matrix must be symmetric.")

    if matshape[0] == 2:
        return l[0][1]
    else:
        if l.dtype != np.float64:
            l = l.astype(np.float64)

        a = l.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rr = np.float64(np.array([0.0,0.0]))
        arr = rr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        res = _calc_hafnian(a, matshape[0], arr)
        return rr[0]
