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
Complex hafnian function
========================

**Module name:** :mod:`hafnian.lhaf`

.. currentmodule:: hafnian.lhaf

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
sofile = os.path.join(path, "lib/lhafnian.so")
cdll = ctypes.CDLL(sofile)


_calc_hafnian = cdll.dhaf
_calc_hafnian.restype = ctypes.c_double


def hafnian(l):
    """Returns the hafnian of a complex matrix l via the C hafnian library.

    Args:
        l (array): a complex, square, symmetric array of even dimensions.

    Returns:
        np.complex128: the hafnian of matrix l
    """
    if l.dtype != np.complex128:
        l = l.astype(np.complex128)
    matshape = l.shape
    a = l.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    rr = np.float64(np.array([0.0, 0.0]))
    arr = rr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    res = _calc_hafnian(a, matshape[0], arr)
    return rr[0] + 1j*rr[1]
