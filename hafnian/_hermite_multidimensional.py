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
Hafnian Python interface
"""
import numpy as np

from .lib.libhaf import hermite_multidimensional as hm



def density_matrix(A, d, resolution, renorm=False):
    r"""Returns photon number statistics of a Gaussian state for a given covariance matrix `A`. 
    

    Args:
        A (array): a square, symmetric :math:`N\times N` array.
        d (array): a one diemensional size :math:`N` array.
        resolution (int): highest number of photons to be resolved.

    """

    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)

    for i in range(dim // 2):
        U[i, i] = 0 - 1j
        U[i + dim // 2, i] = 1 + 0 * 1j
        U[i, i + dim // 2] = 0 + 1j

    U = U / np.sqrt(2)

    U3 = U.transpose()
    U1 = U3.conjugate()
    U2 = U.conjugate()

    tmp1 = np.eye(dim, dtype=complex) + 2 * A
    tmp1_inv = np.linalg.inv(tmp1)

    tmp2 = np.eye(dim, dtype=complex) - 2 * A
    tmp2_inv = np.linalg.inv(tmp2)

    tmp = tmp1_inv @ U2
    tmp = tmp2 @ tmp
    R = U1 @ tmp

    tmpy = tmp2_inv @ d
    y = U3 @ tmpy
    y = 2 * y

    return hm(R, y, resolution, ren=renorm)

def hermite_multidimensional(R, y, resolution, renorm=False):

    return hm(R, y, resolution, ren=renorm)
