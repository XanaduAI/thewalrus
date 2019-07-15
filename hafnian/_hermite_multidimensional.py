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
Hermite Multidimensional Python interface
"""
import numpy as np
from itertools import product

from .lib.libhaf import hermite_multidimensional as hm
from ._hafnian import input_validation

def return_prod(prim, index):
    return np.prod([prim[mode,val] for mode,val in enumerate(index)])

def expansion_coeff(alpha, resolution):
    vals = np.empty([resolution], dtype = type(alpha))
    vals[0] = 1.0
    for i in range(1,resolution):
        vals[i] = vals[i-1]*alpha/np.sqrt(i)
    return vals


def density_matrix(A, d, resolution, renorm=True):
    r"""Returns photon number statistics of a Gaussian state for a given covariance matrix `A`.

    Args:
        A (array): a square, symmetric :math:`N\times N` array.
        d (array): a one diemensional size :math:`N` array.
        resolution (int): highest number of photons to be resolved.
    Returns:
        Density Matrix

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


def hermite_multidimensional(R, resolution, y=None, renorm=False, make_tensor=True):
    r"""
    Returns the multidimensional Hermite polynomials :math:`H_k^{(R)}(y)`. Here :math:`R` is an :math:n \times n: square matrix,
    :math:`y` is an :math:`n` dimensional vector. The polynomials are parametrized by the multi-index :math:`k=(k_0,k_1,\ldots,k_{n-1})
    and are calculated for all values :math:`0 \leq k_j < \text{resolution}`.
    Thus a tensor of dimensions :math:`\text{resolution}^n` is returned. This tensor can either be flattened into a vector or returned as an
    actual tensor with :math:`n` indices.
    Note that is R = np.array([[1.0+0.0j]]) then :math:`H_k^{(R)}(y)` are precisely the well known **probabilists' Hermite polynomials** :math:`He_k(y)`:,
    and if R = 2*np.array([[1.0+0.0j]]) then :math:`H_k^{(R)}(y)` are precisely the well known **physicists' Hermite polynomials** :math:`H_k(y)`:.
    Args:
        R (array): Square matrix parametrizing the Hermite polynomial family
        resolution (int): Maximum size of the subindices in the Hermite polynomial
        y (array): Vector for the argument of the Hermite polynomial
        renorm (bool): If True returns :math:`H_k^{(R)}(y)/\prod(\prod_i k_i!)`
        make_tensor: If False returns a flattened one dimensional array instead of a tensor with the values of the polynomial
    Returns:
        (array): The multidimensional Hermite polynomials.
    """
    input_validation(R, check_symmetry=False)
    n, _ = R.shape
    if y is None:
        y = np.zeros([n], dtype=complex)

    m = y.shape[0]
    if m != n:
        raise ValueError("The matrix R and vector y have incompatible dimensions")

    values = np.array(hm(R, y, resolution, ren=renorm))

    if make_tensor:
        shape = resolution * np.ones([n], dtype=int)
        values = np.reshape(values, shape)

    return values


def hafnian_batched(A, resolution, mu=None, tol=1e-12, renorm=False, make_tensor=True):
    r"""Returns the multidimensional Hermite polynomials :math:`H_k^{(R)}(y)`.
    Args:
        A (array): Square matrix parametrizing the Hermite polynomial family
        resolution (int): Maximum size of the subindices in the Hermite polynomial
        y (array): Vector for the argument of the Hermite polynomial
        renorm (bool): If True returns :math:`H_k^{(R)}(y)/\prod(\prod_i k_i!)`
        make_tensor: If False returns a flattened one dimensional array instead of a tensor with the values of the polynomial
    Returns:
        (array): The multidimensional Hermite polynomials.

    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, tol=tol)
    n, _ = A.shape
    if not np.allclose(A, np.zeros([n,n])):
        if mu is not None:
            try:
                yi = np.linalg.solve(A, mu)
            except np.linalg.LinAlgError:
                raise ValueError("The matrix does not have an inverse")
            return hermite_multidimensional(-A, resolution, y=-yi, renorm=renorm, make_tensor=make_tensor)
        else:
            yi = np.zeros([n],dtype=complex)
            return hermite_multidimensional(-A, resolution, y=-yi, renorm=renorm, make_tensor=make_tensor)
        # Note the minus signs in the arguments. Those are intentional
    else:
        if mu is None:
            tensor = np.zeros([resolution**n],dtype=complex)
            tensor[0] = 1.0
        else:
            vecs = [expansion_coeff(alpha, resolution) for alpha in mu]
            index = resolution*np.ones([n],dtype = int)
            tensor = np.empty(index, dtype = complex)
            prim = np.array([expansion_coeff(alpha,resolution) for alpha in mu])
            for i in product(range(resolution),repeat=n):
                tensor[i] = return_prod(prim,i)
        if make_tensor:
            return tensor
        return tensor.flatten()
#=======
#    return hermite_multidimensional(-A, resolution, y=y, renorm=renorm, make_tensor=make_tensor)
#    # Note the minus signs in the arguments. Those are intentional

