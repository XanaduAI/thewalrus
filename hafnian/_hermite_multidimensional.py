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
from itertools import product
import numpy as np

from .lib.libhaf import hermite_multidimensional as hm
from ._hafnian import input_validation


def return_prod(C, index):
    """Given an array :math:`C_{i,j}` and an array or list of indices :math:`index = [i_1,i_2,i_3,\dots,i_n] `, returns :math:`prod_{k=1}^n C_{1,i_1}`.

    Args:
        C (array): An array
        index (array): A set of indices
    Returns:
        complex: The product of the array elements determines by index
    """
    return np.prod([C[mode, val] for mode, val in enumerate(index)])


def expansion_coeff(alpha, resolution, renorm=True):
    """
    Returns the (quasi) geometric series as a vector with components alpha^i/sqrt(i!) for 0 <= i < resolution.
    If renorm is false it omits the division by factorials
    Args:
        alpha (complex): Ratio of the geometric series
        resoluton (int): Cutoff of the geometric series
        renor (bool): Decides whether to normalize by the factorials
    Returns:
        array: The power series
    """
    vals = np.empty([resolution], dtype=type(alpha))
    vals[0] = 1.0
    if renorm:
        for i in range(1, resolution):
            vals[i] = vals[i - 1] * alpha / np.sqrt(i)
    else:
        for i in range(1, resolution):
            vals[i] = vals[i - 1] * alpha
    return vals


def hermite_multidimensional(R, resolution, y=None, renorm=False, make_tensor=True):
    r"""
    Returns the multidimensional Hermite polynomials :math:`H_k^{(R)}(y)`. Here :math:`R` is an :math:n \times n: square matrix,
    :math:`y` is an :math:`n` dimensional vector. The polynomials are parametrized by the multi-index :math:`k=(k_0,k_1,\ldots,k_{n-1})
    and are calculated for all values :math:`0 \leq k_j < \text{resolution}`, thus a tensor of dimensions :math:`\text{resolution}^n` is returned.
    This tensor can either be flattened into a vector or returned as an actual tensor with :math:`n` indices.
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
    r"""Returns the haf(reduction(A, k)) where k is a vector of (non-negative) integers with the same dimensions as the square matrix A
    :math:`k = (k_0,k_1,\ldots,k_{n-1})` and where :math:`0 \leq k_j < \text{resolution}`.
    If mu is not None the it instead calculates :math:`lhaf(fill_diagonal(reduction(A, k),reduction(mu, k)))`, this calculation can only be performed if
    the matrix A has an inverse.
    Args:
        R (array): Square matrix parametrizing
        resolution (int): Maximum size of the subindices in the Hermite polynomial
        y (array): Vector for the argument of the Hermite polynomial
        renorm (bool): If True returns :math:`haf(reduction(A, k))/\prod(\prod_i k_i!)` or :math:`lhaf(fill_diagonal(reduction(A, k),reduction(mu, k)))` is mu is not None
        make_tensor: If False returns a flattened one dimensional array instead of a tensor with the values of the polynomial
    Returns:
        (array): The values of the hafnians.
    """
    # pylint: disable=too-many-return-statements,too-many-branches
    input_validation(A, tol=tol)
    n, _ = A.shape
    if not np.allclose(A, np.zeros([n, n])):
        if mu is not None:
            try:
                yi = np.linalg.solve(A, mu)
            except np.linalg.LinAlgError:
                raise ValueError("The matrix does not have an inverse")
            return hermite_multidimensional(
                -A, resolution, y=-yi, renorm=renorm, make_tensor=make_tensor
            )
        yi = np.zeros([n], dtype=complex)
        return hermite_multidimensional(
            -A, resolution, y=-yi, renorm=renorm, make_tensor=make_tensor
        )
    # Note the minus signs in the arguments. Those are intentional

    if mu is None:
        tensor = np.zeros([resolution ** n], dtype=complex)
        tensor[0] = 1.0
    else:
        index = resolution * np.ones([n], dtype=int)
        tensor = np.empty(index, dtype=complex)
        prim = np.array([expansion_coeff(alpha, resolution, renorm=renorm) for alpha in mu])
        for i in product(range(resolution), repeat=n):
            tensor[i] = return_prod(prim, i)
    if make_tensor:
        return tensor
    return tensor.flatten()
