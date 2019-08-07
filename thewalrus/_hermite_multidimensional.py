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

from .libwalrus import hermite_multidimensional as hm
from ._hafnian import input_validation


def return_prod(C, index):
    r"""Given an array :math:`C_{i,j}` and an array or list of indices
    :math:`index = [i_1,i_2,i_3,\dots,i_n] `, returns :math:`prod_{k=1}^n C_{k,i_k}`.

    Args:
        C (array): An array
        index (array): A set of indices

    Returns:
        complex: the product of the array elements determined by index
    """
    return np.prod([C[mode, val] for mode, val in enumerate(index)])


def expansion_coeff(alpha, cutoff, renorm=True):
    r"""Returns the (quasi) geometric series as a vector with components
    :math:`\alpha^i/\sqrt{i!}` for :math:`0 \leq i < \texttt{cutoff}`.

    Args:
        alpha (complex): ratio of the geometric series
        cutoff (int): cutoff truncation of the geometric series
        renorm (bool): if ``False``, the components are not normalized
            by the square-root factorials

    Returns:
        array: the (quasi) geometric series
    """
    vals = np.empty([cutoff], dtype=type(alpha))
    vals[0] = 1.0
    if renorm:
        for i in range(1, cutoff):
            vals[i] = vals[i - 1] * alpha / np.sqrt(i)
    else:
        for i in range(1, cutoff):
            vals[i] = vals[i - 1] * alpha
    return vals


def hermite_multidimensional(R, cutoff, y=None, renorm=False, make_tensor=True):
    r"""Returns the multidimensional Hermite polynomials :math:`H_k^{(R)}(y)`.

    Here :math:`R` is an :math:`n \times n` square matrix, and
    :math:`y` is an :math:`n` dimensional vector. The polynomials are
    parametrized by the multi-index :math:`k=(k_0,k_1,\ldots,k_{n-1})`,
    and are calculated for all values :math:`0 \leq k_j < \text{cutoff}`,
    thus a tensor of dimensions :math:`\text{cutoff}^n` is returned.

    This tensor can either be flattened into a vector or returned as an actual
    tensor with :math:`n` indices.

    .. note::

        Note that if :math:`R = (1)` then :math:`H_k^{(R)}(y)`
        are precisely the well known **probabilists' Hermite polynomials** :math:`He_k(y)`,
        and if :math:`R = (2)` then :math:`H_k^{(R)}(y)` are precisely the well known
        **physicists' Hermite polynomials** :math:`H_k(y)`.

    Args:
        R (array): square matrix parametrizing the Hermite polynomial family
        cutoff (int): maximum size of the subindices in the Hermite polynomial
        y (array): vector argument of the Hermite polynomial
        renorm (bool): If ``True``, normalizes the returned multidimensional Hermite
            polynomials such that :math:`H_k^{(R)}(y)/\prod(\prod_i k_i!)`
        make_tensor: If ``False``, returns a flattened one dimensional array
            containing the values of the polynomial

    Returns:
        (array): the multidimensional Hermite polynomials
    """
    input_validation(R)
    n, _ = R.shape
    if y is None:
        y = np.zeros([n], dtype=complex)

    m = y.shape[0]
    if m != n:
        raise ValueError("The matrix R and vector y have incompatible dimensions")

    values = np.array(hm(R, y, cutoff, renorm=renorm))

    if make_tensor:
        shape = cutoff * np.ones([n], dtype=int)
        values = np.reshape(values, shape)

    return values


def hafnian_batched(A, cutoff, mu=None, tol=1e-12, renorm=False, make_tensor=True):
    r"""Calculates the hafnian of :func:`reduction(A, k) <hafnian.reduction>`
    for all possible values of vector ``k`` below the specified cutoff.

    Here,

    * :math:`A` is am :math:`n\times n` square matrix
    * :math:`k` is a vector of (non-negative) integers with the same dimensions as :math:`A`,
      i.e., :math:`k = (k_0,k_1,\ldots,k_{n-1})`, and where :math:`0 \leq k_j < \texttt{cutoff}`.

    The function :func:`~.hafnian_repeated` can be used to calculate the reduced hafnian
    for a *specific* value of :math:`k`; see the documentation for more information.

    .. note::

        If ``mu`` is not ``None``, this function instead returns
        ``hafnian(np.fill_diagonal(reduction(A, k), reduction(mu, k)), loop=True)``.
        This calculation can only be performed if the matrix :math:`A` is invertible.

    Args:
        A (array): a square, symmetric :math:`N\times N` array.
        cutoff (int): maximum size of the subindices in the Hermite polynomial
        mu (array): a vector of length :math:`N` representing the vector of means/displacement
        renorm (bool): If ``True``, the returned hafnians are *normalized*, that is,
            :math:`haf(reduction(A, k))/\prod_i k_i!`
            (or :math:`lhaf(fill\_diagonal(reduction(A, k),reduction(mu, k)))` if
            ``mu`` is not None)
        make_tensor: If ``False``, returns a flattened one dimensional array instead
            of a tensor with the values of the hafnians.

    Returns:
        (array): the values of the hafnians for each value of :math:`k` up to the cutoff
    """
    # pylint: disable=too-many-return-statements,too-many-branches,too-many-arguments
    input_validation(A, tol=tol)
    n, _ = A.shape

    if not np.allclose(A, np.zeros([n, n])):
        if mu is not None:
            try:
                yi = np.linalg.solve(A, mu)
            except np.linalg.LinAlgError:
                raise ValueError("The matrix does not have an inverse")
            return hermite_multidimensional(
                -A, cutoff, y=-yi, renorm=renorm, make_tensor=make_tensor
            )
        yi = np.zeros([n], dtype=complex)
        return hermite_multidimensional(
            -A, cutoff, y=-yi, renorm=renorm, make_tensor=make_tensor
        )
    # Note the minus signs in the arguments. Those are intentional and are due to the fact that Dodonov et al. in PRA 50, 813 (1994) use (p,q) ordering instead of (q,p) ordering

    if mu is None:
        tensor = np.zeros([cutoff ** n], dtype=complex)
        tensor[0] = 1.0
    else:
        index = cutoff * np.ones([n], dtype=int)
        tensor = np.empty(index, dtype=complex)
        prim = np.array([expansion_coeff(alpha, cutoff, renorm=renorm) for alpha in mu])
        for i in product(range(cutoff), repeat=n):
            tensor[i] = return_prod(prim, i)

    if make_tensor:
        return tensor

    return tensor.flatten()
