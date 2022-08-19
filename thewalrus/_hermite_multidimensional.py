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
from typing import Tuple, Generator, Iterable
from numba import jit, njit
from numba.cpython.unsafe.tuple import tuple_setitem
import numpy as np

from ._hafnian import input_validation


# pylint: disable=too-many-arguments
def hermite_multidimensional(
    R, cutoff, y=None, C=1, renorm=False, make_tensor=True, modified=False, rtol=1e-05, atol=1e-08
):
    r"""Returns photon number statistics of a Gaussian state for a given
    covariance matrix as described in *Multidimensional Hermite polynomials
    and photon distribution for polymode mixed light*
    `arxiv:9308033 <https://arxiv.org/abs/hep-th/9308033>`_.

    Here :math:`R` is an :math:`n \times n` square matrix, and
    :math:`y` is an :math:`n` dimensional vector. The polynomials :math:`H_k^{(R)}(y)` are
    parametrized by the multi-index :math:`k=(k_0,k_1,\ldots,k_{n-1})`,
    and are calculated for all values :math:`0 \leq k_j < \text{cutoff}`,
    thus a tensor of dimensions :math:`\text{cutoff}^n` is returned.

    This tensor can either be flattened into a vector or returned as an actual
    tensor with :math:`n` indices.

    This implementation is based on the MATLAB code available at github
    `clementsw/gaussian-optics <https://github.com/clementsw/gaussian-optics>`_.

    .. note::

        Note that if :math:`R = (1)` then :math:`H_k^{(R)}(y)`
        are precisely the well known **probabilists' Hermite polynomials** :math:`He_k(y)`,
        and if :math:`R = (2)` then :math:`H_k^{(R)}(y)` are precisely the well known
        **physicists' Hermite polynomials** :math:`H_k(y)`.

    Args:
        R (array): square matrix parametrizing the Hermite polynomial family
        cutoff (int): maximum size of the subindices in the Hermite polynomial
        y (array): vector argument of the Hermite polynomial
        C (complex): first value of the Hermite polynomials, the default value is 1
        renorm (bool): If ``True``, normalizes the returned multidimensional Hermite
            polynomials such that :math:`H_k^{(R)}(y)/\prod_i k_i!`
        make_tensor (bool): If ``False``, returns a flattened one dimensional array
            containing the values of the polynomial
        modified (bool): whether to return the modified multidimensional Hermite polynomials or the standard ones
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``
    Returns:
        (array): the multidimensional Hermite polynomials
    """

    input_validation(R, atol=atol, rtol=rtol)
    n, _ = R.shape

    if (modified is False) and (y is not None):
        m = y.shape[0]
        if m == n:
            ym = R @ y
            return hermite_multidimensional(
                R, cutoff, y=ym, C=C, renorm=renorm, make_tensor=make_tensor, modified=True
            )

    if y is None:
        y = np.zeros([n], dtype=complex)

    m = y.shape[0]
    if m != n:
        raise ValueError("The matrix R and vector y have incompatible dimensions")

    num_indices = len(y)
    # we want to catch np.ndarray(int) of ndim=0 which cannot be cast to tuple
    if isinstance(cutoff, np.ndarray) and (cutoff.ndim == 0 or len(cutoff) == 1):
        cutoff = int(cutoff)
    if isinstance(cutoff, Iterable):
        cutoffs = tuple(cutoff)
    else:
        cutoffs = tuple([cutoff]) * num_indices

    Rt = np.real_if_close(R)
    yt = np.real_if_close(y)

    dtype = np.find_common_type([Rt.dtype.name, yt.dtype.name], [np.array(C).dtype.name])
    array = np.zeros(cutoffs, dtype=dtype)
    array[(0,) * num_indices] = C

    if renorm:
        values = np.array(_hermite_multidimensional_renorm(Rt, yt, array))
    else:
        values = np.array(_hermite_multidimensional(Rt, yt, array))

    if not make_tensor:
        values = values.flatten()

    return values


def interferometer(R, cutoff, C=1, renorm=True, make_tensor=True, rtol=1e-05, atol=1e-08):
    r"""Returns the matrix elements of an interferometer parametrized in terms of its R matrix.

    Here :math:`R` is an :math:`n \times n` square matrix. The polynomials are
    parametrized by the multi-index :math:`k=(k_0,k_1,\ldots,k_{n-1})`,
    and are calculated for all values :math:`0 \leq k_j < \text{cutoff}`,
    thus a tensor of dimensions :math:`\text{cutoff}^n` is returned.

    This tensor can either be flattened into a vector or returned as an actual
    tensor with :math:`n` indices.

    .. note::

        Note that `interferometer` uses the normalized multidimensional Hermite polynomials.

    Args:
        R (array): square matrix parametrizing the Hermite polynomial family
        cutoff (int): maximum size of the subindices in the Hermite polynomial
        C (complex): first value of the Hermite polynomials, the default value is 1
        renorm (bool): If ``True``, normalizes the returned multidimensional Hermite
            polynomials such that :math:`H_k^{(R)}(y)/\prod_i k_i!`
        make_tensor (bool): If ``False``, returns a flattened one dimensional array
            containing the values of the polynomial
        rtol (float): the relative tolerance parameter used in ``np.allclose``
        atol (float): the absolute tolerance parameter used in ``np.allclose``
    Returns:
        (array): the multidimensional Hermite polynomials
    """

    input_validation(R, atol=atol, rtol=rtol)
    n, num_indices = R.shape

    # we want to catch np.ndarray(int) of ndim=0 which cannot be cast to tuple
    if isinstance(cutoff, np.ndarray) and (cutoff.ndim == 0 or len(cutoff) == 1):
        cutoff = int(cutoff)
    if isinstance(cutoff, Iterable):
        cutoffs = tuple(cutoff)
    else:
        cutoffs = tuple([cutoff]) * num_indices

    Rt = np.real_if_close(R)

    dtype = np.find_common_type([Rt.dtype.name], [np.array(C).dtype.name])
    array = np.zeros(cutoffs, dtype=dtype)
    array[(0,) * num_indices] = C

    if renorm:
        values = np.array(_interferometer_renorm(Rt, array))
    else:
        values = np.array(_interferometer(Rt, array))

    if make_tensor:
        shape = cutoff * np.ones([n], dtype=int)
        values = np.reshape(values, shape)

    return values


# pylint: disable=too-many-arguments
def hafnian_batched(A, cutoff, mu=None, rtol=1e-05, atol=1e-08, renorm=False, make_tensor=True):
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
            :math:`haf(reduction(A, k))/\ \sqrt{prod_i k_i!}`
            (or :math:`lhaf(fill\_diagonal(reduction(A, k),reduction(mu, k)))` if
            ``mu`` is not None)
        make_tensor: If ``False``, returns a flattened one dimensional array instead
            of a tensor with the values of the hafnians.
        rtol (float): the relative tolerance parameter used in ``np.allclose``.
        atol (float): the absolute tolerance parameter used in ``np.allclose``.
    Returns:
        (array): the values of the hafnians for each value of :math:`k` up to the cutoff
    """
    input_validation(A, atol=atol, rtol=rtol)
    n, _ = A.shape

    if mu is None:
        mu = np.zeros([n], dtype=complex)
    # The minus signs are intentional and are due to the fact that
    # Dodonov et al. in PRA 50, 813 (1994) use (p,q) ordering instead of (q,p) ordering
    return hermite_multidimensional(
        -A, cutoff, y=mu, renorm=renorm, make_tensor=make_tensor, modified=True
    )


@jit(nopython=True)
def dec(tup: Tuple[int], i: int) -> Tuple[int, ...]:  # pragma: no cover
    r"""returns a copy of the given tuple of integers where the ith element has been decreased by 1
    Args:
        tup (Tuple[int]): the given tuple
        i (int): the position of the element to be decreased
    Returns:
        Tuple[int,...]: the new tuple with the decrease on i-th element by 1
    """
    copy = tup[:]
    return tuple_setitem(copy, i, tup[i] - 1)


@jit(nopython=True)
def remove(
    pattern: Tuple[int, ...]
) -> Generator[Tuple[int, Tuple[int, ...]], None, None]:  # pragma: no cover
    r"""returns a generator for all the possible ways to decrease elements of the given tuple by 1
    without going below 0.
    Args:
        pattern (Tuple[int, ...]): the pattern given to be decreased
    Returns:
        Generator[Tuple[int, Tuple[int, ...]], None, None]: the generator
    """
    for p, n in enumerate(pattern):
        if n > 0:
            yield p, dec(pattern, p)


SQRT = np.sqrt(np.arange(1000))  # saving the time to recompute square roots


@jit(nopython=True)
def _hermite_multidimensional_renorm(R, y, G):  # pragma: no cover
    r"""Numba-compiled function to fill an array with the Hermite polynomials. It expects an array
    initialized with zeros everywhere except at index (0,...,0) (i.e. the seed value).

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        y (vector[complex]): vector argument of the Hermite polynomial
        G (array[complex]): array to be filled with the Hermite polynomials

    Returns:
        array[complex]: the multidimensional Hermite polynomials
    """
    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,0,...,0) because G[0,0,...,0] is already filled with C
    for idx in indices:
        i = 0
        for i, val in enumerate(idx):
            if val > 0:
                break
        ki = dec(idx, i)  # ki is the pivot index
        u = y[i] * G[ki]
        for l, kl in remove(ki):
            u -= SQRT[ki[l]] * R[i, l] * G[kl]
        G[idx] = u / SQRT[idx[i]]
    return G


@jit(nopython=True)
def _hermite_multidimensional(R, y, G):  # pragma: no cover
    r"""Numba-compiled function to fill an array with the Hermite polynomials. It expects an array
    initialized with zeros everywhere except at index (0,...,0) (i.e. the seed value).

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        y (vector[complex]): vector argument of the Hermite polynomial
        G (array[complex]): array to be filled with the Hermite polynomials

    Returns:
        array[complex]: the multidimensional Hermite polynomials
    """
    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,...,0)
    for idx in indices:
        i = 0
        for i, val in enumerate(idx):
            if val > 0:
                break
        ki = dec(idx, i)
        u = y[i] * G[ki]
        for l, kl in remove(ki):
            u -= ki[l] * R[i, l] * G[kl]
        G[idx] = u
    return G


@jit(nopython=True)
def _interferometer_renorm(R, G):  # pragma: no cover
    r"""Numba-compiled function returning the matrix elements of an interferometer
    parametrized in terms of its R matrix

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        array (array[complex]): array to be filled with the Hermite polynomials

    Returns:
        array[complex]: the multidimensional Hermite polynomials
    """
    dim, _ = R.shape
    num_modes = dim / 2

    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,...,0)
    for idx in indices:
        bran = 0
        for ii in range(0, num_modes):
            bran += idx[ii]

        ketn = 0
        for ii in range(num_modes, dim):
            ketn += idx[ii]

        if bran == ketn:
            i = 0
            for i, val in enumerate(idx):
                if val > 0:
                    break
            ki = dec(idx, i)
            u = 0
            for l, kl in remove(ki):
                u -= SQRT[ki[l]] * R[i, l] * G[kl]
            G[idx] = u / SQRT[idx[i]]

    return G


@jit(nopython=True)
def _interferometer(R, G):  # pragma: no cover
    r"""Numba-compiled function returning the matrix elements of an interferometer
    parametrized in terms of its R matrix

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        G (array[complex]): array to be filled with the Hermite polynomials

    Returns:
        array[complex]: the multidimensional Hermite polynomials
    """
    dim, _ = R.shape
    num_modes = dim / 2

    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,...,0)
    for idx in indices:
        bran = 0
        for ii in range(0, num_modes):
            bran += idx[ii]

        ketn = 0
        for ii in range(num_modes, dim):
            ketn += idx[ii]

        if bran == ketn:
            i = 0
            for i, val in enumerate(idx):
                if val > 0:
                    break
            ki = dec(idx, i)
            u = 0
            for l, kl in remove(ki):
                u -= ki[l] * R[i, l] * G[kl]
            G[idx] = u

    return G


def grad_hermite_multidimensional(G, R, y, C=1, renorm=True, dtype=None):
    # pylint: disable=too-many-arguments
    r"""Calculates the gradients of the renormalized multidimensional Hermite polynomials :math:`C*H_k^{(R)}(y)` with respect to its parameters :math:`C`, :math:`y` and :math:`R`.

    Args:
        G (array): the multidimensional Hermite polynomials
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        y (vector[complex]): vector argument of the Hermite polynomial
        C (complex): first value of the Hermite polynomials
        renorm (bool): If ``True``, uses the normalized multidimensional Hermite
            polynomials such that :math:`H_k^{(R)}(y)/\prod_i k_i!`
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[data type], array[data type], array[data type]: the gradients of the multidimensional Hermite polynomials with respect to C, R and y
    """
    if dtype is None:
        dtype = np.find_common_type(
            [G.dtype.name, R.dtype.name, y.dtype.name], [np.array(C).dtype.name]
        )
    n, _ = R.shape
    if y.shape[0] != n:
        raise ValueError(
            f"The matrix R and vector y have incompatible dimensions ({R.shape} vs {y.shape})"
        )
    dG_dC = np.array(G / C).astype(dtype)
    dG_dR = np.zeros(G.shape + R.shape, dtype=dtype)
    dG_dy = np.zeros(G.shape + y.shape, dtype=dtype)
    if renorm:
        dG_dR, dG_dy = _grad_hermite_multidimensional_renorm(R, y, G, dG_dR, dG_dy)
    else:
        dG_dR, dG_dy = _grad_hermite_multidimensional(R, y, G, dG_dR, dG_dy)

    return dG_dC, dG_dR, dG_dy


@jit(nopython=True)
def _grad_hermite_multidimensional_renorm(R, y, G, dG_dR, dG_dy):  # pragma: no cover
    r"""
    Numba-compiled function to fill two arrays (dG_dR, dG_dy) with the gradients of the renormalized multidimensional Hermite polynomials
    with respect to its parameters :math:`R` and :math:`y`. It needs the `array` of the multidimensional Hermite polynomials.

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        y (vector[complex]): vector argument of the Hermite polynomial
        G (array[complex]): array of the multidimensional Hermite polynomials
        dG_dR (array[complex]): array to be filled with the gradients of the renormalized multidimensional Hermite polynomials with respect to R
        dG_dy (array[complex]): array to be filled with the gradients of the renormalized multidimensional Hermite polynomials with respect to y

    Returns:
        dG_dR[complex], dG_dy[complex]: the gradients of the renormalized multidimensional Hermite polynomials with respect to R and y
    """
    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,...,0)
    for idx in indices:
        i = 0
        for i, val in enumerate(idx):
            if val > 0:
                break
        ki = dec(idx, i)
        dy = y[i] * dG_dy[ki]
        dy[i] += G[ki]
        dR = y[i] * dG_dR[ki]
        for l, kl in remove(ki):
            dy -= SQRT[ki[l]] * dG_dy[kl] * R[i, l]
            dR -= SQRT[ki[l]] * R[i, l] * dG_dR[kl]
            dR[i, l] -= SQRT[ki[l]] * G[kl]
        dG_dR[idx] = dR / SQRT[idx[i]]
        dG_dy[idx] = dy / SQRT[idx[i]]
    return dG_dR, dG_dy


@njit
def pick_pivot(idx):
    for i, val in enumerate(idx):
        if val > 0:
            return i

from numba.cpython.unsafe.tuple import tuple_setitem
@njit
def down(idx, i):
    return tuple_setitem(idx, i, idx[i] - 1)

__t = tuple([(slice(None),)*n for n in range(100)])  # tuples up to 100 elements (overkill, but meh)

@njit
def _shift(A, axis, shift):
    'shifts the elements of an array along an axis'
    return A[tuple_setitem(__t[A.ndim], axis, slice(shift))]

@njit
def shifted_products(G, dL_dG):
    d = len(G.shape)
    D0 = np.sum(G * dL_dG)
    Dm = np.zeros(d, dtype=np.complex128)
    Dmn = np.zeros((d, d), dtype=np.complex128)
    for k in np.ndindex(G.shape):
        for m in range(d):
            km = down(k,m)
            Dm[m] += G[km] * dL_dG[k] * SQRT[k[m]]
            for n in range(d):
                Dmn[m, n] += G[down(km,n)] * dL_dG[k] * SQRT[k[m]] * SQRT[k[n]-np.int(m==n)]
    return D0, Dm, Dmn

@njit
def vjp_hermite_renormalized(XQT, beta, G, dL_dG, full: bool):  # pragma: no cover
    r"""
    Vector-Jacobian product of the upstream gradient of the cost function with respect to the hermite polynomials (i.e. dL_dG) with the 
    gradient of the hermite polynomials with respect to the parameters (i.e. dG_dx where x = R,y,C).
    Effectively it computes dL_dR, dL_dy and dL_dC more efficiently than going through the full Jacobian.

    Args:
        XQT (array[complex]): X @ Q^T matrix where Q is the Husimi covariance matrix in the a,adagger basis.
            If full is false, then XQT is the bottom-right block of the XQT matrix.
        beta (array[complex]): vector of complex displacements. If full is false, then beta is the bottom half of the beta vector.
        G (array[complex]): array of the multidimensional Hermite polynomials
        dL_dG (array[complex]): upstream gradient of the cost function with respect to the multidimensional Hermite polynomials
        full (bool): ``True`` if G is a dm or Choi state, ``False`` if G is a ket or unitary.

    Returns:
        dL_dR[complex], dL_dy[complex]: the gradients of the cost function with respect to R and y
    """
    D0, D1, D2 = shifted_products(G, dL_dG)
    dL_dR = np.zeros(XQT.shape, dtype=np.complex128)
    dL_dy = np.zeros(beta.shape, dtype=np.complex128)
    dL_dC = D0/G[next(np.ndindex(G.shape))]
    
    for m in range(beta.shape[-1]):
        dL_dy[m] = -0.5*beta[m]*D0 + D1[m]
        for n in range(beta.shape[-1]):
            if full:
                dL_dR[m, n] = -0.5*XQT[m,n]*D0 - beta[n]*D1[m] + D2[m,n]
            else:
                dL_dR[m, n] = -0.5*np.real(XQT[m,n]*D0) - beta[n]*D1[m] + D2[m,n]
    return np.conj(dL_dR), np.conj(dL_dy), np.conj(dL_dC)



@jit(nopython=True)
def _grad_hermite_multidimensional(R, y, G, dG_dR, dG_dy):  # pragma: no cover
    r"""
    Numba-compiled function to fill two arrays (dG_dR, dG_dy) with the gradients of the renormalized multidimensional Hermite polynomials
    with respect to its parameters :math:`R` and :math:`y`. It needs the `array` of the multidimensional Hermite polynomials.

    Args:
        R (array[complex]): square matrix parametrizing the Hermite polynomial
        y (vector[complex]): vector argument of the Hermite polynomial
        G (array[complex]): array of the multidimensional Hermite polynomials
        dG_dR (array[complex]): array to be filled with the gradients of the renormalized multidimensional Hermite polynomials with respect to R
        dG_dy (array[complex]): array to be filled with the gradients of the renormalized multidimensional Hermite polynomials with respect to y

    Returns:
        dG_dR[complex], dG_dy[complex]: the gradients of the renormalized multidimensional Hermite polynomials with respect to R and y
    """
    indices = np.ndindex(G.shape)
    next(indices)  # skip the first index (0,...,0)
    for idx in indices:
        i = 0
        for i, val in enumerate(idx):
            if val > 0:
                break
        ki = dec(idx, i)
        dy = y[i] * dG_dy[ki]
        dy[i] += G[ki]
        dR = y[i] * dG_dR[ki]
        for l, kl in remove(ki):
            dy -= ki[l] * dG_dy[kl] * R[i, l]
            dR -= ki[l] * R[i, l] * dG_dR[kl]
            dR[i, l] -= ki[l] * G[kl]
        dG_dR[idx] = dR
        dG_dy[idx] = dy
    return dG_dR, dG_dy
