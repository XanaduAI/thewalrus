"""
Montrealer Python interface
"""
import numpy as np
import numba
from thewalrus.quantum.conversions import Qmat, reduced_gaussian, Xmat
from ._hafnian import nb_ix
from ._torontonian import tor_input_checks
from thewalrus.charpoly import powertrace


@numba.jit(nopython=True, cache=True)
def dec2bin(num, n):  # pragma: no cover
    """Helper function to generate convert an integer into an element of the powerset of n objects.

    Args:
        num (int): label to convert.
        n (int): number of elements in the set.

    Returns:
        (array): array containing the labels of the elements to be selected.

    """
    digits = np.zeros((n), dtype=type(num))
    nn = num
    counter = -1
    while nn >= 1:  # Should be >= 1, not > 1.
        digits[counter] = nn % 2
        counter -= 1
        nn //= 2
    return np.nonzero(digits)[0]


@numba.jit(nopython=True)
def montrealer(Sigma):  # pragma: no cover
    """Calculates the loop-montrealer of the zero-displacement Gaussian state with the given complex covariance matrix.

    Args:
        A (array): adjacency matrix of the Gaussian state.

    Returns:
        (np.complex128): the montrealer of A.

    """
    n = len(Sigma) // 2
    tot_num = 2**n
    val = np.complex128(0)
    for p in numba.prange(tot_num):
        pos = dec2bin(p, n)
        lenpos = len(pos)
        pos = np.concatenate((pos, n + pos))
        submat = nb_ix(Sigma, pos, pos)
        sign = (-1) ** (lenpos + 1)
        val += (sign) * powertrace(submat, n + 1)[-1]
    return (-1) ** (n + 1) * val / (2 * n)


@numba.jit(nopython=True)
def power_loop(Sigma, zeta, n):  # pragma: no cover
    """Auxiliary function to calculate the product np.conj(zeta) @ Sigma^{n-1} @ zeta.
    Args:
        Sigma (array): square complex matrix
        zeta (array): complex vector
        n (int): sought after power

    Returns:
        (np.complex128 or np.float64): the product np.conj(zeta) @ Sigma^{n-1} @ zeta
    """
    vec = zeta
    for i in range(n - 1):
        vec = Sigma @ vec
    return np.conj(zeta) @ vec


@numba.jit(nopython=True, cache=True)
def lmontrealer(Sigma, zeta):  # pragma: no cover
    """Calculates the loop-montrealer of the displaced Gaussian state with the given complex covariance matrix and vector of displacements.

    Args:
        Sigma (array): complex Glauber covariance matrix of the Gaussian state.
        zeta (array): vector of displacements

    Returns:
        (np.complex128): the montrealer of A.
    """
    n = len(Sigma) // 2
    tot_num = 2**n
    val = np.complex128(0)
    val_loops = np.complex128(0)
    for p in numba.prange(tot_num):
        pos = dec2bin(p, n)
        lenpos = len(pos)
        pos = np.concatenate((pos, n + pos))
        subvec = zeta[pos]
        submat = nb_ix(Sigma, pos, pos)
        sign = (-1) ** (lenpos + 1)
        val_loops += sign * power_loop(submat, subvec, n)
        val += sign * powertrace(submat, n + 1)[-1]
    return (-1) ** (n + 1) * (val / (2 * n) + val_loops / 2)


def lmtl(A, zeta):
    """Returns the loop Torontonian of an NxN matrix and an N-length vector.

    Args:
        A (array): an NxN array of even dimensions.
        zeta (array): an N-length vector of even dimensions

    Returns:
        np.float64 or np.complex128: the loop torontonian of matrix A, vector gamma
    """

    tor_input_checks(A, zeta)
    n = len(A) // 2
    Sigma = Xmat(n) @ A
    return lmontrealer(Sigma, zeta)


def mtl(A):
    """Returns the loop Torontonian of an NxN matrix and an N-length vector.

    Args:
        A (array): an NxN array of even dimensions.
        zeta (array): an N-length vector of even dimensions

    Returns:
        np.float64 or np.complex128: the loop torontonian of matrix A, vector gamma
    """

    tor_input_checks(A)
    n = len(A) // 2
    Sigma = Xmat(n) @ A
    return montrealer(Sigma)