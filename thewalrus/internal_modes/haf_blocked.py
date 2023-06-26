import numpy as np
from numba import jit

from thewalrus._hafnian import f_from_matrix as f     # for internal_modes branch of thewalrus

from thewalrus._hafnian import get_AX_S
from thewalrus._hafnian import nb_binom


@jit(nopython=True, cache=True)
def finite_difference_operator_coeffs(der_order, m, u=None, v=None):
    if u is None:
        u = 2 - der_order
    if v is None:
        v = -der_order
    prefac = (-1) ** (der_order - m) * nb_binom(der_order, m) / (u - v) ** der_order
    val = v + m * (u - v)
    return prefac, val


def haf_blocked(A, blocks=None, repeats=None):
    """Calculates the hafnian of the matrix with a given block and repetition pattern.

    Args:
        A (array): input matrix
        blocks (list): how to block (coarse graining) different outcomes
        repeats (list): pattern for repetitions

    Returns:
        value of the hafnian
    """
    if blocks is None:
        blocks = tuple((i,) for i in range(A.shape[0] // 2))

    if repeats is None:
        repeats = (1,) * len(blocks)

    n = sum(repeats)

    repeats_p = [1 + idx for idx in repeats]
    num_indices = sum(map(len, blocks))
    netsum = 0.0 + 0j
    n = sum(repeats)
    coeff_vect = np.zeros([num_indices], dtype=int)
    for index in np.ndindex(*repeats_p):
        coeff_pref = 1.0 + 0j
        for i, block in enumerate(blocks):
            (coeff, val) = finite_difference_operator_coeffs(repeats[i], index[i])
            coeff_pref *= coeff
            for mode in block:
                coeff_vect[mode] = val
        AX_S = get_AX_S(coeff_vect, A)

        netsum += coeff_pref * f(AX_S, 2 * n)[n]
    return netsum




@jit(nopython=True)
def haf_blocked_numba(A, blocks, repeats_p):
    """Calculates the hafnian of the matrix with a given block and repetition pattern.

    Args:
        A (array): input matrix
        blocks (list): how to block (coarse graining) different outcomes
        repeats_p (list): pattern for repetition but with one added in each repetition

    Returns:
        value of the hafnian
    """
    n = sum(repeats_p) - len(repeats_p)
    num_indices = 0
    for block in blocks:
        num_indices += len(block)
    netsum = 0.0 + 0j
    coeff_vect = np.zeros(num_indices, dtype=np.int32)
    for index in np.ndindex(*repeats_p):
        coeff_pref = 1.0 + 0j
        for i, block in enumerate(blocks):
            (coeff, val) = finite_difference_operator_coeffs(repeats_p[i]-1, index[i])
            coeff_pref *= coeff
            for mode in block:
                coeff_vect[mode] = val
        AX_S = get_AX_S(coeff_vect, A)

        netsum += coeff_pref * f(AX_S, 2 * n)[n]
    return netsum
