import numpy as np
from ..decompositions import symplectic_eigenvals
from .gaussian_checks import is_valid_cov, is_pure_cov
from .conversions import reduced_gaussian

def get_partition(modes_A, separation, M):
    r"""Helper function to validate and, if necessary, pre-prepare the inputs
    of ``vonNeumann_entropy()`` and ``log_negativity()``.

    Args:
        modes_A (iterable or int): the subset of modes used for the bipartition
        separation (int): the index of the mode separating the two partitions
            (alternative to ``modes_A``)
        M (int): total number of modes

    Returns:
        (list): mode indices of substate A
    """
    if modes_A is not None:
        if not isinstance(modes_A, (int, range, list, tuple, np.ndarray)):
            raise TypeError("modes_A must be either integer, range, tuple or np.ndarray.")
        if isinstance(modes_A, int):
            modes_A = modes_A
        for mode in modes_A:
            if not isinstance(mode, int) or mode < 0 or mode > M - 1:
                raise ValueError(f"Every element of modes_A must be an integer between 0 and {M - 1}")

    if modes_A is None and separation is not None:
        if not isinstance(separation, int) or separation > M - 1:
            raise ValueError("separation must be an integer smaller than the number of modes.")
        modes_A = range(separation)

    return list(modes_A)

def vonNeumann_entropy(cov):
    r"""Returns the vonNeumann entropy of a covariance matrix.

    Args:
        cov (array): a covariance matrix

    Returns:
        (float): vonNeumann entropy
    """
    if not is_valid_cov(cov):
        raise ValueError("Input is not a valid covariance matrix.")

    nus = symplectic_eigenvals(cov)

    S = 0
    for nu in nus:
        if not np.isclose(nu, 1):
            g = (nu + 1) / 2 * np.log((nu + 1) / 2) - (nu - 1) / 2 * np.log((nu - 1) / 2)
            S += g

    return S

def entanglement_entropy(cov, modes_A=None, separation=None):
    r"""Returns the entanglement entropy of a covariance matrix under a given
    bipartition.

    Args:
        cov (array): a covariance matrix
        modes_A (iterable or int): the subset of modes used for the bipartition
        separation (int): the index of the mode separating the two partitions
            (alternative to ``modes_A``)

    Returns:
        (float): logarithmic negativity
    """
    if not is_pure_cov(cov):
        raise ValueError("Input is not a pure covariance matrix.")

    M = int(len(cov) / 2)

    modes_A = get_partition(modes_A, separation, M)

    _, cov_A = reduced_gaussian(np.zeros(2 * M), cov, modes_A)

    E = vonNeumann_entropy(cov_A)

    return E

def log_negativity(cov, modes_A=None, separation=None):
    r"""Returns the logarithmic negativity of a covariance matrix under a given
    bipartition.

    Args:
        cov (array): a covariance matrix
        modes_A (iterable or int): the subset of modes used for the bipartition
        separation (int): the index of the mode separating the two partitions
            (alternative to ``modes_A``)

    Returns:
        (float): entanglement entropy
    """
    if not is_valid_cov(cov):
        raise ValueError("Input is not a valid covariance matrix.")

    M = int(len(cov) / 2)

    modes_A = get_partition(modes_A, separation, M)

    X = np.ones(M)
    P = np.ones(M)
    P[modes_A] = -1

    S = np.diag(np.concatenate((X, P)))

    cov_tilde = S @ cov @ S

    nus = symplectic_eigenvals(cov_tilde)
    E = np.sum([-np.log(nu) for nu in nus if nu < 1])

    return E