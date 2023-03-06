"""Functions for computing grouped click probabilities"""
import numpy as np
from numba import jit


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
@jit(nopython=True)
def grouped_click_probabilities(
    phn, chn, t_matrix, num_samples, num_groups, seed=1990
):  # pragma: no cover
    """Computes click probabilities and errors for input states sent into a lossy interferometer
    Args:
        phn (array): mean photon numbers of input modes
        chn (array): coherences of input modes
        t_matrix (array): transfer matrix
        num_samples (int): number of samples
        num_groups (int): number of groups into which the samples are divided for error computation
        seed (int): seed of the random number generator
    Returns:
        tuple (prob, error): array of grouped click probabilities and array of corresponding errors
    """
    np.random.seed(seed)
    samp_per_group = num_samples // num_groups
    num_modes, num_input = max(t_matrix.shape), min(t_matrix.shape)
    drp = np.array([(0.5 * np.complex128(phn[i] + chn[i])) ** 0.5 for i in range(num_input)])
    drm = np.array([(0.5 * np.complex128(phn[i] - chn[i])) ** 0.5 for i in range(num_input)])
    delta = 2 * np.pi / (num_modes + 1)
    f_mat = np.asarray(
        [
            [(1 / (num_modes + 1)) * np.exp(1j * q * p * delta) for q in range(num_modes + 1)]
            for p in range(num_modes + 1)
        ]
    )
    acc = np.zeros(num_modes + 1, dtype=np.float64)
    bcc = np.zeros(num_modes + 1, dtype=np.float64)
    qcc = np.zeros(num_modes + 1, dtype=np.float64)
    fix = np.zeros(num_modes + 1, dtype=np.float64)
    for j in range(num_samples):
        wrp = np.array([np.random.normal() for _ in range(num_input)])
        wrm = np.array([np.random.normal() for _ in range(num_input)])
        alpha = t_matrix @ (drp * wrp + 1j * drm * wrm)
        beta = t_matrix.conj() @ (drp * wrp - 1j * drm * wrm)
        gth = np.empty(num_modes + 1, dtype=np.complex128)
        for k in range(num_modes + 1):
            gth[k] = np.prod(
                np.exp(-alpha * beta) + np.exp(-1j * k * delta) * (1 - np.exp(-alpha * beta))
            )
        acc = acc + (f_mat @ gth).real
        if (j + 1) % samp_per_group == 0:
            bcc = bcc + (acc - fix) / samp_per_group
            qcc = qcc + ((acc - fix) / samp_per_group) ** 2
            fix = acc
    return bcc / num_groups, (qcc / num_groups - (bcc / num_groups) ** 2) ** 0.5


@jit(nopython=True)
def grouped_click_probabilities_squeezed(
    input_sq, t_matrix, num_samples, num_groups, seed=1990
):  # pragma: no cover
    """Computes click probabilities for input squeezed states sent into a lossy interferometer
    Args:
        input_sq (array): input squeezing parameters
        t_matrix (array): transfer matrix
        num_samples (int): number of samples
        num_groups (int): number of groups into which the samples are divided for error computation
        seed (int): seed of the random number generator
    Returns:
        tuple (prob, error): array of grouped click probabilities and array of corresponding errors
    """
    phn = np.sinh(input_sq) ** 2
    chn = 0.5 * np.sinh(2 * input_sq)
    return grouped_click_probabilities(phn, chn, t_matrix, num_samples, num_groups, seed)
