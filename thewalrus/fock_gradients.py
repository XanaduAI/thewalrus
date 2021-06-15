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
Fock gradients of Gaussian gates
================================

.. currentmodule:: thewalrus.fock_gradients

This module contains the Fock representation of the standard Gaussian gates as well as their gradients.

.. autosummary::
    :toctree: api

	displacement
	squeezing
	beamsplitter
	two_mode_squeezing
    n_mode_gaussian_gate
	grad_displacement
	grad_squeezing
	grad_beamsplitter
	grad_two_mode_squeezing
    grad_n_mode_gaussian_gate

"""
import numpy as np

from numba import jit


@jit(nopython=True)
def displacement(r, phi, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.

    Args:
        r (float): displacement magnitude
        phi (float): displacement angle
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the displacement operation.
    """
    D = np.zeros((cutoff, cutoff), dtype=dtype)
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    mu = np.array([r * np.exp(1j * phi), -r * np.exp(-1j * phi)])

    D[0, 0] = np.exp(-0.5 * r ** 2)
    for m in range(1, cutoff):
        D[m, 0] = mu[0] / sqrt[m] * D[m - 1, 0]

    for m in range(cutoff):
        for n in range(1, cutoff):
            D[m, n] = mu[1] / sqrt[n] * D[m, n - 1] + sqrt[m] / sqrt[n] * D[m - 1, n - 1]

    return D


@jit(nopython=True)
def grad_displacement(T, r, phi):  # pragma: no cover
    r"""Calculates the gradients of the displacement gate with respect to the displacement magnitude and angle.

    Args:
        T (array[complex]): array representing the gate
        r (float): displacement magnitude
        phi (float): displacement angle

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the displacement gate with respect to r and phi
    """
    cutoff = T.shape[0]
    dtype = T.dtype
    ei = np.exp(1j * phi)
    eic = np.exp(-1j * phi)
    alpha = r * ei
    alphac = r * eic
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_r = np.zeros((cutoff, cutoff), dtype=dtype)
    grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)

    for m in range(cutoff):
        for n in range(cutoff):
            grad_r[m, n] = -r * T[m, n] + sqrt[m] * ei * T[m - 1, n] - sqrt[n] * eic * T[m, n - 1]
            grad_phi[m, n] = sqrt[m] * 1j * alpha * T[m - 1, n] + sqrt[n] * 1j * alphac * T[m, n - 1]

    return grad_r, grad_phi


@jit(nopython=True)
def squeezing(r, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the squeezing gate using a recurrence relation.

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the squeezing gate.
    """
    S = np.zeros((cutoff, cutoff), dtype=dtype)
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    sechr = 1.0 / np.cosh(r)
    R = np.array([[-eitheta_tanhr, sechr], [sechr, np.conj(eitheta_tanhr)],])

    S[0, 0] = np.sqrt(sechr)
    for m in range(2, cutoff, 2):
        S[m, 0] = sqrt[m - 1] / sqrt[m] * R[0, 0] * S[m - 2, 0]

    for m in range(0, cutoff):
        for n in range(1, cutoff):
            if (m + n) % 2 == 0:
                S[m, n] = sqrt[n - 1] / sqrt[n] * R[1, 1] * S[m, n - 2] + sqrt[m] / sqrt[n] * R[0, 1] * S[m - 1, n - 1]
    return S


@jit(nopython=True)
def grad_squeezing(T, r, phi):  # pragma: no cover
    r"""Calculates the gradients of the squeezing gate with respect to the squeezing magnitude and angle

    Args:
        T (array[complex]): array representing the gate
        r (float): squeezing magnitude
        phi (float): squeezing angle

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the squeezing gate with respect to the r and phi
    """
    cutoff = T.shape[0]
    dtype = T.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_r = np.zeros((cutoff, cutoff), dtype=dtype)
    grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)

    sechr = 1.0 / np.cosh(r)
    tanhr = np.tanh(r)
    eiphi = np.exp(1j * phi)
    eiphiconj = np.exp(-1j * phi)

    for m in range(cutoff):
        for n in range(cutoff):
            grad_r[m, n] = (
                -0.5 * tanhr * T[m, n]
                - sechr * tanhr * sqrt[m] * sqrt[n] * T[m - 1, n - 1]
                - 0.5 * eiphi * sechr ** 2 * sqrt[m] * sqrt[m - 1] * T[m - 2, n]
                + 0.5 * eiphiconj * sechr ** 2 * sqrt[n] * sqrt[n - 1] * T[m, n - 2]
            )
            grad_phi[m, n] = (
                -0.5j * eiphi * tanhr * sqrt[m] * sqrt[m - 1] * T[m - 2, n] - 0.5j * eiphiconj * tanhr * sqrt[n] * sqrt[n - 1] * T[m, n - 2]
            )

    return grad_r, grad_phi


@jit(nopython=True)
def two_mode_squeezing(r, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the matrix elements of the two-mode squeezing gate recursively.

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[float]: The Fock representation of the gate

    """
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    Z = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    sc = 1.0 / np.cosh(r)
    eiptr = np.exp(-1j * theta) * np.tanh(r)
    R = -np.array([[0, -np.conj(eiptr), -sc, 0], [-np.conj(eiptr), 0, 0, -sc], [-sc, 0, 0, eiptr], [0, -sc, eiptr, 0],])

    Z[0, 0, 0, 0] = sc

    # rank 2
    for n in range(1, cutoff):
        Z[n, n, 0, 0] = R[0, 1] * Z[n - 1, n - 1, 0, 0]

    # rank 3
    for m in range(cutoff):
        for n in range(m):
            p = m - n
            if 0 < p < cutoff:
                Z[m, n, p, 0] = R[0, 2] * sqrt[m] / sqrt[p] * Z[m - 1, n, p - 1, 0]

    # rank 4
    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                q = p - (m - n)
                if 0 < q < cutoff:
                    Z[m, n, p, q] = R[1, 3] * sqrt[n] / sqrt[q] * Z[m, n - 1, p, q - 1] + R[2, 3] * sqrt[p] / sqrt[q] * Z[m, n, p - 1, q - 1]
    return Z


@jit(nopython=True)
def grad_two_mode_squeezing(T, r, theta):  # pragma: no cover
    """Calculates the gradients of the two-mode squeezing gate with respect to the squeezing magnitude and angle

    Args:
        T (array[complex]): array representing the gate
        r (float): squeezing magnitude
        theta (float): squeezing angle

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the two-mode squeezing gate with respect to r and phi

    """
    cutoff = T.shape[0]
    dtype = T.dtype
    sechr = 1.0 / np.cosh(r)
    tanhr = np.tanh(r)
    ei = np.exp(1j * theta)
    eic = np.exp(-1j * theta)
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))

    grad_r = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    grad_theta = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    grad_r[0, 0, 0, 0] = -sechr * tanhr

    # rank 2
    for n in range(1, cutoff):
        grad_r[n, n, 0, 0] = -tanhr * T[n, n, 0, 0] + sqrt[n] * sqrt[n] * ei * sechr ** 2 * T[n - 1, n - 1, 0, 0]
        grad_theta[n, n, 0, 0] = 1j * ei * tanhr * sqrt[n] * sqrt[n] * T[n - 1, n - 1, 0, 0]

    # rank 3
    for m in range(cutoff):
        for n in range(m):
            p = m - n
            if 0 < p < cutoff:
                grad_r[m, n, p, 0] = (
                    -tanhr * T[m, n, p, 0]
                    + sqrt[m] * sqrt[n] * ei * sechr ** 2 * T[m - 1, n - 1, p, 0]
                    - tanhr * sechr * sqrt[m] * sqrt[p] * T[m - 1, n, p - 1, 0]
                )
                grad_theta[m, n, p, 0] = 1j * ei * tanhr * sqrt[m] * sqrt[n] * T[m - 1, n - 1, p, 0]

    # rank 4
    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                for q in range(cutoff):
                    grad_r[m, n, p, q] = (
                        -tanhr * T[m, n, p, q]
                        + sqrt[m] * sqrt[n] * ei * sechr ** 2 * T[m - 1, n - 1, p, q]
                        - tanhr * sechr * sqrt[m] * sqrt[p] * T[m - 1, n, p - 1, q]
                        - tanhr * sechr * sqrt[n] * sqrt[q] * T[m, n - 1, p, q - 1]
                        - sqrt[p] * sqrt[q] * eic * sechr ** 2 * T[m, n, p - 1, q - 1]
                    )
                    grad_theta[m, n, p, q] = (
                        1j * ei * tanhr * sqrt[m] * sqrt[n] * T[m - 1, n - 1, p, q] + 1j * eic * tanhr * sqrt[p] * sqrt[q] * T[m, n, p - 1, q - 1]
                    )

    return grad_r, grad_theta


@jit(nopython=True)
def beamsplitter(theta, phi, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the Fock representation of the beamsplitter.

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[float]: The Fock representation of the gate
    """
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    R = np.array([[0, 0, ct, -np.conj(st)], [0, 0, st, ct], [ct, st, 0, 0], [-np.conj(st), ct, 0, 0],])

    Z = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    Z[0, 0, 0, 0] = 1.0

    # rank 3
    for m in range(cutoff):
        for n in range(cutoff - m):
            p = m + n
            if 0 < p < cutoff:
                Z[m, n, p, 0] = R[0, 2] * sqrt[m] / sqrt[p] * Z[m - 1, n, p - 1, 0] + R[1, 2] * sqrt[n] / sqrt[p] * Z[m, n - 1, p - 1, 0]

    # rank 4
    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                q = m + n - p
                if 0 < q < cutoff:
                    Z[m, n, p, q] = R[0, 3] * sqrt[m] / sqrt[q] * Z[m - 1, n, p, q - 1] + R[1, 3] * sqrt[n] / sqrt[q] * Z[m, n - 1, p, q - 1]
    return Z


@jit(nopython=True)
def grad_beamsplitter(T, theta, phi):  # pragma: no cover
    r"""Calculates the gradients of the beamsplitter gate with respect to the transmissivity angle and reflection phase

    Args:
        T (array[complex]): array representing the gate
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the beamsplitter gate with respect to theta and phi
    """
    cutoff = T.shape[0]
    dtype = T.dtype
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_theta = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    grad_phi = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    ct = np.cos(theta)
    st = np.sin(theta)
    ei = np.exp(1j * phi)
    eic = np.exp(-1j * phi)

    # rank 3
    for m in range(cutoff):
        for n in range(cutoff - m):
            p = m + n
            if 0 < p < cutoff:
                grad_theta[m, n, p, 0] = -sqrt[m] * sqrt[p] * st * T[m - 1, n, p - 1, 0] + sqrt[n] * sqrt[p] * ei * ct * T[m, n - 1, p - 1, 0]
                grad_phi[m, n, p, 0] = 1j * sqrt[n] * sqrt[p] * ei * st * T[m, n - 1, p - 1, 0]

    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                q = m + n - p
                if 0 < q < cutoff:
                    grad_theta[m, n, p, q] = (
                        -sqrt[m] * sqrt[p] * st * T[m - 1, n, p - 1, q]
                        - sqrt[n] * sqrt[q] * st * T[m, n - 1, p, q - 1]
                        + sqrt[n] * sqrt[p] * ei * ct * T[m, n - 1, p - 1, q]
                        - sqrt[m] * sqrt[q] * eic * ct * T[m - 1, n, p, q - 1]
                    )
                    grad_phi[m, n, p, q] = (
                        1j * sqrt[n] * sqrt[p] * ei * st * T[m, n - 1, p - 1, q] + 1j * sqrt[m] * sqrt[q] * eic * st * T[m - 1, n, p, q - 1]
                    )

    return grad_theta, grad_phi
    
def choi_trick(S, d, m):
    # m: num of modes
    choi_r = np.arcsing(1.0)
    ch = np.cosh(choi_r) * np.identity(m)
    sh = np.sinh(choi_r) * np.identity(m)
    zh = np.zeros([m, m])
    Schoi = np.block(
        [[ch, sh, zh, zh], [sh, ch, zh, zh], [zh, zh, ch, -sh], [zh, zh, -sh, ch]]
    )
    Sxx = S[:m, :m]
    Sxp = S[:m, m:]
    Spx = S[m:, :m]
    Spp = S[m:, m:]
    idl = np.identity(m)
    S_exp = (
        np.block(
            [
                [Sxx, zh, Sxp, zh],
                [zh, idl, zh, zh],
                [Spx, zh, Spp, zh],
                [zh, zh, zh, idl],
            ]
        )
        @ Schoi
    )
    choi_cov = 0.5 * S_exp @ S_exp.T
    idl = np.identity(2 * m)
    R = np.sqrt(0.5) * np.block([[idl, 1j * idl], [idl, -1j * idl]])
    sigma = R @ choi_cov @ R.conj().T
    zh = np.zeros([2 * m, 2 * m])
    X = np.block([[zh, 1j * idl], [idl, zh]])
    sigma_Q = sigma + 0.5 * np.identity(4 * m)
    A_mat = X @ (np.identity(4 * m) - np.linalg.inv(sigma_Q))
#    #TODO: get C from T maybe?
#    d = np.block([d,np.zeros(l//2),d.conj(),np.zeros(l//2)])
#    beta_vector = d.T @ np.linalg.inv(sigma_Q)
#    T = np.expm(- 0.5 * beta_vector.T @ np.linalg.inv(sigma_Q) @ beta_vector) / np.sqrt(np.linalg.det(sigma_Q))
#    C = np.sqrt(T)
    E = np.diag(np.concatenate([np.ones([m]), np.ones([m]) / np.tanh(choi_r)]))
    Sigma = -(E @ A_mat[:2*m, :2*m] @ E).conj()
    mu = np.concatenate([d.conj().T@Sigma[:m,:m]+d.T, d.conj().T@Sigma[m:,:m]])
    cosh_term =1
    for ind in range(0,m+1,2):
        cosh_term = np.cosh(-np.log(np.linalg.svd(S)[1][ind]))*cosh_term
    C = np.exp(- 0.5 * (np.sum((np.abs(d)) **2) + d.conj().T@Sigma[:m,:m]@d.conj()))/np.sqrt(cosh_term)
    return C, mu, Sigma


@lru_cache()
def partition(num_modes: int, n_current: int, cutoff: int)-> Tuple[Tuple[int, ...], ...]:
    return [
        (0,)*(2*num_modes - n_current) + comb for comb in product(*(range(cutoff) for _ in range(n_current)))
    ]
    
@njit
def dec(tup: Tuple[int], i: int) -> Tuple[int, ...]:  # pragma: no cover
    "returns a copy of the given tuple of integers where the ith element has been decreased by 1"
    copy = tup[:]
    return tuple_setitem(copy, i, tup[i] - 1)
    
@njit
def remove(
    pattern: Tuple[int, ...]
) -> Generator[Tuple[int, Tuple[int, ...]], None, None]:  # pragma: no cover
    "returns a generator for all the possible ways to decrease elements of the given tuple by 1"
    for p, n in enumerate(pattern):
        if n > 0:
            yield p, dec(pattern, p)

SQRT = np.sqrt(np.arange(1000))  # saving the time to recompute square roots

def n_mode_gaussian_gate(array, S, d, dtype=np.complex128):
    num_modes = S.shape[0]//2
    cutoff = array.shape[0]
    C, mu, Sigma = choi_trick(S, d, num_modes)
    for n_current in range(1, num_modes+1):
        for idx in partition(num_modes, n_current, cutoff):
            array = fill_n_mode_gaussian_gate_loop(array, idx, C, mu, Sigma)
    return array

def fill_n_mode_gaussian_gate_loop(array, idx, C, mu, Sigma):
    if idx == (0,)*(2*num_modes):
        array[idx] = C
    else:
        for i, val in enumerate(idx):
            if val > 0:
                break
        ki = dec(idx, i)
        u = mu[i] * array[ki]
        for l, kl in remove(ki):
            u -= SQRT[ki[l]] * Sigma[i, l] * array[kl]
        array[idx] = u / SQRT[idx[i]]
    return array
    
    
def grad_n_mode_gaussian_gate(G, S, d):
    
