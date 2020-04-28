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

This module contains the Fock representation of the standard Gaussian gates and
the Kerr gate, as well as their gradients.

.. autosummary::
    :toctree: api

    Dgate
    Sgate
    Rgate
    Kgate
    S2gate
    BSgate

"""
import numpy as np

from numba import jit


@jit(nopython=True)
def displacement(alpha, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculate the matrix elements of the real or complex displacement gate using a recurrence relation.

    Args:
        alpha (float or complex): value of the displacement.
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        (array): matrix representing the displacement operation.
    """
    D = np.zeros((cutoff, cutoff), dtype=dtype)
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    mu = np.array([alpha, -np.conj(alpha)])

    D[0, 0] = np.exp(-0.5 * np.abs(mu[0]) ** 2)
    for m in range(1, cutoff):
        D[m, 0] = mu[0] / sqrt[m] * D[m - 1, 0]

    for m in range(cutoff):
        for n in range(1, cutoff):
            D[m, n] = mu[1] / sqrt[n] * D[m, n - 1] + sqrt[m] / sqrt[n] * D[m - 1, n - 1]

    return D


@jit(nopython=True)
def grad_displacement(T, alpha, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the gradients of the Dgate with respect to alpha and the conjugate of alpha.

    Args:
        T (array[complex]): array representing the gate
        alpha (complex): displacement phase
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the Dgate with respect to alpha and the conjugate of alpha
    """
    cutoff = T.shape[0]
    alphac = np.conj(alpha)
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_alpha = np.zeros((cutoff, cutoff), dtype=dtype)
    grad_alphaconj = np.zeros((cutoff, cutoff), dtype=dtype)

    for m in range(cutoff):
        for n in range(cutoff):
            grad_alpha[m, n] = -0.5 * alphac * T[m, n] + sqrt[m] * T[m - 1, n]
            grad_alphaconj[m, n] = -0.5 * alpha * T[m, n] - sqrt[n] * T[m, n - 1]

    return grad_alpha, grad_alphaconj


@jit(nopython=True)
def squeezing(r, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculate the matrix elements of the real or complex squeezing gate using a recurrence relation.

    Args:
        r (float): squeezing amplitude
        theta (float): squeezing phase
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        (array): matrix representing the squeezing operation.
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
def grad_squeezing(T, r, phi, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the gradients of the squeezing gate with respect to the squeezing amplitude and phase

    Args:
        T (array[complex]): array representing the gate
        r (float): squeezing amplitude
        phi (float): squeezing phase
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the squeezing gate with respect to the squeezing amplitude and phase
    """
    cutoff = T.shape[0]
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
        theta (float): squeezing phase
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
def grad_two_mode_squeezing(T, r, theta, dtype=np.complex128):  # pragma: no cover
    """Calculates the gradients of the two-mode squeezing gate with respect to the squeezing amplitude and phase

    Args:
        T (array[complex]): array representing the gate
        r (float): squeezing magnitude
        theta (float): squeezing phase
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the two-mode squeezing gate with respect to the squeezing amplitude and phase

    """
    cutoff = T.shape[0]
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
def grad_beamsplitter(T, theta, phi, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the Fock representation of the beamsplitter.

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[float]: The Fock representation of the gate
    """
    cutoff = T.shape[0]
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


#####
# POLAR GATES
#####


@jit(nopython=True)
def grad_Dgate(T, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the gradient of the Dgate.

    Args:
        T (array[complex]): array representing the gate
        theta (float): displacement phase
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the Dgate with respect to r (the amplitude) and theta (the phase)
    """
    gradTr = np.zeros((cutoff, cutoff), dtype=dtype)
    gradTtheta = np.zeros((cutoff, cutoff), dtype=dtype)
    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        for m in range(cutoff):
            gradTtheta[n, m] = 1j * (n - m) * T[n, m]
            gradTr[n, m] = np.sqrt(m + 1) * T[n, m + 1] * exptheta
            if m > 0:
                gradTr[n, m] -= np.sqrt(m) * T[n, m - 1] * np.conj(exptheta)
    return gradTr, gradTtheta


def Dgate(r, theta, cutoff, grad=False, dtype=np.complex128):
    """Calculates the Fock representation of the Dgate and its gradient.

    Args:
        r (float): displacement magnitude
        theta (float): displacement phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        dtype (data type): Specifies the data type used for the calculation


    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    if not grad:
        return displacement(r * np.exp(1j * theta), cutoff, dtype=dtype), None, None
    T = displacement(r * np.exp(1j * theta), cutoff + 1)
    (gradTr, gradTtheta) = grad_Dgate(T, theta, cutoff, dtype=dtype)
    return T[:cutoff, :cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_Sgate(T, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the gradient of the Sgate.

    Args:
        T (array[complex]): array representing the gate
        theta (float): squeezing phase
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the Sgate with respect to r (the amplitude) and theta (the phase)

    """
    gradTr = np.zeros((cutoff, cutoff), dtype=dtype)
    gradTtheta = np.zeros((cutoff, cutoff), dtype=dtype)

    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        offset = n % 2
        for m in range(offset, cutoff, 2):
            gradTtheta[n, m] = 0.5j * (n - m) * T[n, m]
            gradTr[n, m] = -0.5 * np.sqrt((m + 1) * (m + 2)) * T[n, m + 2] * exptheta
            if m > 1:
                gradTr[n, m] += 0.5 * np.sqrt(m * (m - 1)) * T[n, m - 2] * np.conj(exptheta)

    return gradTr, gradTtheta


def Sgate(r, theta, cutoff, grad=False, dtype=np.complex128):
    """Calculates the Fock representation of the Sgate and its gradient.

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    if not grad:
        return squeezing(r, theta, cutoff, dtype=dtype), None, None

    T = squeezing(r, theta, cutoff + 2)
    (gradTr, gradTtheta) = grad_Sgate(T, theta, cutoff, dtype=dtype)

    return T[:cutoff, :cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_S2gate(T, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the gradient of the S2gate.

    Args:
        T (array[complex]): array representing the gate
        theta (float): two-mode squeezing phase
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the S2gate with respect to r (the amplitude) and theta (the phase)

    """
    gradTr = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    gradTtheta = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = m - n + k
                if 0 <= l < cutoff:
                    gradTtheta[n, k, m, l] = 1j * (n - m) * T[n, k, m, l]
                    gradTr[n, k, m, l] = np.sqrt((m + 1) * (l + 1)) * T[n, k, m + 1, l + 1] * exptheta
                    if m > 0 and l > 0:
                        gradTr[n, k, m, l] -= np.sqrt(m * l) * T[n, k, m - 1, l - 1] * np.conj(exptheta)
    return gradTr, gradTtheta


# pylint: disable=too-many-arguments
def S2gate(r, theta, cutoff, grad=False, sf_order=False, dtype=np.complex128):
    """Calculates the Fock representation of the S2gate and its gradient.

    Args:
        r (float): two-mode squeezing magnitude
        theta (float): two-mode squeezing phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        sf_order (boolean): whether to use Strawberry Fields ordering for the indices
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """

    if not grad:
        if sf_order:
            index_order = (0, 2, 1, 3)
            return (
                two_mode_squeezing(r, theta, cutoff, dtype=dtype).transpose(index_order),
                None,
                None,
            )
        return two_mode_squeezing(r, theta, cutoff, dtype=dtype), None, None

    T = two_mode_squeezing(r, theta, cutoff + 1, dtype=dtype)
    (gradTr, gradTtheta) = grad_S2gate(T, theta, cutoff, dtype=dtype)

    if sf_order:
        index_order = (0, 2, 1, 3)
        return (
            T[:cutoff, :cutoff, :cutoff, :cutoff].transpose(index_order),
            gradTr.transpose(index_order),
            gradTtheta.transpose(index_order),
        )

    return T[:cutoff, :cutoff, :cutoff, :cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_BSgate(T, phi, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the gradient of the BSgate.

    Args:
        T (array[complex]): array representing the gate
        theta (float): phase angle parametrizing the gate
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the BSgate with respect to r (the amplitude) and theta (the phase)
    """
    expphi = np.exp(1j * phi)
    gradTtheta = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    gradTphi = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = n + k - m
                if 0 <= l < cutoff:
                    gradTphi[n, k, m, l] = -1j * (n - m) * T[n, k, m, l]
                    if m > 0:
                        gradTtheta[n, k, m, l] = np.sqrt(m * (l + 1)) * T[n, k, m - 1, l + 1] * expphi
                    if l > 0:
                        gradTtheta[n, k, m, l] -= np.sqrt((m + 1) * l) * T[n, k, m + 1, l - 1] * np.conj(expphi)
    return gradTtheta, gradTphi


# pylint: disable=too-many-arguments
def BSgate(theta, phi, cutoff, grad=False, sf_order=False, dtype=np.complex128):
    r"""Calculates the Fock representation of the S2gate and its gradient.

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        sf_order (boolean): whether to use Strawberry Fields ordering for the indices
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    if not grad:
        if sf_order:
            index_order = (0, 2, 1, 3)
            return (
                beamsplitter(theta, phi, cutoff, dtype=dtype).transpose(index_order),
                None,
                None,
            )
        return beamsplitter(theta, phi, cutoff, dtype=dtype), None, None

    T = beamsplitter(theta, phi, cutoff + 1, dtype=dtype)
    gradTtheta, gradTphi = grad_BSgate(T, phi, cutoff, dtype=dtype)

    if sf_order:
        index_order = (0, 2, 1, 3)
        return (
            T[:cutoff, :cutoff, :cutoff, :cutoff].transpose(index_order),
            gradTtheta.transpose(index_order),
            gradTphi.transpose(index_order),
        )

    return T[:cutoff, :cutoff, :cutoff, :cutoff], gradTtheta, gradTphi


def Rgate(theta, cutoff, grad=False, dtype=np.complex128):
    """Calculates the Fock representation of the Rgate and its gradient.

    Args:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        dtype (data type): Specifies the data type used for the calculation


    Returns:
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    ns = np.arange(cutoff, dtype=dtype)
    T = np.exp(1j * theta) ** ns
    if not grad:
        return np.diag(T), None
    return np.diag(T), np.diag(1j * ns * T)


def Kgate(theta, cutoff, grad=False, dtype=np.complex128):
    """Calculates the Fock representation of the Kgate and its gradient.

    Args:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    ns = np.arange(cutoff, dtype=dtype)
    T = np.exp(1j * theta) ** (ns ** 2)
    if not grad:
        return np.diag(T), None
    return np.diag(T), np.diag(1j * (ns ** 2) * T)


@jit(nopython=True)
def Ggate_jit(phiR, w, z, cutoff, dtype=np.complex128):
    """Calculates the Fock representation of the single-mode Gaussian gate parametrized
    as the product of the three gates R(phi)D(w)S(z).

    Args:
        phiR (float): angle of the phase rotation gate
        w (complex): displacement parameter
        z (complex): squeezing parameter
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: The Fock representation of the gate
    """

    rS = np.abs(z)
    phiS = np.angle(z)
    EZ = np.exp(1j * phiS)
    T = np.tanh(rS)
    S = 1 / np.cosh(rS)
    ER = np.exp(1j * phiR)
    wc = np.conj(w)

    # 2nd derivatives of G
    R = np.array([[-T * EZ * ER ** 2, ER * S], [ER * S, T * np.conj(EZ)]])

    # 1st derivatives of G
    y = np.array([ER * (w + T * EZ * wc), -wc * S])

    sqrt = np.sqrt(np.arange(cutoff))
    Z = np.zeros((cutoff, cutoff), dtype=dtype)

    Z[0, 0] = np.exp(-0.5 * np.abs(w) ** 2 - 0.5 * wc ** 2 * EZ * T) * np.sqrt(S)

    # rank 1
    for m in range(1, cutoff):
        Z[m, 0] = y[0] / sqrt[m] * Z[m - 1, 0] + R[0, 0] * sqrt[m - 1] / sqrt[m] * Z[m - 2, 0]

    # rank 2
    for m in range(cutoff):
        for n in range(1, cutoff):
            Z[m, n] = y[1] / sqrt[n] * Z[m, n - 1] + R[1, 1] * sqrt[n - 1] / sqrt[n] * Z[m, n - 2] + R[0, 1] * sqrt[m] / sqrt[n] * Z[m - 1, n - 1]

    return Z


@jit(nopython=True)
def Ggate_gradients(phiR, w, z, gate):
    """Computes the complex gradients with respect to all the parameters of the Gaussian gate.
    As the parameters are complex, it returns two gradients per gate (with respect to the parameter and
    with respect to its complex conjugate).

    Args:
        phiR (float): angle of the phase rotation gate
        w (complex): displacement parameter
        z (complex): squeezing parameter
        gate (array): Gaussian gate evaluated at the same parameter values

    Returns:
        tuple[array[complex] x 5]: 1 gradient array for phiR and 2 gradient arrays for (complex) w and z.
    """
    dtype = gate.dtype
    cutoff = gate.shape[0]
    sqrt = np.sqrt(np.arange(cutoff))

    rS = np.abs(z)
    phiS = np.angle(z)
    T = np.tanh(rS)
    S = 1 / np.cosh(rS)
    wc = np.conj(w)

    # Taylor series to avoid division by zero for rS -> 0
    if rS > 1e-3:
        TSplus = T / rS + S ** 2
        TSminus = T / rS - S ** 2
    else:
        TSplus = 1 - rS ** 2 / 3 + S ** 2
        TSminus = 1 - rS ** 2 / 3 - S ** 2

    ### Gradient with respect to phiR
    phi_a = 1j * np.exp(1j * phiR) * (w + wc * np.exp(1j * phiS) * T)
    phi_a2 = -1j * np.exp(1j * (2 * phiR + phiS)) * T
    phi_ab = 1j * S * np.exp(1j * phiR)

    Grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_phi[m, n] = (
                phi_a * sqrt[m] * gate[m - 1, n] + phi_a2 * sqrt[m] * sqrt[m - 1] * gate[m - 2, n] + phi_ab * sqrt[m] * sqrt[n] * gate[m - 1, n - 1]
            )

    ### Gradients with respect to w
    w_a = np.exp(1j * phiR)
    w_1 = -0.5 * wc

    Grad_w = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_w[m, n] = w_a * sqrt[m] * gate[m - 1, n] + w_1 * gate[m, n]

    wc_a = np.exp(1j * (phiR + phiS)) * T
    wc_b = -S + 0.0j
    wc_1 = -0.5 * (w + 2 * wc * np.exp(1j * phiS) * T)

    Grad_wconj = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_wconj[m, n] = wc_a * sqrt[m] * gate[m - 1, n] + wc_b * sqrt[n] * gate[m, n - 1] + wc_1 * gate[m, n]

    ### Gradients with respect to z
    z_a = 0.5 * wc * np.exp(1j * phiR) * TSplus
    z_a2 = -0.25 * np.exp(2j * phiR) * TSplus
    z_b = 0.5 * wc * np.exp(-1j * phiS) * T * S
    z_b2 = -0.25 * np.exp(-2j * phiS) * TSminus
    z_ab = -0.5 * np.exp(1j * (phiR - phiS)) * T * S
    z_1 = -0.25 * wc ** 2 * TSplus - 0.25 * np.exp(-1j * phiS) * T

    Grad_z = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_z[m, n] = (
                z_a * sqrt[m] * gate[m - 1, n]
                + z_b * sqrt[n] * gate[m, n - 1]
                + z_a2 * sqrt[m] * sqrt[m - 1] * gate[m - 2, n]
                + z_b2 * sqrt[n] * sqrt[n - 1] * gate[m, n - 2]
                + z_ab * sqrt[m] * sqrt[n] * gate[m - 1, n - 1]
                + z_1 * gate[m, n]
            )

    zc_a = -0.5 * wc * np.exp(1j * (phiR + 2 * phiS)) * TSminus
    zc_a2 = 0.25 * np.exp(2 * 1j * (phiR + phiS)) * TSminus
    zc_b = 0.5 * wc * np.exp(1j * phiS) * T * S
    zc_b2 = 0.25 * TSplus + 0.0j
    zc_ab = -0.5 * np.exp(1j * (phiR + phiS)) * T * S
    zc_1 = 0.25 * wc ** 2 * np.exp(2 * 1j * phiS) * TSminus - 0.25 * np.exp(1j * phiS) * T

    Grad_zconj = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_zconj[m, n] = (
                zc_a * sqrt[m] * gate[m - 1, n]
                + zc_b * sqrt[n] * gate[m, n - 1]
                + zc_a2 * sqrt[m] * sqrt[m - 1] * gate[m - 2, n]
                + zc_b2 * sqrt[n] * sqrt[n - 1] * gate[m, n - 2]
                + zc_ab * sqrt[m] * sqrt[n] * gate[m - 1, n - 1]
                + zc_1 * gate[m, n]
            )

    return Grad_phi, np.conj(Grad_w), Grad_wconj, np.conj(Grad_z), Grad_zconj


@jit(nopython=True)
def G2gate_jit(phi12, w12, BS1, z12, BS2, cutoff, dtype=np.complex128):
    """Calculates the Fock representation of the two-mode Gaussian gate parametrized
    as the product of the gates R(phi1)xR(phi2) D(w1) x D(w2) BS(th1,vphi1) S(z1)xS(z2) BS(th2,vphi2).

    Args:
        phi12 (float, float): angles of the phase rotation gates
        w12 (complex, complex): displacement parameters
        BS1 (float, float): angles of the first beamsplitter
        z12 (complex, complex): squeezing parameters
        BS2 (float, float): angles of the second beamsplitter
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: The Fock representation of the gate
    """
    # init variables
    phi1, phi2 = phi12
    w1, w2 = w12
    w1c, w2c = np.conj(w12)
    th1, vphi1 = BS1
    th2, vphi2 = BS2

    # derived quantities
    S1, S2 = 1 / np.cosh(np.abs(z12))
    T1, T2 = np.tanh(np.abs(z12)) * np.exp(1j * np.angle(z12))
    c1 = np.cos(th1)
    c2 = np.cos(th2)
    s1 = np.sin(th1)
    s2 = np.sin(th2)
    e1 = np.exp(1j * vphi1)
    e1c = np.exp(-1j * vphi1)
    e2 = np.exp(1j * vphi2)
    e2c = np.exp(-1j * vphi2)
    p1 = np.exp(1j * phi1)
    p2 = np.exp(1j * phi2)

    # 2nd derivatives of Q (order: a1, b1, a2, b2), omitted symmetric lower part
    R = np.array(
        [
            [
                -(p1 ** 2) * (c1 ** 2 * T1 + e1 ** 2 * s1 ** 2 * T2),
                p1 * (c1 * c2 * S1 - e1 * e2 * s1 * s2 * S2),
                c1 * p1 * p2 * e1c * s1 * (T2 * e1 ** 2 - T1),
                -p1 * e2c * (c1 * s2 * S1 + c2 * e1 * e2 * s1 * S2),
            ],
            [
                0,
                c2 ** 2 * np.conj(T1) + e2 ** 2 * s2 ** 2 * np.conj(T2),
                p2 * (c2 * e1c * s1 * S1 + c1 * e2 * s2 * S2),
                c2 * e2 * s2 * np.conj(T2) - c2 * e2c * s2 * np.conj(T1),
            ],
            [0, 0, -(p2 ** 2) * (e1c ** 2 * s1 ** 2 * T1 + c1 ** 2 * T2), p2 * (c1 * c2 * S2 - s1 * s2 * S1 * e1c * e2c),],
            [0, 0, 0, e2c ** 2 * s2 ** 2 * np.conj(T1) + c2 ** 2 * np.conj(T2)],
        ]
    )

    # 1st derivatives of Q
    y = np.array(
        [
            p1 * (w1 + e1 * s1 * T2 * (e1 * s1 * w1c - c1 * w2c) + c1 * T1 * (c1 * w1c + e1c * s1 * w2c)),
            e2 * s2 * S2 * (e1 * s1 * w1c - c1 * w2c) - c2 * S1 * (c1 * w1c + e1c * s1 * w2c),
            p2 * (w2 + c1 * T2 * (c1 * w2c - e1 * s1 * w1c) + e1c ** 2 * s1 * T1 * (c1 * e1 * w1c + s1 * w2c)),
            (c1 * e2c * S1 * s2 + c2 * e1 * s1 * S2) * w1c + (e1c * e2c * s1 * S1 * s2 - c1 * c2 * S2) * w2c,
        ]
    )

    sqrt = np.sqrt(np.arange(cutoff))
    Z = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)

    Z[0, 0, 0, 0] = np.sqrt(S1 * S2) * np.exp(
        0.5 * (-np.abs(w1) ** 2 - np.abs(w2) ** 2 - T1 * (c1 * w1c + e1c * s1 * w2c) ** 2 - T2 * (e1 * s1 * w1c - c1 * w2c) ** 2)
    )

    # rank 1
    for m in range(1, cutoff):
        Z[m, 0, 0, 0] = y[0] / sqrt[m] * Z[m - 1, 0, 0, 0] + R[0, 0] * sqrt[m - 1] / sqrt[m] * Z[m - 2, 0, 0, 0]

    # rank 2
    for m in range(cutoff):
        for n in range(1, cutoff):
            Z[m, n, 0, 0] = (
                y[1] / sqrt[n] * Z[m, n - 1, 0, 0]
                + sqrt[m] / sqrt[n] * R[0, 1] * Z[m - 1, n - 1, 0, 0]
                + sqrt[n - 1] / sqrt[n] * R[1, 1] * Z[m, n - 2, 0, 0]
            )

    # rank 3
    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(1, cutoff):
                Z[m, n, p, 0] = (
                    y[2] / sqrt[p] * Z[m, n, p - 1, 0]
                    + sqrt[m] / sqrt[p] * R[0, 2] * Z[m - 1, n, p - 1, 0]
                    + sqrt[n] / sqrt[p] * R[1, 2] * Z[m, n - 1, p - 1, 0]
                    + sqrt[p - 1] / sqrt[p] * R[2, 2] * Z[m, n, p - 2, 0]
                )

    # rank 4
    for m in range(cutoff):
        for n in range(cutoff):
            for p in range(cutoff):
                for q in range(1, cutoff):
                    Z[m, n, p, q] = (
                        y[3] / sqrt[q] * Z[m, n, p, q - 1]
                        + sqrt[m] / sqrt[q] * R[0, 3] * Z[m - 1, n, p, q - 1]
                        + sqrt[n] / sqrt[q] * R[1, 3] * Z[m, n - 1, p, q - 1]
                        + sqrt[p] / sqrt[q] * R[2, 3] * Z[m, n, p - 1, q - 1]
                        + sqrt[q - 1] / sqrt[q] * R[3, 3] * Z[m, n, p, q - 2]
                    )

    return Z


@jit(nopython=True)
def G2gate_gradients(phi12, w12, BS1, z12, BS2, gate):
    """Computes the complex gradients with respect to all the parameters of the two-mode Gaussian gate.
    For complex parameters it returns two gradients per gate (with respect to the parameter and
    with respect to its complex conjugate).

    Args:
        phi12 (float, float): angles of the phase rotation gates
        w12 (complex, complex): displacement parameters
        BS1 (float, float): first beamsplitter parameters
        z12 (complex, complex): squeezing parameters
        BS2 (float, float): second beamsplitter parameters
        gate (array): Gaussian gate evaluated at the same parameter values

    Returns:
        tuple[array[complex] x 14]: 1 gradient array for the each real parameter (6) and 2 gradient arrays for each complex parameter (4)
    """
    dtype = gate.dtype
    cutoff = gate.shape[0]
    sqrt = np.sqrt(np.arange(cutoff))

    # init variables
    phi1, phi2 = phi12
    w1, w2 = w12
    w1c, w2c = np.conj(w12)
    th1, vphi1 = BS1
    th2, vphi2 = BS2

    # derived quantities
    r12 = np.abs(z12)
    r1, r2 = r12
    zeta1, zeta2 = np.angle(z12)
    S1, S2 = 1 / np.cosh(np.abs(z12))
    T1, T2 = np.tanh(r12) * np.exp(1j * np.angle(z12))
    c1 = np.cos(th1)
    c2 = np.cos(th2)
    s1 = np.sin(th1)
    s2 = np.sin(th2)
    e1 = np.exp(1j * vphi1)
    e1c = np.conj(e1)
    e2 = np.exp(1j * vphi2)
    e2c = np.conj(e2)
    p1 = np.exp(1j * phi1)
    p2 = np.exp(1j * phi2)

    # Gradients with respect to phi1 and phi2
    phi1_a1 = 1j * p1 * (w1 + (c1 ** 2 * T1 + e1 ** 2 * s1 ** 2 * T2) * w1c + 0.5 * e1c * (T1 - e1 ** 2 * T2) * np.sin(2 * th1) * w2c)
    phi1_a1b2 = -1j * p1 * (e1 * s1 * c2 * S2 + e2c * s2 * c1 * S1)
    phi1_a1a2 = -1j * p1 * p2 * c1 * s1 * (T1 * e1c - T2 * e1)
    phi1_a1b1 = 1j * p1 * (c1 * c2 * S1 - s1 * s2 * S2 * e1 * e2)
    phi1_a1a1 = -1j * np.exp(2j * phi1) * (T1 * c1 ** 2 + T2 * s1 ** 2 * e1 ** 2)

    phi2_a2 = 1j * p2 * (w2 + 0.5 * e1 * (T1 * e1c ** 2 - T2) * np.sin(2 * th1) * w1c + (s1 ** 2 * e1c ** 2 * T1 + c1 ** 2 * T2) * w2c)
    phi2_a2b2 = 1j * p2 * (c1 * c2 * S2 - e1c * e2c * S1 * s2 * s1)
    phi2_a2a2 = -1j * np.exp(2j * phi2) * (T1 * s1 ** 2 * e1c ** 2 + T2 * c1 ** 2)
    phi2_a2b1 = 1j * p2 * (e1c * c2 * S1 * s1 + e2 * c1 * S2 * s2)
    phi2_a2a1 = -1j * p1 * p2 * c1 * s1 * (T1 * e1c - T2 * e1)

    Grad_phi1 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_phi1[a1, b1, a2, b2] = (
                        phi1_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + phi1_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + phi1_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + phi1_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + phi1_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                    )

    Grad_phi2 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_phi2[a1, b1, a2, b2] = (
                        phi2_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + phi2_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + phi2_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + phi2_a2b1 * sqrt[a2] * sqrt[b1] * gate[a1, b1 - 1, a2 - 1, b2]
                        + phi2_a2a1 * sqrt[a2] * sqrt[a1] * gate[a1 - 1, b1, a2 - 1, b2]
                    )

    # Gradients with respect to w12
    w1_1 = -0.5 * w1c
    w1_a1 = p1

    w2_1 = -0.5 * w2c
    w2_a2 = p2

    w1c_1 = -0.5 * w1 - 0.5 * np.sin(2 * th1) * w2c * e1c * (T1 - T2 * e1 ** 2) - w1c * (T1 * c1 ** 2 + e1 ** 2 * T2 * s1 ** 2)
    w1c_b2 = e2c * c1 * s2 * S1 + e1 * c2 * s1 * S2
    w1c_a2 = c1 * p2 * e1c * s1 * (T1 - e1 ** 2 * T2)
    w1c_b1 = -c1 * c2 * S1 + e1 * e2 * S2 * s1 * s2
    w1c_a1 = p1 * (c1 ** 2 * T1 + e1 ** 2 * s1 ** 2 * T2)

    w2c_1 = -0.5 * w2 + c1 * e1 * s1 * T2 * w1c - 0.5 * e1c * T1 * np.sin(2 * th1) * w1c - e1c ** 2 * s1 ** 2 * T1 * w2c - c1 ** 2 * T2 * w2c
    w2c_b2 = e1c * e2c * s1 * s2 * S1 - c1 * c2 * S2
    w2c_a2 = p2 * (e1c ** 2 * s1 ** 2 * T1 + c1 ** 2 * T2)
    w2c_b1 = -c2 * e1c * s1 * S1 - c1 * e2 * s2 * S2
    w2c_a1 = c1 * p1 * e1c * s1 * (T1 - e1 ** 2 * T2)

    Grad_w1 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_w1[a1, b1, a2, b2] = w1_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2] + w1_1 * gate[a1, b1, a2, b2]

    Grad_w2 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_w2[a1, b1, a2, b2] = w2_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2] + w2_1 * gate[a1, b1, a2, b2]

    Grad_w1c = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_w1c[a1, b1, a2, b2] = (
                        w1c_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + w1c_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + w1c_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + w1c_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + w1c_1 * gate[a1, b1, a2, b2]
                    )

    Grad_w2c = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_w2c[a1, b1, a2, b2] = (
                        w2c_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + w2c_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + w2c_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + w2c_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + w2c_1 * gate[a1, b1, a2, b2]
                    )

    # Gradients with respect to z12
    # Taylor series to avoid division by zero for r -> 0
    if r1 > 1e-3:
        TS1plus = np.tanh(r1) / r1 + S1 ** 2
        TS1minus = np.tanh(r1) / r1 - S1 ** 2
    else:
        TS1plus = 1 - r1 ** 2 / 3 + S1 ** 2
        TS1minus = 1 - r1 ** 2 / 3 - S1 ** 2
    if r2 > 1e-3:
        TS2plus = np.tanh(r2) / r2 + S2 ** 2
        TS2minus = np.tanh(r2) / r2 - S2 ** 2
    else:
        TS2plus = 1 - r2 ** 2 / 3 + S2 ** 2
        TS2minus = 1 - r2 ** 2 / 3 - S2 ** 2

    z1_1 = -0.25 * TS1plus * (c1 * w1c + e1c * s1 * w2c) ** 2 - 0.25 * np.conj(T1)
    z1_b2 = -0.5 * e2c * S1 * np.conj(T1) * s2 * (c1 * w1c + e1c * s1 * w2c)
    z1_b2b2 = -0.25 * e2c ** 2 * np.exp(-2j * zeta1) * s2 ** 2 * TS1minus
    z1_a2 = 0.5 * p2 * e1c * s1 * TS1plus * (c1 * w1c + e1c * s1 * w2c)
    z1_a2b2 = 0.5 * np.conj(T1) * p2 * e1c * e2c * S1 * s1 * s2
    z1_a2a2 = -0.25 * np.exp(2j * phi2) * e1c ** 2 * s1 ** 2 * TS1plus
    z1_b1 = 0.5 * S1 * np.conj(T1) * c2 * (c1 * w1c + e1c * s1 * w2c)
    z1_b1b2 = 0.25 * np.exp(-2j * zeta1) * e2c * np.sin(2 * th2) * TS1minus
    z1_b1a2 = -0.5 * np.conj(T1) * np.exp(1j * phi2) * e1c * S1 * c2 * s1
    z1_b1b1 = -0.25 * np.exp(-2j * zeta1) * c2 ** 2 * TS1minus
    z1_a1 = 0.5 * p1 * c1 * TS1plus * (c1 * w1c + e1c * s1 * w2c)
    z1_a1b2 = 0.5 * p1 * np.conj(T1) * S1 * e2c * c1 * s2
    z1_a1a2 = -0.25 * np.exp(1j * (phi1 + phi2)) * e1c * np.sin(2 * th1) * TS1plus
    z1_a1b1 = -0.5 * np.conj(T1) * p1 * S1 * c1 * c2
    z1_a1a1 = -0.25 * np.exp(2j * phi1) * c1 ** 2 * TS1plus

    z1c_1 = 0.25 * np.exp(2j * zeta1) * TS1minus * (c1 * w1c + e1c * s1 * w2c) ** 2 - 0.25 * T1
    z1c_b2 = -0.5 * e2c * S1 * T1 * s2 * (c1 * w1c + e1c * s1 * w2c)
    z1c_b2b2 = 0.25 * e2c ** 2 * s2 ** 2 * TS1plus
    z1c_a2 = -0.5 * np.exp(2j * zeta1) * p2 * e1c * s1 * TS1minus * (c1 * w1c + e1c * s1 * w2c)
    z1c_a2b2 = 0.5 * T1 * np.exp(1j * phi2) * e1c * e2c * S1 * s2 * s1
    z1c_a2a2 = 0.25 * np.exp(2j * zeta1) * np.exp(2j * phi2) * e1c ** 2 * s1 ** 2 * TS1minus
    z1c_b1 = 0.5 * S1 * T1 * c2 * (c1 * w1c + e1c * s1 * w2c)
    z1c_b1b2 = -0.25 * e2c * np.sin(2 * th2) * TS1plus
    z1c_b1a2 = -0.5 * T1 * p2 * e1c * S1 * c2 * s1
    z1c_b1b1 = 0.25 * c2 ** 2 * TS1plus
    z1c_a1 = -0.5 * np.exp(2j * zeta1) * np.exp(1j * phi1) * c1 * TS1minus * (c1 * w1c + e1c * s1 * w2c)
    z1c_a1b2 = 0.5 * p1 * T1 * S1 * e2c * c1 * s2
    z1c_a1a2 = 0.25 * np.exp(2j * zeta1) * p1 * p2 * e1c * np.sin(2 * th1) * TS1minus
    z1c_a1b1 = -0.5 * T1 * p1 * S1 * c1 * c2
    z1c_a1a1 = 0.25 * np.exp(2j * zeta1) * np.exp(2j * phi1) * c1 ** 2 * TS1minus

    Grad_z1 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_z1[a1, b1, a2, b2] = (
                        z1_1 * gate[a1, b1, a2, b2]
                        + z1_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + z1_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + z1_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + z1_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + z1_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + z1_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + z1_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + z1_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + z1_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + z1_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + z1_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + z1_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + z1_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + z1_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    Grad_z1c = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_z1c[a1, b1, a2, b2] = (
                        z1c_1 * gate[a1, b1, a2, b2]
                        + z1c_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + z1c_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + z1c_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + z1c_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + z1c_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + z1c_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + z1c_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + z1c_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + z1c_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + z1c_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + z1c_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + z1c_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + z1c_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + z1c_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    z2_1 = -0.25 * TS2plus * (e1 * s1 * w1c - c1 * w2c) ** 2 - 0.25 * np.conj(T2)
    z2_b2 = -0.5 * S2 * np.conj(T2) * c2 * (e1 * s1 * w1c - c1 * w2c)
    z2_b2b2 = -0.25 * np.exp(-2j * zeta2) * c2 ** 2 * TS2minus
    z2_a2 = -0.5 * p2 * c1 * TS2plus * (e1 * s1 * w1c - c1 * w2c)
    z2_a2b2 = -0.5 * p2 * S2 * np.conj(T2) * c1 * c2
    z2_a2a2 = -0.25 * np.exp(2j * phi2) * c1 ** 2 * TS2plus
    z2_b1 = -0.5 * e2 * S2 * np.conj(T2) * s2 * (e1 * s1 * w1c - c1 * w2c)
    z2_b1b2 = -0.25 * np.exp(-2j * zeta2) * e2 * np.sin(2 * th2) * TS2minus
    z2_b1a2 = -0.5 * np.exp(1j * phi2) * e2 * S2 * np.conj(T2) * c1 * s2
    z2_b1b1 = -0.25 * np.exp(-2j * zeta2) * e2 ** 2 * s2 ** 2 * TS2minus
    z2_a1 = 0.5 * np.exp(1j * phi1) * e1 * s1 * TS2plus * (e1 * s1 * w1c - c1 * w2c)
    z2_a1b2 = 0.5 * np.exp(1j * phi1) * e1 * S2 * np.conj(T2) * c2 * s1
    z2_a1a2 = 0.25 * p1 * p2 * e1 * np.sin(2 * th1) * TS2plus
    z2_a1b1 = 0.5 * p1 * e1 * e2 * S2 * np.conj(T2) * s2 * s1
    z2_a1a1 = -0.25 * np.exp(2j * phi1) * e1 ** 2 * s1 ** 2 * TS2plus

    z2c_1 = 0.25 * np.exp(2j * zeta2) * TS2minus * (e1 * s1 * w1c - c1 * w2c) ** 2 - 0.25 * T2
    z2c_b2 = -0.5 * S2 * T2 * c2 * (e1 * s1 * w1c - c1 * w2c)
    z2c_b2b2 = 0.25 * c2 ** 2 * TS2plus
    z2c_a2 = 0.5 * np.exp(2j * zeta2) * p2 * c1 * TS2minus * (e1 * s1 * w1c - c1 * w2c)
    z2c_a2b2 = -0.5 * p2 * S2 * T2 * c1 * c2
    z2c_a2a2 = 0.25 * np.exp(2j * zeta2) * np.exp(2j * phi2) * c1 ** 2 * TS2minus
    z2c_b1 = -0.5 * e2 * S2 * T2 * s2 * (e1 * s1 * w1c - c1 * w2c)
    z2c_b1b2 = 0.25 * e2 * np.sin(2 * th2) * TS2plus
    z2c_b1a2 = -0.5 * p2 * e2 * S2 * T2 * c1 * s2
    z2c_b1b1 = 0.25 * e2 ** 2 * s2 ** 2 * TS2plus
    z2c_a1 = -0.5 * np.exp(2j * zeta2) * p1 * e1 * s1 * TS2minus * (e1 * s1 * w1c - c1 * w2c)
    z2c_a1b2 = 0.5 * p1 * e1 * S2 * T2 * c2 * s1
    z2c_a1a2 = -0.25 * p1 * p2 * np.exp(2j * zeta2) * e1 * np.sin(2 * th1) * TS2minus
    z2c_a1b1 = 0.5 * p1 * e1 * e2 * S2 * T2 * s2 * s1
    z2c_a1a1 = 0.25 * np.exp(2j * phi1) * np.exp(2j * zeta2) * e1 ** 2 * s1 ** 2 * TS2minus

    Grad_z2 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_z2[a1, b1, a2, b2] = (
                        z2_1 * gate[a1, b1, a2, b2]
                        + z2_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + z2_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + z2_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + z2_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + z2_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + z2_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + z2_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + z2_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + z2_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + z2_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + z2_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + z2_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + z2_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + z2_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    Grad_z2c = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_z2c[a1, b1, a2, b2] = (
                        z2c_1 * gate[a1, b1, a2, b2]
                        + z2c_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + z2c_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + z2c_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + z2c_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + z2c_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + z2c_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + z2c_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + z2c_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + z2c_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + z2c_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + z2c_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + z2c_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + z2c_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + z2c_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    # Gradients with respect to theta1 and varphi1
    th1_1 = -(T1 - T2 * e1 ** 2) * (c1 * w2c * e1c - s1 * w1c) * (c1 * w1c + e1c * s1 * w2c)
    th1_b2 = e2c * s2 * S1 * (c1 * w2c * e1c - s1 * w1c) + c2 * S2 * (e1 * c1 * w1c + s1 * w2c)
    th1_a2 = p2 * e1 * (T1 * e1c ** 2 - T2) * ((c1 ** 2 - s1 ** 2) * w1c + 2 * e1c * c1 * s1 * w2c)
    th1_a2b2 = p2 * (-e1c * e2c * c1 * s2 * S1 - c2 * s1 * S2)
    th1_a2a2 = -np.exp(2j * phi2) * c1 * s1 * (e1c ** 2 * T1 - T2)
    th1_b1 = c2 * S1 * (s1 * w1c - e1c * c1 * w2c) + e1 * e2 * s2 * S2 * (c1 * w1c + e1c * s1 * w2c)
    th1_b1a2 = p2 * (e1c * c1 * c2 * S1 - e2 * s1 * s2 * S2)
    th1_a1 = p1 * (T1 - T2 * e1 ** 2) * (e1c * (c1 ** 2 - s1 ** 2) * w2c - 2 * c1 * s1 * w1c)
    th1_a1b2 = p1 * (e2c * s1 * s2 * S1 - e1 * c1 * c2 * S2)
    th1_a1a2 = p1 * p2 * e1c * np.cos(2 * th1) * (e1 ** 2 * T2 - T1)
    th1_a1b1 = -p1 * (c2 * s1 * S1 + e1 * e2 * c1 * s2 * S2)
    th1_a1a1 = np.exp(2j * phi1) * c1 * s1 * (T1 - T2 * e1 ** 2)

    vphi1_1 = -1j * s1 * (T2 * w1c * e1 ** 2 * (s1 * w1c - e1c * c1 * w2c) - T1 * w2c * e1c * (c1 * w1c + e1c * s1 * w2c))
    vphi1_b2 = 1j * s1 * (c2 * S2 * w1c * e1 - e1c * e2c * s2 * S1 * w2c)
    vphi1_a2 = -1j * p2 * e1 * s1 * (c1 * (T1 * e1c ** 2 + T2) * w1c + 2 * T1 * e1c ** 3 * s1 * w2c)
    vphi1_a2b2 = 1j * p2 * e1c * e2c * s1 * s2 * S1
    vphi1_a2a2 = 1j * T1 * np.exp(2j * phi2) * e1c ** 2 * s1 ** 2
    vphi1_b1 = 1j * e1 * s1 * (e2 * s2 * S2 * w1c + e1c ** 2 * c2 * S1 * w2c)
    vphi1_b1a2 = -1j * p2 * e1c * c2 * s1 * S1
    vphi1_a1 = -1j * p1 * e1 ** 2 * s1 * (-2 * s1 * T2 * w1c + e1c * c1 * (T1 * e1c ** 2 + T2) * w2c)
    vphi1_a1b2 = -1j * p1 * e1 * c2 * s1 * S2
    vphi1_a1a2 = 1j * p1 * p2 * e1 * c1 * s1 * (T1 * e1c ** 2 + T2)
    vphi1_a1b1 = -1j * p1 * e1 * e2 * s1 * s2 * S2
    vphi1_a1a1 = -1j * np.exp(2j * phi1) * e1 ** 2 * s1 ** 2 * T2

    Grad_th1 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_th1[a1, b1, a2, b2] = (
                        th1_1 * gate[a1, b1, a2, b2]
                        + th1_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + th1_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + th1_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + th1_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + th1_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + th1_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + th1_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + th1_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + th1_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + th1_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + th1_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                    )

    Grad_vphi1 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_vphi1[a1, b1, a2, b2] = (
                        vphi1_1 * gate[a1, b1, a2, b2]
                        + vphi1_a1 * sqrt[a1] * gate[a1 - 1, b1, a2, b2]
                        + vphi1_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + vphi1_a2 * sqrt[a2] * gate[a1, b1, a2 - 1, b2]
                        + vphi1_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + vphi1_a1a1 * sqrt[a1] * sqrt[a1 - 1] * gate[a1 - 2, b1, a2, b2]
                        + vphi1_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + vphi1_a1a2 * sqrt[a1] * sqrt[a2] * gate[a1 - 1, b1, a2 - 1, b2]
                        + vphi1_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + vphi1_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + vphi1_a2a2 * sqrt[a2] * sqrt[a2 - 1] * gate[a1, b1, a2 - 2, b2]
                        + vphi1_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                    )

    # Gradients with respect to theta2 and varphi2
    th2_b2 = (c1 * c2 * e2c * S1 - e1 * s1 * s2 * S2) * w1c + (c2 * e1c * e2c * s1 * S1 + c1 * s2 * S2) * w2c
    th2_b2b2 = c2 * s2 * (e2c ** 2 * np.conj(T1) - np.conj(T2))
    th2_a2b2 = p2 * (-c2 * e1c * e2c * s1 * S1 - c1 * s2 * S2)
    th2_b1 = (c1 * S1 * s2 + c2 * e1 * e2 * s1 * S2) * w1c + (e1c * s1 * S1 * s2 - c1 * c2 * e2 * S2) * w2c
    th2_b1b2 = e2c * (-np.conj(T1) + e2 ** 2 * np.conj(T2)) * np.cos(2 * th2)
    th2_b1a2 = p2 * (-e1c * s1 * S1 * s2 + c1 * c2 * e2 * S2)
    th2_b1b1 = c2 * s2 * (-np.conj(T1) + e2 ** 2 * np.conj(T2))
    th2_a1b2 = p1 * (-c1 * c2 * e2c * S1 + e1 * s1 * s2 * S2)
    th2_a1b1 = -p1 * (c1 * S1 * s2 + c2 * (e1 * e2 * s1 * S2))

    vphi2_b2 = -1j * e2c * S1 * s2 * (c1 * w1c + e1c * s1 * w2c)
    vphi2_b2b2 = -1j * e2c ** 2 * s2 ** 2 * np.conj(T1)
    vphi2_a2b2 = 1j * p2 * e1c * e2c * s1 * S1 * s2
    vphi2_b1 = 1j * e2 * S2 * s2 * (e1 * s1 * w1c - c1 * w2c)
    vphi2_b1b2 = 1j * c2 * e2c * s2 * (np.conj(T1) + np.conj(T2) * e2 ** 2)
    vphi2_b1a2 = 1j * c1 * p2 * e2 * s2 * S2
    vphi2_b1b1 = 1j * e2 ** 2 * s2 ** 2 * np.conj(T2)
    vphi2_a1b2 = 1j * c1 * p1 * e2c * S1 * s2
    vphi2_a1b1 = -1j * p1 * e1 * e2 * s1 * s2 * S2

    Grad_th2 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_th2[a1, b1, a2, b2] = (
                        th2_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + th2_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + th2_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + th2_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + th2_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + th2_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + th2_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + th2_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + th2_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    Grad_vphi2 = np.zeros((cutoff, cutoff, cutoff, cutoff), dtype=dtype)
    for a1 in range(cutoff):
        for b1 in range(cutoff):
            for a2 in range(cutoff):
                for b2 in range(cutoff):
                    Grad_vphi2[a1, b1, a2, b2] = (
                        vphi2_b1 * sqrt[b1] * gate[a1, b1 - 1, a2, b2]
                        + vphi2_b2 * sqrt[b2] * gate[a1, b1, a2, b2 - 1]
                        + vphi2_a1b1 * sqrt[a1] * sqrt[b1] * gate[a1 - 1, b1 - 1, a2, b2]
                        + vphi2_a1b2 * sqrt[a1] * sqrt[b2] * gate[a1 - 1, b1, a2, b2 - 1]
                        + vphi2_b1b1 * sqrt[b1] * sqrt[b1 - 1] * gate[a1, b1 - 2, a2, b2]
                        + vphi2_b1b2 * sqrt[b1] * sqrt[b2] * gate[a1, b1 - 1, a2, b2 - 1]
                        + vphi2_b1a2 * sqrt[b1] * sqrt[a2] * gate[a1, b1 - 1, a2 - 1, b2]
                        + vphi2_a2b2 * sqrt[a2] * sqrt[b2] * gate[a1, b1, a2 - 1, b2 - 1]
                        + vphi2_b2b2 * sqrt[b2] * sqrt[b2 - 1] * gate[a1, b1, a2, b2 - 2]
                    )

    return (
        Grad_phi1,
        Grad_phi2,
        np.conj(Grad_w1),
        Grad_w1c,
        np.conj(Grad_w2),
        Grad_w2c,
        Grad_th1,
        Grad_vphi1,
        np.conj(Grad_z1),
        Grad_z1c,
        np.conj(Grad_z2),
        Grad_z2c,
        Grad_th2,
        Grad_vphi2,
    )
