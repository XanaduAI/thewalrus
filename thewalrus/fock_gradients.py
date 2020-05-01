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
    r"""Calculate the matrix elements of the real or complex displacement gate using a recursion relation.

    Args:
        alpha (float or complex): value of the displacement.
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        (array): matrix representing the displacement operation.
    """
    D = np.zeros((cutoff, cutoff), dtype=dtype)
    y = np.array([alpha, -np.conj(alpha)])
    sqns = np.sqrt(np.arange(cutoff))

    D[0, 0] = np.exp(-0.5 * np.abs(y[0]) ** 2)
    D[1, 0] = y[0] * D[0, 0]

    for m in range(2, cutoff):
        D[m, 0] = y[0] / sqns[m] * D[m - 1, 0]

    for m in range(0, cutoff):
        for n in range(1, cutoff):
            D[m, n] = y[1] / sqns[n] * D[m, n - 1] + sqns[m] / sqns[n] * D[m - 1, n - 1]

    return D


@jit(nopython=True)
def squeezing(r, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculate the matrix elements of the real or complex squeezing gate using a recursion relation.

    Args:
        r (float): amplitude of the squeezing.
        theta (float): phase of the squeezing.
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        (array): matrix representing the squeezing operation.
    """
    S = np.zeros((cutoff, cutoff), dtype=dtype)
    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    sinhr = 1.0 / np.cosh(r)
    R = np.array([[-eitheta_tanhr, sinhr], [sinhr, np.conj(eitheta_tanhr)],])
    sqns = np.sqrt(np.arange(cutoff))
    S[0, 0] = np.sqrt(sinhr)

    for m in range(2, cutoff, 2):
        S[m, 0] = sqns[m - 1] / sqns[m] * R[0, 0] * S[m - 2, 0]

    for m in range(0, cutoff):
        for n in range(1, cutoff):
            if (m + n) % 2 == 0:
                S[m, n] = (
                    sqns[n - 1] / sqns[n] * R[1, 1] * S[m, n - 2]
                    + sqns[m] / sqns[n] * R[0, 1] * S[m - 1, n - 1]
                )

    return S


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
                    gradTr[n, k, m, l] = (
                        np.sqrt((m + 1) * (l + 1)) * T[n, k, m + 1, l + 1] * exptheta
                    )
                    if m > 0 and l > 0:
                        gradTr[n, k, m, l] -= (
                            np.sqrt(m * l) * T[n, k, m - 1, l - 1] * np.conj(exptheta)
                        )
    return gradTr, gradTtheta


@jit(nopython=True)
def two_mode_squeezing(r, theta, cutoff, dtype=np.complex128):  # pragma: no cover
    """Calculates the matrix elements of the S2gate recursively.

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing phase
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[float]: The Fock representation of the gate

    """
    sc = 1.0 / np.cosh(r)
    eiptr = np.exp(-1j * theta) * np.tanh(r)
    R = -np.array(
        [
            [0, -np.conj(eiptr), -sc, 0],
            [-np.conj(eiptr), 0, 0, -sc],
            [-sc, 0, 0, eiptr],
            [0, -sc, eiptr, 0],
        ]
    )

    sqrt = np.sqrt(np.arange(cutoff))

    Z = np.zeros((cutoff + 1, cutoff + 1, cutoff + 1, cutoff + 1), dtype=dtype)
    Z[0, 0, 0, 0] = sc

    # rank 2
    for n in range(1, cutoff):
        Z[n, n, 0, 0] = R[0, 1] * Z[n - 1, n - 1, 0, 0]

    # rank 3
    for m in range(0, cutoff):
        for n in range(0, m):
            p = m - n
            if 0 < p < cutoff:
                Z[m, n, p, 0] = R[0, 2] * sqrt[m] / sqrt[p] * Z[m - 1, n, p - 1, 0]

    # rank 4
    for m in range(0, cutoff):
        for n in range(0, cutoff):
            for p in range(0, cutoff):
                q = p - (m - n)
                if 0 < q < cutoff:
                    Z[m, n, p, q] = (
                        R[1, 3] * sqrt[n] / sqrt[q] * Z[m, n - 1, p, q - 1]
                        + R[2, 3] * sqrt[p] / sqrt[q] * Z[m, n, p - 1, q - 1]
                    )

    return Z[:cutoff, :cutoff, :cutoff, :cutoff]

#pylint: disable=too-many-arguments
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
                        gradTtheta[n, k, m, l] = (
                            np.sqrt(m * (l + 1)) * T[n, k, m - 1, l + 1] * expphi
                        )
                    if l > 0:
                        gradTtheta[n, k, m, l] -= (
                            np.sqrt((m + 1) * l) * T[n, k, m + 1, l - 1] * np.conj(expphi)
                        )
    return gradTtheta, gradTphi


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
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    R = np.array(
        [[0, 0, ct, -np.conj(st)], [0, 0, st, ct], [ct, st, 0, 0], [-np.conj(st), ct, 0, 0]]
    )

    sqrt = np.sqrt(np.arange(cutoff + 1))

    Z = np.zeros((cutoff + 1, cutoff + 1, cutoff + 1, cutoff + 1), dtype=dtype)
    Z[0, 0, 0, 0] = 1.0

    # rank 3
    for m in range(0, cutoff):
        for n in range(0, cutoff - m):
            p = m + n
            if 0 < p < cutoff:
                Z[m, n, p, 0] = (
                    R[0, 2] * sqrt[m] / sqrt[p] * Z[m - 1, n, p - 1, 0]
                    + R[1, 2] * sqrt[n] / sqrt[p] * Z[m, n - 1, p - 1, 0]
                )

    # rank 4
    for m in range(0, cutoff):
        for n in range(0, cutoff):
            for p in range(0, cutoff):
                q = m + n - p
                if 0 < q < cutoff:
                    Z[m, n, p, q] = (
                        R[0, 3] * sqrt[m] / sqrt[q] * Z[m - 1, n, p, q - 1]
                        + R[1, 3] * sqrt[n] / sqrt[q] * Z[m, n - 1, p, q - 1]
                    )

    return Z[:cutoff, :cutoff, :cutoff, :cutoff]

#pylint: disable=too-many-arguments
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
            return beamsplitter(theta, phi, cutoff, dtype=dtype).transpose(index_order), None, None
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


def cubic_phase(gamma, hbar, cutoff, dtype=np.complex128):
    mu = 1 / np.cbrt(gamma * np.sqrt(hbar))
    eAi, eAip, _, _ = airye(mu ** 4)
    V = np.zeros([cutoff, cutoff], dtype=dtype)
    sqrt = np.sqrt(np.arange(cutoff + 5))
    pref = np.sqrt(np.pi)
    V[0, 0] = 2 * pref * mu * eAi
    V[0, 1] = -2 * pref * 1j * sqrt[2] * (mu ** 2) * (mu ** 2 * eAi + eAip)
    V[1, 0] = V[0, 1]
    V[1, 1] = -8 * mu ** 5 * pref * (mu ** 2 * eAi + eAip)

    for m in range(cutoff):
        for n in range(cutoff - 2):
            V[m, n + 2] = (
                -(
                    sqrt[2]
                    * (sqrt[n - 1] * sqrt[n] * V[m, -2 + n] + (1 + 2 * n) * V[m, n])
                )
                + 4j * mu ** 3 * (sqrt[m] * V[-1 + m, n] - sqrt[1 + n] * V[m, 1 + n])
            ) / (sqrt[2] * sqrt[n + 1] * sqrt[n + 2])
        V[:, m] = V[m, :]

    return V
