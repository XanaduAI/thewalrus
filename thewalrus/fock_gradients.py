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

    Xgate
    Dgate
    Sgate
    Rgate
    Kgate
    S2gate
    BSgate
    Sgate_real
    S2gate_real
    BSgate_real

"""
import numpy as np

from numba import jit


from thewalrus.libwalrus import (
    interferometer,
    squeezing,
    displacement,
    interferometer_real,
    displacement_real,
    squeezing_real,
    two_mode_squeezing,
    two_mode_squeezing_real,
)




@jit(nopython=True)
def displacement_rec(alpha, D):
    y = np.array([alpha, -np.conj(alpha)])
    cutoff, _ = D.shape
    sqns = np.sqrt(np.arange(cutoff))
    D[0,0] = np.exp(-0.5*np.abs(y[0])**2)
    D[1,0] = y[0]*D[0,0]
    for m in range(2,cutoff):
        D[m,0] = (y[0]*D[m-1,0])/sqns[m]
    for n in range(1, cutoff):
        shifted = np.roll(D[:,n-1],1)*sqns
        D[:,n] = (y[1]*D[:,n-1]+shifted)/sqns[n]
    return D



@jit(nopython=True)
def grad_Dgate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the Dgate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r, the displacement magnitude
        gradTtheta (array[complex]): array of zeros that will contain the value of the gradient with respect to theta, the displacement phase
        theta (float): displacement phase
    """
    cutoff = gradTr.shape[0]
    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        for m in range(cutoff):
            gradTtheta[n, m] = 1j * (n - m) * T[n, m]
            gradTr[n, m] = np.sqrt(m + 1) * T[n, m + 1] * exptheta
            if m > 0:
                gradTr[n, m] -= np.sqrt(m) * T[n, m - 1] * np.conj(exptheta)


def Dgate(r, theta, cutoff, grad=False):
    """Calculates the Fock representation of the Dgate and its gradient.

    Args:
        r (float): displacement magnitude
        theta (float): displacement phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    if not grad:
        T = np.empty([cutoff, cutoff], dtype=complex)
        displacement_rec(r * np.exp(1j * theta), T)
        return T, None, None
    T = np.empty([cutoff+1, cutoff+1], dtype=complex)
    displacement_rec(r * np.exp(1j * theta), T)
    gradTr = np.zeros([cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff], dtype=complex)
    grad_Dgate(T, gradTr, gradTtheta, theta)
    return T[0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_Sgate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the Sgate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r, the squeezing amplitude
        gradTtheta (array[complex]): array of zeros that will contain the value of the gradient with respect to theta, the squeezing phase
        theta (float): squeezing phase
    """
    cutoff = gradTr.shape[0]
    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        offset = n % 2
        for m in range(offset, cutoff, 2):
            gradTtheta[n, m] = 0.5j * (n - m) * T[n, m]
            gradTr[n, m] = -0.5 * np.sqrt((m + 1) * (m + 2)) * T[n, m + 2] * exptheta
            if m > 1:
                gradTr[n, m] += 0.5 * np.sqrt(m * (m - 1)) * T[n, m - 2] * np.conj(exptheta)


def Sgate(r, theta, cutoff, grad=False):
    """Calculates the Fock representation of the Sgate and its gradient.

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    mat = np.array(
        [
            [np.exp(1j * theta) * np.tanh(r), -1.0 / np.cosh(r)],
            [-1.0 / np.cosh(r), -np.exp(-1j * theta) * np.tanh(r)],
        ]
    )
    if not grad:
        return squeezing(mat, cutoff), None, None

    T = squeezing(mat, cutoff + 2)
    gradTr = np.zeros([cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff], dtype=complex)

    grad_Sgate(T, gradTr, gradTtheta, theta)
    return T[0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_S2gate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the S2gate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r, the squeezing amplitude
        gradTtheta (array[complex]): array of zeros that will contain the value of the gradient with respect to theta, the squeezing phase
        theta (float): two-mode squeezing phase
    """
    cutoff = gradTr.shape[0]
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


def S2gate(r, theta, cutoff, grad=False):
    """Calculates the Fock representation of the S2gate and its gradient.

    Args:
        r (float): two-mode squeezing magnitude
        theta (float): two-mode squeezing phase
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """

    sc = 1.0 / np.cosh(r)
    eiptr = np.exp(-1j * theta) * np.tanh(r)
    mat = np.array(
        [
            [0, -np.conj(eiptr), -sc, 0],
            [-np.conj(eiptr), 0, 0, -sc],
            [-sc, 0, 0, eiptr],
            [0, -sc, eiptr, 0],
        ]
    )
    if not grad:
        return two_mode_squeezing(mat, cutoff), None, None

    T = two_mode_squeezing(mat, cutoff + 1)
    gradTr = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    grad_S2gate(T, gradTr, gradTtheta, theta)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit(nopython=True)
def grad_BSgate(T, gradTtheta, gradTphi, phi):  # pragma: no cover
    """Calculates the gradient of the BSgate.

    Args:
        T (array[complex]): array representing the gate
        gradTtheta (array[complex]): array of zeros that will contain the value of the gradient with respect to theta, the beamsplitter transmissivity angle
        gradTphi (array[complex]): array of zeros that will contain the value of the gradient with respect to phi, the beamsplitter reflectivity phase
        theta (float): phase angle parametrizing the gate
    """
    cutoff = gradTtheta.shape[0]
    expphi = np.exp(1j * phi)

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


def BSgate(theta, phi, cutoff, grad=False):
    r"""Calculates the Fock representation of the S2gate and its gradient.

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    mat = -np.array(
        [[0, 0, ct, -np.conj(st)], [0, 0, st, ct], [ct, st, 0, 0], [-np.conj(st), ct, 0, 0]]
    )

    if not grad:
        return interferometer(mat, cutoff), None, None

    T = interferometer(mat, cutoff + 1)
    gradTtheta = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    gradTphi = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    grad_BSgate(T, gradTtheta, gradTphi, phi)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradTtheta, gradTphi


@jit(nopython=True)
def grad_Xgate(T, gradT):  # pragma: no cover
    """Calculates the gradient of the Xgate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] = np.sqrt(m + 1) * T[n, m + 1]
            if m > 0:
                gradT[n, m] -= np.sqrt(m) * T[n, m - 1]


def Xgate(x, cutoff, grad=False):
    """Calculates the Fock representation of the Xgate and its gradient.

    Args:
        x (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        hbar (float): value of hbar in the commutation relation

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    if not grad:
        T = np.empty([cutoff, cutoff])
        displacement_rec(x, T)
        return T, None

    T = np.empty([cutoff+1, cutoff+1])
    displacement_rec(x, T)
    gradT = np.zeros([cutoff, cutoff], dtype=float)
    grad_Xgate(T, gradT)
    return T[0:cutoff, 0:cutoff], gradT


@jit(nopython=True)
def grad_Sgate_real(T, gradT):  # pragma: no cover
    """Calculates the gradient of the Sgate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        offset = n % 2
        for m in range(offset, cutoff, 2):
            gradT[n, m] = -0.5 * np.sqrt((m + 1) * (m + 2)) * T[n, m + 2]
            if m > 1:
                gradT[n, m] += 0.5 * np.sqrt(m * (m - 1)) * T[n, m - 2]


def Sgate_real(s, cutoff, grad=False):
    """Calculates the Fock representation of the Sgate and its gradient.

    Args:
        s (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    mat = np.array([[np.tanh(s), -1.0 / np.cosh(s)], [-1.0 / np.cosh(s), -np.tanh(s)]])
    if not grad:
        return squeezing_real(mat, cutoff), None

    T = squeezing_real(mat, cutoff + 2)
    gradT = np.zeros([cutoff, cutoff], dtype=float)
    grad_Sgate_real(T, gradT)

    return T[0:cutoff, 0:cutoff], gradT


def Rgate(theta, cutoff, grad=False):
    """Calculates the Fock representation of the Rgate and its gradient.

    Args:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    ns = np.arange(cutoff)
    T = np.exp(1j * theta) ** ns
    if not grad:
        return np.diag(T), None
    return np.diag(T), np.diag(1j * ns * T)


def Kgate(theta, cutoff, grad=False):
    """Calculates the Fock representation of the Kgate and its gradient.

    Args:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    ns = np.arange(cutoff)
    T = np.exp(1j * theta) ** (ns ** 2)
    if not grad:
        return np.diag(T), None
    return np.diag(T), np.diag(1j * (ns ** 2) * T)


@jit(nopython=True)
def grad_S2gate_real(T, gradT):  # pragma: no cover
    """Calculates the gradient of the S2gate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = m - n + k
                if 0 <= l < cutoff:
                    gradT[n, k, m, l] = np.sqrt((m + 1) * (l + 1)) * T[n, k, m + 1, l + 1]
                    if m > 0 and l > 0:
                        gradT[n, k, m, l] -= np.sqrt(m * l) * T[n, k, m - 1, l - 1]


def S2gate_real(s, cutoff, grad=False):
    """Calculates the Fock representation of the S2gate and its gradient.

    Args:
        s (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    sc = 1.0 / np.cosh(s)
    tr = np.tanh(s)
    mat = np.array([[0, -tr, -sc, 0], [-tr, 0, 0, -sc], [-sc, 0, 0, tr], [0, -sc, tr, 0]])
    if not grad:
        return two_mode_squeezing_real(mat, cutoff), None

    T = two_mode_squeezing_real(mat, cutoff + 1)
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=float)
    grad_S2gate_real(T, gradT)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT


@jit(nopython=True)
def grad_BSgate_real(T, gradT):  # pragma: no cover
    """Calculates the gradient of the BSgate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = n + k - m
                if 0 <= l < cutoff:
                    if m > 0:
                        gradT[n, k, m, l] = np.sqrt(m * (l + 1)) * T[n, k, m - 1, l + 1]
                    if l > 0:
                        gradT[n, k, m, l] -= np.sqrt((m + 1) * l) * T[n, k, m + 1, l - 1]


def BSgate_real(theta, cutoff, grad=False):
    """Calculates the Fock representation of the BSgate and its gradient.

    Args:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    mat = np.array([[0, 0, ct, -st], [0, 0, st, ct], [ct, st, 0, 0], [-st, ct, 0, 0]])

    if not grad:
        return interferometer_real(mat, cutoff), None

    T = interferometer_real(mat, cutoff + 1)
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=float)
    grad_BSgate_real(T, gradT)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT
