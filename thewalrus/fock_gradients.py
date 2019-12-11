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
Gradients of Gaussian gates in the Fock representation
======================================================

**Module name:** :mod:`thewalrus.fock_gradients`

.. currentmodule:: thewalrus.fock_gradients

Contains the Fock representation of the standard Gaussian gates, as well as their gradients.


Fock Gates
----------

.. autosummary::
    Xgate
    Zgate
    Dgate
    Sgate
    Rgate
    S2gate
    BSgate
    Sgate_one_param
    S2gate_one_param
    BSgate_one_param

Code details
^^^^^^^^^^^^
"""
import numpy as np

from numba import jit

from thewalrus.quantum import fock_tensor
from thewalrus.symplectic import squeezing, two_mode_squeezing, beam_splitter


@jit("void(complex128[:,:], complex128[:,:], complex128[:,:], double)")
def grad_Dgate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the Dgate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to theta
        theta (float): phase angle parametrizing the gate
    """
    cutoff = gradTr.shape[0]
    exptheta = np.exp(1j * theta)
    for n in range(cutoff):
        for m in range(cutoff):
            gradTtheta[n, m] = 1j * (n - m) * T[n, m]
            gradTr[n, m] = np.sqrt(m + 1) * T[n, m + 1] * exptheta
            if m > 0:
                gradTr[n, m] -= np.sqrt(m) * T[n, m - 1] * np.conj(exptheta)


def Dgate(r, theta, cutoff, grad=False, s=np.arcsinh(1.0)):
    """Calculates the Fock representation of the Dgate and its gradient.

    Arg:
        r (float): magnitude parameter of the gate
        theta (float): magnitude parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    nmodes = 1
    alpha = np.array([r * np.exp(1j * theta)])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=s), None, None

    T = np.complex128(fock_tensor(S, alpha, cutoff + 1, r=s))
    gradTr = np.zeros([cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff], dtype=complex)
    grad_Dgate(T, gradTr, gradTtheta, theta)
    return T[0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit("void(complex128[:,:], complex128[:,:], complex128[:,:], double)")
def grad_Sgate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the Sgate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to theta
        theta (float): phase angle parametrizing the gate
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


def Sgate(r, theta, cutoff, grad=False, s=np.arcsinh(1.0)):
    """Calculates the Fock representation of the Sgate and its gradient.

    Arg:
        r (float): magnitude parameter of the gate
        theta (float): magnitude parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    alpha = np.array([0.0])
    S = squeezing(r, theta)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=s), None, None

    T = np.complex128(fock_tensor(S, alpha, cutoff + 2, r=s))
    gradTr = np.zeros([cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff], dtype=complex)
    grad_Sgate(T, gradTr, gradTtheta, theta)
    return T[0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit("void(complex128[:,:,:,:],complex128[:,:,:,:], complex128[:,:,:,:], double)")
def grad_S2gate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the gradient of the S2gate.

    Args:
        T (array[complex]): array representing the gate
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to r
        gradTr (array[complex]): array of zeros that will contain the value of the gradient with respect to theta
        theta (float): phase angle parametrizing the gate
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


def S2gate(r, theta, cutoff, grad=False, s=np.arcsinh(1.0)):
    """Calculates the Fock representation of the S2gate and its gradient.

    Arg:
        r (float): magnitude parameter of the gate
        theta (float): magnitude parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    S = two_mode_squeezing(r, theta)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=s), None, None

    T = np.complex128(fock_tensor(S, np.zeros([2]), cutoff + 1, r=r))
    gradTr = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    grad_S2gate(T, gradTr, gradTtheta, theta)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit("void(complex128[:,:,:,:],complex128[:,:,:,:], complex128[:,:,:,:], double)")
def grad_BSgate(T, gradTr, gradTtheta, theta):  # pragma: no cover
    """Calculates the Fock representation of the BSgate and its gradient.

    Arg:
        r (float): magnitude parameter of the gate
        theta (float): magnitude parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[complex], array[complex], array[complex]]: The Fock representations of the gate and its gradients with sizes ``[cutoff]*2``
    """
    cutoff = gradTr.shape[0]
    exptheta = np.exp(1j * theta)

    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = n + k - m
                if 0 <= l < cutoff:
                    gradTtheta[n, k, m, l] = -1j * (n - m) * T[n, k, m, l]
                    if m > 0:
                        gradTr[n, k, m, l] = np.sqrt(m * (l + 1)) * T[n, k, m - 1, l + 1] * exptheta
                    if l > 0:
                        gradTr[n, k, m, l] -= (
                            np.sqrt((m + 1) * l) * T[n, k, m + 1, l - 1] * np.conj(exptheta)
                        )


def BSgate(r, theta, cutoff, grad=False, s=np.arcsinh(1.0)):
    """Calculates the Fock representation of the S2gate and its gradient.

    Arg:
        s (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    S = beam_splitter(r, theta)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=s), None, None

    T = np.complex128(fock_tensor(S, np.zeros([2]), cutoff + 1, r=r))
    gradTr = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=complex)
    grad_BSgate(T, gradTr, gradTtheta, theta)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradTr, gradTtheta


@jit("void(double[:,:], double[:,:], double)")
def grad_Xgate_one_param(T, gradT, pref):  # pragma: no cover
    """Calculates the gradient of the Xgate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
        pref (float): prefactor used to rescale the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] = np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] -= np.sqrt(m) * T[n, m - 1] * pref


def Xgate_one_param(x, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    """Calculates the Fock representation of the Xgate and its gradient.

    Arg:
        x (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        hbar (float): value of hbar in the commutation relation
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([pref * x])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=r), None

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff], dtype=float)
    grad_Xgate_one_param(T, gradT, pref)
    return T[0:cutoff, 0:cutoff], gradT


@jit("void(complex128[:,:], complex128[:,:], double)")
def grad_Zgate_one_param(T, gradT, pref):  # pragma: no cover
    """Calculates the gradient of the Zgate.

    Args:
        T (array[complex]): array representing the gate
        gradT (array[complex]): array of zeros that will contain the value of the gradient
        pref (float): prefactor used to rescale the gradient
    """
    cutoff = gradT.shape[0]
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] = 1j * np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] += 1j * np.sqrt(m) * T[n, m - 1] * pref


def Zgate_one_param(p, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    """Calculates the Fock representation of the Zgate and its gradient.

    Arg:
        p (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        hbar (float): value of hbar in the commutation relation
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([1j * pref * p])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=r), None

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff], dtype=complex)
    grad_Zgate_one_param(T, gradT, pref)
    return T[0:cutoff, 0:cutoff], gradT


@jit("void(double[:,:], double[:,:])")
def grad_Sgate_one_param(T, gradT):  # pragma: no cover
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


def Sgate_one_param(s, cutoff, grad=False, r=np.arcsinh(1.0)):
    """Calculates the Fock representation of the Sgate and its gradient.

    Arg:
        s (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        hbar (float): value of hbar in the commutation relation
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*2``
    """
    S = squeezing(s, 0.0)
    if not grad:
        return fock_tensor(S, np.zeros([1]), cutoff, r=r), None

    T = fock_tensor(S, np.zeros([1]), cutoff + 2, r=r)
    gradT = np.zeros([cutoff, cutoff], dtype=float)
    grad_Sgate_one_param(T, gradT)

    return T[0:cutoff, 0:cutoff], gradT


def Rgate(theta, cutoff, grad=False):
    """Calculates the Fock representation of the Sgate and its gradient.

    Arg:
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


@jit("void(double[:,:,:,:],double[:,:,:,:])")
def grad_S2gate_one_param(T, gradT):  # pragma: no cover
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


def S2gate_one_param(s, cutoff, grad=False, r=np.arcsinh(1.0)):
    """Calculates the Fock representation of the S2gate and its gradient.

    Arg:
        s (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    S = two_mode_squeezing(s, 0)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=r), None

    T = fock_tensor(S, np.zeros([2]), cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=float)
    grad_S2gate_one_param(T, gradT)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT


@jit("void(double[:,:,:,:], double[:,:,:,:])")
def grad_BSgate_one_param(T, gradT):  # pragma: no cover
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


def BSgate_one_param(theta, cutoff, grad=False, r=np.arcsinh(1.0)):
    """Calculates the Fock representation of the BSgate and its gradient.

    Arg:
        theta (float): parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): whether to calculate the gradient or not
        r (float): value of the parameter used internally in fock_tensor

    Returns:
        tuple[array[float], array[float] or None]: The Fock representations of the gate and its gradient with size ``[cutoff]*4``
    """
    S = beam_splitter(theta, 0)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=r), None

    T = fock_tensor(S, np.zeros([2]), cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff], dtype=float)
    grad_BSgate_one_param(T, gradT)

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT
