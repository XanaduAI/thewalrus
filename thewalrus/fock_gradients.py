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
=====================

**Module name:** :mod:`thewalrus.fock_gradients`

.. currentmodule:: thewalrus.fock_gradients

Contains some the Fock representation of the standard Gaussian gates

Auxiliary functions
-------------------

.. autosummary::
    expand
    expand_vector
    reduced_state
    is_symplectic
    sympmat

Gaussian states
---------------

.. autosummary::
    vacuum_state


Gates and operations
--------------------

.. autosummary::
    two_mode_squeezing
    interferometer
    loss
    mean_photon_number
    beam_splitter
    rotation

Code details
^^^^^^^^^^^^
"""
from thewalrus.quantum import fock_tensor
from thewalrus.symplectic import squeezing, two_mode_squeezing, beam_splitter
import numpy as np

# import numba

# @numba.jit
def Xgate(x, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of the Xgate and its gradient
    Arg:
        x (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
        hbar (float): Value of hbar is the commutation relation
        r (float): Value of the parameter used internally in fock_tensor. 
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([pref * x])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=r), None

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff], dtype=complex)
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] = np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] -= np.sqrt(m) * T[n, m - 1] * pref
    return T[0:cutoff, 0:cutoff], gradT


# @numba.jit
def Zgate(p, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of the Zgate and its gradient
    Arg:
        p (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
        hbar (float): Value of hbar is the commutation relation
        r (float): Value of the parameter used internally in fock_tensor.
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([1j * pref * p])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=r), None

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros([cutoff, cutoff], dtype=complex)
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] = 1j * np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] += 1j * np.sqrt(m) * T[n, m - 1] * pref
    return T[0:cutoff, 0:cutoff], gradT


# @numba.jit
def Sgate(s, cutoff, grad=False, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of the Sgate and its gradient
    Arg:
        s (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
        r (float): Value of the parameter used internally in fock_tensor.
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    S = squeezing(s, 0.0)
    if not grad:
        return fock_tensor(S, np.zeros([1]), cutoff, r=r).real, None

    T = fock_tensor(S, np.zeros([1]), cutoff + 2, r=r).real
    gradT = np.zeros([cutoff, cutoff])
    for n in range(cutoff):
        offset = n % 2
        for m in range(offset, cutoff, 2):
            gradT[n, m] = -0.5 * np.sqrt((m + 1) * (m + 2)) * T[n, m + 2]
            if m > 1:
                gradT[n, m] += 0.5 * np.sqrt(m * (m - 1)) * T[n, m - 2]
    return T[0:cutoff, 0:cutoff], gradT


def Rgate(theta, cutoff, grad=False):
    r"""
    Calculates the Fock representation of the Rgate and its gradient
    Arg:
        s (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    ns = np.arange(cutoff)
    if not grad:
        return np.diag(np.exp(1j * ns * theta)), None
    T = np.exp(1j * ns * theta)
    return np.diag(T), np.diag(1j * ns * T)


def S2gate(s, cutoff, grad=False, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of the S2gate and its gradient
    Arg:
        s (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
        r (float): Value of the parameter used internally in fock_tensor.
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    S = two_mode_squeezing(s, 0)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=r).real, None

    T = fock_tensor(S, np.zeros([2]), cutoff + 1, r=r).real
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff])
    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = m - n + k
                if l >= 0 and l < cutoff:
                    gradT[n, k, m, l] = np.sqrt((m + 1) * (l + 1)) * T[n, k, m + 1, l + 1]
                    if m > 0 and l > 0:
                        gradT[n, k, m, l] -= np.sqrt(m * l) * T[n, k, m - 1, l - 1]
    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT


def BSgate(theta, cutoff, grad=False, r=np.arcsinh(1.0)):
    r"""
    Calculates the Fock representation of the BSgate and its gradient
    Arg:
        s (float): Parameter of the gate
        cutoff (int): Fock ladder cutoff
        grad (boolean): Whether to calculate the gradient or not
        r (float): Value of the parameter used internally in fock_tensor.
    Returns:
        tuple(G,dG): The Fock representations of the gate and its gradient
    """
    S = beam_splitter(theta, 0)
    if not grad:
        return fock_tensor(S, np.zeros([2]), cutoff, r=r).real, None

    T = fock_tensor(S, np.zeros([2]), cutoff + 1, r=r).real
    gradT = np.zeros([cutoff, cutoff, cutoff, cutoff])
    for n in range(cutoff):
        for k in range(cutoff):
            for m in range(cutoff):
                l = n + k - m
                if l >= 0 and l < cutoff:
                    if m > 0:
                        gradT[n, k, m, l] = np.sqrt(m * (l + 1)) * T[n, k, m - 1, l + 1]
                    if l > 0:
                        gradT[n, k, m, l] -= np.sqrt((m + 1) * l) * T[n, k, m + 1, l - 1]

    return T[0:cutoff, 0:cutoff, 0:cutoff, 0:cutoff], gradT
