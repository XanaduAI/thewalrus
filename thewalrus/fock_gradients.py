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
import numpy as np


def Xgate(x, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([pref * x])
    S = np.identity(2 * nmodes)

    if not grad:
        T = fock_tensor(S, alpha, cutoff, r=r)
        return T

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros_like(T)
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] += np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] -= np.sqrt(m) * T[n, m - 1] * pref
    return T[0:cutoff, 0:cutoff], gradT[0:cutoff, 0:cutoff]


def Zgate(p, cutoff, grad=False, hbar=2, r=np.arcsinh(1.0)):
    nmodes = 1
    pref = 1.0 / np.sqrt(2 * hbar)
    alpha = np.array([1j*pref * p])
    S = np.identity(2 * nmodes)

    if not grad:
        T = fock_tensor(S, alpha, cutoff, r=r)
        return T

    T = fock_tensor(S, alpha, cutoff + 1, r=r)
    gradT = np.zeros_like(T)
    for n in range(cutoff):
        for m in range(cutoff):
            gradT[n, m] += 1j*np.sqrt(m + 1) * T[n, m + 1] * pref
            if m > 0:
                gradT[n, m] += 1j*np.sqrt(m) * T[n, m - 1] * pref
    return T[0:cutoff, 0:cutoff], gradT[0:cutoff, 0:cutoff]