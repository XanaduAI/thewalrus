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
    Sgate
    Rgate
    S2gate
    BSgate

Code details
^^^^^^^^^^^^
"""
import numpy as np

from numba import jit

from thewalrus.quantum import fock_tensor
from thewalrus.symplectic import squeezing, two_mode_squeezing, beam_splitter


@jit("void(complex128[:,:], complex128[:,:], complex128[:,:], complex128, complex128)")
def grad_Dgate(T, gradTr, gradTtheta, rexptheta, exptheta):# pragma: no cover
    """Calculates the gradient of the Xgate.

    Args:
        T (array[float]): array representing the gate
        gradT (array[float]): array of zeros that will contain the value of the gradient
        pref (float): prefactor used to rescale the gradient
    """

    cutoff = gradTr.shape[0]
    for n in range(cutoff):
        for m in range(cutoff):
            sqm1 = np.sqrt(m + 1)
            sqm = np.sqrt(m)
            gradTtheta[n, m] = 1j*(np.abs(rexptheta)**2)*T[n, m]
            gradTr[n, m] =  sqm1 * T[n, m + 1] * exptheta
            gradTtheta[n, m] += 1j * sqm1 * T[n, m + 1] * rexptheta
            if m > 0:
                gradTr[n, m] -= sqm * T[n, m - 1] * np.conj(exptheta)
                gradTtheta[n, m] += 1j * sqm * T[n, m - 1] * np.conj(rexptheta)


def Dgate(r, theta, cutoff, grad=False, hbar=2, s=np.arcsinh(1.0)):
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
    alpha = np.array([r * np.exp(1j*theta)])
    S = np.identity(2 * nmodes)

    if not grad:
        return fock_tensor(S, alpha, cutoff, r=s), None, None

    T = fock_tensor(S, alpha, cutoff + 1, r=s)
    gradTr = np.zeros([cutoff, cutoff], dtype=complex)
    gradTtheta = np.zeros([cutoff, cutoff], dtype=complex)
    rexptheta = r*np.exp(1j*theta)
    exptheta = np.exp(1j*theta)
    grad_Dgate(T, gradTr, gradTtheta, rexptheta, exptheta)
    return T[0:cutoff, 0:cutoff], gradTr, gradTtheta

