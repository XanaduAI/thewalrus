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
        tuple[array[complex], array[complex] or None]: The Fock representations of the gate
    """

    rS = np.abs(z)
    phiS = np.angle(z)
    EZ = np.exp(1j*phiS)
    T = np.tanh(rS)
    S = 1/np.cosh(rS)
    ER = np.exp(-1j*phiR)
    wc = np.conj(w)

    # 2nd derivatives of G
    R = np.array([
        [-T*EZ*ER**2, ER*S],
        [ER*S, T*np.conj(EZ)]
    ])

    # 1st derivatives of G
    y = np.array([ER*(w + T*EZ*wc), -wc*S])

    sqrt = np.sqrt(np.arange(cutoff))
    Z = np.zeros((cutoff, cutoff), dtype=dtype)

    Z[0, 0] = np.exp(-0.5*np.abs(w)**2 - 0.5*wc**2 * EZ * T)*np.sqrt(S)
    Z[1, 0] = y[0] * Z[0, 0]

    # rank 1
    for m in range(2, cutoff):
        Z[m, 0] = (y[0]/sqrt[m]*Z[m-1, 0] + R[0, 0]*sqrt[m-1]/sqrt[m]*Z[m-2, 0])

    # rank 2
    for m in range(cutoff):
        for n in range(1, cutoff):
            Z[m, n] = (y[1]/sqrt[n]*Z[m, n-1] + R[1, 1]*sqrt[n-1]/sqrt[n]*Z[m, n-2] + R[0, 1]*sqrt[m]/sqrt[n]*Z[m-1, n-1])

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
    phiD = np.angle(w)
    T = np.tanh(rS)
    S = 1/np.cosh(rS)
    wc = np.conj(w)

    # Taylor series to avoid division by zero for rS -> 0
    if rS > 1e-3:
        TSplus = (T/rS + S**2)
        TSminus = (T/rS - S**2)
    else:
        TSplus = (1 - rS**2/3 + S**2)
        TSminus = (1 - rS**2/3 - S**2)

    ### Gradient with respect to phiR

    phi_a = 1j*(w*np.exp(-1j*phiR) + wc*np.exp(1j*(-phiR + phiS))*T)
    phi_a2 = -1j*np.exp(1j*(-2*phiR + phiS))*T
    phi_ab = 1j*S*np.exp(-1j*phiR)

    Grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_phi[m, n] = phi_a*sqrt[m]*gate[m-1, n] + phi_a2*sqrt[m]*sqrt[m-1]*gate[m-2, n] + phi_ab*sqrt[m]*sqrt[n]*gate[m-1, n-1]

    ### Gradients with respect to w

    w_a = np.exp(-1j*phiR)
    w_1 = -0.5*wc

    Grad_w = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_w[m, n] = w_a*sqrt[m]*gate[m-1, n] + w_1*gate[m, n]

    wc_a = np.exp(1j*(-phiR+phiS))*T
    wc_b = -S +0.0j
    wc_1 = -0.5*w * (1 + 2*T*np.exp(1j*(phiS - 2*phiD)))

    Grad_wconj = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_wconj[m, n] = wc_a*sqrt[m]*gate[m-1, n] + wc_b*sqrt[n]*gate[m, n-1] + wc_1*gate[m, n]

    ### Gradients with respect to z

    z_a = 0.5*wc * np.exp(-1j*phiR) * TSplus
    z_a2 = - 0.25*np.exp(-2j*phiR) * TSplus
    z_b = 0.5*wc * np.exp(-1j*phiS) * T*S
    z_b2 = -0.25*np.exp(-2j*phiS) * TSminus
    z_ab = -0.5*np.exp(1j*(-phiR-phiS))*T*S
    z_1 = -0.25*wc**2 * TSplus - 0.25*np.exp(-1j*phiS)*T

    Grad_z = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_z[m, n] = z_a*sqrt[m]*gate[m-1, n] + z_b*sqrt[n]*gate[m, n-1] + z_a2*sqrt[m]*sqrt[m-1]*gate[m-2, n] + z_b2*sqrt[n]*sqrt[n-1]*gate[m, n-2] + z_ab*sqrt[m]*sqrt[n]*gate[m-1, n-1] + z_1*gate[m,n]

    zc_a = -0.5*wc * np.exp(1j*(-phiR + 2*phiS)) * TSminus
    zc_a2 = 0.25*np.exp(2*1j*(-phiR + phiS)) * TSminus
    zc_b = 0.5*wc * np.exp(1j*phiS) * T*S
    zc_b2 = 0.25 * TSplus + 0.0j
    zc_ab = -0.5*np.exp(1j*(-phiR+phiS))*T*S
    zc_1 = 0.25*wc**2 *np.exp(2*1j*phiS)* TSminus - 0.25*np.exp(1j*phiS)*T

    Grad_zconj = np.zeros((cutoff, cutoff), dtype=dtype)
    for m in range(cutoff):
        for n in range(cutoff):
            Grad_zconj[m, n] = zc_a*sqrt[m]*gate[m-1, n] + zc_b*sqrt[n]*gate[m, n-1] + zc_a2*sqrt[m]*sqrt[m-1]*gate[m-2, n] + zc_b2*sqrt[n]*sqrt[n-1]*gate[m, n-2] + zc_ab*sqrt[m]*sqrt[n]*gate[m-1, n-1] + zc_1*gate[m,n]

    return np.conj(Grad_phi), np.conj(Grad_w), Grad_wconj, np.conj(Grad_z), Grad_zconj