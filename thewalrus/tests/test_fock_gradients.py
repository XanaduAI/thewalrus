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
"""Tests for The Walrus fock_gradients functions"""

from thewalrus.fock_gradients import (
    displacement,
    grad_displacement,
    squeezing,
    grad_squeezing,
    two_mode_squeezing,
    grad_two_mode_squeezing,
    beamsplitter,
    grad_beamsplitter
)
import numpy as np


def test_grad_displacement():
    """Tests the value of the analytic gradient for the Dgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    T = displacement(r, theta, cutoff)
    Dr, Dtheta = grad_displacement(T, r, theta)

    dr = 0.001
    dtheta = 0.001
    Drp = displacement(r + dr, theta, cutoff)
    Drm = displacement(r - dr, theta, cutoff)
    Dthetap = displacement(r, theta + dtheta, cutoff)
    Dthetam = displacement(r, theta - dtheta, cutoff)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)

def test_grad_squeezing():
    """Tests the value of the analytic gradient for the Sgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    T = squeezing(r, theta, cutoff)
    Dr, Dtheta = grad_squeezing(T, r, theta)

    dr = 0.001
    dtheta = 0.001
    Drp = squeezing((r + dr), theta, cutoff)
    Drm = squeezing((r - dr), theta, cutoff)
    Dthetap = squeezing(r, theta + dtheta, cutoff)
    Dthetam = squeezing(r, theta - dtheta, cutoff)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_grad_two_mode_squeezing():
    """Tests the value of the analytic gradient for the S2gate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    T = two_mode_squeezing(r, theta, cutoff)
    Dr, Dtheta = grad_two_mode_squeezing(T, r, theta)
    dr = 0.001
    dtheta = 0.001
    Drp = two_mode_squeezing(r + dr, theta, cutoff)
    Drm = two_mode_squeezing(r - dr, theta, cutoff)
    Dthetap = two_mode_squeezing(r, theta + dtheta, cutoff)
    Dthetam = two_mode_squeezing(r, theta - dtheta, cutoff)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)

def test_grad_beamspitter():
    """Tests the value of the analytic gradient for the S2gate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    T = beamsplitter(r, theta, cutoff)
    Dr, Dtheta = grad_beamsplitter(T, r, theta)

    dr = 0.001
    dtheta = 0.001
    Drp = beamsplitter(r + dr, theta, cutoff)
    Drm = beamsplitter(r - dr, theta, cutoff)
    Dthetap = beamsplitter(r, theta + dtheta, cutoff)
    Dthetam = beamsplitter(r, theta - dtheta, cutoff)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-4, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-4, rtol=0)


def test_displacement_values(tol):
    """Tests the correct construction of the single mode displacement operation"""
    cutoff = 5
    alpha = 0.3 + 0.5 * 1j
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.84366482 + 0.00000000e00j, -0.25309944 + 4.21832408e-01j, -0.09544978 - 1.78968334e-01j, 0.06819609 + 3.44424719e-03j, -0.01109048 + 1.65323865e-02j,],
            [0.25309944 + 4.21832408e-01j, 0.55681878 + 0.00000000e00j, -0.29708743 + 4.95145724e-01j, -0.14658716 - 2.74850926e-01j, 0.12479885 + 6.30297236e-03j,],
            [-0.09544978 + 1.78968334e-01j, 0.29708743 + 4.95145724e-01j, 0.31873657 + 0.00000000e00j, -0.29777767 + 4.96296112e-01j, -0.18306015 - 3.43237787e-01j,],
            [-0.06819609 + 3.44424719e-03j, -0.14658716 + 2.74850926e-01j, 0.29777767 + 4.96296112e-01j, 0.12389162 + 1.10385981e-17j, -0.27646677 + 4.60777945e-01j,],
            [-0.01109048 - 1.65323865e-02j, -0.12479885 + 6.30297236e-03j, -0.18306015 + 3.43237787e-01j, 0.27646677 + 4.60777945e-01j, -0.03277289 + 1.88440656e-17j,],
        ]
    )
    T = displacement(np.abs(alpha), np.angle(alpha), cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


def test_squeezing_values(tol):
    """Tests the correct construction of the single mode squeezing operation"""
    r = 0.5
    theta = 0.7
    cutoff = 5
    # This data is obtained by using qutip
    # np.array(squeeze(40,0.5*np.exp(1j*0.7)).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [0.94171062 + 0.0j, 0.0 + 0.0j, 0.23535661 - 0.19823814j, 0.0 + 0.0j, 0.02093159 - 0.12135894j,],
            [0.0 + 0.0j, 0.83512676 + 0.0j, 0.0 + 0.0j, 0.36151137 - 0.30449682j, 0.0 + 0.0j],
            [-0.23535661 - 0.19823814j, 0.0 + 0.0j, 0.64005396 + 0.0j, 0.0 + 0.0j, 0.42261153 - 0.35596078j,],
            [0.0 + 0.0j, -0.36151137 - 0.30449682j, 0.0 + 0.0j, 0.38926873 + 0.0j, 0.0 + 0.0j],
            [0.02093159 + 0.12135894j, 0.0 + 0.0j, -0.42261153 - 0.35596078j, 0.0 + 0.0j, 0.12407853 + 0.0j,],
        ]
    )
    T = squeezing(r, theta, cutoff)
    assert np.allclose(T, expected, atol=tol, rtol=0)


def test_BS_selection_rules(tol):
    r"""Test the selection rules of a beamsplitter.
    If one writes the beamsplitter gate of :math:`U` and its matrix elements as
    :math:`\langle m, n |U|k,l \rangle` then these elements
    are nonzero if and only if :math:`m+n = k+l`. This test checks
    that this selection rule holds.
    """
    cutoff = 4
    T = beamsplitter(np.random.rand(), np.random.rand(), cutoff)
    m = np.arange(cutoff).reshape(-1, 1, 1, 1)
    n = np.arange(cutoff).reshape(1, -1, 1, 1)
    k = np.arange(cutoff).reshape(1, 1, -1, 1)
    l = np.arange(cutoff).reshape(1, 1, 1, -1)

    # create a copy of T, but replace all elements where
    # m+n != k+l with 0.
    S = np.where(m + n != k + l, 0, T)

    # check that S and T remain equal
    assert np.allclose(S, T, atol=tol, rtol=0)


def test_BS_hong_ou_mandel_interference(tol):
    r"""Tests Hong-Ou-Mandel interference for a 50:50 beamsplitter.
    If one writes :math:`U` for the Fock representation of a 50-50 beamsplitter
    then it must hold that :math:`\langle 1,1|U|1,1 \rangle = 0`.
    """
    cutoff = 2
    phi = 2 * np.pi * np.random.rand()
    T = beamsplitter(np.pi / 4, phi, cutoff)  # a 50-50 beamsplitter with phase phi
    assert np.allclose(T[1, 1, 1, 1], 0.0, atol=tol, rtol=0)


def test_S2_selection_rules(tol):
    r"""Tests the selection rules of a two mode squeezing operation.
    If one writes the squeezing gate as :math:`S_2` and its matrix elements as
    :math:`\langle p_0 p_1|S_2|q_0 q_1 \rangle` then these elements are nonzero
    if and only if :math:`p_0 - q_0 = p_1 - q_1`. This test checks that this
    selection rule holds.
    """
    cutoff = 5
    s = np.arcsinh(1.0)
    phi = np.pi / 6
    T = two_mode_squeezing(s, phi, cutoff)
    m = np.arange(cutoff).reshape(-1, 1, 1, 1)
    n = np.arange(cutoff).reshape(1, -1, 1, 1)
    k = np.arange(cutoff).reshape(1, 1, -1, 1)
    l = np.arange(cutoff).reshape(1, 1, 1, -1)

    # create a copy of T, but replace all elements where
    # m+n != k+l with 0.
    S = np.where(m - n != k - l, 0, T)

    # check that S and T remain equal
    assert np.allclose(S, T, atol=tol, rtol=0)


def test_beamsplitter_values(tol):
    r"""Test that the representation of an interferometer in the single
    excitation manifold is precisely the unitary matrix that represents it
    mode in space. This test in particular checks that the BS gate is
    consistent with strawberryfields
    """
    nmodes = 2
    vec_list = np.identity(nmodes, dtype=int).tolist()
    theta = 2 * np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    ct = np.cos(theta)
    st = np.sin(theta) * np.exp(1j * phi)
    U = np.array([[ct, -np.conj(st)], [st, ct]])
    # Calculate the matrix \langle i | U | j \rangle = T[i+j]
    T = beamsplitter(theta, phi, 3)
    U_rec = np.empty([nmodes, nmodes], dtype=complex)
    for i, vec_i in enumerate(vec_list):
        for j, vec_j in enumerate(vec_list):
            U_rec[i, j] = T[tuple(vec_i + vec_j)]
    assert np.allclose(U, U_rec, atol=tol, rtol=0)


def test_two_mode_squeezing_values(tol):
    """Tests the correct construction of the single mode squeezing operation"""
    r = 0.5
    theta = 0.7
    cutoff = 5
    T = two_mode_squeezing(r, theta, cutoff)
    expected = ((np.tanh(r) * np.exp(1j * theta)) ** np.arange(cutoff)) / np.cosh(r)
    assert np.allclose(np.diag(T[:, :, 0, 0]), expected, atol=tol, rtol=0)
