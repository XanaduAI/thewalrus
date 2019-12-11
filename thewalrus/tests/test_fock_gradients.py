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

from thewalrus.fock_gradients import Dgate, BSgate, Sgate, S2gate, Xgate_one_param, Zgate_one_param, Sgate_one_param, S2gate_one_param, BSgate_one_param, Rgate
import numpy as np

def test_Dgate():
    """Tests the value of the analytic gradient for the Dgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    _, Dr, Dtheta = Dgate(r, theta, cutoff, grad=True)
    dr = 0.001
    dtheta = 0.001
    Drp, _, _ = Dgate(r + dr, theta, cutoff, grad=False)
    Drm, _, _ = Dgate(r - dr, theta, cutoff, grad=False)
    Dthetap, _, _ = Dgate(r, theta + dtheta, cutoff, grad=False)
    Dthetam, _, _ = Dgate(r, theta - dtheta, cutoff, grad=False)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_Sgate():
    """Tests the value of the analytic gradient for the Sgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    _, Dr, Dtheta = Sgate(r, theta, cutoff, grad=True)
    dr = 0.001
    dtheta = 0.001
    Drp, _, _ = Sgate(r + dr, theta, cutoff, grad=False)
    Drm, _, _ = Sgate(r - dr, theta, cutoff, grad=False)
    Dthetap, _, _ = Sgate(r, theta + dtheta, cutoff, grad=False)
    Dthetam, _, _ = Sgate(r, theta - dtheta, cutoff, grad=False)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_S2gate():
    """Tests the value of the analytic gradient for the S2gate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    _, Dr, Dtheta = S2gate(r, theta, cutoff, grad=True)
    dr = 0.001
    dtheta = 0.001
    Drp, _, _ = S2gate(r + dr, theta, cutoff, grad=False)
    Drm, _, _ = S2gate(r - dr, theta, cutoff, grad=False)
    Dthetap, _, _ = S2gate(r, theta + dtheta, cutoff, grad=False)
    Dthetam, _, _ = S2gate(r, theta - dtheta, cutoff, grad=False)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_BSgate():
    """Tests the value of the analytic gradient for the BSgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi / 8
    _, Dr, Dtheta = BSgate(r, theta, cutoff, grad=True)
    dr = 0.001
    dtheta = 0.001
    Drp, _, _ = BSgate(r + dr, theta, cutoff, grad=False)
    Drm, _, _ = BSgate(r - dr, theta, cutoff, grad=False)
    Dthetap, _, _ = BSgate(r, theta + dtheta, cutoff, grad=False)
    Dthetam, _, _ = BSgate(r, theta - dtheta, cutoff, grad=False)
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-4, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-4, rtol=0)



def test_Xgate_one_param():
    """Tests the value of the analytic gradient for the Xgate_one_param against finite differences"""
    cutoff = 10
    x = 1.0
    _, dX = Xgate_one_param(x, cutoff, grad=True)
    dx = 0.001
    Xp, _ = Xgate_one_param(x + dx, cutoff)
    Xm, _ = Xgate_one_param(x - dx, cutoff)
    dXfd = (Xp - Xm) / (2 * dx)
    assert np.allclose(dX, dXfd, atol=1e-5, rtol=0)


def test_Zgate_one_param():
    """Tests the value of the analytic gradient for the Zgate_one_param against finite differences"""
    cutoff = 10
    p = 1.0
    _, dZ = Zgate_one_param(p, cutoff, grad=True)
    dp = 0.001
    Zp, _ = Zgate_one_param(p + dp, cutoff)
    Zm, _ = Zgate_one_param(p - dp, cutoff)
    dZfd = (Zp - Zm) / (2 * dp)
    assert np.allclose(dZ, dZfd, atol=1e-5, rtol=0)


def test_Sgate_one_param():
    """Tests the value of the analytic gradient for the Sgate_one_param against finite differences"""
    cutoff = 10
    s = np.arcsinh(1.0)
    _, dS = Sgate_one_param(s, cutoff, grad=True)
    ds = 0.0001
    Ss, _ = Sgate_one_param(s + ds, cutoff)
    Sm, _ = Sgate_one_param(s - ds, cutoff)
    dSfd = (Ss - Sm) / (2 * ds)
    assert np.allclose(dS, dSfd, atol=1e-5, rtol=0)


def test_Rgate():
    """Tests the value of the analytic gradient for the Rgate_one_param against finite differences"""
    theta = 1.0
    cutoff = 9
    _, dR = Rgate(theta, cutoff, grad=True)
    dtheta = 0.0001
    Rs, _ = Rgate(theta + dtheta, cutoff)
    Rm, _ = Rgate(theta - dtheta, cutoff)
    dRfd = (Rs - Rm) / (2 * dtheta)
    assert np.allclose(dR, dRfd, atol=1e-5, rtol=0)


def test_S2gate_one_param():
    """Tests the value of the analytic gradient for the S2gate_one_param against finite differences"""
    cutoff = 10
    s = np.arcsinh(1.0)
    _, dS2 = S2gate_one_param(s, cutoff, grad=True)
    ds = 0.0001
    S2s, _ = S2gate_one_param(s + ds, cutoff)
    S2m, _ = S2gate_one_param(s - ds, cutoff)
    dS2fd = (S2s - S2m) / (2 * ds)
    assert np.allclose(dS2, dS2fd, atol=1e-5, rtol=0)


def test_BSgate_one_param():
    """Tests the value of the analytic gradient for the BSgate_one_param against finite differences"""
    theta = 1.0
    cutoff = 9
    _, dBS = BSgate_one_param(theta, cutoff, grad=True)
    dtheta = 0.0001
    BSs, _ = BSgate_one_param(theta + dtheta, cutoff)
    BSm, _ = BSgate_one_param(theta - dtheta, cutoff)
    dBSfd = (BSs - BSm) / (2 * dtheta)
    assert np.allclose(dBS, dBSfd, atol=1e-5, rtol=0)
