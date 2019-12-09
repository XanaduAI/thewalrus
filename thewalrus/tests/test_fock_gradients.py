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

from thewalrus.fock_gradients import Xgate, Zgate, Sgate, S2gate, BSgate, Rgate
import numpy as np


def test_Xgate():
    """Tests the value of the analytic gradient for the Xgate against finite differences"""
    cutoff = 10
    x = 1.0
    _, dX = Xgate(x, cutoff, grad=True)
    dx = 0.001
    Xp, _ = Xgate(x + dx, cutoff)
    Xm, _ = Xgate(x - dx, cutoff)
    dXfd = (Xp - Xm) / (2 * dx)
    assert np.allclose(dX, dXfd, atol=1e-5, rtol=0)


def test_Zgate():
    """Tests the value of the analytic gradient for the Zgate against finite differences"""
    cutoff = 10
    p = 1.0
    _, dZ = Zgate(p, cutoff, grad=True)
    dp = 0.001
    Zp, _ = Zgate(p + dp, cutoff)
    Zm, _ = Zgate(p - dp, cutoff)
    dZfd = (Zp - Zm) / (2 * dp)
    assert np.allclose(dZ, dZfd, atol=1e-5, rtol=0)


def test_Sgate():
    """Tests the value of the analytic gradient for the Sgate against finite differences"""
    cutoff = 10
    s = np.arcsinh(1.0)
    _, dS = Sgate(s, cutoff, grad=True)
    ds = 0.0001
    Ss, _ = Sgate(s + ds, cutoff)
    Sm, _ = Sgate(s - ds, cutoff)
    dSfd = (Ss - Sm) / (2 * ds)
    assert np.allclose(dS, dSfd, atol=1e-5, rtol=0)


def test_Rgate():
    """Tests the value of the analytic gradient for the Rgate against finite differences"""
    theta = 1.0
    cutoff = 9
    _, dR = Rgate(theta, cutoff, grad=True)
    dtheta = 0.0001
    Rs, _ = Rgate(theta + dtheta, cutoff)
    Rm, _ = Rgate(theta - dtheta, cutoff)
    dRfd = (Rs - Rm) / (2 * dtheta)
    assert np.allclose(dR, dRfd, atol=1e-5, rtol=0)


def test_S2gate():
    """Tests the value of the analytic gradient for the S2gate against finite differences"""
    cutoff = 10
    s = np.arcsinh(1.0)
    _, dS2 = S2gate(s, cutoff, grad=True)
    ds = 0.0001
    S2s, _ = S2gate(s + ds, cutoff)
    S2m, _ = S2gate(s - ds, cutoff)
    dS2fd = (S2s - S2m) / (2 * ds)
    assert np.allclose(dS2, dS2fd, atol=1e-5, rtol=0)


def test_BSgate():
    """Tests the value of the analytic gradient for the BSgate against finite differences"""
    theta = 1.0
    cutoff = 9
    _, dBS = BSgate(theta, cutoff, grad=True)
    dtheta = 0.0001
    BSs, _ = BSgate(theta + dtheta, cutoff)
    BSm, _ = BSgate(theta - dtheta, cutoff)
    dBSfd = (BSs - BSm) / (2 * dtheta)
    assert np.allclose(dBS, dBSfd, atol=1e-5, rtol=0)
