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

from thewalrus.fock_gradients2 import Dgate
import numpy as np
np.set_printoptions(linewidth=200)


def test_Dgate():
    """Tests the value of the analytic gradient for the Xgate against finite differences"""
    cutoff = 4
    r = 1.0
    theta = np.pi/8
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

