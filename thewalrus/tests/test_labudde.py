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
"""Labudde tests"""
import math
import numpy as np
import thewalrus.labudde

import pytest


@pytest.mark.parametrize("phi", [0.1, 0.2, 0.3])
def test_labudde_2by2(phi):
    """Test that the La Budde algorithm produces the correct characteristic polynomial
    from https://en.wikipedia.org/wiki/Characteristic_polynomial."""
    sinh_phi = math.sinh(phi)
    cosh_phi = math.cosh(phi)
    mat = np.array([[cosh_phi, sinh_phi], [sinh_phi, cosh_phi]])
    charpoly = thewalrus.labudde.charpoly_from_labudde(mat)
    assert np.allclose(charpoly[0], -2 * cosh_phi)
    assert np.allclose(charpoly[1], 1)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_powtrace_2by2(n):
    """Consistency test between power_trace_eigen and power_trace_labudde"""
    phi = 0.1 * math.pi
    sinh_phi = math.sinh(phi)
    cosh_phi = math.cosh(phi)
    mat = np.array([[cosh_phi, sinh_phi], [sinh_phi, cosh_phi]])
    pow_trace_lab = thewalrus.labudde.power_trace_labudde(mat, n + 1)

    # Use wolfram alpha to verify with:
    # Trace[[[1.04975523, 0.31935254], [0.31935254, 1.04975523]]^1]
    # Trace[[[1.04975523, 0.31935254], [0.31935254, 1.04975523]]^2]
    # Trace[[[1.04975523, 0.31935254], [0.31935254, 1.04975523]]^3]

    if n == 1:
        assert np.allclose(pow_trace_lab[-1], 2.09951)
    if n == 2:
        assert np.allclose(pow_trace_lab[-1], 2.40794)
    if n == 3:
        assert np.allclose(pow_trace_lab[-1], 2.95599)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_powtrace_4by4(n):
    """Consistency test between power_trace_eigen and power_trace_labudde"""

    mat = np.array(
        [
            [1.04975523, 0.31935254, 1, 2],
            [0.31935254, 1.04975523, 3, 4],
            [0.31635254, 2.444, 5, 6],
            [21.31935254, 3.14975523, 7, 8],
        ]
    )
    pow_trace_lab = thewalrus.labudde.power_trace_labudde(mat, n + 1)
    if n == 1:
        assert np.allclose(pow_trace_lab[-1], 15.0995)
    if n == 2:
        assert np.allclose(pow_trace_lab[-1], 301.18)
    if n == 4:
        assert np.allclose(pow_trace_lab[-1], 81466.1)
