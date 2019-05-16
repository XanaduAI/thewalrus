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
"""Tests for the Torontonian"""
import pytest
import numpy as np
from hafnian import tor

abs_tol = 1.0e-10


def test_torontonian_tmsv():
    """Calculates the torontonian of a two-mode squeezed vacuum
    state squeezed with mean photon number 1.0"""

    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    Omat = np.tanh(r)*np.array([[0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0]])

    tor_val = tor(Omat)
    assert np.abs(tor_val.real - 1.0) < abs_tol


def test_torontonian_vacuum():
    """Calculates the torontonian of a vacuum in n modes
    """
    n_modes = 5
    Omat = np.zeros([2*n_modes, 2*n_modes])
    tor_val = tor(Omat)
    assert np.abs(tor_val.real - 0.0) < abs_tol

