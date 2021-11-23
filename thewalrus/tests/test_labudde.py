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

def test_labudde_2by2():
    """Test that the labudde algorithm produces the correct characteristic polynomial
    from https://en.wikipedia.org/wiki/Characteristic_polynomial."""    
    phi = .1*math.pi
    sinh_phi = math.sinh(phi)
    cosh_phi = math.cosh(phi)
    mat = np.array([[cosh_phi, sinh_phi],[sinh_phi,cosh_phi]])
    charpoly = thewalrus.labudde.charpoly_from_labudde(mat)
    assert np.allclose(charpoly[0],-2*cosh_phi)
    assert np.allclose(charpoly[1],1)
