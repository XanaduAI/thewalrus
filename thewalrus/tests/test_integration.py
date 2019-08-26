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
"""Tests for The Walrus quantum functions"""
# pylint: disable=no-self-use,redefined-outer-name
import pytest

import numpy as np
from thewalrus.quantum import density_matrix_element, density_matrix, pure_state_amplitude, state_vector

mu = np.array([-0.50047867,  0.37373598,  0.01421683,  0.26999427,  0.04450994,  0.01903583])

cov = np.array([[ 1.57884241,  0.81035494,  1.03468307,  1.14908791,  0.09179507, -0.11893174],
       [ 0.81035494,  1.06942863,  0.89359234,  0.20145142,  0.16202296,  0.4578259 ],
       [ 1.03468307,  0.89359234,  1.87560498,  0.16915661,  1.0836528 , -0.09405278],
       [ 1.14908791,  0.20145142,  0.16915661,  2.37765137, -0.93543385, -0.6544286 ],
       [ 0.09179507,  0.16202296,  1.0836528 , -0.93543385,  2.78903152, -0.76519088],
       [-0.11893174,  0.4578259 , -0.09405278, -0.6544286 , -0.76519088,  1.51724222]])

cutoff=15
# the Fock state measurement of mode 0 to be post-selected
m1 = 1
# the Fock state measurement of mode 1 to be post-selected
m2 = 2


@pytest.mark.parametrize("mu", [mu, np.zeros_like(mu)])
def test_preparation_without_displacement(mu):
	psi = state_vector(0*mu, cov, post_select={0: m1, 1: m2}, cutoff=cutoff, hbar=2)
	psi_c = state_vector(0*mu, cov, cutoff=cutoff, hbar=2)
	rho = density_matrix(0*mu, cov, post_select={0: m1, 1: m2}, cutoff=cutoff, hbar=2)
	rho_c = density_matrix(0*mu, cov, cutoff=cutoff, hbar=2)
	ps_psi = np.abs(psi)**2
	ps_psi_c = np.abs(psi_c[m1,m2])**2
	ps_rho = np.diag(rho)
	ps_rho_c = np.diag(rho_c[m1,m1,m2,m2,:,:])
	# Test the probabilities from density matrices are real
	assert np.allclose(ps_rho.imag, np.zeros_like(ps_rho), rtol=0)
	assert np.allclose(ps_rho_c.imag, np.zeros_like(ps_rho_c), rtol=0)
	ps_rho = ps_rho.real
	ps_rho_c = ps_rho_c.real
	# Verify that all the probabilities are equal
	assert np.allclose(ps_rho, ps_rho_c)
	assert np.allclose(ps_rho, ps_psi)
	assert np.allclose(ps_rho, ps_psi_c)