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
"""Tests for the hafnian sampling functions"""
import pytest

import numpy as np
from hafnian.samples import generate_hafnian_sample, hafnian_sample, torontonian_sample


def TMS_cov(r, phi):
    """returns the covariance matrix of a TMS state"""
    cp = np.cos(phi)
    sp = np.sin(phi)
    ch = np.cosh(r)
    sh = np.sinh(r)

    S = np.array([[ch, cp*sh, 0, sp*sh],
                  [cp*sh, ch, sp*sh, 0],
                  [0, sp*sh, ch, -cp*sh],
                  [sp*sh, 0, -cp*sh, ch]])

    return S @ S.T


def test_TMS_hafnian_samples():
    """test sampling from TMS hafnians is correlated"""
    m = 0.432
    phi = 0.546
    V = TMS_cov(np.arcsinh(m), phi)
    res = hafnian_sample(V, samples=20)
    assert np.allclose(res[:, 0], res[:, 1])


def test_TMS_hafnian_samples_cutoff():
    """test sampling from TMS hafnians is correlated"""
    m = 0.432
    phi = 0.546
    V = TMS_cov(np.arcsinh(m), phi)
    res = hafnian_sample(V, samples=20, cutoff=5)
    assert np.allclose(res[:, 0], res[:, 1])


def test_hafnian_samples_nonnumpy():
    """test exception is raised if not a numpy array"""
    with pytest.raises(TypeError):
        hafnian_sample(5, samples=20)


def test_hafnian_samples_nonsquare():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Covariance matrix must be square."):
        hafnian_sample(np.array([[0, 5, 3], [0, 1, 2]]), samples=20)


def test_hafnian_samples_nans():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Covariance matrix must not contain NaNs."):
        hafnian_sample(np.array([[0, 5], [0, np.NaN]]), samples=20)


def test_hafnian_samples_wronglength():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Means vector must be the same length as the covariance matrix."):
        hafnian_sample(np.array([[0, 5], [0, 2]]), mu=np.array([5]), samples=20)


def test_torontonian_samples_nonnumpy():
    """test exception is raised if not a numpy array"""
    with pytest.raises(TypeError):
        torontonian_sample(5, samples=20)


def test_torontonian_samples_nonsquare():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Covariance matrix must be square."):
        torontonian_sample(np.array([[0, 5, 3], [0, 1, 2]]), samples=20)


def test_torontonian_samples_nans():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Covariance matrix must not contain NaNs."):
        torontonian_sample(np.array([[0, 5], [0, np.NaN]]), samples=20)


def test_torontonian_samples_wronglength():
    """test exception is raised if not a numpy array"""
    with pytest.raises(ValueError, match="Means vector must be the same length as the covariance matrix."):
        torontonian_sample(np.array([[0, 5], [0, 2]]), mu=np.array([5]), samples=20)
