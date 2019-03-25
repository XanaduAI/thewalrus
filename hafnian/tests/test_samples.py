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
from scipy.stats import nbinom

from hafnian.samples import generate_hafnian_sample, hafnian_sample, torontonian_sample


rel_tol = 3.0
abs_tol = 1.0e-10


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


def test_single_squeezed_state_hafnian():
    """Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a single mode squeezed vacuum state
    """
    n_samples = 1000
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    sigma = np.array([[np.exp(2*r), 0.],
                      [0., np.exp(-2*r)]])

    n_cut = 10
    samples = hafnian_sample(sigma, samples=n_samples, cutoff=n_cut)
    bins = np.arange(0, max(samples), 1)
    (freq, _) = np.histogram(samples, bins=bins)
    rel_freq = freq/n_samples
    nm = max(samples)//2

    x = nbinom.pmf(np.arange(0, nm, 1), 0.5, np.tanh(np.arcsinh(np.sqrt(mean_n)))**2)
    x2 = np.zeros(2*len(x))
    x2[::2] = x
    rel_freq = freq[0:-1]/n_samples
    x2 = x2[0:len(rel_freq)]

    assert np.all(np.abs(x2 - rel_freq) < rel_tol/np.sqrt(n_samples))


def test_single_squeezed_state_torontonian():
    """Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a single mode squeezed vacuum state
    """
    n_samples = 1000
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    sigma = np.array([[np.exp(2*r), 0.],
                      [0., np.exp(-2*r)]])
    samples = torontonian_sample(sigma, samples=n_samples)
    samples_list = list(samples)

    rel_freq = np.array([samples_list.count(0), samples_list.count(1)])/n_samples
    x2 = np.empty([2])

    x2[0] = 1.0/np.sqrt(1.0+mean_n)
    x2[1] = 1.0 - x2[0]
    assert np.all(np.abs(x2 - rel_freq) < rel_tol/np.sqrt(n_samples))


def test_two_mode_squeezed_state_hafnian():
    """Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a two mode squeezed vacuum state
    """
    n_samples = 1000
    n_cut = 5
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    c = np.cosh(2*r)
    s = np.sinh(2*r)
    sigma = np.array([[c, s, 0, 0],
                      [s, c, 0, 0],
                      [0, 0, c, -s],
                      [0, 0, -s, c]])

    samples = hafnian_sample(sigma, samples=n_samples, cutoff=n_cut)
    assert np.all(samples[:, 0] == samples[:, 1])

    samples1d = samples[:, 0]
    bins = np.arange(0, max(samples1d), 1)
    (freq, _) = np.histogram(samples1d, bins=bins)
    rel_freq = freq/n_samples

    probs = (1.0/(1.0+mean_n))*(mean_n/(1.0+mean_n))**bins
    assert np.all(np.abs(rel_freq - probs[0:-1]) < rel_tol/np.sqrt(n_samples))


def test_two_mode_squeezed_state_torontonian():
    """Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a two mode squeezed vacuum state
    """
    n_samples = 1000
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    c = np.cosh(2*r)
    s = np.sinh(2*r)
    sigma = np.array([[c, s, 0, 0],
                      [s, c, 0, 0],
                      [0, 0, c, -s],
                      [0, 0, -s, c]])

    samples = torontonian_sample(sigma, samples=n_samples)
    assert np.all(samples[:, 0] == samples[:, 1])

    samples1d = samples[:, 0]
    bins = np.arange(0, max(samples1d), 1)
    (freq, _) = np.histogram(samples1d, bins=bins)
    rel_freq = freq/n_samples

    probs = np.empty([2])
    probs[0] = 1.0/(1.0+mean_n)
    probs[1] = 1.0-probs[0]
    assert np.all(np.abs(rel_freq - probs[0:-1]) < rel_tol/np.sqrt(n_samples))
