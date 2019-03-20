""" 
Sampler class tests
===================
This is a set of unit tests for the sampler class
"""

from scipy.stats import nbinom
import numpy as np
import pytest
import hafnian.samples



rel_tol = 3.0
abs_tol = 1.0e-10

    
def test_single_squeezed_state_hafnian():
    """
    Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a single mode squeezed vacuum state
    """
    n_samples = 1000
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    sigma = np.array([[np.exp(2*r), 0.        ],
                   [0.        , np.exp(-2*r)]])
    n_cut = 10
    samples = hafnian.samples.hafnian_sample(sigma, samples = n_samples, cutoff = n_cut)
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
    """
    Test the sampling routines by comparing the photon number frequencies and the exact
    probability distribution of a single mode squeezed vacuum state
    """
    n_samples = 1000
    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    sigma = np.array([[np.exp(2*r), 0.        ],
                   [0.        , np.exp(-2*r)]])
    n_cut = 10
    samples = hafnian.samples.torontonian_sample(sigma, samples = n_samples)
    samples_list = list(samples)
    rel_freq = np.array([samples_list.count(0), samples_list.count(1)])/n_samples
    x2 = np.empty([2])
    x2[0] = 1.0/np.sqrt(1.0+mean_n)
    x2[1] = 1.0 - x2[0]
    assert np.all(np.abs(x2 - rel_freq) < rel_tol/np.sqrt(n_samples))
    
    
   
def test_two_mode_squeezed_state_hafnian():
    """
    Test the sampling routines by comparing the photon number frequencies and the exact
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
    samples = hafnian.samples.hafnian_sample(sigma, samples = n_samples, cutoff = n_cut)
    assert np.all(samples[:, 0] == samples[:, 1])
    samples1d = samples[:, 0]
    bins = np.arange(0, max(samples1d), 1)
    (freq, _) = np.histogram(samples1d, bins=bins)
    rel_freq = freq/n_samples

    probs = (1.0/(1.0+mean_n))*(mean_n/(1.0+mean_n))**bins
    print(rel_freq)
    print(probs)
    assert np.all(np.abs(rel_freq - probs[0:-1]) < rel_tol/np.sqrt(n_samples))


def test_two_mode_squeezed_state_torontonian():
    """
    Test the sampling routines by comparing the photon number frequencies and the exact
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
    samples = hafnian.samples.torontonian_sample(sigma, samples = n_samples)
    assert np.all(samples[:, 0] == samples[:, 1])
    samples1d = samples[:, 0]
    bins = np.arange(0, max(samples1d), 1)
    (freq, _) = np.histogram(samples1d, bins=bins)
    rel_freq = freq/n_samples

    probs = np.empty([2])
    probs[0] = 1.0/(1.0+mean_n)
    probs[1] = 1.0-probs[0]
    print(rel_freq)
    print(probs)
    assert np.all(np.abs(rel_freq - probs[0:-1]) < rel_tol/np.sqrt(n_samples))
