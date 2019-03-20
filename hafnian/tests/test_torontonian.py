""" 
Sampler class tests
===================
This is a set of unit tests for the sampler class
"""


import numpy as np
import pytest
from hafnian import tor

abs_tol = 1.0e-10
    
def test_torontonian_tmsv():
    """
    Calculates the torontonian of a two-mode squeezed vacuum
    state squeezed with mean photon number 1.0
    """

    mean_n = 1.0
    r = np.arcsinh(np.sqrt(mean_n))
    Omat = np.tanh(r)*np.array([[0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0]])

    tor_val = tor(Omat)
    assert np.abs(tor_val.real - 1.0) < abs_tol


def test_torontonian_vacuum():
    """
    Calculates the torontonian of a vacuum in n modes
    """
    n_modes = 5
    Omat = np.zeros([2*n_modes, 2*n_modes])
    tor_val = tor(Omat)
    assert np.abs(tor_val.real - 0.0) < abs_tol


