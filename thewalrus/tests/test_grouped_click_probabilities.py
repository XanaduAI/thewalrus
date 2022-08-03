"""Test for the grouped click probability function"""
from itertools import product
import numpy as np
import pytest
from thewalrus.random import random_interferometer
from thewalrus.symplectic import passive_transformation, squeezing
from thewalrus.quantum import mean_clicks, variance_clicks
from thewalrus._torontonian import threshold_detection_prob
from thewalrus.grouped_click_probabilities import grouped_click_probabilities_squeezed


@pytest.mark.parametrize("num_modes", [4, 6, 8])
@pytest.mark.parametrize("eta", [0.2, 0.4, 0.6])
@pytest.mark.parametrize("num_samples", [10 ** 3, 10 ** 4, 10 ** 5])
@pytest.mark.parametrize("num_groups", [5, 10, 100])
def test_mean_var(num_modes, eta, num_samples, num_groups):
    """This function tests the mean and variance of the number of clicks"""
    sq_vec = np.random.rand(num_modes)
    tmat = eta * random_interferometer(num_modes)
    sq_cov = squeezing(sq_vec) @ squeezing(sq_vec)
    _, out_cov = passive_transformation(np.zeros(2 * num_modes), sq_cov, tmat)
    mean_n = mean_clicks(out_cov)
    var_n = variance_clicks(out_cov)
    probs = grouped_click_probabilities_squeezed(sq_vec, tmat, num_samples, num_groups)[0]
    mean_np = probs @ np.arange(num_modes + 1)
    var_np = probs @ (np.arange(num_modes + 1)) ** 2 - mean_np ** 2
    assert np.allclose(mean_n, mean_np, rtol=10 * (num_samples) ** (-0.5))
    assert np.allclose(var_n, var_np, rtol=10 * (num_samples) ** (-0.5))
    assert np.allclose(probs.sum(), 1.0)


@pytest.mark.parametrize("num_modes", [2, 3, 4])
@pytest.mark.parametrize("eta", [0.4, 0.6, 0.8])
@pytest.mark.parametrize("num_samples", [10 ** 3, 10 ** 4, 10 ** 5])
@pytest.mark.parametrize("num_groups", [5, 10, 100])
def test_probs(num_modes, eta, num_samples, num_groups):
    """This function tests the click probabilities"""
    sq_vec = np.random.rand(num_modes)
    tmat = eta * random_interferometer(num_modes)
    sq_cov = squeezing(sq_vec) @ squeezing(sq_vec)
    out_mu, out_cov = passive_transformation(np.zeros(2 * num_modes), sq_cov, tmat)
    t_probs = np.zeros([num_modes + 1])
    for cmb in product([0, 1], repeat=num_modes):
        prob = threshold_detection_prob(out_mu, out_cov, np.array(cmb)).real
        t_probs[sum(cmb)] += prob
    s_probs = grouped_click_probabilities_squeezed(sq_vec, tmat, num_samples, num_groups)[0]
    assert np.allclose(t_probs, s_probs, rtol=10 * (num_samples) ** (-0.5), atol=min(t_probs))
