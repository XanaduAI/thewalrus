# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
tests for code in thewalrus.internal_modes
"""

import pytest

import numpy as np 
import strawberryfields as sf 

from scipy.stats import unitary_group
from scipy.special import factorial

from copy import deepcopy

from itertools import combinations_with_replacement

from thewalrus import low_rank_hafnian, reduction

from thewalrus.internal_modes import (
    pnr_prob,
    distinguishable_pnr_prob

    )

from thewalrus.random import random_covariance
from thewalrus.quantum import density_matrix_element, Amat, Qmat, state_vector
from thewalrus.symplectic import squeezing, passive_transformation
from thewalrus.symplectic import autonne as takagi


### auxilliary functions for testing ###
def vacuum_prob_distinguishable(rs, T):
    """Calculates the vacuum probability when distinguishable squeezed states go into an interferometer.
    Args:
        rs (array): squeezing parameters
        T (array): transmission matrix
    Returns:
        (float): vacuum probability
    """
    prob = 1
    M = len(rs)
    mu = np.zeros([2 * M])
    for i in range(M):
        localr = [0] * M
        localr[i] = 2 * rs[i]
        _, lcov = passive_transformation(mu, squeezing(localr), T)
        prob *= 1 / np.sqrt(np.linalg.det(Qmat(lcov)).real)
    return prob


def _list_to_freq_dict(words):
    """Convert between a list which of "words" and a dictionary
    which shows how many times each word appears in word

    Args:
        words (list): list of words
    Returns:
        dict : how many times a word appears. key is word, value is multiplicity
    """
    return {i: words.count(i) for i in set(words)}


def duplicate(x):
    """duplicate a list"""
    y = list(x)
    return y + y


def generate_origins(keys, num_pairs):
    """Find all the possible ways in which num_pairs could have come from a number of different squeezed source specified in keys"""
    origins = [
        _list_to_freq_dict(duplicate(x)) for x in combinations_with_replacement(keys, num_pairs)
    ]
    return origins


def generate_origins_unpaired(keys, num_pairs):
    """Find all the possible ways in which num_pairs could have come from a number of different squeezed source specified in keys"""
    origins = [_list_to_freq_dict(x) for x in combinations_with_replacement(keys, num_pairs)]
    return origins


####################################################################
# Mikhail's magic OOP way of dealing with boxes and coloured balls #
####################################################################
class Box:
    def __init__(self, size: int):
        self.size = size
        self.contents = ""

    def __repr__(self):
        return self.contents


def _generate(boxes, colour, index, red, blue):
    if index == len(boxes):
        yield deepcopy(boxes)
        return

    box = boxes[index]

    # Compute the remaining (effective) box size.
    size = box.size - len(box.contents)

    # Don't take too many red balls.
    max_r = min(red, size)

    # Don't take too many blue balls.
    min_r = max(0, size - blue)

    for r in range(min_r, max_r + 1):
        b = size - r
        box.contents += colour * r
        yield from _generate(boxes, colour, index + 1, red - r, blue - b)
        box.contents = box.contents[: len(box.contents) - r]


def generate(colours, total_balls, boxes, index=0):
    """A generator with all the possible colour/boxes combinations"""
    if index == len(colours):
        yield deepcopy(boxes)
        return

    colour, red = colours[index]
    blue = total_balls - red
    for combo in _generate(boxes, colour, 0, red, blue):
        yield from generate(colours, blue, combo, index + 1)


def prob_low_rank(r, vec, pattern, norm=True):
    """Calculate the probability of event with photon number pattern
    when a squeezed state with squeezing parameter r enter in a interferometer
    The interferometer is represented by a vector vec corresponding to the
    relevant column of the unitary matrix"""
    vec = vec * np.sqrt(np.tanh(r))
    expvec = reduction(vec, pattern)
    pref = 1.0
    if norm is True:
        pref = 1 / np.cosh(r)
    if len(expvec) == 0:
        return pref
    amp = low_rank_hafnian(np.array([expvec]).T)
    denom = np.prod(factorial(pattern))
    return pref * np.abs(amp) ** 2 / denom


def generate_all_origins(input_labels, events):
    """Generate all the possible combinations of photon events/photon colour (label) origins
    This is returnes as dictionary.
    The input labels are a list of integers specifying where the squeezed light went into the unitary
    events is a dictionary specifying where the photons were detected and how many of them were counted.
    """
    input_labels = list(map(chr, input_labels))

    total_n = sum([events[key] for key in events])
    assert total_n % 2 == 0
    total_p = total_n // 2
    origins = generate_origins([str(i) for i in input_labels], total_p)
    result = []
    for origin in origins:
        keys = origin.keys()
        colours = [(key, origin[key]) for key in keys]
        boxes = [Box(events[key]) for key in events.keys()]
        result.append(list(generate(colours, total_n, boxes)))
    flat_list = [item for sublist in result for item in sublist]
    return [[list(map(ord, list(x.contents))) for x in y] for y in flat_list]


def generate_all_origins_unpaired(input_labels, events):
    """Generate all the possible combinations of photon events/photon colour (label) origins
    This is returnes as dictionary.
    The input labels are a list of integers specifying where the squeezed light went into the unitary
    events is a dictionary specifying where the photons were detected and how many of them were counted.
    """
    input_labels = list(map(chr, input_labels))

    total_n = sum([events[key] for key in events])
    total_p = total_n
    origins = generate_origins_unpaired([str(i) for i in input_labels], total_p)
    result = []
    for origin in origins:
        keys = origin.keys()
        colours = [(key, origin[key]) for key in keys]
        boxes = [Box(events[key]) for key in events.keys()]
        result.append(list(generate(colours, total_n, boxes)))
    flat_list = [item for sublist in result for item in sublist]
    return [[list(map(ord, list(x.contents))) for x in y] for y in flat_list]


def dict_to_pattern(input_dict, num_modes):
    """Converts a dictionary of photon events into a pattern in a total of num_modes"""
    pattern = [0] * num_modes
    for i in input_dict:
        pattern[i] = input_dict[i]
    return pattern


def prob_distinguishable(U, input_labels, input_squeezing, events):
    """Calculate the probability of distinguishable event when an interferometer U is used.
    input_labels tell the function where the squeezed light was injected while
    input_squeezing gives the squeezing parameters
    events is a dictionary describing the detection pattern
    """
    n_modes = len(U)
    pvac = 1.0 / np.prod(np.cosh(input_squeezing))
    total_n = sum([events[key] for key in events])
    if total_n % 2 != 0:
        return 0
    modes = list(events.keys())
    origins = generate_all_origins(input_labels, events)
    lists = [
        [{i: y.count(label) for i, y in zip(modes, origin)} for label in input_labels]
        for origin in origins
    ]
    mappable_dict_to_pattern = lambda x: dict_to_pattern(x, n_modes)
    patterns = [list(map(mappable_dict_to_pattern, l)) for l in lists]
    net_sum = 0.0
    for pattern in patterns:
        term = np.prod(
            [
                prob_low_rank(input_squeezing[i], U[:, input_labels[i]], pattern[i], norm=False)
                for i in range(len(input_squeezing))
            ]
        )
        net_sum += term
    net_sum *= pvac
    return net_sum


def prob_low_rank_lossy(GT, pattern, norm=False):
    """Calculate the probability of event with photon number pattern
    when a squeezed state with squeezing parameter r enter in a interferometer
    The interferometer + squeezed state is presented by GT matrix such that GT.T @ GT gives
    the rank two adjacency matrix of the state.
    """
    if sum(pattern) == 0:
        if norm is False:
            return 1.0
        return norm
    denom = np.prod(factorial(pattern))
    pat = list(pattern) + list(pattern)
    GTpat = np.array([reduction(GT[i], pat) for i in [0, 1]])
    num = low_rank_hafnian(GTpat.T).real
    if norm is False:
        return num / denom
    return num * norm / denom


def prob_distinguishable_lossy(T, input_labels, input_squeezing, events):
    """Calculate the probability of a distinguishable event when a lossy interferometer T is used.
    input_labels tell the function where the squeezed light was injected while
    input_squeezing gives the squeezing parameters
    events is a dictionary describing the detection pattern
    # Note that it is always assumed that the squeezing happens in the first modes, i.e., input labels should be 0,1,2....
    """
    rs = np.zeros(len(T))
    for num, i in enumerate(input_labels):
        rs[i] = input_squeezing[num]
    n_modes = len(T)
    mu = np.zeros([2 * n_modes])
    vac_probs = []
    GTs = []
    for i in range(n_modes):
        localr = [0] * n_modes
        localr[i] = 2 * rs[i]
        _, lcov = passive_transformation(mu, squeezing(localr), T)
        vac_probs.append(1 / np.sqrt(np.linalg.det(Qmat(lcov)).real))
        Am = Amat(lcov)
        l, U = takagi(Am)
        GTs.append((U[:, 0:2] * np.sqrt(l[0:2])).T)

    modes = list(events.keys())
    origins = generate_all_origins_unpaired(input_labels, events)
    lists = [
        [{i: y.count(label) for i, y in zip(modes, origin)} for label in input_labels]
        for origin in origins
    ]
    mappable_dict_to_pattern = lambda x: dict_to_pattern(x, n_modes)
    patterns = [list(map(mappable_dict_to_pattern, l)) for l in lists]

    net_sum = 0.0
    for pattern in patterns:
        term = np.prod(
            [
                prob_low_rank_lossy(GTs[i], pattern[i], norm=vac_probs[i])
                for i in range(len(input_squeezing))
            ]
        )
        net_sum += term
    return net_sum


#### Test functions start here ####

@pytest.mark.parametrize("M", [3,4,5,6])
def test_pnr_prob_single_internal_mode(M):
    """
    test internal modes functionality against standard method for pnr probabilities
    """

    cov = random_covariance(M)
    mu = np.zeros(2 * M)

    pattern = [2,3,0] + [1] * (M - 3)
        
    p1 = pnr_prob(cov, pattern)
    p2 = density_matrix_element(mu, cov, pattern, pattern).real
    
    assert np.isclose(p1, p2)

@pytest.mark.parametrize("M", [3,4,5,6])
def test_distinguishable_pnr_prob(M):
    hbar = 2

    pattern = [3,2,0] + [1] * (M - 3)

    mu = np.zeros(2 * M)

    rs = [1] * M
    T = 0.5 * unitary_group.rvs(M)

    big_cov = np.zeros((2*M**2, 2*M**2))
    covs = []
    for i, r in enumerate(rs):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, T)
        covs.append(cov)
        big_cov[i::M,i::M] = cov

    p1 = pnr_prob(covs, pattern, hbar=hbar)
    p2 = pnr_prob(big_cov, pattern, hbar=hbar)
    p3 = distinguishable_pnr_prob(pattern, rs, T)

    assert np.isclose(p1,p2)
    assert np.isclose(p1,p3)
    assert np.isclose(p2,p3)


@pytest.mark.parametrize("M", range(2, 7))
def test_distinguishable_probs(M):
    """test distinguishability code against combinatorial version"""
    U = unitary_group.rvs(M)
    r = 0.4

    rs = r * np.ones(M)
    input_labels = np.arange(M)

    pattern = [1] * M

    events = dict(enumerate(pattern))

    p1 = prob_distinguishable(U, input_labels, rs, events)

    p2 = distinguishable_pnr_prob(pattern, rs, U)

    assert np.allclose(p1, p2, atol=1e-6)

@pytest.mark.parametrize("M", range(2, 7))
def test_distinguishable_vacuum_probs(M):
    """test distinguishability code against combinatorial version for vacuum outcome"""
    U = unitary_group.rvs(M)
    r = 0.4

    rs = r * np.ones(M)
    input_labels = np.arange(M)

    pattern = [0] * M

    events = dict(enumerate(pattern))

    p1 = prob_distinguishable(U, input_labels, rs, events)

    p2 = distinguishable_pnr_prob(pattern, rs, U)

    assert np.allclose(p1, p2, atol=1e-6)

@pytest.mark.parametrize("M", range(2, 7))
def test_distinguishable_probs_collisions(M):
    """test distinguishability code against combinatorial version"""
    U = unitary_group.rvs(M)
    r = 0.4

    rs = r * np.ones(M)
    input_labels = np.arange(M)

    pattern = [2] * 2 + [0] * (M - 2)
    events = dict(enumerate(pattern))
    p1 = prob_distinguishable(U, input_labels, rs, events)

    p2 = distinguishable_pnr_prob(pattern, rs, U)

    assert np.allclose(p1, p2, atol=1e-6)

@pytest.mark.parametrize("M", range(2, 7))
@pytest.mark.parametrize("collisions", [False, True])
def test_distinguishable_probabilitites_single_input(M, collisions):
    """test the distinguishable code is correct with a single squeezer"""
    if collisions:
        # Make patterns with at least one collision
        pattern = np.random.randint(0, 3, M)
        pattern[0] = 2
    else:
        # Make patterns with zeros and ones only
        pattern = np.random.randint(0, 2, M)
    rs = [1] + [0] * (M - 1)

    T = (unitary_group.rvs(M) * np.random.rand(M)) @ unitary_group.rvs(M)
    singular_values = np.linalg.svd(T, compute_uv=False)
    T /= max(singular_values)

    cov_in = squeezing(2 * np.array(rs))
    mu_in = np.zeros([2 * M])
    mu_out, cov_out = passive_transformation(mu_in, cov_in, T)
    expected = np.real_if_close(
        density_matrix_element(mu_out, cov_out, list(pattern), list(pattern))
    )
    obtained = distinguishable_pnr_prob(pattern, rs, T)
    assert np.allclose(expected, obtained)

@pytest.mark.parametrize("M", range(2, 10))
def test_distinguishable_vacuum_probs_lossy(M):
    """test distinguishability code against combinatorial version for vacuum outcome (with loss)"""
    T = (unitary_group.rvs(M) * np.random.rand(M)) @ unitary_group.rvs(M)
    rs = np.random.rand(M)

    p1 = vacuum_prob_distinguishable(rs, T)
    pattern = [0] * M
    p2 = distinguishable_pnr_prob(pattern, rs, T)

    assert np.allclose(p1, p2, atol=1e-6)

@pytest.mark.parametrize("M", range(7, 15))
@pytest.mark.parametrize("pat_dict", [{1: 3, 2: 2}, {1: 1, 2: 1, 3: 1, 4: 1}])
def test_distinguishable_probs_lossy(M, pat_dict):
    """test distinguishable code against combinatorial version for lossy systems"""
    T = (unitary_group.rvs(M) * np.random.rand(M)) @ unitary_group.rvs(M)
    rs_vec = [0] * M
    rs_ind = [0, 1, 2]
    rs_vals = [0.7, 0.8, 0.9]
    for i, ind in enumerate(rs_ind):
        rs_vec[ind] = rs_vals[i]
    pat_list = [0] * M
    for val in pat_dict:
        pat_list[val] = pat_dict[val]
    expected = prob_distinguishable_lossy(T, rs_ind, rs_vals, pat_dict)
    obtained = distinguishable_pnr_prob(pat_list, rs_vec, T)
    assert np.allclose(expected, obtained)
    assert np.allclose(expected, obtained)

