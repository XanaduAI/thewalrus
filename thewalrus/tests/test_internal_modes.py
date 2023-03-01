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
from copy import deepcopy
from itertools import chain, combinations_with_replacement, product

from random import gauss

import pytest

import numpy as np

from scipy.stats import unitary_group
from scipy.special import factorial

from repoze.lru import lru_cache

from thewalrus import low_rank_hafnian, reduction

from thewalrus.decompositions import takagi
from thewalrus.random import random_covariance
from thewalrus.quantum import (
    density_matrix_element,
    density_matrix,
    Amat,
    Qmat,
    state_vector,
)
from thewalrus.symplectic import (
    beam_splitter,
    expand,
    expand_vector,
    interferometer,
    passive_transformation,
    reduced_state,
    squeezing,
)

from thewalrus.internal_modes import pnr_prob, density_matrix_single_mode
from thewalrus.internal_modes.prepare_cov import (
    O_matrix,
    orthonormal_basis,
    state_prep,
    prepare_cov,
    LO_overlaps,
)

### auxilliary functions for testing ###
# if we want to have less auxilliary functions, we can remove a few tests and get rid of it all

### Nico's code for combinatorial approach for distinguishable calculations ###
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

# pylint: disable=too-few-public-methods
class Box:
    """A class to represent boxes"""

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


# pylint: disable=consider-using-generator
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


# pylint: disable=consider-using-generator
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


# pylint: disable=consider-using-generator
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

    def mappable_dict_to_pattern(x):
        """Convenience function"""
        return dict_to_pattern(x, n_modes)

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

    def mappable_dict_to_pattern(x):
        """Convenience function for mapping modes and labels"""
        return dict_to_pattern(x, n_modes)

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


#################################################################################
# David Phillip's code for combinatorial approach to internal mode calculations #
#################################################################################


def loss(cov, efficiency, hbar=2):
    r"""Implements spatial mode loss on a covariance matrix whose modes are grouped by spatial modes.
    Works for any number of Schmidt/orthonormal modes.

    Args:
        cov (array): covariance matrix
        efficiency (array): array of efficiencies of each spatial mode
        hbar (int/float): the value of hbar, either 0.5, 1 or 2 (default 2)

    Returns:
        covariance matrix updated for loss
    """
    M = efficiency.shape[0]
    R = cov.shape[0] // (2 * M)
    T = np.array([])
    for i in range(M):
        T = np.append(T, np.array(R * [np.sqrt(efficiency[i])]))
    T = np.diag(np.append(T, T))
    return T @ cov @ T + (hbar / 2) * (np.identity(cov.shape[0]) - T @ T)


@lru_cache(maxsize=1000000)
def combos(N, R):
    r"""Returns a list of all partitions of detecting N photons with a mode-insensitive detector into R modes.

    Args:
        N (int): total number of detected photons
        R (int): number of modes in which to split the photons

    Returns:
        all of the possible partitions
    """
    if R == 1:
        return [[N]]

    new_combos = []
    for first_val in range(N + 1):
        rest = combos(N - first_val, R - 1)
        new = [p[0] + p[1] for p in product([[first_val]], rest)]
        new_combos += new
    new_combos.reverse()
    return new_combos


def dm_MD_2D(dm_MD):
    r"""For R effective modes and when computing up to Ncutoff, this function converts a 2R-dimensional
    density matrix (i.e. 2 dimensions for each Schmidt mode) into a 2-dimensional density matrix.
    The initial density matrix dm_MD has entries dm_MD[j_{0}, k_{0}, ..., j_{R-1}, k_{R-1}] where j_{0} etc. run from 0 to Ncutoff-1.
    The final matrix dm_2D has Ncutoff**R x Ncutoff**R entries.

    Args:
        dm_MD (array): 2R-dimensional density matrix

    Returns:
        2-dimensional density matrix
    """
    R = len(dm_MD.shape) // 2
    if np.allclose(R, 1):
        return dm_MD
    Ncutoff = len(dm_MD)
    new_ax = np.arange(2 * R).reshape([R, 2]).T.flatten()
    dm_2D = dm_MD.transpose(new_ax).reshape([Ncutoff**R, Ncutoff**R])
    return dm_2D


def dm_2D_MD(dm_2D, R):
    r"""Converts a 2-dimensional density matrix into a 2R-dimensional density matrix (i.e. 2 dimensions for each effective mode).
    When computing for up to Ncutoff photons with R effective modes, the initial matrix has Ncutoff**R x Ncutoff**R entries.
    The final density matrix dm_MD has entries dm_MD[j_{0}, k_{0}, ..., j_{R-1}, k_{R-1}] where j_{0} etc. run from 0 to Ncutoff-1.

    Args:
        dm_2D (array): 2-dimensional density matrix
        R (int): effective number of modes

    Returns:
        2R-dimensional density matrix
    """
    if np.allclose(R, 1):
        return dm_2D
    Ncutoff = len(dm_2D) ** (1 / R)
    assert abs(round(Ncutoff) - Ncutoff) < 1e-5
    Ncutoff = round(Ncutoff)
    dim = 2 * R * [Ncutoff]
    new_ax = np.arange(2 * R).reshape([R, 2]).T.flatten()
    dm_MD = np.reshape(dm_2D, dim).transpose(new_ax)
    return dm_MD


def swap_matrix(M, R):
    r"""Computes the matrix that swaps the ordering of modes from grouped by orthonormal modes to grouped by spatial modes.
    The inverse of the swap matrix is its transpose.

    Args:
        M (int): number of spatial modes.
        R (int): number of orthonormal modes in each spatial mode.

    Returns:
        M*R x M*R swap matrix.
    """
    P = np.zeros((M * R, M * R), dtype=int)
    in_modes = range(M * R)
    out_modes = list(chain.from_iterable(range(i, M * R, M) for i in range(M)))
    P[out_modes, in_modes] = 1
    return P


def implement_U(cov, U):
    r"""Implements a spatial mode linear optical transofrmation (flat in orthonormal modes) described by U on a covariance matrix.
    Assumes the modes of the input covariance matrix are grouped by spatial modes.

    Args:
        cov (array): covariance matrix.
        U (array): unitary transformation of spatial modes.

    Returns:
        transformed covariance matrix.
    """
    M = U.shape[0]
    R = cov.shape[0] // (2 * M)
    Ubig = np.identity(M * R, dtype=np.complex128)
    for j in range(R):
        Ubig[M * j : M * (j + 1), M * j : M * (j + 1)] = U
    X = swap_matrix(M, R)
    Usymp = interferometer(X.T @ Ubig @ X)
    return Usymp @ cov @ Usymp.T


# pylint: disable=too-many-statements, too-many-branches, consider-using-generator
def heralded_density_matrix(
    rjs,
    O,
    U,
    N,
    efficiency=None,
    noise=None,
    Ncutoff=None,
    MD=True,
    normalize=False,
    thr=1e-3,
    thresh=1e-4,
    hbar=2,
):
    r"""Returns the density matrix of the specified spatial mode when heralding on N (dict) photons in defined spatial modes for the given inupt parameters.
    The initial state has squeezing parameters rjs (list for each spatial mode of squeezing parameters of each Schmidt mode for that spatial mode),
    and mode overlaps described by the O matrix. The whole system is evolved under a unitary U on the spatial modes.
    Output density matrix has dimensions for each orthonormal mode.

    Args:
        rjs (list[array]): list for each spatial mode of list/array of squeezing parameters for each Schmidt mode in that spatial mode.
        O (array): 2-dimensional matrix of the overlaps between each Schmidt mode in all spatial modes combined.
        U (array): unitary matrix expressing the three spatial mode interferometer.
        N (dict): post selection total photon number in the spatial modes (int), indexed by spatial mode.
        efficiency (array): total efficiency/transmission of the three spatial modes.
        noise (array): Poissonian noise amplitude in each spatial mode after loss (sqrt of <n>).
        Ncutoff (int): cutoff dimension for each density matrix.
        MD (bool): return multidimensional density matrix.
        normalize (bool): whether to normalise the output density matrix.
        thr (float): eigenvalue threshold under which orthonormal mode is discounted.
        thresh (float): fidelity distance away from vacuum for an orthonormal mode to be discarded.
        hbar (int/float): the value of hbar, either 0.5, 1.0 or 2.0 (default 2.0).

    Returns:
        density matrix of heralded spatial mode.
    """
    # pylint: disable=too-many-arguments
    if not np.allclose(U.shape[0], len(rjs)):
        raise ValueError("Unitary is the wrong size, it must act on all spatial modes")
    if not np.allclose(U.shape[0], len(N) + 1):
        raise ValueError("Mismatch between expected system size and heralding modes")
    if not set(list(N.keys())).issubset(set(list(np.arange(U.shape[0])))):
        raise ValueError("Keys of N must correspond to all but one spatial mode")
    if not np.allclose(O.shape[0], sum([len(listElem) for listElem in rjs])):
        raise ValueError(
            "Length of O must equal the total number of Schmidt modes accross all spatial modes"
        )
    if not np.allclose(U @ U.T.conj(), np.identity(len(rjs))):
        raise ValueError("U must be a unitary matrix")
    if efficiency is not None:
        if not np.allclose(U.shape[0], efficiency.shape[0]):
            raise ValueError(
                "If giving an efficiency, a value for each spatial mode must be provided"
            )
    if noise is not None:
        if not np.allclose(U.shape[0], noise.shape[0]):
            raise ValueError(
                "If giving a noise values, a value for each spatial mode must be provided"
            )
    M = U.shape[0]
    N_nums = list(N.values())
    HM = list(set(list(np.arange(M))).difference(list(N.keys())))[0]
    if efficiency is None:
        efficiency = np.ones(M)
    eps, W = orthonormal_basis(rjs, O=O, thr=thr)
    covinit = state_prep(eps, W, thresh=thresh, hbar=hbar)
    R = covinit.shape[0] // (2 * M)
    covu = implement_U(covinit, U)
    covfinal = loss(covu, efficiency, hbar=hbar)

    combos_list = []
    totals = []
    if noise is not None:
        for ii in N_nums:
            combos_temp = combos(ii, R + 1)
            combos_list.append(combos_temp)
            totals.append(len(combos_temp))
    else:
        for ii in N_nums:
            combos_temp = combos(ii, R)
            combos_list.append(combos_temp)
            totals.append(len(combos_temp))

    list_of_lists = [list(range(i)) for i in totals]
    indices = product(*list_of_lists)

    Nmax = max(N_nums)
    if Ncutoff is None:
        Ncutoff = int(np.ceil(2 * Nmax))

    post_select_dicts_sig = []
    if noise is not None:
        post_select_dicts_noise = []

    for idx in indices:
        temp_dict_sig = {}
        for jj in range(M - 1):
            if jj < HM:
                for kk in range(R):
                    temp_dict_sig[kk + jj * R] = combos_list[jj][idx[jj]][kk]
            else:
                for kk in range(R):
                    temp_dict_sig[kk + (jj + 1) * R] = combos_list[jj][idx[jj]][kk]
        post_select_dicts_sig.append(temp_dict_sig)
        if noise is not None:
            temp_dict_noise = {}
            for mm in range(M - 1):
                if mm < HM:
                    temp_dict_noise[mm] = combos_list[mm][idx[mm]][-1]
                else:
                    temp_dict_noise[mm + 1] = combos_list[mm][idx[mm]][-1]
            post_select_dicts_noise.append(temp_dict_noise)

    total_dm_list = []
    for i, _ in enumerate(post_select_dicts_sig):
        dm_temp = density_matrix(
            np.zeros(covfinal.shape[0]),
            covfinal,
            post_select=post_select_dicts_sig[i],
            cutoff=Ncutoff,
            hbar=hbar,
        )
        if noise is not None:
            dm_temp *= np.trace(
                density_matrix(
                    expand_vector(noise, hbar=hbar),
                    (hbar / 2) * np.identity(2 * M),
                    post_select=post_select_dicts_noise[i],
                    cutoff=Ncutoff,
                    hbar=hbar,
                )
            )
        total_dm_list.append(dm_temp)

    dm_tot = sum(total_dm_list)
    if normalize:
        dm_tot = dm_2D_MD(dm_MD_2D(dm_tot) / np.trace(dm_MD_2D(dm_tot)), R)
    if not MD:
        return dm_MD_2D(dm_tot)
    return dm_tot


# pylint: disable=too-many-arguments, consider-using-generator
def heralded_density_matrix_LO(
    rjs,
    F,
    U,
    N,
    LO_shape,
    efficiency=None,
    noise=None,
    Ncutoff=None,
    normalize=True,
    thr=1e-3,
    thresh=1e-4,
    hbar=2,
):
    r"""Returns the density matrix of the specified spatial mode when heralding on N (dict) photons in defined spatial modes for the given inupt parameters.
    The initial state has squeezing parameters rjs (list for each spatial mode of squeezing parameters of each Schmidt mode for that spatial mode),
    and mode overlaps described by the O matrix. The whole system is evolved under a unitary U on the spatial modes. The resulting density matrix
    is single-mode in the local oscillator basis.
    Args:
        rjs: list for each spatial mode of list/array of squeezing parameters for each Schmidt mode in that spatial mode.
        F: List of arrays of the temporal modes of the Schmidt modes in each spatial mode, must be normalized and
           ordered by spatial mode.
        U: unitary matrix expressing the three spatial mode interferometer.
        N: post selection total photon number in the spatial modes (int), indexed by spatial mode.
        LO_shape: temporal profile of local oscillator.
        efficiency: total efficiency/transmission of the three spatial modes.
        noise: Poissonian noise amplitude in each spatial mode after loss (sqrt of <n>).
        Ncutoff: cutoff dimension for each density matrix.
        normalize: whether to normalise the output density matrix.
        thr: eigenvalue threshold under which orthonormal mode is discounted.
        thresh: fidelity distance away from vacuum for an orthonormal mode to be discarded.
        hbar: the value of hbar, either 0.5, 1.0 or 2.0 (default 2.0).
    Returns:
        density matrix of heralded spatial mode in local oscillator basis.
    """

    if not np.allclose(U.shape[0], len(rjs)):
        raise ValueError("Unitary is the wrong size, it must act on all spatial modes")
    if not np.allclose(U.shape[0], len(N) + 1):
        raise ValueError("Mismatch between expected system size and heralding modes")
    if not set(list(N.keys())).issubset(set(list(np.arange(U.shape[0])))):
        raise ValueError("Keys of N must correspond to all but one spatial mode")
    Fs = [f for fj in F for f in fj]
    if not np.allclose(len(Fs), sum([len(listElem) for listElem in rjs])):
        raise ValueError(
            "Length of F must equal the total number of Schmidt modes accross all spatial modes"
        )
    if not np.allclose(Fs[0].shape[0], LO_shape.shape[0]):
        raise ValueError(
            "Schmidt temporal functions and LO temporal profile must have the same number of components"
        )
    if not np.allclose(U @ U.T.conj(), np.identity(len(rjs))):
        raise ValueError("U must be a unitary matrix")
    if efficiency is not None:
        if not np.allclose(U.shape[0], efficiency.shape[0]):
            raise ValueError(
                "If giving an efficiency, a value for each spatial mode must be provided"
            )
    if noise is not None:
        if not np.allclose(U.shape[0], noise.shape[0]):
            raise ValueError(
                "If giving a noise values, a value for each spatial mode must be provided"
            )
    M = U.shape[0]
    N_nums = list(N.values())
    HM = list(set(list(np.arange(M))).difference(list(N.keys())))[0]
    if efficiency is None:
        efficiency = np.ones(M)
    chis, eps, W = orthonormal_basis(rjs, F=F, thr=thr)
    covinit = state_prep(eps, W, thresh=thresh, hbar=hbar)
    R = covinit.shape[0] // (2 * M)
    covu = implement_U(covinit, U)
    covfinal = loss(covu, efficiency, hbar=hbar)

    Uswap = np.zeros((M, M))
    swapV = np.concatenate((np.arange(HM), np.arange(HM + 1, M), np.array([HM])))
    for j, k in enumerate(swapV):
        Uswap[j][k] = 1
    covreordered = implement_U(covfinal, Uswap)  # Putting heralded spatial mode in position M

    LO_shape /= np.linalg.norm(LO_shape)
    T_LO = np.identity(R, dtype=np.complex128)
    T_LO[0] = np.array([np.inner(LO_shape.conj(), chis[j]) for j in range(R)])
    T_tot = np.identity(M * R, dtype=np.complex128)
    T_tot[(M - 1) * R : M * R, (M - 1) * R : M * R] = T_LO
    _, covlo = passive_transformation(
        np.zeros(covreordered.shape[0]), covreordered, T_tot, hbar=hbar
    )
    if R > 1:
        _, covtraced = reduced_state(np.zeros(covlo.shape[0]), covlo, np.arange((M - 1) * R + 1))
    else:
        covtraced = covlo[:]

    combos_list = []
    totals = []
    if noise is not None:
        for ii in range(M - 1):
            combos_temp = combos(N_nums[ii], R + 1)
            combos_list.append(combos_temp)
            totals.append(len(combos_temp))
    else:
        for ii in range(M - 1):
            combos_temp = combos(N_nums[ii], R)
            combos_list.append(combos_temp)
            totals.append(len(combos_temp))

    list_of_lists = [list(range(i)) for i in totals]
    indices = product(*list_of_lists)

    Nmax = max(N_nums)
    if Ncutoff is None:
        Ncutoff = int(np.ceil(2 * Nmax))

    post_select_dicts_sig = []
    if noise is not None:
        post_select_dicts_noise = []

    for idx in indices:
        temp_dict_sig = {}
        for jj in range(M - 1):
            for kk in range(R):
                temp_dict_sig[kk + jj * R] = combos_list[jj][idx[jj]][kk]
        post_select_dicts_sig.append(temp_dict_sig)
        if noise is not None:
            temp_dict_noise = {}
            for mm in range(M - 1):
                temp_dict_noise[mm] = combos_list[mm][idx[mm]][-1]
            post_select_dicts_noise.append(temp_dict_noise)

    L = len(post_select_dicts_sig)
    total_dm_list = []
    for i in range(L):
        dm_temp = density_matrix(
            np.zeros(covtraced.shape[0]),
            covtraced,
            post_select=post_select_dicts_sig[i],
            cutoff=Ncutoff,
            hbar=hbar,
        )
        if noise is not None:
            dm_temp *= np.trace(
                density_matrix(
                    expand_vector(noise, hbar=hbar),
                    (hbar / 2) * np.identity(2 * M),
                    post_select=post_select_dicts_noise[i],
                    cutoff=Ncutoff,
                    hbar=hbar,
                )
            )
        total_dm_list.append(dm_temp)

    dm_tot = sum(total_dm_list)
    if normalize:
        return dm_tot / np.trace(dm_tot)
    return dm_tot


#############################
# Test functions start here #
#############################


@pytest.mark.parametrize("M", [3, 4, 5, 6])
def test_pnr_prob_single_internal_mode(M):
    """
    test internal modes functionality against standard method for pnr probabilities
    """

    cov = random_covariance(M)
    mu = np.zeros(2 * M)

    pattern = [2, 3, 0] + [1] * (M - 3)

    p1 = pnr_prob(cov, pattern)
    p2 = density_matrix_element(mu, cov, pattern, pattern).real

    assert np.isclose(p1, p2)


@pytest.mark.parametrize("M", [3, 4, 5, 6])
def test_pnr_prob_fully_distinguishable(M):
    """Testing the photon number distribution of fully distinguishable squeezed states"""
    hbar = 2

    pattern = [3, 2, 0] + [1] * (M - 3)

    mu = np.zeros(2 * M)

    rs = [1] * M
    T = 0.5 * unitary_group.rvs(M)

    big_cov = np.zeros((2 * M**2, 2 * M**2))
    covs = []
    for i, r in enumerate(rs):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, T)
        covs.append(cov)
        big_cov[i::M, i::M] = cov
    big_cov2 = prepare_cov([[r] for r in rs], T, O=np.identity(M))
    assert np.allclose(big_cov, big_cov2)

    p1 = pnr_prob(covs, pattern, hbar=hbar)
    p2 = pnr_prob(big_cov, pattern, hbar=hbar)
    p3 = pnr_prob(big_cov2, pattern, hbar=hbar)

    assert np.isclose(p1, p2)
    assert np.isclose(p2, p3)


@pytest.mark.parametrize("M", range(2, 7))
@pytest.mark.parametrize("pat", [0, 1, [2, 2]])
def test_distinguishable_probs(M, pat):
    """test distinguishability code against combinatorial version for vacuum outcome"""
    U = unitary_group.rvs(M)
    r = 0.4

    rs = r * np.ones(M)
    input_labels = np.arange(M)

    if type(pat) is int:
        pattern = [pat] * M
    else:
        pattern = pat + [0] * (M - len(pat))

    events = dict(enumerate(pattern))

    p1 = prob_distinguishable(U, input_labels, rs, events)

    hbar = 2
    mu = np.zeros(2 * M)
    covs = []
    for i, r in enumerate(rs):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, U)
        covs.append(cov)
    p2 = pnr_prob(covs, pattern, hbar=hbar)

    assert np.isclose(p1, p2, atol=1e-6)

    if sum(pattern) == 0:
        p3 = vacuum_prob_distinguishable(rs, U)
        assert np.isclose(p1, p3, atol=1e-6)
        assert np.isclose(p2, p3, atol=1e-6)


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
    obtained = pnr_prob(cov_out, pattern, hbar=2)
    assert np.allclose(expected, obtained)


@pytest.mark.parametrize("M", range(2, 10))
def test_distinguishable_vacuum_probs_lossy(M):
    """test distinguishability code against combinatorial version for vacuum outcome (with loss)"""
    T = (unitary_group.rvs(M) * np.random.rand(M)) @ unitary_group.rvs(M)
    rs = np.random.rand(M)

    p1 = vacuum_prob_distinguishable(rs, T)
    pattern = [0] * M

    hbar = 2
    mu = np.zeros(2 * M)
    covs = []
    for i, r in enumerate(rs):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, T)
        covs.append(cov)
    p2 = pnr_prob(covs, pattern, hbar=hbar)

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

    hbar = 2
    mu = np.zeros(2 * M)
    covs = []
    for i, r in enumerate(rs_vals):
        r_vec = np.zeros(M)
        r_vec[i] = r
        S = squeezing(r_vec)
        cov = 0.5 * hbar * S @ S.T
        mu, cov = passive_transformation(mu, cov, T)
        covs.append(cov)
    obtained = pnr_prob(covs, pat_list, hbar=hbar)
    assert np.allclose(expected, obtained)


@pytest.mark.parametrize("S", [0.1, 0.4, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.1, 3.1])
def test_O_matrix(S, phi):
    """test code for forming O matrix from temporal modes"""
    F = [[np.array([np.exp(1j * phi), 0])], [np.array([S, np.sqrt(1 - S**2)])]]
    O_direct = np.array([[1, S * np.exp(-1j * phi)], [S * np.exp(1j * phi), 1]])
    O = O_matrix(F)
    assert np.allclose(O_direct, O)


@pytest.mark.parametrize("r", [0.1, 0.6, 1.3, 2.6])
@pytest.mark.parametrize("S", [0.1, 0.4, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.1, 3.1])
def test_orthonormal_basis(r, S, phi):
    """test code for forming orthonormal basis with two spatial modes with a single temporal mode squeezer in each with the same squeezing parameter.
    Variable overlap and phase."""
    rjs = [np.array([r]), np.array([r])]
    F = [[np.array([np.exp(1j * phi), 0])], [np.array([S, np.sqrt(1 - S**2)])]]
    O = np.array([[1, S * np.exp(-1j * phi)], [S * np.exp(1j * phi), 1]])
    chis, eps, W = orthonormal_basis(rjs, F=F, O=O)
    chi0 = np.exp(1j * phi) * np.array([np.sqrt(1 + S), np.sqrt(1 - S)]) / np.sqrt(2)
    chi1 = np.exp(1j * phi) * np.array([np.sqrt(1 - S), -np.sqrt(1 + S)]) / np.sqrt(2)
    W0 = np.array([[np.sqrt(1 + S), np.sqrt(1 - S)], [np.sqrt(1 - S), -np.sqrt(1 + S)]]) / np.sqrt(
        2
    )
    W1 = (
        np.array([[np.sqrt(1 + S), -np.sqrt(1 - S)], [np.sqrt(1 - S), np.sqrt(1 + S)]])
        * np.exp(-1j * phi)
        / np.sqrt(2)
    )
    assert np.allclose(chi0, chis[0])
    assert np.allclose(chi1, chis[1])
    assert np.allclose(np.array([r, 0]), eps[0])
    assert np.allclose(np.array([r, 0]), eps[1])
    assert np.allclose(np.abs(W0), np.abs(W[0]))
    assert np.allclose(np.abs(W1), np.abs(W[1]))


def test_orthonormal_basis_error():
    """Tests the value errors of orthonormal_basis"""

    rjs = 2 * [np.ones(2)]
    f = np.ones(5)
    F1 = [[f, f, f], [f, f]]
    with pytest.raises(
        ValueError,
        match="Length of F must equal the total number of Schmidt modes accross all spatial modes",
    ):
        orthonormal_basis(rjs, F=F1)

    F2 = [[f, f], [f, f]]
    O1 = unitary_group.rvs(4)
    with pytest.raises(ValueError, match="Both O and F were given but are not compatible"):
        orthonormal_basis(rjs, O=O1, F=F2)

    U1 = unitary_group.rvs(4)
    O2 = U1 + U1.T
    with pytest.raises(ValueError, match="O must be a Hermitian matrix"):
        orthonormal_basis(rjs, O=O2)

    U2 = unitary_group.rvs(5)
    O3 = U2 + U2.conj().T
    with pytest.raises(
        ValueError,
        match="Length of O must equal the total number of Schmidt modes accross all spatial modes",
    ):
        orthonormal_basis(rjs, O=O3)

    with pytest.raises(ValueError, match="Either F or O must be given"):
        orthonormal_basis(rjs)


@pytest.mark.parametrize("r", [0.1, 0.6, 1.3, 2.6])
@pytest.mark.parametrize("S", [0.1, 0.4, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.1, 3.1])
def test_state_prep(r, S, phi):
    """test code for forming state from orthonormalised system of 2 squeezers. Variable overlap and phase."""
    ### This test fails
    ### r = 0.1, S = 0.9, phi = 0.0
    ### r = 0.1, S = 0.9, phi = 0.9
    ### r = 0.1, S = 0.9, phi = 2.1
    ### r = 0.1, S = 0.9, phi = 3.1

    hbar = 2
    W0 = np.array([[np.sqrt(1 + S), np.sqrt(1 - S)], [np.sqrt(1 - S), -np.sqrt(1 + S)]]) / np.sqrt(
        2
    )
    W1 = (
        np.array([[np.sqrt(1 + S), -np.sqrt(1 - S)], [np.sqrt(1 - S), np.sqrt(1 + S)]])
        * np.exp(-1j * phi)
        / np.sqrt(2)
    )
    eps = [np.array([r, 0]), np.array([r, 0])]
    W = [W0, W1]
    covsp = state_prep(eps, W, thresh=0.0, hbar=hbar)
    covinit = (hbar / 2) * np.diag(
        np.array([np.exp(-2 * r), 1, np.exp(-2 * r), 1, np.exp(2 * r), 1, np.exp(2 * r), 1])
    )
    U = np.block([[W0.T.conj(), np.zeros(W0.shape)], [np.zeros(W1.shape), W1.T.conj()]])
    covorth = interferometer(U) @ covinit @ interferometer(U).T
    assert np.allclose(covsp, covorth)


def test_state_prep_error():
    """Tests the value errors of state_prep"""

    Wtemp = unitary_group.rvs(2)
    eps = [np.array([1, 0]), np.array([1, 0])]
    W = [Wtemp, Wtemp, Wtemp]
    with pytest.raises(ValueError, match="len of eps must be equal to len of W"):
        state_prep(eps, W)


@pytest.mark.parametrize("r", [0.1, 0.6, 1.3, 2.6])
@pytest.mark.parametrize("S", [0.1, 0.4, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.1, 3.1])
def test_prepare_cov(r, S, phi):
    """test code for forming state from orthonormalised system of 2 squeezers. Variable overlap and phase."""
    ### This test fails
    ### r = 0.1, S = 0.9, phi = 0.0
    ### r = 0.1, S = 0.9, phi = 0.9
    ### r = 0.1, S = 0.9, phi = 2.1
    ### r = 0.1, S = 0.9, phi = 3.1
    hbar = 2
    rjs = [np.array([r]), np.array([r])]
    O = np.array([[1, S * np.exp(-1j * phi)], [S * np.exp(1j * phi), 1]])
    U = unitary_group.rvs(len(rjs))
    cov = prepare_cov(rjs, U, O=O, thresh=0.0, hbar=hbar)
    W0 = np.array([[np.sqrt(1 + S), np.sqrt(1 - S)], [np.sqrt(1 - S), -np.sqrt(1 + S)]]) / np.sqrt(
        2
    )
    W1 = (
        np.array([[np.sqrt(1 + S), -np.sqrt(1 - S)], [np.sqrt(1 - S), np.sqrt(1 + S)]])
        * np.exp(-1j * phi)
        / np.sqrt(2)
    )
    covinit = (hbar / 2) * np.diag(
        np.array([np.exp(-2 * r), 1, np.exp(-2 * r), 1, np.exp(2 * r), 1, np.exp(2 * r), 1])
    )
    Uw = np.block([[W0.T.conj(), np.zeros(W0.shape)], [np.zeros(W1.shape), W1.T.conj()]])
    covorth = interferometer(Uw) @ covinit @ interferometer(Uw).T
    covu = implement_U(covorth, U)
    assert np.allclose(cov, covu)


def prepare_cov_error():
    """Tests the value errors of state_prep"""

    U = unitary_group.rvs(2)
    rjs1 = 3 * [np.ones(2)]
    with pytest.raises(
        ValueError, match="Unitary is the wrong size, it must act on all spatial modes"
    ):
        prepare_cov(rjs1, U)

    rjs2 = 2 * [np.ones(2)]
    with pytest.raises(ValueError, match="T must be have singular values <= 1"):
        prepare_cov(rjs2, 2 * U)

    with pytest.raises(ValueError, match="Either O or F must be supplied"):
        prepare_cov(rjs2, 0.9 * U)


@pytest.mark.parametrize("r", [0.1, 0.6, 1.3, 2.6])
@pytest.mark.parametrize("S", [0.1, 0.4, 0.7, 0.9])
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.1, 3.1])
def test_LO_overlaps(r, S, phi):
    """test code for local oscillator overlaps from orthonormalised system of 2 squeezers with a random LO shape. Variable overlap and phase"""
    F = [[np.array([np.exp(1j * phi), 0])], [np.array([S, np.sqrt(1 - S**2)])]]
    chis, _, _ = orthonormal_basis([np.array([r]), np.array([r])], F=F)
    LO_shape = np.array(
        [gauss(0, 1) * np.exp(1j * gauss(0, 1)), gauss(0, 1) * np.exp(1j * gauss(0, 1))]
    )
    LO_shape /= np.linalg.norm(LO_shape)
    assert np.allclose(
        LO_overlaps(chis, LO_shape),
        np.array([np.inner(LO_shape.conj(), chis[j]) for j in range(len(chis))]),
    )


def test_mixed_heralded_photon():
    """test code for generating heralded single photon state from squeezed states with 2 internal modes"""
    na = 1
    nb = 0.5
    ns = np.array([na, nb])
    rs = np.arcsinh(np.sqrt(ns))
    gs = ns / (1 + ns)
    cutoff = 5
    ps = np.array([g ** np.arange(cutoff) / (1 + n) for g, n in zip(gs, ns)])
    herald_val = 1
    dm_modea = np.array([ps[0, i] * ps[1, herald_val - i] for i in range(herald_val + 1)])
    dm_modeb = dm_modea[::-1]
    dm_modea = np.diag(dm_modea) / np.sum(dm_modea)
    dm_modeb = np.diag(dm_modeb) / np.sum(dm_modeb)

    F = [np.array([1, 0]), np.array([0, 1])]
    theta = np.pi / 4
    phi = -np.pi / 2
    U_TMSV = np.array(
        [
            [np.cos(theta), np.exp(-1j * phi) * np.sin(theta)],
            [-np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
        ]
    )
    cov, chis = prepare_cov([rs, rs], U_TMSV, F=[F, F])
    LO_overlapa = LO_overlaps(chis, chis[0])
    LO_overlapb = LO_overlaps(chis, chis[1])
    rho_a = density_matrix_single_mode(cov, {1: 1}, LO_overlap=LO_overlapa, cutoff=2)
    rho_b = density_matrix_single_mode(cov, {1: 1}, LO_overlap=LO_overlapb, cutoff=2)

    assert np.allclose(dm_modea, rho_a)
    assert np.allclose(dm_modeb, rho_b)


def test_pure_gkp():
    """test pure gkp state density matrix using 2 methods from the walrus against
    internal_modes.density_matrix_single_mode (but with only 1 temporal mode)"""

    m1, m2 = 5, 7
    params = np.array(
        [
            -1.38155106,
            -1.21699567,
            0.7798817,
            1.04182349,
            0.87702211,
            0.90243916,
            1.48353639,
            1.6962906,
            -0.24251599,
            0.1958,
        ]
    )
    sq_r = params[:3]
    bs_theta1, bs_theta2, bs_theta3 = params[3:6]
    bs_phi1, bs_phi2, bs_phi3 = params[6:9]
    sq_virt = params[9]

    S1 = squeezing(np.abs(sq_r), phi=np.angle(sq_r))
    BS1, BS2, BS3 = (
        beam_splitter(bs_theta1, bs_phi1),
        beam_splitter(bs_theta2, bs_phi2),
        beam_splitter(bs_theta3, bs_phi3),
    )
    Usymp1, Usymp2, Usymp3 = (
        expand(BS1, [0, 1], 3),
        expand(BS2, [1, 2], 3),
        expand(BS3, [0, 1], 3),
    )
    Usymp = Usymp3 @ Usymp2 @ Usymp1
    r2 = np.array([0, 0, sq_virt])
    S2 = squeezing(np.abs(r2), phi=np.angle(r2))
    Z = S2 @ Usymp @ S1
    cov = Z @ Z.T

    cutoff = 26
    mu = np.zeros([len(cov)])

    psi = state_vector(mu, cov, post_select={1: m1, 2: m2}, cutoff=cutoff)

    rho1 = np.outer(psi, psi.conj())
    rho1 /= np.trace(rho1)
    # get density matrix directly using The Walrus
    rho2 = density_matrix(mu, cov, post_select={1: m1, 2: m2}, cutoff=cutoff)
    rho2 /= np.trace(rho2)

    # get density matrix using new code
    rho3 = density_matrix_single_mode(cov, {1: m1, 2: m2}, cutoff=cutoff)
    rho3 /= np.trace(rho3)
    assert np.allclose(rho1, rho2, atol=2.5e-4)
    assert np.allclose(rho1, rho3, atol=4.7e-4)
    assert np.allclose(rho2, rho3, atol=4.8e-4)
    #### Note that the tolerances are higher than they should be.


def test_lossy_gkp():
    """
    test against thewalrus for lossy gkp state generation
    """

    m1, m2 = 5, 7
    params = np.array(
        [
            -1.38155106,
            -1.21699567,
            0.7798817,
            1.04182349,
            0.87702211,
            0.90243916,
            1.48353639,
            1.6962906,
            -0.24251599,
            0.1958,
        ]
    )
    sq_r = params[:3]
    bs_theta1, bs_theta2, bs_theta3 = params[3:6]
    bs_phi1, bs_phi2, bs_phi3 = params[6:9]
    sq_virt = params[9]

    S1 = squeezing(np.abs(sq_r), phi=np.angle(sq_r))
    BS1, BS2, BS3 = (
        beam_splitter(bs_theta1, bs_phi1),
        beam_splitter(bs_theta2, bs_phi2),
        beam_splitter(bs_theta3, bs_phi3),
    )
    Usymp1, Usymp2, Usymp3 = (
        expand(BS1, [0, 1], 3),
        expand(BS2, [1, 2], 3),
        expand(BS3, [0, 1], 3),
    )
    Usymp = Usymp3 @ Usymp2 @ Usymp1
    r2 = np.array([0, 0, sq_virt])
    S2 = squeezing(np.abs(r2), phi=np.angle(r2))
    Z = S2 @ Usymp @ S1
    cov = Z @ Z.T
    eta = 0.95
    T = np.diag([np.sqrt(eta)] * 3)
    mu = np.zeros([len(cov)])
    mu, cov_lossy = passive_transformation(mu, cov, T)
    cutoff = 26
    # get density matrix using The Walrus
    rho_loss1 = density_matrix(mu, cov_lossy, post_select={1: m1, 2: m2}, cutoff=cutoff)
    rho_loss1 /= np.trace(rho_loss1)

    # get density matrix using new code
    rho_loss2 = density_matrix_single_mode(cov_lossy, {1: m1, 2: m2}, cutoff=cutoff)
    rho_loss2 /= np.trace(rho_loss2)
    assert np.allclose(rho_loss1, rho_loss2, atol=2.7e-4)


def test_vac_schmidt_modes_gkp():
    """
    add vacuum schmidt modes and check it doesn't change the state
    """
    m1, m2 = 5, 7
    params = np.array(
        [
            -1.38155106,
            -1.21699567,
            0.7798817,
            1.04182349,
            0.87702211,
            0.90243916,
            1.48353639,
            1.6962906,
            -0.24251599,
            0.1958,
        ]
    )
    sq_r = params[:3]
    bs_theta1, bs_theta2, bs_theta3 = params[3:6]
    bs_phi1, bs_phi2, bs_phi3 = params[6:9]
    sq_virt = params[9]

    S1 = squeezing(np.abs(sq_r), phi=np.angle(sq_r))
    BS1, BS2, BS3 = (
        beam_splitter(bs_theta1, bs_phi1),
        beam_splitter(bs_theta2, bs_phi2),
        beam_splitter(bs_theta3, bs_phi3),
    )
    Usymp1, Usymp2, Usymp3 = (
        expand(BS1, [0, 1], 3),
        expand(BS2, [1, 2], 3),
        expand(BS3, [0, 1], 3),
    )
    Usymp = Usymp3 @ Usymp2 @ Usymp1
    r2 = np.array([0, 0, sq_virt])
    S2 = squeezing(np.abs(r2), phi=np.angle(r2))
    Z = S2 @ Usymp @ S1
    cov = Z @ Z.T
    mu = np.zeros([len(cov)])

    cutoff = 26
    psi = state_vector(mu, cov, post_select={1: m1, 2: m2}, cutoff=cutoff)

    rho1 = np.outer(psi, psi.conj())
    rho1 /= np.trace(rho1)

    M = 3
    K = 5
    big_cov = np.eye(2 * M * K, dtype=np.complex128)
    big_cov[::K, ::K] = cov

    rho_big = density_matrix_single_mode(big_cov, {1: m1, 2: m2}, cutoff=cutoff)
    rho_big /= np.trace(rho_big)

    assert np.allclose(rho1, rho_big, atol=4e-4)


def test_density_matrix_error():
    """Testing value errors in density_matrix_single_mode"""
    U = unitary_group.rvs(2)
    zs0 = np.array([np.arcsinh(np.sqrt(2.927))])
    zs1 = np.array([np.arcsinh(np.sqrt(2.592))])
    rjs = [zs0, zs1]

    O = np.identity(2, dtype=np.complex128)
    S = 0.8 * np.exp(0 * 1j)
    O[0, 1] = S.conj()
    O[1, 0] = S

    cov = prepare_cov(rjs, U, O=O, thresh=5e-3)
    pattern = {0: 1, 3: 2}

    with pytest.raises(
        ValueError, match="Keys of pattern must correspond to all but one spatial mode"
    ):
        density_matrix_single_mode(cov, pattern)

    K = cov.shape[0] // (2 * len(pattern))

    N = {0: 3}
    LO_overlap1 = np.ones(K + 1)
    LO_overlap1 /= np.linalg.norm(LO_overlap1)

    with pytest.raises(
        ValueError,
        match="Number of overlaps with LO must match number of internal modes",
    ):
        density_matrix_single_mode(cov, N, LO_overlap=LO_overlap1)

    LO_overlap2 = 2 * np.ones(K)

    with pytest.raises(ValueError, match="Norm of overlaps must not be greater than 1"):
        density_matrix_single_mode(cov, N, LO_overlap=LO_overlap2)


@pytest.mark.parametrize("cutoff", [8, 9])
def test_density_matrix(cutoff):
    """
    test generation of heralded density matrix against combinatorial calculation
    """
    U = unitary_group.rvs(2)

    N = {0: 3}

    efficiency = 1 * np.ones(2)

    noise = None

    n0 = 2.9267754749886055
    n1 = 2.592138225047742
    zs0 = np.array([np.arcsinh(np.sqrt(n0))])
    zs1 = np.array([np.arcsinh(np.sqrt(n1))])
    rjs = [zs0, zs1]

    O = np.identity(2, dtype=np.complex128)
    S = 0.8 * np.exp(0 * 1j)
    O[0, 1] = S.conj()
    O[1, 0] = S

    dm = heralded_density_matrix(
        rjs, O, U, N, efficiency=efficiency, noise=noise, Ncutoff=cutoff, thresh=5e-3
    )

    rho = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for i in range(cutoff):
        for j in range(cutoff):
            rho[i, j] = sum(dm[i, j, m, m] for m in range(cutoff))

    rho_norm = rho / np.trace(rho)

    cov = prepare_cov(rjs, U, O=O, thresh=5e-3)

    rho2_norm = density_matrix_single_mode(cov, N, cutoff=cutoff)

    assert np.allclose(rho_norm, rho2_norm, atol=1e-6, rtol=1e-6)


def test_density_matrix_LO():
    """
    test generation of heralded density matrix in LO basis against combinatorial calculation
    """
    U = unitary_group.rvs(2)

    N = {0: 3}

    efficiency = 1 * np.ones(2)

    noise = None

    n0 = 2.9267754749886055
    n1 = 2.592138225047742
    zs0 = np.array([np.arcsinh(np.sqrt(n0))])
    zs1 = np.array([np.arcsinh(np.sqrt(n1))])
    rjs = [zs0, zs1]

    S = 0.8
    phi = 0.5
    F = [[np.array([np.exp(1j * phi), 0])], [np.array([S, np.sqrt(1 - S**2)])]]
    LO_shape = np.array(
        [gauss(0, 1) * np.exp(1j * gauss(0, 1)), gauss(0, 1) * np.exp(1j * gauss(0, 1))]
    )
    LO_shape /= np.linalg.norm(LO_shape)

    cutoff = 8

    rho = heralded_density_matrix_LO(
        rjs,
        F,
        U,
        N,
        LO_shape,
        efficiency=efficiency,
        noise=noise,
        Ncutoff=cutoff,
        thresh=5e-3,
    )

    cov, chis = prepare_cov(rjs, U, F=F, thresh=5e-3)
    LO_overlap = LO_overlaps(chis, LO_shape)

    rho2 = density_matrix_single_mode(cov, N, LO_overlap=LO_overlap, cutoff=cutoff)

    assert np.allclose(rho, rho2, atol=1e-6, rtol=1e-6)
