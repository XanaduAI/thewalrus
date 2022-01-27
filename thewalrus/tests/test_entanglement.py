# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entanglement tests"""
# pylint: disable=no-self-use, assignment-from-no-return
import numpy as np

from thewalrus.quantum import entanglement_entropy, log_negativity
from thewalrus.random import random_interferometer
from thewalrus.symplectic import interferometer, passive_transformation


def random_cov(num_modes=200, pure=False, nonclassical=True):
    r"""Creates a random covariance matrix for testing.

    Args:
        num_modes (int): number of modes
        pure (bool): Gaussian state is pure
        nonclassical (bool): Gaussian state is nonclassical

    Returns:
        (array): a covariance matrix
    """
    M = num_modes
    O = interferometer(random_interferometer(M))

    if pure:
        eta = 1
    elif not pure:
        eta = 0.123

    r = 1.234
    if nonclassical:
        # squeezed inputs
        cov_in = np.diag(np.concatenate([np.exp(2 * r) * np.ones(M), np.exp(-2 * r) * np.ones(M)]))
    elif not nonclassical and pure:
        # vacuum inputs
        cov_in = np.eye(2 * M)
    elif not nonclassical and not pure:
        # squashed inputs
        cov_in = np.diag(np.concatenate([np.exp(2 * r) * np.ones(M), np.ones(M)]))

    cov_out = O @ cov_in @ O.T
    _, cov = passive_transformation(
        np.zeros([len(cov_out)]), cov_out, np.sqrt(eta) * np.identity(len(cov_out) // 2)
    )

    return cov


def test_entanglement_entropy_bipartition():
    """Tests if both substates of a bipartition will yield the same
    entanglement entropy"""
    M = 200
    cov = random_cov(num_modes=M, pure=True, nonclassical=True)
    modes_A = 111
    # make a list that contains all modes but 111:
    modes_B = list(range(M))
    modes_B.pop(111)
    E_A = entanglement_entropy(cov, modes_A=modes_A)
    E_B = entanglement_entropy(cov, modes_A=modes_B)
    assert np.isclose(E_A, E_B)


def test_log_negativity_bipartition():
    """Tests if both substates of a bipartition will yield the same log
    negativity"""
    M = 200
    cov = random_cov(num_modes=M, pure=False, nonclassical=True)
    modes_A = 111
    # make a list that contains all modes but 111:
    modes_B = list(range(M))
    modes_B.pop(111)
    E_A = log_negativity(cov, modes_A=modes_A)
    E_B = log_negativity(cov, modes_A=modes_B)
    assert np.isclose(E_A, E_B)


def test_entanglement_entropy_classical():
    """Tests if a classical state will return entanglement entropy zero"""
    cov = random_cov(pure=True, nonclassical=False)
    E = entanglement_entropy(cov, modes_A=[10, 20, 30, 40])
    assert np.isclose(E, 0)


def test_log_negativity_classical():
    """Tests if a classical state will return log negativity zero"""
    cov = random_cov(pure=False, nonclassical=False)
    E = log_negativity(cov, modes_A=[10, 20, 30, 40])
    assert np.isclose(E, 0)


def test_entanglement_entropy_split():
    """Tests the equivalence of the two alternative inputs ``modes_A`` and
    ``split``"""
    cov = random_cov(pure=True, nonclassical=True)
    split = 123
    E_0 = entanglement_entropy(cov, split=split)
    E_1 = entanglement_entropy(cov, modes_A=range(split))
    assert np.isclose(E_0, E_1)


def test_log_negativity_split():
    """Tests the equivalence of the two alternative inputs ``modes_A`` and
    ``split``"""
    cov = random_cov(pure=False, nonclassical=True)
    split = 123
    E_0 = log_negativity(cov, split=split)
    E_1 = log_negativity(cov, modes_A=range(split))
    assert np.isclose(E_0, E_1)
