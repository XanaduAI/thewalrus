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
Functions for calculating properties of states of distinguihable squeezed states of
light having passed through an interferometer.
"""

import numpy as np
from .photon_number_distributions import _squeezed_state_distribution


def sample(T, rs, n_samples=100, input_cutoff=50):
    """
    Calculates a possible resultant photon number distribution when distinguishable
    squeezers are sent into an interferometer.

    Args:
        T (numpy.ndarray): interferometer transmission matrix
        rs (numpy.ndarray): input squeezing parameters
        n_samples (int): number of samples to return
        input_cutoff (int): Fock basis photon number cutoff

    Returns:
        outputs (numpy.ndarray): a resultant photon number distribution
    """
    M = T.shape[0]
    abs2T = (T * T.conj()).real

    detection_probs = abs2T.sum(0)
    probs = np.array([abs2T[:, i] / detection_probs[i] for i in range(M)])

    p_n = np.array([_squeezed_state_distribution(r, cutoff=input_cutoff) for r in rs])
    p_n = np.array([p / p.sum() for p in p_n])

    outputs = np.empty((n_samples, M), dtype=np.int64)
    for sample in range(n_samples):
        output = np.zeros(M, dtype=np.int64)
        for i in range(M):
            n = np.random.choice(np.arange(input_cutoff), p=p_n[i])
            n_detected = np.random.binomial(n, min(1, detection_probs[i]))
            if n_detected > 0:
                output_modes_i = np.random.choice(
                    np.arange(M), p=probs[i], size=n_detected
                )
                output_i = np.bincount(output_modes_i, minlength=M)
                output += output_i
        outputs[sample] = output
    return outputs


def number_means(T, rs):
    """
    Calculates the resultant vector of mean photon numbers when distinguishable
    squeezers are sent into an interferometer.

    Args:
        T (numpy.ndarray): interferometer transmission matrix
        rs (numpy.ndarray): input squeezing parameters

    Returns:
        (numpy.ndarray): resultant mean photon numbers
    """
    n = len(rs)
    return np.array(
        [
            np.sum([(np.sinh(rs[k]) * np.abs(T[i, k])) ** 2 for k in range(n)])
            for i in range(n)
        ]
    )


def number_cov(T, rs):
    """
    Calculates the resultant photon number covariance matrix when distinguishable
    squeezers are sent into an interferometer.

    Args:
        T (numpy.ndarray): interferometer transmission matrix
        rs (numpy.ndarray): input squeezing parameters

    Returns:
        (numpy.ndarray): resultant covariance matrix
    """
    absT = np.abs(T) ** 2
    covN = (absT * np.sinh(rs) ** 4) @ absT.T
    covM = (absT * (0.5 * np.sinh(2 * rs)) ** 2) @ absT.T
    return covN + covM + np.diag(distinguishable_number_means(T, rs))
