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
r"""
Classical Sampling Algorithms
=============================

.. currentmodule:: thewalrus.csamples

This submodule provides access to classical sampling algorithms for thermal states going through
an interferometer specified by a real orthogonal matrix. The quantum state to be sampled is
specified by a positive semidefinite real matrix and a mean photon number. The algorithm implemented
here was first derived in

* Saleh Rahimi-Keshari, Austin P Lund, and Timothy C Ralph.
  "What can quantum optics say about computational complexity theory?" `Physical Review Letters, 114(6):060501, (2015).
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.060501>`_

For more precise details of the implementation see

* Soran Jahangiri, Juan Miguel Arrazola, Nicolás Quesada, and Nathan Killoran.
  "Point processes with Gaussian boson sampling" `Phys. Rev. E 101, 022134, (2020).
  <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.022134>`_.



Summary
-------
.. autosummary::
    rescale_adjacency_matrix_thermal
    rescale_adjacency_matrix
    generate_thermal_samples

Code details
------------
"""
# pylint: disable=too-many-arguments

import numpy as np
from scipy.optimize import root_scalar

def rescale_adjacency_matrix_thermal(
    A, n_mean, check_positivity=True, check_symmetry=True, rtol=1e-05, atol=1e-08
):
    r"""Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that encodes it has
    a total mean photon number n_mean for thermal sampling.

    Args:
        A (array): Adjacency matrix, assumed to be positive semi-definite and real
        n_mean (float): Mean photon number of the Gaussian state
        check_positivity (bool): Checks if the matrix A is positive semidefinite
        check_symmetry (bool): Checks if the matrix is symmetric
        rtol: relative tolerance for the checks
        atol: absolute tolerance for the checks

    Returns:
        tuple(array,array): rescaled eigenvalues and eigenvectors of the matrix A
    """
    return rescale_adjacency_matrix(
        A, n_mean, 1.0, check_positivity=check_positivity, check_symmetry=check_symmetry, rtol=rtol, atol=atol
    )

def rescale_adjacency_matrix(
    A, n_mean, scale, check_positivity=True, check_symmetry=True, rtol=1e-05, atol=1e-08
):
    r"""Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that encodes it has
    a total mean photon number n_mean.

    Args:
        A (array): Adjacency matrix, assumed to be positive semi-definite and real
        n_mean (float): Mean photon number of the Gaussian state
        scale (float): Determines whether to rescale the matrix for thermal sampling (scale = 1.0)
            or for squashed sampling (scale = 2.0)
        check_positivity (bool): Checks if the matrix A is positive semidefinite
        check_symmetry (bool): Checks if the matrix is symmetric
        rtol: relative tolerance for the checks
        atol: absolute tolerance for the checks

    Returns:
        tuple(array,array): rescaled eigenvalues and eigenvectors of the matrix A
    """
    ls, O = np.linalg.eigh(A)
    ls[np.abs(ls) < atol] = 0.0

    if check_symmetry is True:
        assert np.allclose(A, A.T, rtol=rtol, atol=atol)

    if check_positivity is True:
        assert np.all(ls >= 0)

    max_sv = ls[-1]
    a_lim = 0.0
    b_lim = 1 / (atol + scale * max_sv)
    x_init = 0.5 * b_lim

    def mean_photon_number(x, vals):
        r"""Returns the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A where vals are the eigenvalues of positive semidefinite A

        Args:
            x (float): Scaling parameter
            vals (array): Eigenvalues of the matrix A

        Returns:
            n_mean: Mean photon number in the Gaussian state
        """

        vals2 = scale * x * vals
        n = np.sum(vals2 / (1.0 - vals2))
        return n


    # The following function is implicitly tested in test_rescaling_thermal
    def grad_mean_photon_number(x, vals): # pragma: no cover
        r"""Returns the gradient of the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A with respect to x.
        vals are the eigenvalues of A

        Args:
            x (float): Scaling parameter
            vals (array): Eigenvalues of the matrix A

        Returns:
            d_n_mean: Derivative of the mean photon number in the Gaussian state
                with respect to x
        """
        vals1 = scale * x * vals
        dn = np.sum((scale * vals) / (1 - vals1) ** 2)
        return dn

    f = lambda x: mean_photon_number(x, ls) - n_mean
    df = lambda x: grad_mean_photon_number(x, ls)
    res = root_scalar(f, fprime=df, x0=x_init, bracket=(a_lim, b_lim))
    assert res.converged
    return res.root * ls, O

def generate_thermal_samples(ls, O, num_samples=1):
    r"""Generates samples of the Gaussian state in terms of the mean photon number parameter ls and the interferometer O.

        Args:
            ls (array): squashing parameters
            O (array): Orthogonal matrix representing the interferometer
            num_samples: Number of samples to generate

        Returns:
            list(array: samples
    """
    rs = 0.5 * ls / (1 - ls)
    return [
        np.random.poisson(
            np.abs(
                O @ (np.random.normal(0, np.sqrt(rs)) + 1j * np.random.normal(0, np.sqrt(rs)))
            )
            ** 2
        )
        for _ in range(num_samples)
    ]
