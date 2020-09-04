# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Quantum algorithms
==================

.. currentmodule:: thewalrus.quantum

This submodule provides access to various utility functions that act on Gaussian
quantum states.

For more details on how the hafnian relates to various properties of Gaussian quantum
states, see:

* Kruse, R., Hamilton, C. S., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "Detailed study of Gaussian boson sampling." `Physical Review A 100, 032326 (2019)
  <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.032326>`_

* Hamilton, C. S., Kruse, R., Sansoni, L., Barkhofen, S., Silberhorn, C., & Jex, I.
  "Gaussian boson sampling." `Physical Review Letters, 119(17), 170501 (2017)
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.170501>`_

* Quesada, N.
  "Franck-Condon factors by counting perfect matchings of graphs with loops."
  `Journal of Chemical Physics 150, 164113 (2019) <https://aip.scitation.org/doi/10.1063/1.5086387>`_

* Quesada, N., Helt, L. G., Izaac, J., Arrazola, J. M., Shahrokhshahi, R., Myers, C. R., & Sabapathy, K. K.
  "Simulating realistic non-Gaussian state preparation." `Physical Review A 100, 022341 (2019)
  <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.022341>`_


Fock states and tensors
-----------------------

.. autosummary::

    pure_state_amplitude
    state_vector
    density_matrix_element
    density_matrix
    fock_tensor
    probabilities
    update_probabilities_with_loss
    update_probabilities_with_noise
    normal_ordered_expectation
    fidelity

Details
^^^^^^^

.. autofunction::
    pure_state_amplitude

.. autofunction::
    state_vector

.. autofunction::
    density_matrix_element

.. autofunction::
    density_matrix

.. autofunction::
    fock_tensor

.. autofunction::
    probabilities

.. autofunction::
    update_probabilities_with_loss

.. autofunction::
    update_probabilities_with_noise

.. autofunction::
    normal_ordered_expectation

.. autofunction::
    fidelity


Utility functions
-----------------

.. autosummary::

    reduced_gaussian
    Xmat
    Qmat
    Covmat
    Amat
    Beta
    Means
    prefactor
    find_scaling_adjacency_matrix
    mean_number_of_clicks
    mean_number_of_clicks_graph
    find_scaling_adjacency_matrix_torontonian
    gen_Qmat_from_graph
    photon_number_mean
    photon_number_mean_vector
    photon_number_covar
    photon_number_covmat
    is_valid_cov
    is_pure_cov
    is_classical_cov
    find_classical_subsystem
    total_photon_num_dist_pure_state
    gen_single_mode_dist
    gen_multi_mode_dist
    normal_ordered_complex_cov


Details
^^^^^^^
"""

from .fock_states_and_tensors import (
    pure_state_amplitude,
    state_vector,
    density_matrix_element,
    density_matrix,
    fock_tensor,
    probabilities,
    update_probabilities_with_loss,
    update_probabilities_with_noise,
    fidelity,

    find_classical_subsystem,
    prefactor,
    loss_mat,
)

from .adjacency_matrices import (
    mean_number_of_clicks,
    mean_number_of_clicks_graph,
    variance_number_of_clicks,
    find_scaling_adjacency_matrix,
    find_scaling_adjacency_matrix_torontonian,
    gen_Qmat_from_graph,
)

from .covariance_matrices import (
    reduced_gaussian,

    Xmat,
    Qmat,
    Covmat,
    Amat,
    normal_ordered_expectation,
    Beta,
    Means,

    is_valid_cov,
    is_pure_cov,
    is_classical_cov
)

from .means_and_variances import (
    photon_number_mean,
    photon_number_mean_vector,
    photon_number_covar,
    photon_number_covmat,
    photon_number_expectation,
    photon_number_squared_expectation,
)

from .photon_number_distributions import (
    total_photon_num_dist_pure_state,
    gen_single_mode_dist,
    gen_multi_mode_dist,
)

__all__ = [
    "pure_state_amplitude",
    "state_vector",
    "density_matrix_element",
    "density_matrix",
    "fock_tensor",
    "probabilities",
    "update_probabilities_with_loss",
    "update_probabilities_with_noise",
    "normal_ordered_expectation",
    "fidelity",
]
