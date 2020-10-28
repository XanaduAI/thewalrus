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
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    pure_state_amplitude
    state_vector
    density_matrix_element
    density_matrix
    fock_tensor
    probabilities
    loss_mat
    update_probabilities_with_loss
    update_probabilities_with_noise
    find_classical_subsystem
    tvd_cutoff_bounds

Adjacency matrices
^^^^^^^^^^^^^^^^^^

.. autosummary::

    adj_scaling
    adj_scaling_torontonian
    adj_to_qmat

Gaussian checks
^^^^^^^^^^^^^^^

.. autosummary::

    is_valid_cov
    is_pure_cov
    is_classical_cov
    fidelity

Conversions
^^^^^^^^^^^

.. autosummary::

    reduced_gaussian
    Xmat
    Qmat
    Covmat
    Amat
    complex_to_real_displacements
    real_to_complex_displacements

Means and variances
^^^^^^^^^^^^^^^^^^^

.. autosummary::

    photon_number_mean
    photon_number_mean_vector
    photon_number_covar
    photon_number_covmat
    photon_number_expectation
    photon_number_squared_expectation
    normal_ordered_expectation
    mean_clicks
    variance_clicks

Photon number distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    pure_state_distribution

Details
^^^^^^^
"""
import warnings
import functools

from .fock_tensors import (
    pure_state_amplitude,
    state_vector,
    density_matrix_element,
    density_matrix,
    fock_tensor,
    probabilities,
    loss_mat,
    update_probabilities_with_loss,
    update_probabilities_with_noise,
    find_classical_subsystem,
    tvd_cutoff_bounds,
)

from .adjacency_matrices import (
    adj_scaling,
    adj_scaling_torontonian,
    adj_to_qmat,
)

from .gaussian_checks import (
    is_valid_cov,
    is_pure_cov,
    is_classical_cov,
    fidelity,
)

from .conversions import (
    reduced_gaussian,
    Xmat,
    Qmat,
    Covmat,
    Amat,
    complex_to_real_displacements,
    real_to_complex_displacements,
)

from .means_and_variances import (
    photon_number_mean,
    photon_number_mean_vector,
    photon_number_covar,
    photon_number_covmat,
    photon_number_expectation,
    photon_number_squared_expectation,
    normal_ordered_expectation,
    mean_clicks,
    variance_clicks,
)

from .photon_number_distributions import (
    pure_state_distribution,
)

__all__ = [
    "pure_state_amplitude",
    "state_vector",
    "density_matrix_element",
    "density_matrix",
    "fock_tensor",
    "probabilities",
    "loss_mat",
    "update_probabilities_with_loss",
    "update_probabilities_with_noise",
    "find_classical_subsystem",
    "adj_scaling",
    "adj_scaling_torontonian",
    "adj_to_qmat",
    "is_valid_cov",
    "is_pure_cov",
    "is_classical_cov",
    "fidelity",
    "reduced_gaussian",
    "Xmat",
    "Qmat",
    "Covmat",
    "Amat",
    "complex_to_real_displacements",
    "real_to_complex_displacements",
    "photon_number_mean",
    "photon_number_mean_vector",
    "photon_number_covar",
    "photon_number_covmat",
    "photon_number_expectation",
    "photon_number_squared_expectation",
    "normal_ordered_expectation",
    "mean_clicks",
    "variance_clicks",
    "pure_state_distribution",
]

def deprecate(new_func):
    """Wrapper for deprecated functions to raise warning"""

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"This function is deprecated and will be removed. Use {new_func.__name__} instead.",
            DeprecationWarning
        )
        return new_func(*args, **kwargs)
    return wrapper

# old names for functions; remove in due time
Means = deprecate(real_to_complex_displacements)
Beta = deprecate(complex_to_real_displacements)
total_photon_num_dist_pure_state = deprecate(pure_state_distribution)
gen_Qmat_from_graph = deprecate(adj_to_qmat)
find_scaling_adjacency_matrix = deprecate(adj_scaling)
find_scaling_adjacency_matrix_torontonian = deprecate(adj_scaling_torontonian)
mean_number_of_clicks = deprecate(mean_clicks)
variance_number_of_clicks = deprecate(variance_clicks)
