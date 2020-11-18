# Version 0.15.0-dev

### New features

* Adds the function `random_banded_interferometer` to generate unitary matrices with a given bandwidth. [#208](https://github.com/XanaduAI/thewalrus/pull/208)

* Adds the function `tvd_cutoff_bounds` to calculate bounds in the total variation distance between a Fock-truncated and an ideal GBS distribution. [#210](https://github.com/XanaduAI/thewalrus/pull/210)

### Improvements

* The hafnians and loop hafnians of diagonal matrices are now calculated in polynomial time. [#212](https://github.com/XanaduAI/thewalrus/pull/212)

### Bug fixes

* Removes unnecessary `np.real_if_close` statements in `quantum/fock_tensors.py` causing the `probabilities` to not be normalized.

### Breaking changes

### Contributors

This release contains contributions from (in alphabetical order):

Nicolas Quesada

---

# Version 0.14.0

### New features

* Adds the function `find_classical_subsystem` that tries to find a subset of the modes with a classical covariance matrix. [#193](https://github.com/XanaduAI/thewalrus/pull/193)

* Adds the functions `mean_number_of_clicks` and `variance_number_of_clicks` that calculate the first and second statistical moments of the total number of clicks in a Gaussian state centered at the origin. [#195](https://github.com/XanaduAI/thewalrus/pull/195)

* Adds the module `decompositions` with the function `williamson` to find the Williamson decomposition of an even-size positive-semidefinite matrix. [#200](https://github.com/XanaduAI/thewalrus/pull/200)

* Adds the `loop_hafnian_quad` function to the Python interface for converting double into quad, do the calculations in quad and then return a double. [#201](https://github.com/XanaduAI/thewalrus/pull/201)

### Improvements

* Introduces a new faster and significantly more accurate algorithm to calculate power traces allowing to speed up the calculation of loop hafnians [#199](https://github.com/XanaduAI/thewalrus/pull/199)

* The `quantum` module has been refactored and organized into sub-modules. Several functions have been renamed, while the old names are being deprecated. [#197](https://github.com/XanaduAI/thewalrus/pull/197)

* Adds support for C++14 [#202](https://github.com/XanaduAI/thewalrus/pull/202)

* `pytest-randomly` is added to the test suite to improve testing and avoid stochastically failing tests. [#205](https://github.com/XanaduAI/thewalrus/pull/205)

* Modifies the function `input_validation` to use `np.allclose` for checking the symmetry of the input matrices. [#206](https://github.com/XanaduAI/thewalrus/pull/205)

* Modifies the function `_hafnian` to calculate efficiently loop hafnians of diagonal matrices. [#206](https://github.com/XanaduAI/thewalrus/pull/205)

### Breaking changes

* Removes the redundant function `normal_ordered_complex_cov`. [#194](https://github.com/XanaduAI/thewalrus/pull/194)

* Renames the function `mean_number_of_clicks` to be `mean_number_of_click_graph`. [#195](https://github.com/XanaduAI/thewalrus/pull/195)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Nicolas Quesada, Trevor Vincent


---


# Version 0.13.0

### New features

* Adds a new algorithm for hafnians of matrices with low rank. [#166](https://github.com/XanaduAI/thewalrus/pull/166)

* Adds a function to calculate the fidelity between two Gaussian quantum states. [#169](https://github.com/XanaduAI/thewalrus/pull/169)

* Adds a new module, `thewalrus.random`, to generate random unitary, symplectic and covariance matrices. [#169](https://github.com/XanaduAI/thewalrus/pull/169)

* Adds new functions `normal_ordered_expectation`, `photon_number_expectation` and `photon_number_squared_expectation` in `thewalrus.quantum` to calculate expectation values of products of normal ordered expressions and number operators and their squares. [#175](https://github.com/XanaduAI/thewalrus/pull/175)

* Adds the function `hafnian_sample_graph_rank_one` in `thewalrus.samples` to sample from rank-one adjacency matrices. [#174](https://github.com/XanaduAI/thewalrus/pull/174)

### Improvements

* Adds parallelization support using Dask for `quantum.probabilities`. [#161](https://github.com/XanaduAI/thewalrus/pull/161)

* Removes support for Python 3.5. [#163](https://github.com/XanaduAI/thewalrus/pull/163)

* Changes in the interface and speed ups in the functions in the `thewalrus.fock_gradients` module. [#164](https://github.com/XanaduAI/thewalrus/pull/164/files)

* Improves documentation of the multidimensional Hermite polynomials. [#166](https://github.com/XanaduAI/thewalrus/pull/166)

* Improves speed of `fock_tensor` when the symplectic matrix passed is also orthogonal. [#166](https://github.com/XanaduAI/thewalrus/pull/166)

### Bug fixes

* Fixes Numba decorated functions not rendering properly in the documentation. [#173](https://github.com/XanaduAI/thewalrus/pull/173)

* Solves the issue with `quantum` and `samples` not being rendered in the documentation or the TOC. [#173](https://github.com/XanaduAI/thewalrus/pull/173)

* Fix bug where quantum and samples were not showing up in the documentation. [#182](https://github.com/XanaduAI/thewalrus/pull/182)

### Breaking changes

* The functions in `thewalrus.fock_gradients` are now separated into functions for the gradients and the gates. Moreover, they are renamed, for instance `Dgate` becomes `displacement` and its gradient is now `grad_displacement`. [#164](https://github.com/XanaduAI/thewalrus/pull/164/files)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Josh Izaac, Filippo Miatto, Nicolas Quesada

---


# Version 0.12.0

### New features

* Adds the ability to calculate the mean number of photons in a given mode of a Gaussian state. [#148](https://github.com/XanaduAI/thewalrus/pull/148)

* Adds the ability to calculate the photon number distribution of a pure or mixed state using `generate_probabilities`. [#152](https://github.com/XanaduAI/thewalrus/pull/152)

* Allows to update the photon number distribution when undergoing loss by using `update_probabilities_with_loss`. [#152](https://github.com/XanaduAI/thewalrus/pull/152)

* Allows to update the photon number distribution when undergoing noise `update_probabilities_with_noise`. [#153](https://github.com/XanaduAI/thewalrus/pull/153)

* Adds a brute force sampler `photon_number_sampler` that given a (multi-)mode photon number distribution generates photon number samples. [#152](https://github.com/XanaduAI/thewalrus/pull/152)

* Adds the ability to perform the Autonne-Takagi decomposition of a complex-symmetric matrix using `autonne` from the `symplectic` module. [#154](https://github.com/XanaduAI/thewalrus/pull/154)

### Improvements


* Improves the efficiency of Hermite polynomial calculation in `hermite_multidimensional.hpp`. [#141](https://github.com/XanaduAI/thewalrus/pull/141)

* Implements parallelization with Dask for sampling from the Hafnian/Torontonian of a Gaussian state. [#145](https://github.com/XanaduAI/thewalrus/pull/145)

### Bug fixes

* Corrects the issue with hbar taking a default value when calling `state_vector`, `pure_state_amplitude`, and `density_matrix_element` [#149](https://github.com/XanaduAI/thewalrus/pull/149)

### Contributors

This release contains contributions from (in alphabetical order):


Theodor Isacsson, Nicolas Quesada, Kieran Wilkinson


---


# Version 0.11.0

### New features

* Introduces the renormalized hermite polynomials. These new polynomials improve the speed and accuracy of `thewalrus.quantum.state_vector` and `thewalrus.quantum.density_matrix` and also `hafnian_batched` and `hermite_multimensional` when called with the optional argument `renorm=True`. [#108](https://github.com/XanaduAI/thewalrus/pull/108)

* Adds functions for calculating the covariance for the photon number distribution of a Gaussian state including a function for the full covariance matrix. [#137](https://github.com/XanaduAI/thewalrus/pull/137)

* Adds support for Python 3.8. [#138](https://github.com/XanaduAI/thewalrus/pull/138)

### Improvements

* Updates the reference that should be used when citing The Walrus. [#102](https://github.com/XanaduAI/thewalrus/pull/102)

* Updates and improves the speed and accuracy of `thewalrus.quantum.fock_tensor`. [#107](https://github.com/XanaduAI/thewalrus/pull/107)

* Add OpenMP support to the repeated moment hafnian code. [#120](https://github.com/XanaduAI/thewalrus/pull/120)

* Improves speed of the functions in `hermite_multidimensional.hpp`. [#123](https://github.com/XanaduAI/thewalrus/pull/123)

* Improves speed of the functions in `thewalrus.fock_gradients` by doing calls to optimized functions in `hermite_multidimensional.hpp`. [#123](https://github.com/XanaduAI/thewalrus/pull/123)

* Further improves speed of the functions `thewalrus.fock_gradients` by writing explicit recursion relations for a given number of modes. [#129](https://github.com/XanaduAI/thewalrus/pull/129)

* Adds the functions `find_scaling_adjacency_matrix_torontonian` and `mean_number_of_clicks` that allow to fix the mean number of clicks when doing threshold detection sampling and allow to calculate the mean of clicks generated by a scaled adjacency matrix. [#136](https://github.com/XanaduAI/thewalrus/pull/136/)


### Bug fixes

* Corrects typos in the random number generation in the C++ unit tests. [#118](https://github.com/XanaduAI/thewalrus/pull/118)

* Corrects typos in describing the repeated-moment algorithm of Kan in the documentation. [#104](https://github.com/XanaduAI/thewalrus/pull/104)

* Removes paper.{md,pdf,bib} from the repository now that The Walrus paper is published in Journal of Open Source Software [#106](https://github.com/XanaduAI/thewalrus/pull/106)

* Updates the S2gate to use the correct definition. [#130](https://github.com/XanaduAI/thewalrus/pull/130)

* Corrects the issue with hbar taking a default value when calculating mu in the density matrix function [#134] (https://github.com/XanaduAI/thewalrus/pull/134)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Josh Izaac, Filippo Miatto, Nicolas Quesada, Trevor Vincent, Kieran Wilkinson


---

# Version 0.10.0

### New features
* Adds the function `thewalrus.quantum.fock_tensor` that returns the Fock space tensor corresponding to a Symplectic transformation in phase space. [#90](https://github.com/XanaduAI/thewalrus/pull/90)

* Adds the `thewalrus.fock_gradients` module which provides the Fock representation of a set of continuous-variable universal gates in the Fock representation and their gradients. [#96](https://github.com/XanaduAI/thewalrus/pull/96)

### Improvements

* Unifies return values of all symplectic gates in the `thewalrus.symplectic` module. [#81](https://github.com/XanaduAI/thewalrus/pull/81)

* Removes unnecessary citations in the tutorials. [#92](https://github.com/XanaduAI/thewalrus/pull/92)

* Improves the efficiency of the multidimensional Hermite polynomials implementation and simplifies a number of derived functions. [#93](https://github.com/XanaduAI/thewalrus/pull/93)

### Bug fixes

* Fixes a bug in the calculation of the state vector in `thewalrus.quantum.state_vector`. This bug was found and fixed while implementing `thewalrus.quantum.fock_tensor`. [#90](https://github.com/XanaduAI/thewalrus/pull/90)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Nicolas Quesada


---

# Version 0.9.0

### New features
* Adds a symplectic module `symplectic` which allows easy access to symplectic transformations and covariance matrices of Gaussian states. [#78](https://github.com/XanaduAI/thewalrus/pull/78)

### Improvements

* Adds a quick reference section in the documentation. [#75](https://github.com/XanaduAI/thewalrus/pull/75)

### Bug fixes

* Solves issue [#70](https://github.com/XanaduAI/thewalrus/issues/70) related to the index ordering in `thewalrus.quantum.density_matrix`.

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Nicolas Quesada


---

# Version 0.8.0

### New features

* Adds classical sampling of permanents of positive definite matrices in `csamples`. [#61](https://github.com/XanaduAI/thewalrus/pull/61)

* The Walrus gallery with examples of nongaussian state preparations that can be studied using the functions from the `quantum` module. [#55](https://github.com/XanaduAI/thewalrus/pull/55)

### Improvements

* Updates the bibliography of the documentation with recently published articles. [#51](https://github.com/XanaduAI/thewalrus/pull/51)

### Bug fixes

* Important bugfix in `quantum`. This bug was detected by running the tests in `test_integration`. [#48](https://github.com/XanaduAI/thewalrus/pull/48)

* Corrects the Makefile so that it uses the environment variable pointing to Eigen (if available). [#58](https://github.com/XanaduAI/thewalrus/pull/58)

* Removes any reference to Fortran in the Makefile. [#58](https://github.com/XanaduAI/thewalrus/pull/58)

### Contributors
This release contains contributions from (in alphabetical order):

Luke Helt, Josh Izaac, Soran Jahangiri, Nicolas Quesada, Guillaume Thekkadath


---

# Version 0.7.0

### New features

* Hafnian library has been renamed to The Walrus, and the low-level libhafnian C++ library renamed to `libwalrus`. [#34](https://github.com/XanaduAI/thewalrus/pull/34)

* Added a seed function `thewalrus.samples.seed`, in order to make the sampling algorithms deterministic for testing purposes. [#29](https://github.com/XanaduAI/thewalrus/pull/29)

* Added support for the batched hafnian; `hafnian_batched` (in Python) and `libwalrus::hermite_multidimensional_cpp` (in C++). This is a newly added algorithm that allows a batch of photon number statistics to be computed for a given quantum Gaussian state. [#21](https://github.com/XanaduAI/thewalrus/pull/21)

* Adds the ability to sample from Gaussian states with finite means. [#25](https://github.com/XanaduAI/thewalrus/pull/25)

### Improvements

* Permanent Fortran code was ported to C++, with improvements including support for quadruple precision and summation using the `fsum` algorithm. [#20](https://github.com/XanaduAI/thewalrus/pull/20)

* Reorganization of the repository structure; C++ source code folder has been renamed from `src` to `include`, C++ tests have been moved into their own top-level directory `tests`, and the `hafnian/lib` subpackage has been removed in favour of a top-level `thewalrus/libwalrus.so` module. [#34](https://github.com/XanaduAI/thewalrus/pull/34)

* Added additional references to the bibliography [#30](https://github.com/XanaduAI/thewalrus/pull/30)

* Adds documentation related to permanents, sampling states with nonzero displacement, sampling of classical states, and multidimensional Hermite polynomials and batched hafnians. [#27](https://github.com/XanaduAI/thewalrus/pull/27)

* Simplifies the hafnian sampling functions [#25](https://github.com/XanaduAI/thewalrus/pull/25)

* Test improvements, including replacing custom tolerance checks with `np.allclose()`. [#23](https://github.com/XanaduAI/thewalrus/pull/23)

### Bug fixes

* Minor typos corrected in the documentation. [#33](https://github.com/XanaduAI/thewalrus/pull/33) [#28](https://github.com/XanaduAI/thewalrus/pull/28) [#34](https://github.com/XanaduAI/thewalrus/pull/31)

### Contributors

This release contains contributions from (in alphabetical order):

Brajesh Gupt, Josh Izaac, Nicolas Quesada

---

# Version 0.6.1

### New features

- Added two new sampling functions in `hafnian.sample`, allowing efficient sampling from classical Gaussian states:

  * `hafnian_sample_classical_state`
  * `torontonian_sample_classical_state`

- Added functions to calculate the probability amplitudes directly for pure states:

  * `pure_state_amplitude`
  * `state_vector`

- Added utility functions to check the validity of covariance matrices:

  * `is_valid_cov`
  * `is_pure_cov`
  * `is_classical_cov`

### Improvements

* All functions in the `quantum` package now take as arguments xp-covariance matrices and vectors of means for consistency.

### Contributors

This release contains contributions from (in alphabetical order):

Nicolas Quesada

---

# Version 0.6.0

### New features

- Added a new sampling submodule `hafnian.sample`, allowing sampling from the underlying hafnian/Torontonian distribution of graphs/Gaussian states

- Documentation overhaul: now contains some of the best and most up-to-date information about hafnians, loop hafnians, and Torontonians

- C++ library has been significantly improved, tested, and refactored

### Improvements

- Ported the `hafnian_approx.F90` Fortran file to C++

- The Torontonian function is now parallelized via OpenMP

- Tests and the C++ header library have been refactored

- Addition of new C++ tests using Googletest

- C++ library is now documented via Doxygen; this is integrated directly into Sphinx

### Contributors

This release contains contributions from (in alphabetical order):

Brajesh Gupt, Josh Izaac, Nicolas Quesada
