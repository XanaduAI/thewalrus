# Version 0.10.0-dev

### New features

### Improvements

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):


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
