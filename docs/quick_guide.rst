.. _quick_guide:

Quick guide
###########

This section provides a quick guide to find which function does what in The Walrus.


================================================================================ =============================
**What you want**                                                                **Corresponding function**
-------------------------------------------------------------------------------- -----------------------------
Numerical hafnian                                                                :func:`thewalrus.hafnian()`
Symbolic hafnian                                                                 :func:`thewalrus.reference.hafnian()`
Hafnian of a matrix with repeated rows and columns                               :func:`thewalrus.hafnian_repeated()`
Hafnians of all possible reductions of a given matrix                            :func:`thewalrus.hafnian_batched()`
Hafnian samples of a Gaussian state                                              :func:`thewalrus.samples.hafnian_sample_state()`
Torontonian samples of a Gaussian state                                          :func:`thewalrus.samples.torontonian_sample_state()`
Hafnian samples of a graph                                                       :func:`thewalrus.samples.hafnian_sample_graph()`
Torontonian samples of a graph                                                   :func:`thewalrus.samples.torontonian_sample_graph()`
All probability amplitudes of a pure Gaussian state                              :func:`thewalrus.quantum.state_vector()`
All matrix elements of a general Gaussian state                                  :func:`thewalrus.quantum.density_matrix()`
A particular probability amplitude of pure Gaussian state                        :func:`thewalrus.quantum.pure_state_amplitude()`
A particular matrix element of a general Gaussian state                          :func:`thewalrus.quantum.density_matrix_element()`
The Fock representation of a Gaussian unitary                                    :func:`thewalrus.quantum.fock_tensor()`
================================================================================ =============================

Note that all the hafnian functions listed above generalize to loop hafnians.
