Overview
========

The Walrus contains a Python interface, and low-level C++ ``libwalrus`` library.

Python interface
----------------

* The :mod:`thewalrus` Python interface provides access to various highly optimized hafnian, permanent, and torontonian algorithms

* The :mod:`thewalrus.quantum` submodule provides access to various utility functions that act on Gaussian quantum states

* The :mod:`thewalrus.samples` submodule provides access to algorithms to sample from the hafnian or the torontonian of Gaussian quantum states

* The :mod:`thewalrus.symplectic` submodule provides access to a convenient set of symplectic matrices and utility functions to manipulate them

* The :mod:`thewalrus.random` submodule provides access to random unitary, symplectic and covariance matrices

* The :mod:`thewalrus.fock_gradients` submodule provides access to the Fock representation of certain continuous-variable gates and their gradients

* The :mod:`thewalrus.reference` submodule provides access to pure-Python reference implementations of the hafnian, loop hafnian, and torontonian


Low-level libraries
-------------------

The low-level ``libwalrus`` :ref:`C++ library <libwalrus_cpp>` is a header-only library containing various parallelized algorithms for computing the hafnian, loop hafnian, permanent, and Torontonian calculation of complex, real, and integer matrices. This library is used under-the-hood by the Python :mod:`thewalrus` module.

You can also use the ``libwalrus`` library directly in your C++ projects - just ensure that the ``include`` folder is in your include path, and add

.. code-block:: cpp

	#include <libwalrus.hpp>

at the top of your C++ source file. See the file :download:`example.cpp <../examples/example.cpp>`, as well as the corresponding Makefile, for an example of how the ``libwalrus`` library can be accessed directly from C++ code.

Alternatively, if you install the The Walrus package as a python wheel using ``pip``, you can link against the static pre-built library provided.

Octave
------

In addition, two auxiallary Octave functions are provided: :download:`octave/hafnian.m <../octave/hafnian.m>` and :download:`octave/loophafnian.m <../octave/loophafnian.m>`.
