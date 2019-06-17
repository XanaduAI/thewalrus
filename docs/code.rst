Overview
========

The Hafnian library contains a Python interface, and low-level C/Fortran libraries.

Python interface
----------------

* The :mod:`hafnian` Python interface provides access to various highly optimized hafnian, permanent, and torontonian algorithms

* The :mod:`hafnian.quantum` submodule provides access to various utility functions that act on Gaussian quantum states

* The :mod:`hafnian.samples` submodule provides access to algorithms to sample from the hafnian or the torontonian of Gaussian quantum states

* The :mod:`hafnian.reference` submodule provides access to pure-Python reference implementations of the hafnian, loop hafnian, and torontonian


Low-level libraries
-------------------

* A :ref:`C++ library <hafnian_cpp>` containing various parallelized algorithms for computing the hafnian, loop hafnian, and Torontonian calculation of complex, real, and integer matrices.

* A :ref:`Fortran library <perm_f90>` for the permanent calculation of real and complex matrices.

You can also use the C++ Hafnian library directly in your C++ projects - just ensure that the ``src`` folder is in your include path, and add

.. code-block:: cpp

	#include <hafnian.hpp>

at the top of your C++ source file. See the file :download:`example.cpp <../src/example.cpp>`, as well as the corresponding Makefile, for an example of how the hafnian library can be accessed directly from C++ code.

Alternatively, if you install the Hafnian package as a python wheel using ``pip``, you can link against the static pre-built library provided.

Octave
------

In addition, two auxiallary Octave functions are provided: :download:`octave/hafnian.m <../octave/hafnian.m>` and :download:`octave/loophafnian.m <../octave/loophafnian.m>`.
