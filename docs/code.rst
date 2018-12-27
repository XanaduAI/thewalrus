Code documentation and overiew
===============================

The Hafnian library contains a Python frontend interface, and backend C/Fortran libraries:

Frontend
---------

* The Python interface provides access to the C library hafnian algorithms, as well as the Fortran library permanent algorithms.

Backends
--------

* The C library, ``libhaf.so`` for the hafnian calculation of complex, real, and integer matrices.

* The Fortran library, ``libperm.so`` for the permanent calculation of real matrices.

You can also use the C++ Hafnian library directly in your C++ projects - just ensure that the ``src`` folder is in your include path, and add

.. code-block:: cpp

	#include <hafnian.hpp>

at the top of your C++ source file. See the file :download:`timing.cpp <../src/timing.cpp>`, as well as the corresponding Makefile, for an example of how the hafnian library can be accessed directly from C code.

Alternatively, if you install the Hafnian package as a python wheel using pip, you can link against the static pre-built library provided.

Octave
------

In addition, two auxiallary Octave functions are provided: :download:`octave/hafnian.m <../octave/hafnian.m>` and :download:`octave/loophafnian.m <../octave/loophafnian.m>`.
