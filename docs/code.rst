Code documentation and overiew
===============================

The Hafnian library contains a Python frontend interface, and backend C/Fortran libraries:

Frontend
---------

* The Python interface provides access to the C library hafnian algorithms, as well as the Fortran library permanent algorithms.

Backends
--------

* The C libraries, ``libhaf.so`` for the hafnian calculation of complex matrices, and ``librhaf.so``, for the hafnian calculation of real matrices.

* The Fortran library, ``libperm.so`` for the permanent calculation of real matrices.

If you would like to access the C  or Fortran libraries directly, download the source code and navigate to the ``src`` directory. The corresponding libraries can be installed by running

.. code-block:: bash

  $ make library
  $ make rlibrary
  $ make libperm

See the file :download:`timing.c <../src/timing.c>` for an example of how the hafnian library can be accessed directly from C code.

Alternatively, if you install the Hafnian package as a python wheel using pip, you can link against the static pre-built library provided.

Octave
------

In addition, two auxiallary Octave functions are provided: :download:`octave/hafnian.m <../octave/hafnian.m>` and :download:`octave/loophafnian.m <../octave/loophafnian.m>`.
