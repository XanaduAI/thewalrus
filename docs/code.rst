Code documentation and overiew
===============================

The Hafnian library contains three components:

* The Python interface :mod:`hafnian`. This provides access to the C library hafnian algorithms through Python. See the next page for more details on the Python interface.

* The underlying C library, ``lhafnian.so`` for the hafnian calculation of complex matrices, and ``rlhafnian.so``, for the hafnian calculation of real matrices.

  - If you would like to access the C library directly, download the source code and navigate to the ``src`` directory. The C library can be installed by running

    .. code-block:: bash

      $ make library
      $ make rlibrary

    See the file :download:`timing.c <../src/timing.c>` for an example of how the hafnian library can be accessed directly from C code.

* In addition, two auxiallary Octave functions are provided: :download:`octave/hafnian.m <../octave/hafnian.m>` and :download:`octave/loophafnian.m <../octave/loophafnian.m>`.
