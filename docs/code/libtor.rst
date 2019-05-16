Torontonian Fortran library
===========================

The torontonian Fortran interface allows calculation of the torontonian.

.. note:: The torontonian Fortran interface only provides functions for calculating the torontonian. For calculating the hafnian, either use the :mod:`Python interface <hafnian>`, or the C++ interface.

Compiling the library
---------------------

To compile the Fortran torontonian library, navigate to the top directory and simply run

.. code-block:: console

    $ make libtor

This will generate the shared library ``src/libtor.so``, as well as the Fortran module files ``src/perm.mod``, ``src/vars.mod``, and ``src/kinds.mod``.

Example
-------

For instance, consider the following example :download:`example.f90 <../../src/example.f90>`, which calculates the torontonian of several all ones matrices:

.. code-block:: cpp

    program tor
        use torontonian
        implicit none

        integer, parameter :: nmax = 10;
        integer :: n, m

        complex(8) :: p
        complex(8), allocatable :: mat(:, :)

        do m = 1, nmax
            ! create a 2m*2m all ones matrix
            allocate(mat(2*m, 2*m))
            mat = 1.d0
            ! calculate the torontonian
            p = tor(mat)
            ! print out the result
            write(*,*)p
            deallocate(mat)
        end do
    end program


This can be compiled using the ``gfortran`` compiler as follows,

.. code-block:: console

    $ gfortran src/example.f90 -o example -O3 -Wall -Isrc/ -Lsrc/ -ltor

where the flags ``-I`` and ``-L`` are required to allow the compiler to find both the ``.mod`` module files and the ``libtor.so`` library respectively.

Below, the main interface (available as templated functions) as well as all auxiliary functions are summarized and listed.


Main interface
--------------

The following subroutines are intended as the main interface to the Fortran torontonian library. All support parallelization via OpenMP.


.. rst-class:: longtable docutils

================  ==============================================
:cpp:func:`tor`   Returns the torontonian of a complex matrix in quadruple precision
================  ==============================================



Code details
------------



.. cpp:function:: function tor(mat)

    Returns the torontonian of a quadruple precision complex matrix.

    :param complex(8) mat(\:, \:): *(input)* an array of size :math:`n\times n`
    :param complex(8) torontonian: *(output)* the resulting torontonian.
