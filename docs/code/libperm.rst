Permanent Fortran interface
===========================

The Permanent Fortran interface allows calculation of the permanent of both real and complex matrices, using Ryser's algorithm.

.. note:: The Permanent Fortran interface only provides functions for calculating the permanent. For calculating the hafnian, either use the :mod:`Python interface <hafnian>`, or the C++ interface.

Compiling the library
---------------------

To compile the Fortran Permanent library, navigate to the top directory and simply run

.. code-block:: console

    $ make libperm

This will generate the shared library ``src/libperm.so``, as well as the Fortran module files ``src/perm.mod``, ``src/vars.mod``, and ``src/kinds.mod``.

Example
-------

For instance, consider the following example :download:`example.f90 <../../src/example.f90>`, which calculates the permanent of several all ones matrices:

.. code-block:: cpp

    program permanent
        use perm
        implicit none

        integer, parameter :: nmax = 10;
        integer :: n, m

        real(8) :: p
        real(8), allocatable :: mat(:, :)

        do m = 1, nmax
            ! create a 2m*2m all ones matrix
            allocate(mat(2*m, 2*m))
            mat = 1.d0
            ! calculate the permanent
            call re(mat, p)
            ! print out the result
            write(*,*)p
            deallocate(mat)
        end do
    end program


This can be compiled using the ``gfortran`` compiler as follows,

.. code-block:: console

    $ gfortran src/example.f90 -o example -O3 -Wall -Isrc/ -Lsrc/ -lperm

where the flags ``-I`` and ``-L`` are required to allow the compiler to find both the ``.mod`` module files and the ``libperm.so`` library respectively.

Below, the main interface (available as templated functions) as well as all auxiliary functions are summarized and listed.


Main interface
--------------

The following subroutines are intended as the main interface to the Fortran Permanent library. All support parallelization via OpenMP.


.. rst-class:: longtable docutils

================  ==============================================
:cpp:func:`re`    Returns the permanent of a real matrix using Ryser's algorithm.
:cpp:func:`comp`  Returns the permanent of a complex matrix using Ryser's algorithm.
================  ==============================================



Code details
------------



.. cpp:function:: subroutine re(mat, permanent)

    Returns the permanent of a double precision real matrix using Ryser's algorithm.

    :param real(8) mat(\:, \:): *(input)* a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.
    :param real(8) permanent: *(output)* the resulting permanent.

.. cpp:function:: subroutine comp(mat, permanent)

    Returns the permanent of a double precision real matrix using Ryser's algorithm.

    :param complex(8) mat(\:, \:): *(input)* a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.
    :param complex(8) permanent: *(output)* the resulting permanent.
