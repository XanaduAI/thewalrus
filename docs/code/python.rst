.. role:: html(raw)
   :format: html

Hafnian Python interface
========================

.. currentmodule:: hafnian

This is the top level module of the Hafnian Python interface,
containing the functions :func:`hafnian` and :func:`perm`.
These wrapper functions determine,
based on the input matrix, whether to use the complex or real
C++/Fortran library.

Algorithm terminology
---------------------

Eigenvalue hafnian algorithm
    The algorithm described in
    *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*,
    `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
    This algorithm scales like :math:`\mathcal{O}(n^3 2^{n/2})`, and supports caclulation of
    the loop hafnian.

Recursive hafnian algorithm
    The algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
    This algorithm scales like :math:`\mathcal{O}(n^4 2^{n/2})`. This algorithm does not
    currently support the loop hafnian.

Repeating hafnian algorithm
    The algorithm described in *From moments of sum to moments of product*,
    `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.
    This method is more efficient for matrices with repeated rows and columns, and supports caclulation of
    the loop hafnian.


Python wrappers
---------------

.. autosummary::
    hafnian
    hafnian_repeated
    hafnian_approx
    tor
    perm
    permanent_repeated
    det
    kron_reduced
    version

.. automodule:: hafnian
    :members:
    :inherited-members:
    :private-members:


----

Low level hafnian interface
---------------------------

For more advanced usage, direct access to the hafnian C++ library are provided
via the functions:


.. rst-class:: longtable docutils

=========================== ================================
:func:`haf_real`             Returns the hafnian or loop hafnian of a real symmetric matrix A by directly querying the C++ hafnian library.
:func:`haf_complex`          Returns the hafnian or loop hafnian of a complex symmetric matrix A by directly querying the C++ hafnian library.
:func:`haf_int`              Returns the hafnian of an integer matrix A by directly querying the C++ hafnian library recursive algorithm.
:func:`haf_rpt_real`         Returns the hafnian of a real matrix A via the C++ hafnian library using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.
:func:`haf_rpt_complex`      Returns the hafnian of a complex matrix A via the C++ hafnian library using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.
:func:`hafnian_nonneg`       Returns the approximate hafnian of a real matrix A with non-negative entries via the Fortran hafnian approximation library.
=========================== ================================

.. note::

    As these functions interface directly with the underlying C++ library, the arguments will need to be the exact type requested, or an error will occur. For example, if a function expects a ``np.float64``
    array argument, but the array to be passed is an integer array, this will have to be converted manually:

    >>> A = np.ones([6, 6])
    >>> haf_real(np.float64(A))

.. py:function:: haf_int(A)

    Returns the hafnian of an integer symmetric matrix A by directly querying the C++ hafnian library recursive algorithm.

    Modified with permission from https://github.com/eklotek/Hafnian.

    .. note:: Currently does not support calculation of the loop hafnian.

    :param array A: a ``np.int64``, square, symmetric array of even dimensions.
    :returns: the hafnian of matrix A.
    :rtype: np.float64


.. py:function:: haf_real(A, loop=False, recursive=True, quad=True)

    Returns the hafnian or loop hafnian of a real symmetric matrix A by directly querying the C++ hafnian library.

    :param array A: a ``np.float64``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :param bool quad: If ``True``, the input matrix is cast to a ``long double complex``
        matrix internally for a quadruple precision hafnian computation.
    :return: the hafnian of matrix A
    :rtype: ``np.float64``


.. py:function:: haf_complex(A, loop=False, recursive=True, quad=True)

    Returns the hafnian or loop hafnian of a complex symmetric matrix A by directly querying the C++ hafnian library.

    :param array A: a ``np.complex128``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :param bool quad: If ``True``, the input matrix is cast to a ``long double complex``
        matrix internally for a quadruple precision hafnian computation.
    :return: the hafnian of matrix A
    :rtype: ``np.complex128``


.. py:function:: haf_rpt_real(A, rpt, mu=None, loop=False, use_eigen=True)

    Returns the hafnian of a real matrix A via the C++ hafnian library
    using the algorithm described in algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.

    :param array A: a ``np.float64``, square, symmetric :math:`n\times n` array.
    :param array rpt: a ``np.int32`` length-:math:`n` array, corresponding to the number of times each row/column of matrix A is repeated.
    :param array mu: a ``np.float64`` vector of length :math:`N` representing the vector of means/displacement. If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this only affects the loop hafnian.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool use_eigen: If ``True``, he Eigen linear algebra library is used for matrix multiplication.

.. py:function:: haf_rpt_complex(A, rpt, mu=None, loop=False, use_eigen=True)

    Returns the hafnian of a real matrix A via the C++ hafnian library
    using the algorithm described in algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.

    :param array A: a ``np.complex128``, square, symmetric :math:`n\times n` array.
    :param array rpt: a ``np.int32`` length-:math:`n` array, corresponding to the number of times each row/column of matrix A is repeated.
    :param array mu: a ``np.complex128`` vector of length :math:`N` representing the vector of means/displacement. If not provided, ``mu`` is set to the diagonal of matrix ``A``. Note that this only affects the loop hafnian.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool use_eigen: If ``True``, he Eigen linear algebra library is used for matrix multiplication

.. py:function:: hafnian_nonneg(A, nsample)

    Returns the approximate hafnian of a real matrix A with non-negative entries via the Fortran hafnian approximation library.

    :param array A: a ``np.float``, square, symmetric :math:`n\times n` array.
    :param int nsample: the number of times to sample the determinant of various
        submatrices in order to approximate the hafnian. The larger the number
        of samples, the more accurate the result, at the cost of a longer computation time.

Low level torontonian interface
-------------------------------

For more advanced usage, direct access to the torontonian Fortran library are provided
via the functions:

.. rst-class:: longtable docutils

=========================== =================================================
:func:`tor_complex`         Returns the torontonian of complex matrix A directly querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.tor`.
:func:`det_real`            Returns the determinant of real matrix A calculated in quadruple precision by querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.det_real`.
:func:`det_complex`         Returns the determinant of complex matrix A calculated in quadruple precision by querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.det_complex`.
=========================== =================================================

.. py:function:: tor_complex(A)

    Returns the torontonian of complex matrix A by directly querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.tor`.

    :param array A: a square array.
    :returns: the torontonian of matrix A.
    :rtype: complex


.. py:function:: det_real(A)

    Returns the determinant of real matrix A calculated in quadruple precision by querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.det_real`.

    .. note::

        This function uses a modified version of the Fortran LINPACK_Q library
        in order to calculate the determinant of the matrix using quadruple precision.

    :param array A: an ``np.float`` square array.
    :returns: the determinant of matrix A.
    :rtype: float

.. py:function:: det_complex(A)

    Returns the determinant of complex matrix A calculated in quadruple precision by querying the Fortran subroutine :func:`hafnian.lib.libtor.torontonian.det_complex`.

    .. note::

        This function uses a modified version of the Fortran LINPACK_Q library
        in order to calculate the determinant of the matrix using quadruple precision.

    :param array A: an ``np.complex`` square array.
    :returns: the determinant of matrix A.
    :rtype: complex



Low level permanent interface
-----------------------------

For more advanced usage, direct access to the permanent Fortran library are provided
via the functions:


.. rst-class:: longtable docutils

=========================== =================================================
:func:`perm_real`           Returns the permanent of real matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.
:func:`perm_complex`        Returns the permanent of complex matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.
=========================== =================================================


.. py:function:: perm_real(A)

    Returns the permanent of real matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.

    :param array A: a ``np.float``, square array.
    :returns: the permanent of matrix A.
    :rtype: float


.. py:function:: perm_complex(A)

    Returns the permanent of complex matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.comp`.

    :param array A: a ``np.complex``, square array.
    :returns: the permanent of matrix A.
    :rtype: complexs
