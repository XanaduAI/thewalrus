.. role:: html(raw)
   :format: html

.. automodule:: hafnian
    :members:
    :inherited-members:
    :private-members:

:html:`<h2>Low level hafnian interface</h2>`

For more advanced usage, direct access to the hafnian C++ library are provided
via the functions:


.. rst-class:: longtable docutils

=========================== ================================
:func:`haf_real`             Returns the hafnian or loop hafnian of a real symmetric matrix A by directly querying the C++ hafnian library.
:func:`haf_complex`          Returns the hafnian or loop hafnian of a complex symmetric matrix A by directly querying the C++ hafnian library.
:func:`haf_int`              Returns the hafnian of an integer matrix A by directly querying the C++ hafnian library recursive algorithm.
:func:`haf_rpt_real`         Returns the hafnian of a real matrix A via the C++ hafnian library using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.
:func:`haf_rpt_complex`      Returns the hafnian of a complex matrix A via the C++ hafnian library using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.
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


.. py:function:: haf_real(A, loop=False, recursive=False)

    Returns the hafnian or loop hafnian of a real symmetric matrix A by directly querying the C++ hafnian library.

    :param array A: a ``np.float64``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :return: the hafnian of matrix A
    :rtype: ``np.float64``


.. py:function:: haf_complex(A, loop=False, recursive=False)

    Returns the hafnian or loop hafnian of a complex symmetric matrix A by directly querying the C++ hafnian library.

    :param array A: a ``np.complex128``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :return: the hafnian of matrix A
    :rtype: ``np.complex128``


.. py:function:: haf_rpt_real(A, rpt, loop=False, use_eigen=True)

    Returns the hafnian of a real matrix A via the C++ hafnian library
    using the algorithm described in algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.

    :param array A: a ``np.float64``, square, symmetric :math:`n\times n` array.
    :param array rpt: a ``np.int32`` length-:math:`n` array, corresponding to the number of times each row/column of matrix A is repeated.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool use_eigen: If ``True``, he Eigen linear algebra library is used for matrix multiplication.

.. py:function:: haf_rpt_complex(A, rpt, loop=False, use_eigen=True)

    Returns the hafnian of a real matrix A via the C++ hafnian library
    using the algorithm described in algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__. This method is more efficient for matrices with repeated rows and columns.

    :param array A: a ``np.complex128``, square, symmetric :math:`n\times n` array.
    :param array rpt: a ``np.int32`` length-:math:`n` array, corresponding to the number of times each row/column of matrix A is repeated.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool use_eigen: If ``True``, he Eigen linear algebra library is used for matrix multiplication


:html:`<h2>Low level permanent interface</h2>`

For more advanced usage, direct access to the permanent Fortran library are provided
via the functions:


.. rst-class:: longtable docutils

=========================== =================================================
:func:`perm_real`           Returns the permanent of real matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.
:func:`perm_complex`        Returns the permanent of complex matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.
=========================== =================================================


.. note::

    As these functions interface directly with the underlying Fortran library, the arguments will need to be the exact type requested, or an error will occur. For example, if a function expects a ``np.float64``
    array argument, but the array to be passed is an integer array, this will have to be converted manually:

    >>> A = np.ones([6, 6])
    >>> perm_real(np.float64(A))


.. py:function:: perm_real(A)

    Returns the permanent of real matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.

    :param array A: a ``np.float64``, square array.
    :returns: the permanent of matrix A.
    :rtype: np.float64


.. py:function:: perm_complex(A)

    Returns the permanent of complex matrix A using the Ryser algorithm by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.comp`.

    :param array A: a ``np.complex64``, square array.
    :returns: the permanent of matrix A.
    :rtype: np.complex128
