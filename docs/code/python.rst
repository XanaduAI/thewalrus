.. automodule:: hafnian
    :members:
    :inherited-members:
    :private-members:


.. py:function:: haf_int(A)

    Returns the hafnian of real symmetric matrix A by directly querying the C++ function :func:`hafnian.lib.libhaf.haf_int`.

    Modified with permission from https://github.com/eklotek/Hafnian.

    .. note:: Currently does not support calculation of the loop hafnian.

    :param array A: a ``np.int64``, square, symmetric array of even dimensions.
    :returns: the hafnian of matrix A.
    :rtype: np.float64


.. py:function:: haf_real(A, loop=False, recursive=False)

    Returns the hafnian of real symmetric matrix A by directly querying the C++ function :func:`hafnian.lib.libhaf.haf_real`.

    :param array A: a ``np.float64``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :return: the hafnian of matrix A
    :rtype: ``np.float64``


.. py:function:: haf_complex(A, loop=False, recursive=False)

    Returns the hafnian of complex symmetric matrix A by directly querying the C++ function :func:`hafnian.lib.libhaf.haf_complex`.

    :param array A: a ``np.complex128``, square, symmetric array of even dimensions.
    :param bool loop: If ``True``, the loop hafnian is returned. Default false.
    :param bool recursive: If ``True``, the recursive algorithm is used. Note:
        the recursive algorithm does not currently support the loop hafnian.
    :return: the hafnian of matrix A
    :rtype: ``np.complex128``


.. py:function:: perm_real(A)

    Returns the permanent of real matrix A by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.re`.

    :param array A: a ``np.float64``, square array.
    :returns: the permanent of matrix A.
    :rtype: np.float64


.. py:function:: perm_complex(A)

    Returns the permanent of complex matrix A by directly querying the Fortran subroutine :func:`hafnian.lib.libperm.perm.comp`.

    :param array A: a ``np.complex64``, square array.
    :returns: the permanent of matrix A.
    :rtype: np.complex128
