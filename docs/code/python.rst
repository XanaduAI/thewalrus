.. automodule:: hafnian
    :members:
    :inherited-members:
    :private-members:


.. py:function:: haf_real(A, loop=False)

    Returns the hafnian of real symmetric matrix A by directly querying the C library :func:`hafnian.lib.libhaf.haf_real`.

    Args:
        A (array): a np.float64, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.float64: the hafnian of matrix A


.. py:function:: haf_complex(A, loop=False)

    Returns the hafnian of complex symmetric matrix A by directly querying the C library :func:`hafnian.lib.libhaf.haf_complex`.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        loop (bool): If ``True``, the loop hafnian is returned. Default false.

    Returns:
        np.complex128: the hafnian of matrix A


.. py:function:: perm_real(A)

    Returns the permanent of real matrix A by directly querying the Fortran library :func:`hafnian.lib.libperm.perm.re`.

    Args:
        A (array): an np.float64 square array.

    Returns:
        np.float64: the permanent of matrix A.


.. py:function:: perm_complex(A)

    Returns the permanent of complex matrix A by directly querying the Fortran library :func:`hafnian.lib.libperm.perm.comp`.

    Args:
        A (array): an np.complex64 square array.

    Returns:
        np.complex128: the permanent of matrix A.