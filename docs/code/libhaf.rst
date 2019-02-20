Hafnian C++ library
===================

The Hafnian C++ interface is provided as a header-only library, :download:`hafnian.hpp <../../src/hafnian.hpp>`, which can be included at the top of your source file:

.. code-block:: cpp

    #include <hafnian.hpp>

The following templated functions are then available for use within the ``hafnian`` namespace.

.. note:: The Hafnian C++ interface only provides functions for calculating the hafnian. For calculating the permanent via Ryser's algorithm, either use the :mod:`Python interface <hafnian>`, or the Fortran interface.

Example
-------

For instance, consider the following example :download:`example.cpp <../../src/example.cpp>`, which calculates the loop hafnian of several all ones matrices:

.. code-block:: cpp

    #include <iostream>
    #include <complex>
    #include <vector>
    #include <hafnian.hpp>


    int main() {
        int nmax = 10;

        for (int m = 1; m <= nmax; m++) {
            // create a 2m*2m all ones matrix
            int n = 2*m;
            std::vector<std::complex<double>> mat(n*n, 1.0);

            // calculate the hafnian
            std::complex<double> hafval = hafnian::loop_hafnian(mat);
            // print out the result
            std::cout << hafval << std::endl;
        }

        return 0;
    };


This can be compiled using the gcc ``g++`` compiler as follows,

.. code-block:: console

    $ g++ example.cpp -o example -std=c++11 -O3 -Wall -I/path/to/hafnian.hpp -I/path/to/Eigen -fopenmp

where ``/path/to/hafnian.hpp`` is the path to the directory containing ``hafnian.hpp``, ``/path/to/Eigen`` is the path to the Eigen C++ linear algebra header library, and the ``-fopenmp`` flag instructs the compiler to parallelize the compiled program using OpenMP.

Additionally, you may instruct Eigen to simply act as a 'frontend' to an installed LAPACKE library. To do so, you must pass additional flags:

.. code-block:: console

    $ g++ example.cpp -o example -std=c++11 -O3 -Wall -I/path/to/hafnian.hpp -I/path/to/Eigen -fopenmp \
    -DLAPACKE -llapacke -lblas

Below, the main interface (available as templated functions) as well as all auxiliary functions are summarized and listed.

.. note::

    If compiling using the ``clang`` compiler provided by Xcode on MacOS, OpenMP is natively supported, however the ``libomp.so`` library must be installed and linked to separately. One approach is to use the Homebrew packaging manager:

    .. code-block:: console

        $ brew install eigen libomp
        $ clang example.cpp -o example -O3 -Wall -fPIC -shared -Xpreprocessor -fopenmp -lomp \
        -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/


Main interface
--------------

The following functions are intended as the main interface to the C++ Hafnian library. All three support parallelization via OpenMP.


.. rst-class:: longtable docutils

=============================    ==============================================
:cpp:func:`hafnian_recursive`    Returns the hafnian of a matrix using the recursive algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
:cpp:func:`hafnian`              Returns the hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
:cpp:func:`loop_hafnian`         Returns the loop hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
:cpp:func:`hafnian_rpt`          Returns the hafnian of a matrix with repeated rows and columns using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.
=============================    ==============================================


Auxiliary functions
-------------------

The following auxiliary functions are used in the calculation of the hafnian.


.. rst-class:: longtable docutils

============================ =====
:cpp:func:`powtrace`            Calculates the power trace of matrix ``z``.
:cpp:func:`dec2bin`             Convert a base-10 integer to character vector representing the corresponding binary number.
:cpp:func:`find2`               Convert a base-10 integer ``x`` to character vector ``dst`` of length ``len`` representing the corresponding binary number.
:cpp:func:`do_chunk`            Calculates and sums parts :math:`X,X+1,\dots,X+\text{chunksize}` using the Cygan and Pilipczuk formula for the hafnian of matrix ``mat``.
:cpp:func:`do_chunk_loops`      Calculates and sums parts :math:`X,X+1,\dots,X+\text{chunksize}` using the Cygan and Pilipczuk formula for the loop hafnian of matrix ``mat``.
============================ =====



Code details
------------



.. cpp:function:: template\<typename T> T hafnian_recursive(std::vector<T> &mat)

    Returns the hafnian of a matrix using the recursive algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`, where it is labelled as 'Algorithm 2'.

    .. note:: Modified with permission from https://github.com/eklotek/Hafnian.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.


.. cpp:function:: template\<typename T> T hafnian(std::vector<T> &mat)

    Returns the hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.



.. cpp:function:: template\<typename T> T loop_hafnian(std::vector<T> &mat)

    Returns the loop hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.


.. cpp:function:: template\<typename T> T hafnian_rpt(std::vector<T> &mat, std::vector<int> &rpt, bool use_eigen=true)

    Returns the hafnian of a matrix using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.

    Note that this algorithm, while generally slower than the the above, can be more efficient
    in the cases where the matrix has repeated rows and columns.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.
    :param std\:\:vector<int> &rpt: a vector of integers, representing the number of times eacg row/column in ``mat`` is repeated. For example, ``mat = {1}`` and ``rpt = {6}`` represents a :math:`6\times 6` matrix of all ones.
    :param bool use_eigen: whether to use the Eigen linear algebra library to compute matrix multiplication. If ``true`` (default) then Eigen is used, if ``false`` then pure C++ loops are used.


.. cpp:function:: template\<typename T> T loop_hafnian_rpt(std::vector<T> &mat, std::vector<int> &rpt, bool use_eigen=true)

    Returns the loop hafnian of a matrix using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.

    Note that this algorithm, while generally slower than the the above, can be more efficient
    in the cases where the matrix has repeated rows and columns.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered symmetric matrix.
    :param std\:\:vector<int> &rpt: a vector of integers, representing the number of times eacg row/column in ``mat`` is repeated. For example, ``mat = {1}`` and ``rpt = {6}`` represents a :math:`6\times 6` matrix of all ones.
    :param bool use_eigen: whether to use the Eigen linear algebra library to compute matrix multiplication. If ``true`` (default) then Eigen is used, if ``false`` then pure C++ loops are used.


.. cpp:function:: std::vector<std::complex<double>> powtrace(std::vector<std::complex<double>> &z, int n, int l)

    Given a (flattened) complex matrix ``z`` of dimensions :math:`n\times n`, calculates :math:`Tr(z^j)~\forall~1\leq j\leq l`.

    .. note:: this function makes use of either the :cpp:class:`Eigen::ComplexEigenSolver` or the LAPACKE routine :cpp:func:`zgees` depending on the compilation.

    :param std\:\:vector<std\:\:complex<double>> z: a flattened complex vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered matrix.
    :param int n: size of the matrix ``z``.
    :param int l: maximum matrix power when calculating the power trace.
    :return: returns a vector containing the power traces of matrix ``z`` to power :math:`1\leq j \leq l`.
    :rtype: std::vector<std::complex<double>>

.. cpp:function:: std::vector<double> powtrace(std::vector<double> &z, int n, int l)

    Given a (flattened) real matrix ``z`` of dimensions :math:`n\times n`, calculates :math:`Tr(z^j)~\forall~1\leq j\leq l`.

    .. note:: this function makes use of either the :cpp:class:`Eigen::EigenSolver` or the LAPACKE routine :cpp:func:`dgees` depending on the compilation.

    :param std\:\:vector<std\:\:complex<double>> z: a flattened real vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered matrix.
    :param int n: size of the matrix ``z``.
    :param int l: maximum matrix power when calculating the power trace.
    :return: returns a vector containing the power traces of matrix ``z`` to power :math:`1\leq j \leq l`.
    :rtype: std::vector<double>


.. cpp:function:: void dec2bin(char* dst, unsigned long long int x, unsigned char len)

    Convert a base-10 integer ``x`` to character vector ``dst`` of length ``len`` representing the corresponding binary number.

    :param char* dst: resulting character array representing the resulting binary digits.
    :param unsigned long long int x: base-10 input.
    :param unsigned char len: length of the array ``dst``.


.. cpp:function:: unsigned char find2(char* dst, unsigned char len, unsigned char* pos)

    Given a string of length ``len`` it finds in which positions it has a 1 and stores its position ``i``, as ``2*i`` and ``2*i+1`` in consecutive slots of the array ``pos``.

    It also returns (twice) the number of ones in array ``dst``.

    :param char* dst: character array representing binary digits.
    :param unsigned char len: length of the array ``dst``.
    :param unsigned char* len: resulting character array of length ``2*len`` storing the indices at which ``dst`` contains the values 1.
    :return: returns twice the number of ones in array ``dst``.`.
    :rtype: unsigned char


.. cpp:function:: template\<typename T> T do_chunk(std::vector<T> &mat, int n, unsigned long long int X, unsigned long long int chunksize)

    This function calculates and sums parts :math:`X,X+1,\dots,X+\text{chunksize}` using the Cygan and Pilipczuk formula for the hafnian of matrix ``mat``.

    Note that if ``X=0`` and ``chunksize=pow(2.0, n/2)``, then the full hafnian is calculated.

    This function uses OpenMP (if available) to parallelize the reduction.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered matrix.

    :param int n: size of the matrix represented by ``z``.
    :param unsigned long long int X: the starting integer of the summation loop.
    :param unsigned long long int chunksize: the number of consecutive summations to perform.
    :return: returns the sum of parts :math:`X,X+1,\dots,X+\text{chunksize}` of the hafnian of matrix ``z``.
    :rtype: T


.. cpp:function:: template\<typename T> T do_chunk_loops(std::vector<T> &mat, std::vector<T> &C, std::vector<T> &D, int n, unsigned long long int X, unsigned long long int chunksize)

    This function calculates and sums parts :math:`X,X+1,\dots,X+\text{chunksize}` using the Cygan and Pilipczuk formula for the loop hafnian of matrix ``mat``.

    Note that if ``X=0`` and ``chunksize=pow(2.0, n/2)``, then the full loop hafnian is calculated.

    This function uses OpenMP (if available) to parallelize the reduction.

    :tparam T: template parameter accepts any (signed) numeric type, including ``int``, ``long int``, ``long long int``, ``float``, ``double``, ``std::complex<float>``, ``std::complex<double>``, etc.

    :param std\:\:vector<T> &mat: a flattened vector of size :math:`n^2`, representing an :math:`n\times n` row-ordered matrix.
    :param std\:\:vector<T> &C: contains the diagonal elements of matrix ``z``.
    :param std\:\:vector<T> &D: the diagonal elements of matrix ``z``, with every consecutive pair swapped (i.e., ``C[0]==D[1]``, ``C[1]==D[0]``, ``C[2]==D[3]``, ``C[3]==D[2]``, etc.).

    :param int n: size of the matrix represented by ``z``.
    :param unsigned long long int X: the starting integer of the summation loop.
    :param unsigned long long int chunksize: the number of consecutive summations to perform.
    :return: returns the sum of parts :math:`X,X+1,\dots,X+\text{chunksize}` of the loop hafnian of matrix ``z``.
    :rtype: T
