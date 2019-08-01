.. _libwalrus_cpp:

The libwalrus C++ library
=========================

The ``libwalrus`` C++ library is provided as a header-only library, :download:`libwalrus.hpp <../../include/libwalrus.hpp>`, which can be included at the top of your source file:

.. code-block:: cpp

    #include <libwalrus.hpp>

The following templated functions are then available for use within the ``libwalrus`` namespace.

Example
-------

For instance, consider the following example :download:`example.cpp <../../examples/example.cpp>`, which calculates the loop hafnian of several all ones matrices:

.. code-block:: cpp

    #include <iostream>
    #include <complex>
    #include <vector>
    #include <libwalrus.hpp>


    int main() {
        int nmax = 10;

        for (int m = 1; m <= nmax; m++) {
            // create a 2m*2m all ones matrix
            int n = 2 * m;
            std::vector<std::complex<double>> mat(n * n, 1.0);

            // calculate the hafnian
            std::complex<double> hafval = libwalrus::loop_hafnian(mat);
            // print out the result
            std::cout << hafval << std::endl;
        }

        return 0;
    };

This can be compiled using the gcc ``g++`` compiler as follows,

.. code-block:: console

    $ g++ example.cpp -o example -std=c++11 -O3 -Wall -I/path/to/libwalrus.hpp -I/path/to/Eigen -fopenmp

where ``/path/to/libwalrus.hpp`` is the path to the directory containing ``libwalrus.hpp``, ``/path/to/Eigen`` is the path to the Eigen C++ linear algebra header library, and the ``-fopenmp`` flag instructs the compiler to parallelize the compiled program using OpenMP.

Additionally, you may instruct Eigen to simply act as a 'frontend' to an installed LAPACKE library. To do so, you must pass additional flags:

.. code-block:: console

    $ g++ example.cpp -o example -std=c++11 -O3 -Wall -I/path/to/libwalrus.hpp -I/path/to/Eigen \
    -fopenmp -DLAPACKE -llapacke -lblas

Below, the main interface (available as templated functions) as well as all auxiliary functions are summarized and listed.

.. note::

    If compiling using the ``clang`` compiler provided by Xcode on MacOS, OpenMP is natively supported, however the ``libomp.so`` library must be installed and linked to separately. One approach is to use the Homebrew packaging manager:

    .. code-block:: console

        $ brew install eigen libomp
        $ clang example.cpp -o example -O3 -Wall -fPIC -shared -Xpreprocessor -fopenmp -lomp \
        -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/


Main interface
--------------

The following functions are intended as the main interface to the C++ ``libwalrus`` library. Most support parallelization via OpenMP.


.. rst-class:: longtable docutils

=============================================================  ==============================================
:cpp:func:`libwalrus::hafnian_recursive`                       Returns the hafnian of a matrix using the recursive algorithm described in *Counting perfect matchings as fast as Ryser* :cite:`bjorklund2012counting`.
:cpp:func:`libwalrus::hafnian`                                 Returns the hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
:cpp:func:`libwalrus::loop_hafnian`                            Returns the loop hafnian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
:cpp:func:`libwalrus::hafnian_rpt`                             Returns the hafnian of a matrix with repeated rows and columns using the algorithm described in *From moments of sum to moments of product*, `doi:10.1016/j.jmva.2007.01.013 <https://dx.doi.org/10.1016/j.jmva.2007.01.013>`__.
:cpp:func:`libwalrus::hafnian_approx`                          Returns the approximate hafnian of a matrix with non-negative entries by sampling over determinants. The higher the number of samples, the better the accuracy.
:cpp:func:`libwalrus::torontonian`                             Returns the Torontonian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__.
:cpp:func:`libwalrus::torontonian_fsum`                        Returns the torontonian of a matrix using the algorithm described in *A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer*, `arxiv:1805.12498 <https://arxiv.org/abs/1805.12498>`__, with increased accuracy via the ``fsum`` summation algorithm.
:cpp:func:`libwalrus::permanent`                               Returns the permanent of a matrix using Ryser's algorithm with Gray code ordering.
:cpp:func:`libwalrus::perm_fsum`                               Returns the permanent of a matrix using Ryser's algorithm with Gray code ordering, with increased accuracy via the ``fsum`` summation algorithm.
:cpp:func:`libwalrus::hermite_multidimensional_cpp`            Returns photon number statistics of a Gaussian state for a given covariance matrix as described in *Multidimensional Hermite polynomials and photon distribution for polymode mixed light* `arxiv:9308033 <https://arxiv.org/abs/hep-th/9308033>`__.
=============================================================  ==============================================


API
---

`See here <../libwalrus_cpp_api/library_root.html>`_ for full details on the C++ API
and the ``libwalrus`` namespace.


.. toctree::
   :maxdepth: 2
   :caption: C++ API
   :hidden:

   ../libwalrus_cpp_api/library_root
