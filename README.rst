The Walrus
##########

.. image:: https://circleci.com/gh/XanaduAI/thewalrus/tree/master.svg?style=svg
    :alt: CircleCI
    :target: https://circleci.com/gh/XanaduAI/thewalrus/tree/master

.. image:: https://ci.appveyor.com/api/projects/status/9udscqldo1xd25yk/branch/master?svg=true
    :alt: Appveyor
    :target: https://ci.appveyor.com/project/josh146/hafnian/branch/master

.. image:: https://img.shields.io/codecov/c/github/xanaduai/thewalrus/master.svg?style=flat
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/thewalrus

.. image:: https://img.shields.io/codacy/grade/df94d22534cf4c05b1bddcf697011a82.svg?style=flat
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/thewalrus?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/thewalrus&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/the-walrus.svg?style=flat
    :alt: Read the Docs
    :target: https://the-walrus.readthedocs.io

.. image:: https://img.shields.io/pypi/pyversions/thewalrus.svg?style=flat
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/thewalrus

.. image:: https://joss.theoj.org/papers/10.21105/joss.01705/status.svg
    :alt: JOSS - The Journal of Open Source Software
    :target: https://doi.org/10.21105/joss.01705

A library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling. For more information, please see the `documentation <https://the-walrus.readthedocs.io>`_.

Features
========

* Fast calculation of hafnians, loop hafnians, and torontonians of general and certain structured matrices.

* An easy to use interface to use the loop hafnian for Gaussian quantum state calculations.

* Sampling algorithms for hafnian and torontonians of graphs.

* Efficient classical methods for approximating the hafnian of non-negative matrices.

* Easy to use implementations of the multidimensional Hermite polynomials, which can also be used to calculate hafnians of all reductions of a given matrix.


Installation
============

Pre-built binary wheels are available for the following platforms:

+------------+-------------+------------------+---------------+
|            | macOS 10.6+ | manylinux x86_64 | Windows 64bit |
+============+=============+==================+===============+
| Python 3.6 |      X      |        X         |       X       |
+------------+-------------+------------------+---------------+
| Python 3.7 |      X      |        X         |       X       |
+------------+-------------+------------------+---------------+
| Python 3.8 |      X      |        X         |       X       |
+------------+-------------+------------------+---------------+

To install, simply run

.. code-block:: bash

    pip install thewalrus


Compiling from source
=====================

The Walrus depends on the following Python packages:

* `Python <http://python.org/>`_ >= 3.6
* `NumPy <http://numpy.org/>`_  >= 1.13.3
* `Numba <https://numba.pydata.org/>`_ >= 0.43.1

In addition, to compile the C++ extension, the following dependencies are required:

* A C++11 compiler, such as ``g++`` >= 4.8.1, ``clang`` >= 3.3, ``MSVC`` >= 14.0/2015
* `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ - a C++ header library for linear algebra.
* `Cython <https://cython.org/>`_ an optimising static compiler for the Python programming language.

On Debian-based systems, these can be installed via ``apt`` and ``curl``:

.. code-block:: console

    $ sudo apt install g++ libeigen3-dev
    $ pip install Cython

or using Homebrew on MacOS:

.. code-block:: console

    $ brew install gcc eigen
    $ pip install Cython

Alternatively, you can download the Eigen headers manually:

.. code-block:: console

    $ mkdir ~/.local/eigen3 && cd ~/.local/eigen3
    $ wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz -O eigen3.tar.gz
    $ tar xzf eigen3.tar.gz eigen-eigen-323c052e1731/Eigen --strip-components 1
    $ export EIGEN_INCLUDE_DIR=$HOME/.local/eigen3

Note that we export the environment variable ``EIGEN_INCLUDE_DIR`` so that The Walrus can find the Eigen3 header files (if not provided, The Walrus will by default look in ``/use/include/eigen3`` and ``/usr/local/include/eigen3``).

You can compile the latest development version by cloning the git repository, and installing using pip in development mode.

.. code-block:: console

    $ git clone https://github.com/XanaduAI/thewalrus.git
    $ cd thewalrus && python -m pip install -e .


OpenMP
------

``libwalrus`` uses OpenMP to parallelize both the permanent and the hafnian calculation. **At the moment, this is only supported on Linux using the GNU g++ compiler, due to insufficient support using Windows/MSCV and MacOS/Clang.**



Using LAPACK, OpenBLAS, or MKL
------------------------------

If you would like to take advantage of the highly optimized matrix routines of LAPACK, OpenBLAS, or MKL, you can optionally compile the ``libwalrus`` such that Eigen uses these frameworks as backends. As a result, all calls in the ``libwalrus`` library to Eigen functions are silently substituted with calls to LAPACK/OpenBLAS/MKL.

For example, for LAPACK integration, make sure you have the ``lapacke`` C++ LAPACK bindings installed (``sudo apt install liblapacke-dev`` in Ubuntu-based Linux distributions), and then compile with the environment variable ``USE_LAPACK=1``:

.. code-block:: console

    $ USE_LAPACK=1 python -m pip install thewalrus --no-binary :all:

Alternatively, you may pass ``USE_OPENBLAS=1`` to use the OpenBLAS library.


Software tests
==============

To ensure that The Walrus library is working correctly after installation, the test suite can be run by navigating to the source code folder and running

.. code-block:: console

    $ make test

To run the low-level C++ test suite, `Googletest <https://github.com/google/googletest>`_
will need to be installed. In Ubuntu-based distributions, this can be done as follows:

.. code-block:: console

    sudo apt-get install cmake libgtest-dev

Alternatively, the latest Googletest release can be installed from source:

.. code-block:: console

    sudo apt install cmake
    wget -qO - https://github.com/google/googletest/archive/release-1.8.1.tar.gz | tar -xz
    cmake -D CMAKE_INSTALL_PREFIX:PATH=$HOME/googletest -D CMAKE_BUILD_TYPE=Release googletest-release-1.8.1
    make install

If installing Googletest from source, make sure that the included headers and
libraries are available on your include/library paths.

Documentation
=============

The Walrus documentation is available online on `Read the Docs <https://the-walrus.readthedocs.io>`_.

To build it locally, you need to have the following packages installed:

* `Sphinx <http://sphinx-doc.org/>`_ >= 1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >= 0.3.6
* `nbsphinx <https://github.com/spatialaudio/nbsphinx>`_
* `Pandoc <https://pandoc.org/>`_
* `breathe <https://breathe.readthedocs.io/en/latest/>`_ >= 4.12.0
* `exhale <https://exhale.readthedocs.io/en/latest/>`_
* `Doxygen <http://www.doxygen.nl/>`_

They can be installed via a combination of ``pip`` and ``apt`` if on a Debian-based system:
::

    $ sudo apt install pandoc doxygen
    $ pip3 install sphinx sphinxcontrib-bibtex nbsphinx breathe exhale

To build the HTML documentation, go to the top-level directory and run the command

.. code-block:: console

    $ make doc

The documentation can then be found in the ``docs/_build/html/`` directory.

Contributing to The Walrus
==========================

We welcome contributions - simply fork The Walrus repository, and then make a pull request containing your contribution. All contributors to The Walrus will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to projects, applications or scientific publications that use The Walrus.

Authors
=======

Brajesh Gupt, Josh Izaac and Nicolas Quesada.

All contributions are acknowledged in the `acknowledgments page <https://github.com/XanaduAI/thewalrus/blob/master/.github/ACKNOWLEDGMENTS.md>`_.

If you are doing research using The Walrus, please cite `our paper <https://joss.theoj.org/papers/10.21105/joss.01705>`_:

 Brajesh Gupt, Josh Izaac and Nicolas Quesada. The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling. Journal of Open Source Software, 4(44), 1705 (2019)


Support
=======

- **Source Code:** https://github.com/XanaduAI/thewalrus
- **Issue Tracker:** https://github.com/XanaduAI/thewalrus/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

The Walrus is **free** and **open source**, released under the Apache License, Version 2.0.
