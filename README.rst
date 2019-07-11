Hafnian
#######
    
.. image:: https://circleci.com/gh/XanaduAI/hafnian/tree/master.svg?style=svg&circle-token=209b57390082a2b2fe2cdc9ee49a301ddc29ca5b
    :alt: CircleCI
    :target: https://circleci.com/gh/XanaduAI/hafnian/tree/master

.. image:: https://ci.appveyor.com/api/projects/status/9udscqldo1xd25yk/branch/master?svg=true
    :alt: Appveyor
    :target: https://ci.appveyor.com/project/josh146/hafnian/branch/master

.. image:: https://img.shields.io/codecov/c/github/xanaduai/hafnian/master.svg?style=flat
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/hafnian

.. image:: https://img.shields.io/codacy/grade/df94d22534cf4c05b1bddcf697011a82.svg?style=flat
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/hafnian?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/hafnian&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/hafnian.svg?style=flat
    :alt: Read the Docs
    :target: https://hafnian.readthedocs.io

.. image:: https://img.shields.io/pypi/pyversions/hafnian.svg?style=flat
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/hafnian

The fastest exact hafnian library. For more information, please see the `documentation <https://hafnian.readthedocs.io>`_.

Features
========

* The fastest calculation of the hafnians, loop hafnians, and torontonians of general and certain structured matrices.

* An easy to use interface to use the loop hafnian for quantum state calculations

* State of the art algorithms to sample from hafnian and torontonians of graphs.

* Efficient classical methods for approximating the hafnian of non-negative matrices.

Installation
============

Pre-built binary wheels are available for the following platforms:

+------------+-------------+------------------+---------------+
|            | macOS 10.6+ | manylinux x86_64 | Windows 64bit |
+============+=============+==================+===============+
| Python 3.5 |  ✅         |  ✅              |   ✅          |
+------------+-------------+------------------+---------------+
| Python 3.6 |  ✅         |  ✅              |   ✅          |
+------------+-------------+------------------+---------------+
| Python 3.7 |  ✅         |  ✅              |   ✅          |
+------------+-------------+------------------+---------------+

To install, simply run

.. code-block:: bash

    pip install hafnian


Compiling from source
=====================

Hafnian depends on the following Python packages:

* `Python <http://python.org/>`_ >=3.5
* `NumPy <http://numpy.org/>`_  >=1.13.3

In addition, to compile the C++ extension, the following dependencies are required:

* A C++11 compiler, such as ``g++`` >= 4.8.1, ``clang`` >= 3.3, ``MSVC`` >= 14.0/2015
* `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ - a C++ header library for linear algebra.

On Debian-based systems, these can be installed via ``apt`` and ``curl``:

.. code-block:: console

    $ sudo apt install g++ libeigen3-dev

or using Homebrew on MacOS:

.. code-block:: console

    $ brew install gcc eigen

Alternatively, you can download the Eigen headers manually:

.. code-block:: console

    $ mkdir ~/.local/eigen3 && cd ~/.local/eigen3
    $ wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz -O eigen3.tar.gz
    $ tar xzf eigen3.tar.gz eigen-eigen-323c052e1731/Eigen --strip-components 1
    $ export EIGEN_INCLUDE_DIR=$HOME/.local/eigen3

Note that we export the environment variable ``EIGEN_INCLUDE_DIR`` so that Hafnian can find the Eigen3 header files (if not provided, Hafnian will by default look in ``/use/include/eigen3`` and ``/usr/local/include/eigen3``).

Once all dependencies are installed, you can compile the latest stable version of the Hafnian library as follows:

.. code-block:: console

    $ python -m pip install hafnian --no-binary :all:

Alternatively, you can compile the latest development version by cloning the git repository, and installing using pip in development mode.

.. code-block:: console

    $ git clone https://github.com/XanaduAI/hafnian.git
    $ cd hafnian && python -m pip install -e .


OpenMP
------

The Hafnian library uses OpenMP to parallelize both the permanent and the hafnian calculation. **At the moment, this is only supported on Linux using the GNU g++ compiler, due to insufficient support using Windows/MSCV and MacOS/Clang.**



Using LAPACK, OpenBLAS, or MKL
------------------------------

If you would like to take advantage of the highly optimized matrix routines of LAPACK, OpenBLAS, or MKL, you can optionally compile the Hafnian library such that Eigen uses these frameworks as backends. As a result, all calls in the Hafnian library to Eigen functions are silently substituted with calls to LAPACK/OpenBLAS/MKL.

For example, for LAPACK integration, make sure you have the ``lapacke`` C++ LAPACK bindings installed (``sudo apt install liblapacke-dev`` in Ubuntu-based Linux distributions), and then compile with the environment variable ``USE_LAPACK=1``:

.. code-block:: console

    $ USE_LAPACK=1 python -m pip install hafnian --no-binary :all:

Alternatively, you may pass ``USE_OPENBLAS=1`` to use the OpenBLAS library.


Software tests
==============

To ensure that the Hafnian library is working correctly after installation, the test suite can be run by navigating to the source code folder and running

.. code-block:: console

    $ make test

To run the low-level C++ test suite, `Googletest <https://github.com/google/googletest>`_
will need to be installed. In Ubuntu-based distributions, this can be done as follows:

.. code-block:: console

    sudo apt-get install cmake libgtest-dev
    cd /usr/src/googletest/googletest
    sudo cmake
    sudo make
    sudo cp libgtest* /usr/lib/
    sudo mkdir /usr/local/lib/googletest
    sudo ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a
    sudo ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a

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

The Hafnian+ documentation is currently not hosted online. To build it locally, you need to have the following packages installed:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6
* `nbsphinx <https://github.com/spatialaudio/nbsphinx>`_
* `Pandoc <https://pandoc.org/>`_
* `breathe <https://breathe.readthedocs.io/en/latest/>`_ >=4.12.0
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



Authors
=======

Nicolás Quesada, Brajesh Gupt, and Josh Izaac.

All contributions are acknowledged in the `acknowledgments page <https://github.com/XanaduAI/hafnian/blob/master/.github/ACKNOWLEDGMENTS.md>`_.

If you are doing research using Hafnian, please cite `our paper <https://dl.acm.org/citation.cfm?id=3325111>`_:

 Andreas Björklund, Brajesh Gupt, and Nicolás Quesada. A faster hafnian formula for complex matrices and its benchmarking on a supercomputer, Journal of Experimental Algorithmics (JEA) 24 (1), 11 (2019)


Support
=======

- **Source Code:** https://github.com/XanaduAI/hafnian
- **Issue Tracker:** https://github.com/XanaduAI/hafnian/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

Hafnian is **free** and **open source**, released under the Apache License, Version 2.0.
