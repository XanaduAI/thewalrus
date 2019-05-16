Hafnian
#######

.. image:: https://img.shields.io/travis/XanaduAI/hafnian/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/hafnian

.. image:: https://ci.appveyor.com/api/projects/status/6wt68c81f8ly583s/branch/master?svg=true
    :alt: Appveyor
    :target: https://ci.appveyor.com/project/josh146/hafnianplus/branch/master

.. image:: https://img.shields.io/codecov/c/github/xanaduai/hafnian/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/hafnian

.. image:: https://img.shields.io/codacy/grade/df94d22534cf4c05b1bddcf697011a82.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/hafnian?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/hafnian&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/hafnian.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://hafnian.readthedocs.io

.. image:: https://img.shields.io/pypi/pyversions/hafnian.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/hafnian

The fastest exact hafnian library for real and complex matrices. For more information, please see the `documentation <https://hafnian.readthedocs.io>`_.

Features
========

* The fastest calculation of the hafnian, loop hafnian, permanent, and torontonian,
  of general and certain structured matrices.

* An easy to use interface to use the loop hafnian to calculate Fock matrix
  elements of Gaussian states via the included quantum module.

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

These can be installed using ``pip``:

.. code-block:: bash

    pip install hafnian


Compiling from source
=====================

Hafnian depends on the following Python packages:

* `Python <http://python.org/>`_ >=3.5
* `NumPy <http://numpy.org/>`_  >=1.13.3

In addition, to compile the included Fortran and C++ extensions, the following dependencies are required:

* A Fortran compiler, such as ``gfortran``
* The LINPACK_Q quadruple precision linear algebra library
* A C++11 compiler, such as ``g++`` >= 4.8.1, ``clang`` >= 3.3, ``MSVC`` >= 14.0/2015
* `Eigen3 <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ - a C++ header library for linear algebra.

On Debian-based systems, these can be installed via ``apt`` and ``curl``:

.. code-block:: console

    $ sudo apt install g++ gfortran libeigen3-dev
    $ curl -sL -o src/linpack_q_complex.f90 https://raw.githubusercontent.com/josh146/linpack_q_complex/master/linpack_q_complex.f90

or using Homebrew on MacOS:

.. code-block:: console

    $ brew install gcc eigen
    $ curl -sL -o src/linpack_q_complex.f90 https://raw.githubusercontent.com/josh146/linpack_q_complex/master/linpack_q_complex.f90

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

Documentation
=============

The Hafnian+ documentation is currently not hosted online. To build it locally, you need to have the following packages installed:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6
* `nbsphinx <https://github.com/spatialaudio/nbsphinx>`_
* `Pandoc <https://pandoc.org/>`_

They can be installed via a combination of ``pip`` and ``apt`` if on a Debian-based system:
::

    $ sudo apt install pandoc
    $ pip3 install sphinx sphinxcontrib-bibtex nbsphinx --user

To build the HTML documentation, go to the top-level directory and run the command

.. code-block:: console

    $ make doc

The documentation can then be found in the ``docs/_build/html/`` directory.



Authors
=======

Nicolás Quesada, Brajesh Gupt, and Josh Izaac.

If you are doing research using Hafnian, please cite `our paper <https://arxiv.org/abs/1805.12498>`_:

 Andreas Björklund, Brajesh Gupt, and Nicolás Quesada. A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer *arXiv*, 2018. arxiv:1805.12498


Support
=======

- **Source Code:** https://github.com/XanaduAI/hafnian
- **Issue Tracker:** https://github.com/XanaduAI/hafnian/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

Hafnian is **free** and **open source**, released under the Apache License, Version 2.0.
