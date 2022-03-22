The Walrus
##########

.. image:: https://github.com/XanaduAI/thewalrus/actions/workflows/tests.yml/badge.svg
    :alt: Tests
    :target: https://github.com/XanaduAI/thewalrus/actions/workflows/tests.yml

.. image:: https://img.shields.io/codecov/c/github/xanaduai/thewalrus/master.svg?style=flat
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/thewalrus

.. image:: https://img.shields.io/codefactor/grade/github/XanaduAI/thewalrus/master?style=flat
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/xanaduai/thewalrus

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

The Walrus requires Python version 3.7, 3.8, 3.9, or 3.10. Installation of The Walrus, as
well as all dependencies, can be done using pip:

.. code-block:: bash

    pip install thewalrus


Compiling from source
=====================

The Walrus has the following dependencies:

* `Python <http://python.org/>`_ >= 3.7
* `NumPy <http://numpy.org/>`_  >= 1.19.2
* `Numba <https://numba.pydata.org/>`_ >= 0.49.1
* `SciPy <https://scipy.org/>`_ >=1.2.1
* `SymPy <https://www.sympy.org/>`_ >=1.5.1
* `Dask[delayed] <https://docs.dask.org/>`

You can compile the latest development version by cloning the git repository, and installing using
pip in development mode.

.. code-block:: console

    $ git clone https://github.com/XanaduAI/thewalrus.git
    $ cd thewalrus && python -m pip install -e .


Software tests
==============

To ensure that The Walrus library is working correctly after installation, the test
suite can be run locally using pytest.

Additional packages are required to run the tests. These dependencies can be found in
``requirements-dev.txt`` and can be installed using ``pip``:

.. code-block:: console

    $ pip install -r requirements-dev.txt

To run the tests, navigate to the source code folder and run the command

.. code-block:: console

    $ make test


Documentation
=============

The Walrus documentation is available online on `Read the Docs <https://the-walrus.readthedocs.io>`_.

Additional packages are required to build the documentation locally as specified in `doc/requirements.txt`.
These packages can be installed using:

.. code-block:: console

    $ sudo apt install pandoc
    $ pip install -r docs/requirements.txt

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

The Walrus is the work of `many contributors <https://github.com/XanaduAI/thewalrus/blob/master/.github/ACKNOWLEDGMENTS.md>`_.

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
