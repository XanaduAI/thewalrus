Hafnian
########

.. image:: https://img.shields.io/travis/XanaduAI/hafnian/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/hafnian

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

The fastest exact hafnian library for real and complex full rank matrices. For more information, please see the `documentation <https://hafnian.readthedocs.io>`_.

Features
========

* Provides the fastest calculation of the hafnian and loop hafnian.

* The algorithms in this library are what Ryser's formula is to the permanent.


Dependencies
============

Hafnian depends on the following Python packages:

* `Python <http://python.org/>`_ >=3.5
* `NumPy <http://numpy.org/>`_  >=1.13.3

These can be installed using pip, or, if on linux, using your package manager (i.e. ``apt`` if on a Debian-based system.)

Note that the C extension may need to be compiled; you will need the following libraries:

* BLAS
* LAPACKe
* OpenMP

On Debian-based systems, this can be done via ``apt`` before installation:
::

    $ sudo apt install liblapacke-dev


Installation
============

Installation of Hafnian, as well as all required Python packages mentioned above, can be done using pip:
::

    $ python -m pip install hafnian


Software tests
==============

To ensure that the Hafnian library is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

  make test

Documentation
=============

The Hafnian documentation is built automatically and hosted at `Read the Docs <https://hafnian.readthedocs.io>`_. To build it locally, you need to have the following packages installed:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6
* `nbsphinx <https://github.com/spatialaudio/nbsphinx>`_
* `Pandoc <https://pandoc.org/>`_

They can be installed via a combination of ``pip`` and ``apt`` if on a Debian-based system:
::

    $ sudo apt install pandoc
    $ pip3 install sphinx sphinxcontrib-bibtex nbsphinx --user

To build the HTML documentation, go to the top-level directory and run the command
::

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
