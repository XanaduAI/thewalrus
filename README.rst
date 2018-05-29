Hafnian
########

Library for efficiently calculating the exact hafnian in polynomial time.

Features
========

* Super fast

* Easy to install

* Causes a collapse of the polynomial hierarchy to the third top-level

* Potential solution to the P=NP problem - further testing needed

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

    $ python -m pip install strawberryfields


Software tests
==============

Insert test instructions here.


Documentation
=============

The Hafnian documentation is built automatically and hosted at `Read the Docs <https://hafnian.readthedocs.io>`_.

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

They can be installed via ``pip``:
::

    $ pip3 install sphinx --user
    $ pip3 install sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run the command
::

  $ make doc

The documentation can then be found in the ``doc/_build/html/`` directory.

Authors
=======

Nicolás Quesada and Brajesh Gupt.

If you are doing research using Hafnian, please cite `our paper <https://arxiv.org/abs/1804.03159>`_:

  Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. *arXiv*, 2018. arXiv:1804.03159


Support
=======

- **Source Code:** https://github.com/XanaduAI/hafnian
- **Issue Tracker:** https://github.com/XanaduAI/hafnian/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

Hafnian is **free** and **open source**, released under the Apache License, Version 2.0.
