Hafnian
################################

:Release: |release|
:Date: |today|

The fastest exact hafnian library.

Features
========

* Provides the fastest calculation of the hafnian and loop hafnian.

* The algorithms in this library are what Ryser's formula is to the permanent.

* (We also provide an efficient function for calculating the permanent via Ryser's formula.)

Getting started
===============

To get the Hafnian package installed and running on your system, begin at the :ref:`download and installation guide <installation>`. Then, familiarise yourself with some :ref:`background information on the Hafnian <hafnian>` and :ref:`the computational algorithm <algorithms>`.

For getting started with using the Hafnian library in your own code, have a look at the `Python tutorial <hafnian_tutorial.ipynb>`_.

Finally, detailed documentation on the code and API is provided.

Support
=======

- **Source Code:** https://github.com/XanaduAI/hafnian
- **Issue Tracker:** https://github.com/XanaduAI/hafnian/issues

If you are having issues, please let us know, either by email or by posting the issue on our Github issue tracker.

Authors
=======

Nicolás Quesada, Brajesh Gupt, and Josh Izaac.

If you are doing research using Hafnian, please cite `our paper <https://arxiv.org/abs/1805.12498>`_:

  Andreas Björklund, Brajesh Gupt, and Nicolás Quesada. A faster hafnian formula for complex matrices and its benchmarking on the Titan supercomputer *arXiv*, 2018. arxiv:1805.12498


License
=======

The hafnian library is **free** and **open source**, released under the Apache License, Version 2.0.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installing
   research


.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   hafnian
   algorithms
   references

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   hafnian_tutorial.ipynb
   permanent_tutorial.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Hafnian API
   :hidden:

   code
   code/python
   code/quantum


.. toctree::
   :maxdepth: 2
   :caption: Hafnian libraries
   :hidden:

   code/libhaf
   code/libperm
   code/libtor


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
