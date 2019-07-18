Hafnian
################################

:Release: |release|
:Date: |today|

The fastest hafnian library.

Features
========

* The fastest calculation of the hafnians, loop hafnians, and torontonians of general and certain structured matrices.

* An easy to use interface to use the loop hafnian for Gaussian quantum state calculations

* State of the art algorithms to sample from (loop) hafnian and torontonians of graphs.

* Efficient classical methods for approximating the hafnian of non-negative matrices.

* Easy to use implementations of the multidimensional Hermite polynomials, which can also be used to calculate hafnians of all reductions of a given matrix.

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
   loop_hafnian
   algorithms
   hermite
   gbs
   gbs_sampling
   notation
   references

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   basics.ipynb
   hafnian_tutorial.ipynb
   permanent_tutorial.ipynb
   gaussian_states.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Hafnian API
   :hidden:

   code
   code/hafnian
   code/quantum
   code/samples
   code/reference

.. toctree::
   :maxdepth: 2
   :caption: Low-level libraries
   :hidden:

   code/libhaf


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
