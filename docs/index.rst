The Walrus
##########

:Release: |release|
:Date: |today|

A library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling.

Features
========

* Fast calculation of hafnians, loop hafnians, and torontonians of general and certain structured matrices.

* An easy to use interface to use the loop hafnian for Gaussian quantum state calculations.

* Sampling algorithms for hafnian and torontonians of graphs.

* Efficient classical methods for approximating the hafnian of non-negative matrices.

* Easy to use implementations of the multidimensional Hermite polynomials, which can also be used to calculate hafnians of all reductions of a given matrix.

Getting started
===============

To get the The Walrus installed and running on your system, begin at the :ref:`download and installation guide <installation>`. Then, familiarise yourself with some :ref:`background information on the Hafnian <hafnian>` and :ref:`the computational algorithm <algorithms>`.

For getting started with using the The Walrus in your own code, have a look at the `Python tutorial <hafnian_tutorial.ipynb>`_.

Finally, detailed documentation on the code and API is provided.

Support
=======

- **Source Code:** https://github.com/XanaduAI/thewalrus
- **Issue Tracker:** https://github.com/XanaduAI/thewalrus/issues

If you are having issues, please let us know, either by email or by posting the issue on our Github issue tracker.

Authors
=======

Nicolás Quesada, Brajesh Gupt, and Josh Izaac.

If you are doing research using The Walrus, please cite `our paper <https://joss.theoj.org/papers/10.21105/joss.01705>`_:

 Brajesh Gupt, Josh Izaac and Nicolás Quesada. The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling. Journal of Open Source Software, 4(44), 1705 (2019)

License
=======

The Walrus library is **free** and **open source**, released under the Apache License, Version 2.0.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installing
   research
   quick_guide


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

.. toctree::
   :maxdepth: 2
   :caption: The Walrus Gallery
   :hidden:

   gallery/gallery

.. toctree::
   :maxdepth: 2
   :caption: The Walrus API
   :hidden:

   code
   code/thewalrus
   code/quantum
   code/samples
   code/symplectic
   code/fock_gradients
   code/reference
   code/libwalrus
