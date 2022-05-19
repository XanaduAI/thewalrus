The Walrus Documentation
########################

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        p.lead.grey-text {
            margin-bottom: 30px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>

    <div class="container mt-2 mb-2">
        <div class="row container-fluid">
            <div class="col-lg-4 col-12 mb-2 text-center">
                <img src="_static/walrus.svg" class="img-fluid" alt="Responsive image" style="width:100%; max-width: 300px;"></img>
            </div>
            <div class="col-lg-8 col-12 mb-2" style="display: flex;justify-content: center;align-items: center;flex-flow: column;">
                <p class="lead grey-text">
                    A library for the calculation of hafnians, Hermite polynomials, and Gaussian boson sampling.
                </p>
            </div>
        </div>
        <div class="row mt-3">

.. index-card::
    :name: Using The Walrus
    :link: quick_guide.html
    :description: See the quick guide for an overview of available functions in The Walrus

.. index-card::
    :name: Background
    :link: hafnian.html
    :description: Learn about the hafnian, loop hafnian, and its relationship to quantum photonics

.. index-card::
    :name: API
    :link: code.html
    :description: Explore The Walrus Python API

.. raw:: html

        </div>
    </div>

Features
========

* Fast calculation of hafnians, loop hafnians, and torontonians of general and certain structured matrices powered by `Numba <https://numba.pydata.org/>`_.

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
   gallery/gallery

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
   :caption: The Walrus API
   :hidden:

   code
   code/thewalrus
   code/quantum
   code/samples
   code/csamples
   code/symplectic
   code/charpoly
   code/random
   code/fock_gradients
   code/decompositions
   code/reference
