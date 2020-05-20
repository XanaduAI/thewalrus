The Walrus Documentation
########################

.. rst-class:: lead grey-text ml-2

:Release: |release|

.. raw:: html

    <style>
        #right-column.card {
            box-shadow: none!important;
        }
        #right-column.card:hover {
            box-shadow: none!important;
        }
        .breadcrumb {
            display: none;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
        }
        .footer-relations {
            border-top: 0px;
        }
    </style>
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
    <div style='clear:both'></div>
    <div class="container mt-2 mb-2">
        <div class="row mt-3">
            <div class="col-lg-4 mb-2 adlign-items-stretch">
                <a href="quick_guide.html">
                    <div class="card rounded-lg" style="height:100%;">
                        <div class="d-flex">
                            <div>
                                <h3 class="card-title pl-3 mt-4">
                                Using The Walrus
                                </h3>
                                <p class="mb-3 grey-text px-3">
                                    See the quick guide for an overview of available functions in The Walrus <i class="fas fa-angle-double-right"></i>
                                </p>
                            </div>
                        </div>
                    </div>
                </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="hafnian.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            Background
                            </h3>
                            <p class="mb-3 grey-text px-3">Learn about the hafnian, loop hafnian, and its relationship to quantum photonics <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
            <div class="col-lg-4 mb-2 align-items-stretch">
                <a href="code.html">
                <div class="card rounded-lg" style="height:100%;">
                    <div class="d-flex">
                        <div>
                            <h3 class="card-title pl-3 mt-4">
                            API
                            </h3>
                            <p class="mb-3 grey-text px-3">Explore The Walrus Python and C++ APIs <i class="fas fa-angle-double-right"></i></p>
                        </div>
                    </div>
                </div>
            </a>
            </div>
        </div>
    </div>


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
   code/random
   code/fock_gradients
   code/reference
   code/libwalrus
