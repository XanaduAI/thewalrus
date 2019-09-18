---
title: 'The Walrus: the fastest calculation of hafnians, Hermite polynomials and Gaussian boson sampling'
tags:
  - Python
  - quantum computing
  - quantum optics
  - graph theory
authors:
  - name: Brajesh Gupt
    orcid: 0000-0002-6352-8342
    affiliation: 1
  - name: Josh Izaac
    orcid: 0000-0003-2640-0734
    affiliation: 1
  - name: Nicolás Quesada
    orcid: 0000-0002-0175-1688
    affiliation: 1
affiliations:
 - name: Xanadu, Toronto, Canada
   index: 1
date: 3 August 2019
bibliography: paper.bib
---

>> In [The Walrus](https://github.com/XanaduAI/thewalrus), we provide a highly optimized implementation of the best known algorithms for hafnians, loop hafnians, multidimensional Hermite polynomials, and torontonians of generic real and complex matrices. We also provide access to recently proposed methods to generate samples of a Gaussian boson sampler. These methods have exponential time complexity in the number of bosons measured. For ease of use, a Python interface to the library's low-level C++ implementations is also provided, as well as pre-compiled static libraries installable via the Python package manager `pip`. This short paper provides a high-level description of the library and its rationale; in-depth information on the algorithms, implementations and interface can be found in its [documentation](https://the-walrus.readthedocs.io/en/latest/).


The hafnian matrix function was introduced by @caianiello1953 as a generalization of
the permanent while studying problems in bosonic quantum field theory.
For a symmetric matrix $\mathbf{A} = \mathbf{A}^T$of size $2n$, the hafnian (haf) is defined as
$$ \text{haf}\left(\mathbf{A} \right) = \sum_{\sigma \in \text{PMP}(2n)}
\prod_{i=1}^n A_{\sigma(2i-1),\sigma(2i)},$$
where $\text{PMP}(2n)$ is the set of perfect matching permutations of $2n$ elements, i.e.,
permutations  $\sigma: [2n] \to [2n]$ such that $\sigma(2i-1) < \sigma(2i)$ and $\sigma(2i-1) < \sigma(2i+1)$ for all $i$ [@barvinok2016].


While the permanent counts the number of perfect matchings of a *bipartite* graph encoded in an adjacency matrix
$\mathbf{B}$, the hafnian counts the number of perfect matchings of an *arbitrary undirected graph*, and thus the permanent is a special case of the hafnian; this relation is encapsulated in the
identity $\text{perm}(\mathbf{B}) = \text{haf}\left( \left[ \begin{smallmatrix} \mathbf{0} &
\mathbf{B}   \\   \mathbf{B}^T & \mathbf{0} \end{smallmatrix} \right] \right).$


The permanent has received a significant amount of attention, especially after @valiant1979 proved
that it is #P-hard to compute, giving one
of the first examples of a problem in this complexity class. This important complexity-theoretic
observation was predated by @ryser1963, who provided an algorithm to calculate
the permanent of an arbitrary matrix of size $n \times n$ in $O(2^n)$ time, which is still
to date the fastest algorithm for calculating permanents.

Surprisingly, it took almost half a century to derive a formula for hafnians that matched the
complexity of the one for permanents. Indeed, it was only
@bjorklund2012 who derived an algorithm that computed the hafnian of a
$2n \times 2n$ matrix in time $O(2^n)$.

The interest in hafnians was recently reignited by findings in quantum computing.
Gaussian Boson Sampling (GBS) [@hamilton2017; @kruse2018] is a non-universal model of quantum computation in which it is possible to show that there are computations a quantum computer can do in polynomial time that a classical computer cannot.
Experimentally, GBS is based on the idea that a certain subset of quantum
states, so-called Gaussian states, can be easily prepared in physical devices, and then those
states can be measured with particle-number resolving detectors. Because these are quantum
mechanical particles, the outcomes of the measurements are stochastic in nature and it is precisely
the simulation of these random "samples" that requires superpolynomial time to
simulate on a classical computer.

The relation between GBS and hafnians stems from the fact that the probability of a given experimental outcome
is proportional to the hafnian of a matrix constructed from the covariance matrix of the Gaussian
state. This observation requires the Gaussian state to have zero mean and the detector to be able to
resolve any number of particles. More generally, one can consider Gaussian states with
finite mean [@quesada2019; @quesada2019a], in which the probability is given by a loop hafnian,
a matrix function that counts the number of perfect matchings of a graph that has loops
[@bjorklund2019]. Moreover, if the particle detectors can only decide whether there were zero or
more than zero particles -- so-called threshold detectors -- then the probability is given by the torontonian,
a matrix function that acts as a generating function for the hafnian [@quesada2018]. One can also show that the probabilities
of a Gaussian state probed in the number basis are related to multidimensional Hermite
polynomials [@dodonov1994]. Calculating the probabilities
of a GBS experiment in terms of multidimensional Hermite polynomials [@kok2001] is often suboptimal since they have worse space and time scaling than the corresponding calculation in terms of hafnians.


In The Walrus, we provide a highly optimized implementation of the best known algorithms for hafnians,
loop hafnians, Hermite polynomials, and torontonians of generic real and complex matrices. We also implement
algorithms that specialize to certain matrices with structure, for example having repeated rows and
columns [@kan2008] or non-negative entries [@barvinok1999]. For increased efficiency, these
algorithms are implemented in C++ as a templated header-only library, allowing them to
be applied to arbitrary numeric types and are also parallelized via OpenMP. Common linear algebra algorithms are
applied using the Eigen C++ template library, which may also act as a frontend to an optimized
BLAS/LAPACKE library installation if the user so chooses. For ease of use, a Python interface to the
low-level C++ algorithms is also provided, as well as pre-compiled static libraries installable via the
Python package manager `pip` for Windows, MacOS, and Linux users.
We also provide implementations of multidimensional Hermite polynomials, that are, to the best of our
knowledge, the first ones implemented in a fast open source library.
With this underlying machinery we also
provide two extra Python-only modules. The first one, *quantum*, allows one to calculate in a straightforward manner the
probabilities or probability amplitudes of Gaussian states in the particle representation. The second
one, *samples*, allows one to generate GBS samples. This module implements state-of-the-art algorithms that
have been recently developed [@quesada2019c]. Of course, given the promise that GBS should be a hard
problem for classical computers, the complexity of the algorithm we provide for GBS is, like the
complexity of the hafnian, still exponential in the size of the number of particles generated.

Our package has already been used in several research efforts to understand how to generate resource
states for universal quantum computing [@quesada2019a; @valson2019franck], study the dynamics of vibrational quanta in
molecules [@quesada2019], and develop the applications of GBS to molecular docking [@banchi2019],
graph theory [@schuld2019], and point processes [@jahangiri2019]. More importantly, it has been
useful in delineating when quantum computation can be simulated by classical computing resources and
when it cannot [@gupt2018; @quesada2019c; @wu2019speedup; @killoran2019].

# Acknowledgements

The authors thank our colleagues at Xanadu, especially, J.M. Arrazola, T.R. Bromley, S. Jahangiri and
N. Killoran, for valuable feedback.

N.Q. thanks A. Björklund and W.R. Clements for insightful correspondence and C. Ducharme and L.G. Helt for comments
on the manuscript.

# References
