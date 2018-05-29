.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _algorithms:

Algorithms
===========

.. sectionauthor:: Nicolas Quesada <nicolas@xanadu.ai>

The exact calculation of the number of perfect matchings for general graphs has been investigated by several authors in recent years. An algorithm running in :math:`O(n^2 2^n)` time was given by Björklund and Husfeldt in Ref. :cite:`bjorklund2008exact`. In the same paper an algorithm running in :math:`O(1.733^n)` time was presented using fast matrix multiplication. Nederlof :cite:`nederlof2009fast` and Koivisto :cite:`koivisto2009partitioning` provided polynomial space algorithms running in :math:`O(1.942^n)` and :math:`O^*(\phi^n)` space respectively. In the last sentence :math:`\phi = (1+\sqrt{5})/2 \approx 1.618` is the Golden ratio and the notation :math:`O^*` is used to indicate that polylogarithmic corrections have been suppressed in the scaling.

More recently Björklund :cite:`bjorklund2012counting` and later Cygan and Pilipczuk :cite:`cygan2015faster` provided :math:`O(\text{poly}(n) 2^{n/2})` time algorithms for the calculation of the hafnian. These two algorithms can also be used to count (up to polynomial corrections) the number of perfect matchings for bipartite graphs with the same exponential growth as Ryser's algorithm for the permanent :cite:`ryser1963combinatorial`. Equivalently, if one could construct an algorithm that calculates hafnians in time :math:`O(\alpha^{n/2})` with :math:`\alpha<2` one could calculate permanents faster than Ryser's algorithm (which is the fastest known algorithm to calculate the permanent :cite:`rempala2007symmetric`). This is because of the identity

.. math::
   
   \haf \left( \left[
   \begin{array}{cc}
   0 & \mathbf{W} \\
   \mathbf{W}^T & 0 \\
   \end{array}
   \right]\right) = \text{per}(\mathbf{W}),

   
which states that a bipartite graph with two parts having :math:`n/2` elements can always be thought as a simple graph with :math:`n` vertices.
Since the exact calculation of the permanent of 0-1 matrices is  in the \#P complete class
:cite:`valiant1979complexity` the last identity shows that the hafnian of (at least) certain 0-1 block matrices is also in the \#P complete class.
   

In this library we implement an algorithm that allows to count the perfect matchings of a graph with :math:`n` vertices in time :math:`O(n^3 2^{n/2})` for graphs with and without loops. This two algorithms are encapsulated in the hafnian and loophafnian function.
