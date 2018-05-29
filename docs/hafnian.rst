.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _hafnian:


The hafnian
======================

.. sectionauthor:: Nicolas Quesada <nicolas@xanadu.ai>

The hafnian of an :math:`n \times n ` symmetric matrix :math:`\mathbf{A} =\mathbf{A}^T` is defined as
		   
.. math::
\label{eq:hafA}
\haf(\mathbf{A}) = \sum_{M \in \text{PMP}(n)} \prod_{\scriptscriptstyle (i, j) \in M} A_{i, j}

where :math:`PMP$(n)` stands for the set of perfect matching permutations of :math:`n` (even) objects.
For :math:`n=4` the set of perfect matchings is

.. math::
\label{eq:PMP4}
\text{PMP}(4) = \big\{ (0,1)(2,3),\ (0,2)(1,3),\ (0,3),(1,2) \big\},

and the hafnian of a :math:`4 \times 4` matrix :math:`\mathbf{B}` is

.. math::
\label{eq:hafB}
\haf(\mathbf{B}) = B_{0,1} B_{2,3}+B_{0,2}B_{1,3}+B_{0,3} B_{1,2}.


More generally, the set PMP(:math:`n`) contains

.. math::
\label{eq:haf1}
|\text{PMP}(n)|=(n -1)!! = 1 \times 3 \times 5 \times \ldots \times (n -1)

elements and thus as defined it takes :math:`(n-1)!!` additions of products of :math:`n/2` numbers to calculate the hafnian of :math:`\mathbf{A}`.
Note that the diagonal elements of the matrix :math:`\mathbf{A}` do not appear in the calculation of the hafnian and are (conventionally) taken to be zero. 
