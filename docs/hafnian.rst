.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _hafnian:


The hafnian
======================

.. sectionauthor:: Nicolas Quesada <nicolas@xanadu.ai>

The hafnian of an :math:`n \times n` symmetric matrix :math:`\mathbf{A} =\mathbf{A}^T` is defined as
		   
.. math::
   \label{eq:hafA}
   \haf(\mathbf{A}) = \sum_{M \in \text{PMP}(n)} \prod_{\scriptscriptstyle (i, j) \in M} A_{i, j}

where :math:`PMP(n)` stands for the set of perfect matching permutations of :math:`n` (even) objects.
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


We will also be interested in a generalization of the hafnian function where we will consider graphs that have loops, henceforth referred to as lhaf (loop hafnian). The weight associated with said loops will be allocated in the diagonal elements of the adjacency matrix :math:`\mathbf{A}` (which were previously ignored in the definition of the hafnian). To account for the possibility of loops we generalize the set of perfect matching permutations PMP to the single-pair matchings (SPM). This is simply the set of perfect matchings of a complete graph with loops. Thus we define

.. math::
   
   \lhaf(\mathbf{A}) = \sum_{M \in \text{SPM}(n)} \prod_{\scriptscriptstyle (i,j) \in M} A_{i,j}.

Considering again a graph with 4 vertices we get a total of 10 SPMs:
   
.. math::
      
   \text{SPM}(4)=\big\{ &(0,1)(2,3),\ (0,2)(1,3), \ (0,3),(1,2), \ (0,0)(1,1)(2,3),\ (0,1)(2,2)(3,3),
   \\&  (0,2)(1,1)(3,3), \ (0,0)(2,2)(1,3), \ (0,0)(3,3)(1,2),\ (0,3)(1,1)(2,2),  \ (0,0)(1,1)(2,2)(3,3) \big\}. 

   
and the the lhaf of a :math:`4 \times 4` matrix :math:`\mathbf{B}` is

.. math::

   \lhaf(\mathbf{B}) =& B_{0,1} B_{2,3}+B_{0,2}B_{1,3}+B_{0,3} B_{1,2}\\
   &+ B_{0,0} B_{1,1} B_{2,3}+B_{0,1} B_{2,2} B_{3,3}+B_{0,2}B_{1,1}B_{3,3}\nonumber\\
   &+ B_{0,0} B_{2,2} B_{1,3}+B_{0,0}B_{3,3}B_{1,2}+B_{0,3} B_{1,1} B_{2,2}\nonumber\\
   &+ B_{0,0} B_{1,1} B_{2,2} B_{3,3}. \nonumber

More generally for a graph with :math:`n` vertices (:math:`n` even) the number of SPMs is

.. math::
   
   |\text{SPM}(n)| = (n-1)!! \  _1F_1\left(-\frac{n}{2};\frac{1}{2};-\frac{1}{2}\right)

where :math:`_1F_1\left(a;b;z\right)` is the Kummer confluent hypergeometric function. Note that :math:`_1F_1\left(-\frac{n}{2};\frac{1}{2};-\frac{1}{2}\right)` scales superpolynomially in :math:`n`

Finally, let us comment on the scaling properties of the :math:`\haf` and :math:`\lhaf`.
Unlike the hafnian the function loop hafnian is not homogeneous in its matrix entries, i.e.

.. math::
   
   \haf(\mu \mathbf{A}) &= \mu ^{n/2} \haf(\mathbf{A}) \text{  but},\\
   \lhaf(\mu \mathbf{A}) &\neq \mu ^{n/2} \lhaf(\mathbf{A}).

where :math:`n` is the size of the matrix :math:`\mathbf{A}` and :math:`\mu \geq 0`. However if we split the matrix :math:`\mathbf{A}`  in terms of its diagonal :math:`\mathbf{A}_{\text{diag}}` part and its offdiagonal part :math:`\mathbf{A}_{\text{off-diag}}`

.. math::
   
   \mathbf{A} = \mathbf{A}_{\text{diag}}+\mathbf{A}_{\text{off-diag}}

then it holds that
   
.. math::
   
   \lhaf(\sqrt{\mu} \mathbf{A}_{\text{diag}}+ \mu \mathbf{A}_{\text{off-diag}}) = \mu^{n/2} \lhaf(\mathbf{A}_{\text{diag}}+ \mathbf{A}_{\text{off-diag}}) =\mu^{n/2} \lhaf(\mathbf{A}).
