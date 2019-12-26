.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html
.. _hermite:


Multidimensional Hermite polynomials
====================================
.. sectionauthor:: Nicolás Quesada <nicolas@xanadu.ai>

In this section we study the multidimensional Hermite polynomials originally introduced by C. Hermite in 1865. Mizrahi :cite:`mizrahi1975generalized` provides an exhaustive reference on this subject. We however, follow the succinct treatment of Kok and Braunstein :cite:`kok2001multi`.

In the next section, where we discuss quantum Gaussian states, we will explain how these polynomials relate to hafnians and loop hafnians. For the moment just let us introduce them and study their formal properties.

Generating function definition
******************************
Given two complex vectors :math:`\alpha,\beta \in \mathbb{C}^\ell` and a symmetric matrix :math:`\mathbf{B} = \mathbf{B}^T \in \mathbb{C}^{\ell \times \ell}`,

.. math::
    G_B(\alpha,\beta) = \exp\left( \alpha \mathbf{B} \beta^T - \tfrac{1}{2}\beta \mathbf{B} \beta^T\right) = \sum_{\mathbf{m} \geq \mathbf{0}} \prod_{i=1}^{\ell} \frac{\beta_i^{m_i}}{m_i!} H_{\mathbf{m}}^{(\mathbf{B})}(\alpha),

where the notation :math:`\mathbf{m} \geq \mathbf{0}` is used to indicate that the sum goes over all vectors  in :math:`\mathbb{N}^{\ell}_0` (the set of vectors of nonnegative integers of size :math:`\ell`). This generating function provides an implicit definition of the multidimensional Hermite polynomials.
It is also straightforward to verify that :math:`H_{\mathbf{0}}^{(\mathbf{B})}(\alpha) = 1`.

In the one dimensional case, :math:`\ell=1`, one can compare the generating function above with the ones for the "probabilists' Hermite polynomials" :math:`He_n(x)` and "physicists' Hermite polynomials" :math:`H_n(x)` to find

.. math::
    He_n(x) = H_{n}^{([1])}(x), \\
    H_n(x) = H_{n}^{([2])}(x).

.. tip::
   The multidimensional Hermite polynomials are *Implemented as* :func:`thewalrus.hermite_multidimensional`.


Recursion relation
******************
Based on the generating function introduced in the previous section one can derive the following recursion relation

.. math::
    H_{\mathbf{m}+\mathbf{e}_i}^{(\mathbf{B})}(\alpha) - \left(\sum_{j=1}^\ell B_{i,j} \alpha_j \right) H_{\mathbf{m}}^{(\mathbf{B})}(\alpha) + \sum_{j=1}^\ell B_{i,j} m_j H_{\mathbf{m}-\mathbf{e}_j}^{(\mathbf{B})}(\alpha) = 0,


where :math:`\mathbf{e}_j` is a vector with zeros in all its entries except in the :math:`i^{\text{th}}` entry where it has a one.




From this recursion relation, or by Taylor expanding the generating function, one easily finds

.. math::
    H_{\mathbf{e}_i}^{(\mathbf{B})}(\alpha) = \sum_{j=1}^\ell B_{ij} \alpha_j.


Using this recursion relation one can calculate all the multidimensional Hermite polynomials up to a given cutoff.


The connection between the multidimensional Hermite polynomials and **pure** Gaussian states was reported by Wolf :cite:`wolf1974canonical`, and later by Kramer, Moshinsky and Seligman :cite:`kramer1975group`. This same connection was also pointed out by Doktorov, Malkin and Man'ko in the context of vibrational modes of molecules :cite:`doktorov1977dynamical`.
Furthermore, this connection was later generalized to **mixed** Gaussian states by Dodonov, Man'ko and Man'ko :cite:`dodonov1994multidimensional`.
