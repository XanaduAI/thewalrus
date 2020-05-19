.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html
.. _hermite:


Multidimensional Hermite polynomials
====================================
.. sectionauthor:: Nicolás Quesada <nicolas@xanadu.ai>

In this section we study the multidimensional Hermite polynomials originally introduced by C. Hermite in 1865. See Mizrahi :cite:`mizrahi1975generalized`, Berkowitz et al. :cite:`berkowitz1970calculation` and Kok and Braunstein :cite:`kok2001multi` for more details.

In the next section, where we discuss quantum Gaussian states, we will explain how these polynomials relate to hafnians and loop hafnians. For the moment just let us introduce them and study their formal properties.

Generating function
*******************
Given two complex vectors :math:`\alpha,\beta \in \mathbb{C}^\ell` and a symmetric matrix :math:`\bm{B} = \bm{B}^T \in \mathbb{C}^{\ell \times \ell}`,

.. math::
    F_B(\alpha,\beta) &= \exp\left( \alpha \bm{B} \beta^T - \tfrac{1}{2}\beta \bm{B} \beta^T\right) = \sum_{\bm{m} \geq \bm{0}} \prod_{i=1}^{\ell} \frac{\beta_i^{m_i}}{m_i!} H_{\bm{m}}^{(\bm{B})}(\alpha),\\
    K_B(\alpha,\beta) &= \exp\left( \alpha  \beta^T - \tfrac{1}{2}\beta \bm{B} \beta^T\right) = \sum_{\bm{m} \geq \bm{0}} \prod_{i=1}^{\ell} \frac{\beta_i^{m_i}}{m_i!} G_{\bm{m}}^{(\bm{B})}(\alpha),

where the notation :math:`\bm{m} \geq \bm{0}` is used to indicate that the sum goes over all vectors  in :math:`\mathbb{N}^{\ell}_0` (the set of vectors of nonnegative integers of size :math:`\ell`). This generating function provides an implicit definition of the multidimensional Hermite polynomials.
It is also straightforward to verify that :math:`H_{\bm{0}}^{(\bm{B})}(\alpha) = G_{\bm{0}}^{(\bm{B})}(\alpha) 1`. Finally, one can connect the standard Hermite polynomials
:math:`H_{\bm{m}}^{(\bm{B})}(\alpha)` to the modified Hermite polynomials :math:`G_{\bm{m}}^{(\bm{B})}(\alpha)` via

.. math::
	H_{\bm{m}}^{(\bm{B})}(\alpha) = G_{\bm{m}}^{(\bm{B})}(\alpha \bm{B}).

In the one dimensional case, :math:`\ell=1`, one can compare the generating function above with the ones for the "probabilists' Hermite polynomials" :math:`He_n(x)` and "physicists' Hermite polynomials" :math:`H_n(x)` to find

.. math::
    He_n(x) = H_{n}^{([1])}(x), \\
    H_n(x) = H_{n}^{([2])}(x).

.. tip::
   *The standard multidimensional Hermite polynomials are implemented as* :func:`thewalrus.hermite_multidimensional`. *The modified Hermite polynomials can be obtained by passing the extra argument* :code:`modified=True`.


Recursion relation
******************
Based on the generating functions introduced in the previous section one can derive the following recursion relations

.. math::
    H_{\bm{m}+\bm{e}_i}^{(\bm{B})}(\alpha) - \left(\sum_{j=1}^\ell B_{i,j} \alpha_j \right) H_{\bm{m}}^{(\bm{B})}(\alpha) + \sum_{j=1}^\ell B_{i,j} m_j H_{\bm{m}-\bm{e}_j}^{(\bm{B})}(\alpha) &= 0,\\
    G_{\bm{m}+\bm{e}_i}^{(\bm{B})}(\alpha) -  \alpha_i  G_{\bm{m}}^{(\bm{B})}(\alpha) + \sum_{j=1}^\ell B_{i,j} m_j G_{\bm{m}-\bm{e}_j}^{(\bm{B})}(\alpha) &= 0,


where :math:`\bm{e}_j` is a vector with zeros in all its entries except in the :math:`i^{\text{th}}` entry where it has a one.




From this recursion relation, or by Taylor expanding the generating function, one easily finds

.. math::
    H_{\bm{e}_i}^{(\bm{B})}(\alpha) &= \sum_{j=1}^\ell B_{ij} \alpha_j,\\
    G_{\bm{e}_i}^{(\bm{B})}(\alpha) &= \alpha_i.


Using this recursion relation one can calculate all the multidimensional Hermite polynomials up to a given cutoff.


The connection between the multidimensional Hermite polynomials and **pure** Gaussian states was reported by Wolf :cite:`wolf1974canonical`, and later by Kramer, Moshinsky and Seligman :cite:`kramer1975group`. This same connection was also pointed out by Doktorov, Malkin and Man'ko in the context of vibrational modes of molecules :cite:`doktorov1977dynamical`.
Furthermore, this connection was later generalized to **mixed** Gaussian states by Dodonov, Man'ko and Man'ko :cite:`dodonov1994multidimensional`. These matrix elements have the form

.. math::
	C \times \frac{H_{\bm{m}}^{(\bm{B})}(\alpha)}{\sqrt{\bm{m}!}} = C \times \frac{G_{\bm{m}}^{(\bm{B})}(\alpha \bm{B})}{\sqrt{\bm{m}!}}.

To obtain the standard or modified Hermite polynomials renormalized by the square root of the factorial of its index :math:`\sqrt{\bm{m}!}` one can pass the optional argument :code:`renorm=True`.



Multidimensional Hermite polynomials and hafnians
*************************************************
By connecting the results in page 815 of Dodonov et al. :cite:`dodonov1994multidimensional` with the results in page 546 of Kan :cite:`kan2008moments` one obtains the following relation between the hafnian and the multidimensional Hermite polynomials

.. math::
	H_{\bm{m}}^{(-\bm{B})}(\bm{0}) = G_{\bm{m}}^{(-\bm{B})}(\bm{0})= \text{haf}(\bm{B}_{\bm{m}}),

and moreover one can generalize it to

.. math::
	G_{\bm{m}}^{(-\bm{B})}(\alpha) = \text{lhaf}\left(\text{vid}(\bm{B}_{\bm{m}},\alpha_{\bm{m}})\right),

for loop hafnians. With these two identifications one can use the recursion relations of the multidimensional Hermite polynomials to calculate all the hafnians of the reductions of a given matrix up to a given cutoff.

With these observations and using the recursion relations for the Hermite polynomials and setting :math:`\bm{m}=\bm{1} - \bm{e}_i, \  \alpha = 0` one easily derives the well known Laplace expansion for the hafnian (cf. Sec. 4.1 of :cite:`barvinok2016combinatorics`)

.. math::
	\text{haf}(\bm{B}) = \sum_{j \neq i} B_{i,j} \text{haf}(\bm{B}_{-i-j}),

where :math:`j` is a fixed index and :math:`\bm{B}_{-i-j}` is the matrix obtained from :math:`\bm{B}` by removing rows and columns :math:`i` and :math:`j`.
