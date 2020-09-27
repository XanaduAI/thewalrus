.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html
.. _gbs:


Gaussian States in the Fock basis
=================================
.. sectionauthor:: Nicolás Quesada <nicolas@xanadu.ai>

In this section we show how the matrix elements of so-called Gaussian states in the Fock basis are related to the hafnians and loop hafnians introduced in the previous sections.

.. note:: The results presented in this page use the conventions introduced in the :ref:`notation <notation>` page and the reader is strongly encouraged to get familiar with them.

Wigner functions and Gaussian states
************************************
The quantum state :math:`\varrho` of an :math:`\ell`-mode system can be uniquely characterized by its Wigner function :cite:`serafini2017quantum`

.. math::
	\label{Eq: Wigner}
	W(\vec \alpha; \varrho) = \int \frac{\text{d}\vec \xi}{\pi^{2\ell}} \text{Tr}[\varrho \hat D(\vec \xi)] \exp\left(\vec \alpha^T \bm{\Omega} \  \vec \xi\right),

where :math:`\vec \alpha = (\alpha_0,\ldots, \alpha_{\ell-1},\alpha_0^*,\ldots, \alpha_{\ell-1}^*)` and similarly :math:`\vec \xi = (\xi_0,\ldots, \xi_{\ell-1},\xi_0^*,\ldots, \xi_{\ell-1}^*)` are bivectors of complex amplitudes where the second half of the vector is the complex conjugate of the first half. The displacement operator is defined as :math:`\hat D(\xi):=\exp(\vec{\xi}^T \bm{\Omega} \hat \zeta)`, where :math:`\bm{\Omega}= \left[   \begin{smallmatrix} 	0 &  \mathbb{I} \\ 	-\mathbb{I} & 0  \end{smallmatrix} \right]` is the symplectic form and :math:`\hat\zeta_j` is an operator vector of the mode creation and annihilation operators. Recall,  :math:`\ell` being the number of modes, we have :math:`\hat\zeta_j=\hat a_j` and :math:`\hat \zeta_{\ell+j}=\hat a_j^\dagger` for  :math:`j=1,\cdots,\ell`.
These bosonic creation and annihilation operators satisfy the canonical commutation relations :math:`[\hat a_i, \hat a_j]` = 0 and :math:`[\hat a_i, \hat a_j^\dagger] = \delta_{ij}`.

A quantum state is called Gaussian if its Wigner function is Gaussian :cite:`weedbrook2012gaussian`. Any multimode Gaussian state :math:`\varrho` is completely parametrized by its first and second moments, namely the  vector of  means :math:`\vec{\beta}` with components

.. math:: \label{eq:WignerMeans}
	\vec \beta_j = \text{Tr}[\varrho\hat\zeta_j],

and the Wigner-covariance matrix :math:`\bm{\sigma}` with entries

.. math:: \label{eq:WignerCovMat}
	\bm{\sigma}_{jk} = \text{Tr}[\varrho \{\hat{\zeta}_j,\hat{\zeta}_k^\dagger \}]/2 - \vec \beta_j \vec \beta_k^*,

where :math:`\{x,y\} := xy +yx` denotes the anti-commutator.

The variables :math:`(\alpha_0,\ldots,\alpha_{\ell-1})` are said to be complex normal distributed with mean :math:`(\beta_0,\ldots,\beta_{\ell-1})   \in \mathbb{C}^{\ell}` and covariance matrix :math:`{\bm{\sigma}}`  :cite:`picinbono1996second` which furthermore needs to satisfy the uncertainty relation :cite:`simon1994quantum`

.. math:: \label{eq:uncertainty}
	{\bm{\sigma}} + \bm{Z}/2 \geq 0,

where :math:`\bm{Z} = \left( \begin{smallmatrix} \mathbb{I} & 0\\ 0& -\mathbb{I} \end{smallmatrix} \right)`. The covariance matrix :math:`\bm{\sigma}` is customarily parametrized in the following block form :cite:`picinbono1996second`

.. math:: \bm{\sigma} = \left[\begin{array}{c|c}
	\bm{\Gamma} & \bm{C} \\
	\hline
	\bm{C}^* & \bm{\Gamma}^*
	\end{array} \right],

where :math:`\bm{\Gamma}` is hermitian and positive definite, while :math:`\bm{C}` is symmetric.



Gaussian states in the quadrature basis
***************************************

Historically, Gaussian states are parametrized not in terms of the covariance matrix :math:`\bm{\sigma}` of the complex amplitudes :math:`\alpha_j`, but rather in terms of its quadrature components, the canonical positions :math:`q_j` and canonical momenta :math:`p_j`,

.. math::
	\alpha_j = \frac{1}{\sqrt{2 \hbar}} \left( q_j+ i p_j \right),

where :math:`\hbar` is a positive constant. There are at least three conventions for the value of this constant, :math:`\hbar \in \{1/2,1,2 \}`.

It is convenient to write the following vector :math:`\bm{r} = (q_0,\ldots,q_{\ell-1},p_0,\ldots,p_{\ell-1})` whose mean is related to the vector of means :math:`\vec \beta` as

.. math::
	\vec \beta &= \frac{1}{\sqrt{\hbar}}\bm{R} \bm{r}, \\
	\bm{\bar{r}} &= \sqrt{\hbar} \bm{R}^\dagger \vec \beta, \\
	\bm{R} &= \frac{1}{\sqrt{2}}\begin{bmatrix}
		\mathbb{I} & i \mathbb{I}  \\
		\mathbb{I} & -i \mathbb{I}
		\end{bmatrix}.

Similarly the complex normal covariance matrix :math:`\bm{\sigma}` of the variables :math:`(\alpha_0,\ldots,\alpha_{\ell-1})` is related to the normal covariance matrix :math:`\bm{V}` of the variables :math:`\bm{r} = (q_0,\ldots,q_{\ell-1},p_0,\ldots,p_{\ell-1})` as

.. math::
	\bm{\sigma} &= \frac{1}{\hbar} \ \bm{R} \bm{V} \bm{R}^\dagger \\
	\bm{V} &= {\hbar} \ \bm{R}^\dagger \bm{\sigma} \bm{R}.

.. tip::

   To convert between the complex covariance matrix :math:`\bm{\sigma}` and the quadrature covariance matrix :math:`\bm{V}` use the functions :func:`thewalrus.quantum.Qmat` and :func:`thewalrus.quantum.Covmat`


An important property of Gaussian states is that reduced (or marginal) states of a global Gaussian state are also Gaussian. This implies that the reduced covariance matrix of a subsystem of a Gaussian state together with a reduced vector of means fully characterize a reduced Gaussian state. The reduced covariance matrix for modes :math:`S = i_1,\ldots,i_n` is obtained from the covariance matrix of the global state :math:`\bm{\sigma}` or :math:`\bm{V}` by keeping the columns and rows  :math:`i_1,\ldots,i_n` and :math:`i_1+\ell,\ldots,i_n+\ell` of the original covariance matrix :math:`\bm{\sigma}`. Similarly one obtains the vector of means by keeping only entries :math:`i_1,\ldots,i_n` and :math:`i_1+\ell,\ldots,i_n+\ell` of the original vector of means :math:`\vec \beta` or :math:`\bm{\bar{r}}`. Using the :ref:`notation <notation>` previously introduced, one can succinctly write the covariance matrix of modes :math:`S=i_1,\ldots,i_m` as :math:`\bm{\sigma}_{(S)}` or :math:`\bm{V}_{(S)}` , and similarly the vector of means as :math:`\vec{\beta}_{(S)}` or :math:`\bm{\bar{r}}_{(S)}`.

.. tip::

   To obtain the reduced covariance matrix and vector of means for a certain subset of the modes use :func:`thewalrus.quantum.reduced_gaussian`.


Note that for :math:`\bm{V}` to be a valid **quantum** covariance matrix it needs to be symmetric and satisfy the uncertainty relation

.. math::
	\bm{V} + i \frac{\hbar}{2} \bm{\Omega} \geq 0.


.. tip::

   To verify if a given quadrature covariance matrix is a valid quantum covariance matrix use the function :func:`thewalrus.quantum.is_valid_cov`

A Gaussian state is pure :math:`\varrho = \ket{\psi} \bra{\psi}` if and only if :math:`\text{det}\left( \tfrac{2}{\hbar} \bm{V} \right) = 1`.

.. tip::

   To verify if a given quadrature covariance matrix is a valid quantum covariance matrix and corresponds to a pure state use the function :func:`thewalrus.quantum.is_pure_cov`

Finally, there is a special subset of Gaussian states called **classical** whose covariance matrix satisfies

.. math::
	\bm{V} \geq \tfrac{\hbar}{2}\mathbb{I}.

This terminology is explained in the next section when sampling is discussed.

.. tip::

   To verify if a given quadrature covariance matrix is a valid quantum covariance matrix and corresponds to a classical state use the function :func:`thewalrus.quantum.is_classical_cov`



Gaussian states in the Fock basis
*********************************
In this section we use a generalization :cite:`quesada2019franck,quesada2019simulating` of the results of Hamilton et al. :cite:`hamilton2017gaussian` by providing an explicit expression for Fock basis matrix elements :math:`\langle \bm{m} | \rho | \bm{n} \rangle`, :math:`\bm{n} = (n_0,\ldots, n_{\ell-1}), \bm{m} = (m_0,\ldots, m_{\ell-1})`, of an :math:`\ell`-mode Gaussian state :math:`\rho` with covariance matrix :math:`\bm{\sigma}` and displacement vector :math:`\vec \beta`.
Note that these matrix elements can also be calculated using multidimensional Hermite polynomials as shown by Dodonov et al. :cite:`dodonov1994multidimensional`. Depending on how many of these elements are required one can prefer to calculate loop hafnians or multidimensional Hermite polynomials. In particular if one only needs a few matrix elements it is more advantageous to use the formulas derived below. On the other hand if one requires **all** the matrix elements up to a certain Fock occupation cutoff it is more efficient to use the methods of Dodonov et al., which are also implemented in this library.


We first define the following useful quantities:

.. math:: \bm{X} &=  \begin{bmatrix}
		0 &  \mathbb{I} \\
		\mathbb{I} & 0
		\end{bmatrix} , \\
	\bm{\Sigma} &= \bm{\sigma} +\tfrac{1}{2} \mathbb{I}_{2\ell},\\
	T &=\frac{\exp\left(-\tfrac{1}{2} \vec \beta^\dagger \bm{\Sigma}^{-1} \vec \beta \right)}{ \sqrt{\text{det}(\bm{\Sigma}) \prod_{s=1}^\ell n_s! m_s!}},\\
	\bm{p} &= (n_0,\ldots,n_{\ell-1},m_0,\ldots,m_{\ell-1}).

We refer to :math:`\bm{\Sigma}` as the **Husimi** covariance matrix.

As shown in detail in  Appendix A of Ref. :cite:`quesada2019simulating`, the Fock matrix elements of a Gaussian state :math:`\rho` are given by the expression

.. math:: \label{Eq: lhaf}
	\langle \bm{m} | \rho | \bm{n} \rangle  = T \times  \text{lhaf}( \text{vid}(\bm{A}_{\bm{p}}, \gamma_{ \bm{p}}) ),

where :math:`\text{lhaf}` is the :ref:`loop hafnian<loophafnian>` and :math:`\text{vid}` is the vector in diagonal notation introduced in the :ref:`notation<notation>` section.


Note that one can also obtain the probability of detecting a certain photon number pattern :math:`\bm{n} = (n_0,\ldots,n_{\ell-1})` by calculating

.. math:: p(\bm{n}|\varrho) = \langle \bm{n} | \varrho | \bm{n} \rangle.


.. tip::

   To obtain the matrix element of gaussian state with quadrature covariance matrix :math:`\bm{V}` and vector of means :math:`\bm{r}` use the function :func:`thewalrus.quantum.density_matrix_element`.

.. tip::

   To obtain the Fock space density matrix of gaussian state with quadrature covariance matrix :math:`\bm{V}` and vector of means :math:`\bm{r}` use the function :func:`thewalrus.quantum.density_matrix`.


In the case where the Gaussian state :math:`\varrho = |\psi \rangle \langle \psi|` is pure then the matrix element

.. math:: \langle \bm{m} | \varrho | \bm{n} \rangle = \langle \bm{m} | \psi \rangle \langle \psi| \bm{n} \rangle

factorizes into a product of two amplitudes. In Ref. :cite:`quesada2019franck` it was shown that the Fock **amplitude** of a gaussian state is also given by a loop hafnian. Then, for pure states the matrix :math:`\bm{\bar{A}} = \bm{\bar{B}} \oplus \bm{\bar{B}}^*`.



.. tip::

   To obtain the overlap of a *pure* gaussian state with quadrature covariance matrix :math:`\bm{V}` and vector of means :math:`\bm{r}` and a given Fock state :math:`\langle \bm{n}|` use the function :func:`thewalrus.quantum.pure_state_amplitude`.

.. tip::

   To obtain the Fock space state vector (ket) of a pure gaussian state with quadrature covariance matrix :math:`\bm{V}` and vector of means :math:`\bm{r}` use the function :func:`thewalrus.quantum.state_vector`.




Gaussian states and threshold detection
***************************************
In the last section we sketched how to obtain the probability that a certain photon-number outcome is obtained when a Gaussian state is measured with photon-number detectors. In this section we show how to obtain the analogous probability for the case of threshold detectors. These binary outcome detectors can only distinguish between the the vacuum state and occupied states, and thus for a single mode they are described by the POVM elements

.. math::
	\hat{\Pi}_0^{(i)} = \ket{0_i} \bra{0_i} \text{ and } \hat{\Pi}_1^{(i)} = 1_i - \hat{\Pi}_0^{(i)},

where :math:`\ket{0_i}` is the vacuum state of mode :math:`i` and :math:`1_i` is the identity in the Hilbert space of mode :math:`i`.

For an :math:`\ell` mode Gaussian state with zero mean, the outcome of threshold detection in all of its modes is described by a bitstring vector :math:`\bm{n} = (n_0,\ldots,n_{\ell-1})` and the probability of the event is given by Born's rule according to

.. math::
	p(\bm{n}|\varrho) &= \text{Tr} \left( \prod_{i=1}^{\ell} \Pi_{n_i}^{(i)} \varrho  \right) = \frac{\text{tor} \left(\bm{O}_{\{ \bm{n}\}} \right)}{\sqrt{\text{det}(\bm{\sigma})}}, \\
	\bm{O} &= 	\left(\mathbb{I}_{2\ell} - \bm{\Sigma}^{-1} \right)

where :math:`\text{tor}` is the Torontonian. For :math:`2 \ell \times 2 \ell` matrix :math:`\bm{O}` the Torontonian is defined as

.. math::
	\text{tor}(\bm{O}) = \sum_{S \in P([\ell])} (-1)^{|S|} \frac{1}{\sqrt{\det\left(\mathbb{I} - \bm{O}_{(S)}\right)}}

The torontonian can be thought of as a generating function for hafnians (cf. the trace algorithm formula in :ref:`algorithms <algorithms>` section).

.. tip::

   The torontonian is implemented as :func:`thewalrus.tor`.





