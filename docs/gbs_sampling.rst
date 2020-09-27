.. role:: raw-latex(raw)
   :format: latex

.. role:: html(raw)
   :format: html

.. _sampling:


Gaussian Boson Sampling
=======================
.. sectionauthor:: Nicolas Quesada <nicolas@xanadu.ai>

What is Gaussian Boson Sampling
*******************************

Gaussian Boson Sampling was introduced in Ref. :cite:`hamilton2017gaussian` as a problem that could potentially show how a non-universal quantum computer shows exponential speed ups over a classical computer. In the ideal scenario a certain number :math:`N` of squeezed states are sent into :math:`M \times M` interferometer and are then proved by photon-number resolving detectors. Hamilton et al. argue that under certain conjectures and assumptions it should be extremely inefficient for a classical computer to generate the **samples** that the quantum optical experiment just sketched generates by construction. Note that the setup described by Hamilton et al. is a special case of a Gaussian state. We consider Gaussian states, that except for having zero mean, are arbitrary.



In this section we describe a classical algorithm introduced in Ref. :cite:`quesada2019classical`, that generates GBS samples in exponential time in the number of photons measured. This algorithm reduces the generation of a sample to the calculation of a chain of conditional probabilities, in which subsequent modes are interrogated as to how many photons should be detected conditioned on the detection record of previously interrogated modes. The exponential complexity of the algorithm stems from the fact that the conditional probabilities necessary to calculate a sample are proportional to hafnians of matrices of size :math:`2N\times 2N`, where :math:`N` is the number of photons detected.


An exponential-time classical algorithm for GBS
***********************************************
As explained in the :ref:`previous <gbs>` section the reduced or marginal density matrices of a Gaussian state are also Gaussian, and it is straightforward to compute their marginal covariance matrices given the covariance matrix of the global quantum state.

Let :math:`\bm{\Sigma}` be the Husimi covariance matrix of the Gaussian state being measured with photon-number resolving detectors. Let :math:`S = (i_0,i_1,\ldots,i_{m-1})` be a set of indices specifying a subset of the modes. In particular we write :math:`S=[k] = (0,1,2,\ldots, k-1)` for the first :math:`k` modes. We write :math:`\bm{\Sigma}_{(S)}` to indicate the covariance matrix of the modes specified by the index set :math:`S` and define :math:`\bm{A}^{S} := \bm{X} \left[\mathbb{I} - \left( \bm{\Sigma}_{(S)}\right)^{-1} \right]`.

As shown in the previous section, these matrices can be used to calculate photon-number probabilities as

.. math::
	p(\bm{N} = \bm{n}) = \frac{\text{haf}(\bm{A}^{S}_{(\bm{n})})}{  \bm{n}! \sqrt{\det(\bm{\Sigma}_{(S)})}}

where :math:`\bm{N}=\left\{N_{i_1},N_{i_2},\ldots,N_{i_m} \right\}` is a random variable denoting the measurement outcome, and :math:`\bm{n} = \left\{n_{i_1},n_{i_2},\ldots,n_{i_m} \right\}` is a set of integers that represent the actual outcomes of the photon number measurements and :math:`\bm{n}! = \prod_{j=0}^{m-1} n_j!`

Now, we want to introduce an algorithm to generate samples of the random variable :math:`\{N_0,\ldots,N_{M-1}\}` distributed according to the GBS probability distribution. To generate samples, we proceed as follows: First, we can always calculate the following probabilities

.. math::
	p(N_0=n_0) = \frac{\text{Haf}\left(\bm{A}^{[0]}_{(n_0)}\right)}{ n_0! \sqrt{\det(\bm{\Sigma}_{(0)})}},

for :math:`n_0=0,1,\ldots, n_{\max}`, where :math:`n_{\max}` is a cut-off on the maximum number of photons we can hope to detect in this mode.
Having constructed said probabilities, we can always generate a sample for the number of photons in the first mode. This will fix :math:`N_0 = n_0`. Now we want to sample from :math:`N_1|N_0=n_0`. To this end, we use the definition of conditional probability

.. math::
	p(N_1=n_1|N_0=n_0)= \frac{p(N_1=n_1,N_0=n_0)}{p(N_0=n_0)} =\frac{\text{haf}\left(\bm{A}^{[1]}_{(n_0,n_1)}\right)}{n_0! n_1! \sqrt{\det(\bm{\Sigma}_{([2])})}} \frac{1}{p(N_0=n_0)}.

We can, as before, use this formula to sample the number of photons in the second mode conditioned on observing :math:`n_0` photons in the first mode. Note that the factor :math:`p(N_1=n_1)` is already known from the previous step. By induction, the conditional probability of observing :math:`n_k` photons in the :math:`k`-th mode satisfies given observations of :math:`n_0,\ldots,n_{k-1}` photons in the previous :math:`k` modes is given by

.. math::
	p(N_k=n_k|N_{k-1}=n_{k-1},\ldots,N_0=n_0) &=    \frac{p(N_k=n_k,N_{k-1}=n_{k-1},\ldots,N_0=n_0) }{p(N_{k-1}=n_{k-1},\ldots,N_0=n_0)}  \\
	&=\frac{\text{haf}\left(\bm{A}^{[k+1]}_{(n_0,n_2,\ldots,n_k)}\right)}{n_0! n_1! \ldots n_{k+1}! \sqrt{\det(\bm{\Sigma}_{([k])})}} \frac{1}{p(N_{k-1}=n_{k-1},\ldots,N_0=n_0)},

where :math:`p(N_{k-1}=n_{k-1},\ldots,N_0=n_0)` has already been calculated from previous steps. The algorithm then proceeds as follows: for mode :math:`k`, we use the previous equation to sample the number of photons in that mode conditioned on the number of photons in the previous :math:`k` modes. Repeating this sequentially for all modes produces the desired sample.



.. tip::

   To generate samples from a gaussian state specified by a quadrature covariance matrix use :func:`thewalrus.samples.generate_hafnian_sample`.

      Note that the above algorithm can also be generalized to states with finite means for which one only needs to provide the mean with the optional argument ``mean``.


Threshold detection samples
***************************
Note the arguments presented in the previous section can also be generalized to threshold detection. In this case one simple need to replace :math:`\text{haf} \to \text{tor}` and :math:`\bm{A}^{[k+1]}_{(n_0,n_2,\ldots,n_k)} \to \bm{O}^{[k+1]}_{(n_0,n_2,\ldots,n_k)}` where :math:`\bm{O}^{S} = \left[\mathbb{I} - \left( \bm{\Sigma}_{(S)}\right)^{-1} \right]`.

.. tip::

   To generate threshold samples from a gaussian state specified by a quadrature covariance matrix use :func:`thewalrus.samples.generate_torontonian_sample`.


Sampling of classical states
****************************

In the previous section it was mentioned that states whose covariance matrix satisfies :math:`\bm{V} \geq \frac{\hbar}{2}\mathbb{I}` are termed classical. These designation is due to the fact that for these states it is possible to obtain a polynomial (cubic) time algorithm to generate photon number or threshold samples :cite:`rahimi2015can`.

.. tip::

   To generate photon number or threshold samples from a classical gaussian state specified by a quadrature covariance matrix use :func:`thewalrus.samples.hafnian_sample_classical_state` or :func:`thewalrus.samples.torontonian_sample_classical_state`.

Note that one can use this observation to sample from a probability distribution that is proportional to the permanent of a positive semidefinite matrix, for details on how this is done cf. Ref. :cite:`jahangiri2020point`.

.. tip::

   To generate photon number samples from a thermal state parametrized by a positive semidefinite *real* matrix use the module :func:`thewalrus.csamples`.

