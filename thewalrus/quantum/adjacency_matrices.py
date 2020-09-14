# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions for rescaling adjacency matrices as well as calculating the Qmat
covariance matrix from an adjacency matrix.
"""

import numpy as np
from scipy.optimize import root_scalar

from .conversions import Xmat, Covmat
from .means_and_variances import mean_clicks


def adj_scaling_torontonian(A, c_mean):
    r""" Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that encodes it has
    give a mean number of clicks equal to ``c_mean`` when measured with
    threshold detectors.

    Args:
        A (array): adjacency matrix
        c_mean (float): mean photon number of the Gaussian state

    Returns:
        float: scaling parameter
    """
    n, _ = A.shape
    if c_mean < 0 or c_mean > n:
        raise ValueError("The mean number of clicks should be smaller than the number of modes")

    vals = np.linalg.svd(A, compute_uv=False)
    localA = A / vals[0]  # rescale the matrix so that the singular values are between 0 and 1.

    def cost(x):
        r""" Cost function giving the difference between the wanted number of clicks and the number
        of clicks at a given scaling value. It assumes that the adjacency matrix has been rescaled
        so that it has singular values between 0 and 1.

        Args:
            x (float): scaling value

        Return:
            float: difference between desired and obtained mean number of clicks
        """
        if x >= 1.0:
            return c_mean - n
        if x <= 0:
            return c_mean
        return c_mean - _mean_clicks_adj(x * localA)

    res = root_scalar(cost, x0=0.5, bracket=(0.0, 1.0))  # Do the optimization

    if not res.converged:
        raise ValueError("The search for a scaling value failed")
    return res.root / vals[0]


def _mean_clicks_adj(A):
    r""" Given an adjacency matrix this function calculates the mean number of clicks.
    For this to make sense the user must provide a matrix with singular values
    less than or equal to one. See Appendix A.3 of <https://arxiv.org/abs/1902.00462>`_
    by Banchi et al.

    Args:
        A (array): rescaled adjacency matrix

    Returns:
        float: mean number of clicks
    """
    n, _ = A.shape
    idn = np.identity(n)
    X = np.block([[0 * idn, idn], [idn, 0 * idn]])
    B = np.block([[A, 0 * A], [0 * A, np.conj(A)]])
    Q = np.linalg.inv(np.identity(2 * n) - X @ B)
    return mean_clicks(Covmat(Q))


def adj_scaling(A, n_mean):
    r""" Returns the scaling parameter by which the adjacency matrix A
    should be rescaled so that the Gaussian state that endodes it has
    a total mean photon number n_mean.

    Args:
        A (array): Adjacency matrix
        n_mean (float): Mean photon number of the Gaussian state

    Returns:
        float: Scaling parameter
    """
    eps = 1e-10
    ls = np.linalg.svd(A, compute_uv=False)
    max_sv = ls[0]
    a_lim = 0.0
    b_lim = 1.0 / (eps + max_sv)
    x_init = 0.5 * b_lim

    if 1000 * eps >= max_sv:
        raise ValueError("The singular values of the matrix A are too small.")

    def mean_photon_number(x, vals):
        r""" Returns the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A where vals are the singular values of A

        Args:
            x (float): Scaling parameter
            vals (array): Singular values of the matrix A

        Returns:
            n_mean: Mean photon number in the Gaussian state
        """
        vals2 = (x * vals) ** 2
        n = np.sum(vals2 / (1.0 - vals2))
        return n

    # The following function is implicitly tested in test_adj_scaling
    def grad_mean_photon_number(x, vals):  # pragma: no cover
        r""" Returns the gradient od the mean number of photons in the Gaussian state that
        encodes the adjacency matrix x*A with respect to x.
        vals are the singular values of A

        Args:
            x (float): Scaling parameter
            vals (array): Singular values of the matrix A

        Returns:
            d_n_mean: Derivative of the mean photon number in the Gaussian state
                with respect to x
        """
        vals1 = vals * x
        dn = (2.0 / x) * np.sum((vals1 / (1 - vals1 ** 2)) ** 2)
        return dn

    f = lambda x: mean_photon_number(x, ls) - n_mean
    df = lambda x: grad_mean_photon_number(x, ls)
    res = root_scalar(f, fprime=df, x0=x_init, bracket=(a_lim, b_lim))

    if not res.converged:
        raise ValueError("The search for a scaling value failed")

    return res.root


def adj_to_qmat(A, n_mean):
    r""" Returns the Qmat xp-covariance matrix associated to a graph with
    adjacency matrix :math:`A` and with mean photon number :math:`n_{mean}`.

    Args:
        A (array): a :math:`N\times N` ``np.float64`` (symmetric) adjacency matrix
        n_mean (float): mean photon number of the Gaussian state

    Returns:
        array: the :math:`2N\times 2N` Q matrix.
    """
    n, m = A.shape

    if n != m:
        raise ValueError("Matrix must be square.")

    sc = adj_scaling(A, n_mean)
    Asc = sc * A
    A = np.block([[Asc, 0 * Asc], [0 * Asc, Asc.conj()]])
    I = np.identity(2 * n)
    X = Xmat(n)
    Q = np.linalg.inv(I - X @ A)
    return Q
