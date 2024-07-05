# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decomposition tests"""
# pylint: disable=no-self-use, assignment-from-no-return
import numpy as np
import pytest
from scipy.linalg import block_diag

from thewalrus.random import random_interferometer as haar_measure
from thewalrus.random import random_symplectic
from thewalrus.decompositions import williamson, blochmessiah, takagi, pre_iwasawa, iwasawa
from thewalrus.symplectic import sympmat as omega
from thewalrus.quantum.gaussian_checks import is_symplectic


class TestWilliamsonDecomposition:
    """Tests for the Williamson decomposition"""

    @pytest.fixture
    def create_cov(self, hbar, tol):
        """create a covariance state for use in testing.

        Args:
            nbar (array[float]): vector containing thermal state values

        Returns:
            tuple: covariance matrix and symplectic transform
        """

        def _create_cov(nbar):
            """wrapped function"""
            n = len(nbar)
            O = omega(n)

            # initial vacuum state
            cov = np.diag(2 * np.tile(nbar, 2) + 1) * hbar / 2

            # interferometer 1
            U1 = haar_measure(n)
            S1 = np.vstack([np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])])

            # squeezing
            r = np.log(0.2 * np.arange(n) + 2)
            Sq = block_diag(np.diag(np.exp(-r)), np.diag(np.exp(r)))

            # interferometer 2
            U2 = haar_measure(n)
            S2 = np.vstack([np.hstack([U2.real, -U2.imag]), np.hstack([U2.imag, U2.real])])

            # final symplectic
            S_final = S2 @ Sq @ S1

            # final covariance matrix
            cov_final = S_final @ cov @ S_final.T

            # check valid symplectic transform
            assert np.allclose(S_final.T @ O @ S_final, O)

            # check valid state
            eigs = np.linalg.eigvalsh(cov_final + 1j * (hbar / 2) * O)
            eigs[np.abs(eigs) < tol] = 0
            assert np.all(eigs >= 0)

            if np.allclose(nbar, 0):
                # check pure
                assert np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))
            else:
                # check not pure
                assert not np.allclose(np.linalg.det(cov_final), (hbar / 2) ** (2 * n))

            return cov_final, S_final

        return _create_cov

    def test_square_validation(self):
        """Test that the graph_embed decomposition raises exception if not square"""
        A = np.random.rand(4, 5) + 1j * np.random.rand(4, 5)
        with pytest.raises(ValueError, match="matrix is not square"):
            williamson(A)

    def test_symmetric_validation(self):
        """Test that the graph_embed decomposition raises exception if not symmetric"""
        A = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        with pytest.raises(ValueError, match="matrix is not symmetric"):
            williamson(A)

    def test_even_validation(self):
        """Test that the graph_embed decomposition raises exception if not even number of rows"""
        A = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
        A += A.T
        with pytest.raises(ValueError, match="must have an even number of rows/columns"):
            williamson(A)

    def test_positive_definite_validation(self):
        """Test that the graph_embed decomposition raises exception if not positive definite"""
        A = np.diag([-2, 0.1, 2, 3])
        with pytest.raises(ValueError, match="matrix is not positive definite"):
            williamson(A)

    def test_vacuum_state(self, tol):
        """Test vacuum state"""
        V = np.identity(4)
        Db, S = williamson(V)
        assert np.allclose(Db, np.identity(4), atol=tol, rtol=0)
        assert np.allclose(S, np.identity(4), atol=tol, rtol=0)

    def test_pure_state(self, create_cov, hbar, tol):
        """Test pure state"""
        n = 3
        O = omega(n)

        cov, _ = create_cov(np.zeros([n]))

        Db, S = williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar = 0
        assert np.allclose(nbar, 0, atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)

    def test_mixed_state(self, create_cov, hbar, tol):
        """Test mixed state"""
        n = 3
        O = omega(n)
        nbar_in = np.abs(np.random.rand(n))

        cov, _ = create_cov(nbar_in)

        Db, S = williamson(cov)
        nbar = np.diag(Db) / hbar - 0.5

        # check decomposition is correct
        assert np.allclose(S @ Db @ S.T, cov, atol=tol, rtol=0)
        # check nbar
        assert np.allclose(sorted(nbar[:n]), sorted(nbar_in), atol=tol, rtol=0)
        # check S is symplectic
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)


class TestBlochMessiahDecomposition:
    """Tests for the Bloch Messiah decomposition"""

    @pytest.fixture
    def create_transform(self):
        """create a symplectic transform for use in testing.

        Args:
            n (int): number of modes
            passive (bool): whether transform should be passive or not

        Returns:
            array: symplectic matrix
        """

        def _create_transform(n, passive=True):
            """wrapped function"""
            O = omega(n)

            # interferometer 1
            U1 = haar_measure(n)
            S1 = np.vstack([np.hstack([U1.real, -U1.imag]), np.hstack([U1.imag, U1.real])])

            Sq = np.identity(2 * n)
            if not passive:
                # squeezing
                r = np.log(0.2 * np.arange(n) + 2)
                Sq = block_diag(np.diag(np.exp(-r)), np.diag(np.exp(r)))

            # interferometer 2
            U2 = haar_measure(n)
            S2 = np.vstack([np.hstack([U2.real, -U2.imag]), np.hstack([U2.imag, U2.real])])

            # final symplectic
            S_final = S2 @ Sq @ S1

            # check valid symplectic transform
            assert np.allclose(S_final.T @ O @ S_final, O)
            return S_final

        return _create_transform

    @pytest.mark.parametrize("N", range(50, 500, 50))
    def test_blochmessiah_rand(self, N):
        """Tests blochmessiah function for different matrix sizes."""
        S = random_symplectic(N)
        u, d, v = blochmessiah(S)
        assert np.allclose(u @ d @ v, S)
        assert np.allclose(u.T @ u, np.eye(len(u)))
        assert np.allclose(v.T @ v, np.eye(len(v)))
        assert is_symplectic(u)
        assert is_symplectic(v)

    @pytest.mark.parametrize("M", [np.random.rand(4, 5), np.random.rand(4, 4)])
    def test_blochmessiah_error(self, M):
        """Tests that non-symplectic matrices raise a ValueError in blochmessiah."""
        with pytest.raises(ValueError, match="Input matrix is not symplectic."):
            blochmessiah(M)

    def test_identity(self, tol):
        """Test identity"""
        n = 2
        S_in = np.identity(2 * n)
        O1, S, O2 = blochmessiah(S_in)

        assert np.allclose(O1 @ O2, np.identity(2 * n), atol=tol, rtol=0)
        assert np.allclose(S, np.identity(2 * n), atol=tol, rtol=0)

        # test orthogonality
        assert np.allclose(O1.T, O1, atol=tol, rtol=0)
        assert np.allclose(O2.T, O2, atol=tol, rtol=0)

        # test symplectic
        O = omega(n)
        assert np.allclose(O1 @ O @ O1.T, O, atol=tol, rtol=0)
        assert np.allclose(O2 @ O @ O2.T, O, atol=tol, rtol=0)

    @pytest.mark.parametrize("passive", [True, False])
    def test_transform(self, passive, create_transform, tol):
        """Test decomposition agrees with transform. also checks that passive transform has no squeezing.
        Note: this test also tests the case with degenerate symplectic values"""
        n = 3
        S_in = create_transform(3, passive=passive)
        O1, S, O2 = blochmessiah(S_in)

        # test decomposition
        assert np.allclose(O1 @ S @ O2, S_in, atol=tol, rtol=0)

        # test no squeezing
        if passive:
            assert np.allclose(O1 @ O2, S_in, atol=tol, rtol=0)
            assert np.allclose(S, np.identity(2 * n), atol=tol, rtol=0)

        # test orthogonality
        assert np.allclose(O1.T @ O1, np.identity(2 * n), atol=tol, rtol=0)
        assert np.allclose(O2.T @ O2, np.identity(2 * n), atol=tol, rtol=0)

        # test symplectic
        O = omega(n)
        assert np.allclose(O1.T @ O @ O1, O, atol=tol, rtol=0)
        assert np.allclose(O2.T @ O @ O2, O, atol=tol, rtol=0)
        assert np.allclose(S @ O @ S.T, O, atol=tol, rtol=0)


@pytest.mark.parametrize("n", [5, 10, 50])
@pytest.mark.parametrize("datatype", [np.complex128, np.float64])
@pytest.mark.parametrize("svd_order", [True, False])
def test_takagi(n, datatype, svd_order):
    """Checks the correctness of the Takagi decomposition function"""
    if datatype is np.complex128:
        A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    if datatype is np.float64:
        A = np.random.rand(n, n)
    A += A.T
    r, U = takagi(A, svd_order=svd_order)
    assert np.allclose(A, U @ np.diag(r) @ U.T)
    assert np.all(r >= 0)
    if svd_order is True:
        assert np.all(np.diff(r) <= 0)
    else:
        assert np.all(np.diff(r) >= 0)


# pylint: disable=too-many-arguments
@pytest.mark.parametrize("n", [5, 10, 50])
@pytest.mark.parametrize("datatype", [np.complex128, np.float64])
@pytest.mark.parametrize("svd_order", [True, False])
@pytest.mark.parametrize("half_rank", [0, 1])
@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("null_space", [0, 5, 10])
@pytest.mark.parametrize("offset", [0, 0.5])
def test_degenerate(n, datatype, svd_order, half_rank, phase, null_space, offset):
    """Tests Takagi produces the correct result for very degenerate cases"""
    nhalf = n // 2
    diags = (
        [half_rank * np.random.rand()] * nhalf
        + [np.random.rand() - offset] * (n - nhalf)
        + [0] * null_space
    )
    if datatype is np.complex128:
        U = haar_measure(n + null_space)
    if datatype is np.float64:
        U = np.exp(1j * phase) * haar_measure(n + null_space, real=True)
    A = U @ np.diag(diags) @ U.T
    r, U = takagi(A, svd_order=svd_order)
    assert np.allclose(A, U @ np.diag(r) @ U.T)
    assert np.allclose(U @ U.T.conj(), np.eye(n + null_space))
    assert np.all(r >= 0)
    if svd_order is True:
        assert np.all(np.diff(r) <= 0)
    else:
        assert np.all(np.diff(r) >= 0)


def test_zeros():
    """Verify that the Takagi decomposition returns a zero vector and identity matrix when
    input a matrix of zeros"""
    dim = 4
    a = np.zeros((dim, dim))
    rl, U = takagi(a)
    assert np.allclose(rl, np.zeros(dim))
    assert np.allclose(U, np.eye(dim))


def test_takagi_error():
    """Tests the value errors of Takagi"""
    n = 10
    m = 11
    A = np.random.rand(n, m)
    with pytest.raises(ValueError, match="The input matrix is not square"):
        takagi(A)

def test_takagi_diagonal_matrix():
    """Test the takagi decomposition works well for a specific matrix that was not deecomposed accuratelyin a previous version.
    See more info in PR #393 (https://github.com/XanaduAI/thewalrus/pull/393)"""
    A = np.load('test_matrix_for_takagi.npy')
    d, U = takagi(A)
    assert np.allclose(A, U @ np.diag(d) @ U.T)
    assert np.allclose(U @ np.conjugate(U).T, np.eye(len(U)))
    assert np.all(d >= 0)

def test_real_degenerate():
    """Verify that the Takagi decomposition returns a matrix that is unitary and results in a
    correct decomposition when input a real but highly degenerate matrix. This test uses the
    adjacency matrix of a balanced tree graph."""

    vals = [
        1,
        2,
        31,
        34,
        35,
        62,
        67,
        68,
        94,
        100,
        101,
        125,
        133,
        134,
        157,
        166,
        167,
        188,
        199,
        200,
        220,
        232,
        233,
        251,
        265,
        266,
        283,
        298,
        299,
        314,
        331,
        332,
        346,
        364,
        365,
        377,
        397,
        398,
        409,
        430,
        431,
        440,
        463,
        464,
        472,
        503,
        535,
        566,
        598,
        629,
        661,
        692,
        724,
        755,
        787,
        818,
        850,
        881,
        913,
        944,
    ]
    mat = np.zeros([31 * 31])
    mat[vals] = 1
    mat = mat.reshape(31, 31)
    # The lines above are equivalent to:
    # import networkx as nx
    # g = nx.balanced_tree(2, 4)
    # a = nx.to_numpy_array(g)
    rl, U = takagi(mat)
    assert np.allclose(U @ U.conj().T, np.eye(len(mat)))
    assert np.allclose(U @ np.diag(rl) @ U.T, mat)


@pytest.mark.parametrize("n", [5, 10, 50])
@pytest.mark.parametrize("datatype", [np.complex128, np.float64])
@pytest.mark.parametrize("svd_order", [True, False])
def test_autonne_takagi(n, datatype, svd_order):
    """Checks the correctness of the Autonne decomposition function"""
    if datatype is np.complex128:
        A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    if datatype is np.float64:
        A = np.random.rand(n, n)
    A += A.T
    r, U = takagi(A, svd_order=svd_order)
    assert np.allclose(A, U @ np.diag(r) @ U.T)
    assert np.all(r >= 0)
    if svd_order is True:
        assert np.all(np.diff(r) <= 0)
    else:
        assert np.all(np.diff(r) >= 0)


@pytest.mark.parametrize("size", [10, 20, 100])
def test_flat_phase(size):
    """Test that the correct decomposition is obtained even if the first entry is 0"""
    A = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    A += A.T
    A[0, 0] = 0
    l, u = takagi(A)
    assert np.allclose(A, u * l @ u.T)


def test_real_input_edge():
    """Adapted from https://math.stackexchange.com/questions/4418925/why-does-this-algorithm-for-the-takagi-factorization-fail-here"""
    rng = np.random.default_rng(0)  # Important for reproducibility
    A = (rng.random((100, 100)) - 0.5) * 114
    A = A * A.T  # make A symmetric
    l, u = takagi(A)
    # Now, reconstruct A, see
    Ar = u * l @ u.T
    assert np.allclose(A, Ar)


@pytest.mark.parametrize("rank1", [2, 4, 5])
@pytest.mark.parametrize("rank2", [2, 4, 5])
@pytest.mark.parametrize("rankrand", [2, 4, 5])
@pytest.mark.parametrize("rankzero", [2, 4, 5])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("unitary", [True, False])
def test_pre_iwasawa(rank1, rank2, rankrand, rankzero, symmetric, unitary):
    """Tests the pre_iwasawa decomposition"""
    vals = np.array(
        [np.random.rand(1)[0]] * rank1
        + [np.random.rand(1)[0]] * rank2
        + list(np.random.rand(rankrand))
        + [1] * rankzero
    )
    if unitary is True:
        vals = np.ones_like(vals)
    dd = np.concatenate([vals, 1 / vals])
    dim = len(vals)
    U = haar_measure(dim)
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    if symmetric is False:
        V = haar_measure(dim)
        P = np.block([[V.real, -V.imag], [V.imag, V.real]])
    else:
        P = O.T

    S = (O * dd) @ P
    EE, DD, FF = pre_iwasawa(S)
    assert np.allclose(EE @ DD @ FF, S)
    assert is_symplectic(EE)
    assert is_symplectic(FF)
    assert is_symplectic(FF)
    assert np.allclose(FF @ FF.T, np.identity(2 * dim))
    assert np.allclose(DD[:dim, :dim] @ DD[dim:, dim:], np.identity(dim))
    assert np.allclose(DD[:dim, dim:], 0)
    assert np.allclose(DD[dim:, :dim], 0)
    A = EE[:dim, :dim]
    B = EE[:dim, dim:]
    C = EE[dim:, :dim]
    D = EE[dim:, dim:]
    assert np.allclose(A, np.eye(dim))
    assert np.allclose(B, 0)
    assert np.allclose(C, C.T)
    assert np.allclose(D, np.eye(dim))


@pytest.mark.parametrize("rank1", [2, 4, 5])
@pytest.mark.parametrize("rank2", [2, 4, 5])
@pytest.mark.parametrize("rankrand", [2, 4, 5])
@pytest.mark.parametrize("rankzero", [2, 4, 5])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("unitary", [True, False])
def test_iwasawa(rank1, rank2, rankrand, rankzero, symmetric, unitary):
    """Tests the Iwasawa decomposition"""
    vals = np.array(
        [np.random.rand(1)[0]] * rank1
        + [np.random.rand(1)[0]] * rank2
        + list(np.random.rand(rankrand))
        + [1] * rankzero
    )
    if unitary is True:
        vals = np.ones_like(vals)
    dd = np.concatenate([vals, 1 / vals])
    dim = len(vals)
    U = haar_measure(dim)
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    if symmetric is False:
        V = haar_measure(dim)
        P = np.block([[V.real, -V.imag], [V.imag, V.real]])
    else:
        P = O.T
    S = (O * dd) @ P
    EE, DD, FF = iwasawa(S)
    assert np.allclose(EE @ DD @ FF, S)
    assert is_symplectic(EE)
    assert is_symplectic(FF)
    assert is_symplectic(FF)
    assert np.allclose(FF @ FF.T, np.identity(2 * dim))
    assert np.allclose(DD, np.diag(np.diag(DD)))
    assert np.allclose(DD[:dim, :dim] @ DD[dim:, dim:], np.identity(dim))
    A = EE[:dim, :dim]
    B = EE[:dim, dim:]
    C = EE[dim:, :dim]
    D = EE[dim:, dim:]
    assert np.allclose(B, 0)
    XX = A.T @ C
    assert np.allclose(XX, XX.T)
    assert np.allclose(A @ D.T, np.eye(dim))
    assert np.allclose(np.diag(EE), 1)
    assert np.allclose(np.tril(A), A)
    assert np.allclose(np.triu(D), D)


def test_pre_iwasawa_error():
    """Tests error is raised when input not symplectic"""
    M = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Input matrix is not symplectic."):
        pre_iwasawa(M)


def test_iwasawa_error():
    """Tests error is raised when input not symplectic"""
    M = np.random.rand(4, 5)
    with pytest.raises(ValueError, match="Input matrix is not symplectic."):
        iwasawa(M)


def test_iwasawa2x2():
    """Compares numerics against exact result for 2x2 matrices in Arvind 1995"""
    num_tests = 100
    for _ in range(num_tests):
        S = random_symplectic(1)
        A, N, K = iwasawa(S)
        a = S[0, 0]
        b = S[0, 1]
        c = S[1, 0]
        d = S[1, 1]
        eta = a**2 + b**2
        xi = (a * c + b * d) / eta
        eta = np.sqrt(eta)
        AA = np.array([[1, 0], [xi, 1]])
        NN = np.diag([eta, 1 / eta])
        KK = np.array([[a, b], [-b, a]]) / eta
        assert np.allclose(A, AA)
        assert np.allclose(K, KK)
        assert np.allclose(N, NN)
