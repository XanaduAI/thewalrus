import pytest
import numpy as np
from thewalrus import mtl, lmtl
from thewalrus.reference import rspm, rpmp
from thewalrus.reference import mtl as mtl_symb
from thewalrus.reference import lmtl as lmtl_symb
#from thewalrus.quantum import
from scipy.special import factorial2
from scipy.stats import unitary_group


@pytest.mark.parametrize("n", range(1,6))
def test_montrealer_all_ones(n):
    """Test that the Montrealer of a matrix of ones gives (2n-2)!!"""
    A = np.ones([2*n, 2*n])
    mtl_val = mtl(A)
    mtl_expect = factorial2(2*n-2)
    assert np.allclose(mtl_val, mtl_expect)


@pytest.mark.parametrize("n", range(1,8))
def test_size_of_pmpr(n):
    """rpmp(2n) should have (2n-2)!! elements"""
    terms_rpmp = sum(1 for _ in rpmp(range(2*n)))
    terms_theo = factorial2(2*n-2)
    assert terms_rpmp == terms_theo


@pytest.mark.parametrize("n", range(1,8))
def test_size_of_rspm(n):
    """rspm(2n) should have (n+1)(2n-2)!! elements"""
    terms_rspm = sum(1 for _ in rspm(range(2*n)))
    terms_theo = (n+1)*factorial2(2*n-2)
    assert terms_rspm == terms_theo


@pytest.mark.parametrize("n", range(2,8))
def test_size_of_rpmp(n):
    """The rpmp must form a Y-alternating walk without loops"""
    test = True
    for perfect in rpmp(range(1,2*n+1)):
        last = perfect[0][1] #starting point
        reduced_last = last-n if last>n else last
        #different mode in every tuple
        if reduced_last == 1: test = False
        
        for i in perfect[1:]:
            reduced = i[0]-n if i[0]>n else i[0], i[1]-n if i[1]>n else i[1]
            #different mode in every tuple
            if reduced[0]==reduced[1]: test = False
            #consecutive tuple contain the same mode
            if reduced_last not in reduced: test = False

            last = i[0] if reduced[1] == reduced_last else i[1]
            reduced_last = last-n if last>n else last
        
        #last mode most coincide with the first one
        if reduced_last != 1: test=False

    assert test


@pytest.mark.parametrize("n", range(2,8))
def test_mtl_functions_agree(n):
    """Make sure both mtl functions agree with one another"""
    U_N = np.matrix(unitary_group.rvs(n))
    U_N = U_N + U_N.conj().T 
    U_M = np.matrix(unitary_group.rvs(n))
    U_M = U_M + U_M.T
    A = np.block([[U_M.conj(), U_N], [U_N.T, U_M]])
    assert np.allclose(mtl_symb(A), mtl(A))


@pytest.mark.parametrize("n", range(2,8))
def test_lmtl_functions_agree(n):
    """Make sure both lmtl functions agree with one another"""
    n = 4
    U_N = np.matrix(unitary_group.rvs(n))
    U_N = U_N + U_N.conj().T 
    U_M = np.matrix(unitary_group.rvs(n))
    U_M = U_M + U_M.T
    A = np.block([[U_M.conj(), U_N], [U_N.T, U_M]])
    zeta = np.diag(A)
    assert np.allclose(lmtl(A, zeta), lmtl_symb(A, zeta))


@pytest.mark.parametrize("n", range(2,8))
def test_mtl_lmtl_agree(n):
    """Make sure mtl and lmtl give the same result if zeta = 0"""


@pytest.mark.parametrize("n", range(2,8))
def test_mtl_lmtl_agree(n):
    """Make sure mtl and lmtl from .reference give the same result if zeta = 0"""