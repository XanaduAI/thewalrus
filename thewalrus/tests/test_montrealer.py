import pytest
import numpy as np
from thewalrus import mtl, lmtl
from thewalrus._montrealer import montrealer, lmontrealer
from thewalrus.reference import rspm, rpmp
#from thewalrus.quantum import
from scipy.special import factorial2


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