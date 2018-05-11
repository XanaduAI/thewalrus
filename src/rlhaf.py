import ctypes
import os
import sys
import numpy as np


tol=1e-12
path = os.path.dirname(__file__)
sofile = os.path.join(path, "rlhafnian.so")
cdll = ctypes.CDLL(sofile)
#cdll1 = ctypes.CDLL("libgsl.so")
#cdll2 = ctypes.CDLL("liblapacke.so")

_calc_hafnian = cdll.dhaf
_calc_hafnian.restype = ctypes.c_double




def hafnian(l):
    """ Takes a matrix l, checks it is symmetric and of even 
    dimensions and calculates its hafnian by invoking the C routine"""
    matshape=l.shape
    assert matshape[0] == matshape[1], "Check if it is a square matrix"
    assert isinstance(l, np.ndarray), "Check if it is a numpy array"
    assert matshape[0]%2 ==0, "Check that is of even dimensions"
    assert np.linalg.norm(l-np.transpose(l)) < tol, "Check that it is symmetric"
    if matshape[0]==2:
        return l[0][1]
    else:
        if l.dtype != np.float64:
            l = l.astype(np.float64)
        a = l.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rr=np.float64(np.array([0.0,0.0]))
        arr=rr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        res = _calc_hafnian(a, matshape[0], arr)
        return rr[0]+1j*rr[1]


