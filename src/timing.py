
import time
import sys
import numpy as np
import lhaf
import time 


header=["Size", "Time(complex128)", "Result(complex128)"]
print("{: >5} {: >15} {: >25} ".format(*header))
for n in range(2,23):
    mat2=np.ones([2*n,2*n],dtype=np.complex128)
    init2=time.clock()
    x2=np.real(lhaf.hafnian(mat2))
    end2=time.clock()
    row=[2*n,end2-init2,x2]
    
    print("{: >5} {: >15} {: >25}".format(*row))
