
import time
import sys
import numpy as np
import rlhaf
import time 


header=["Size", "Time(complex128)", "Result(complex128)"]
print("{: >5} {: >15} {: >25} ".format(*header))
for n in range(2,23):
    mat2=np.ones([2*n,2*n],dtype=np.float64)
    init2=time.clock()
    x2=np.real(rlhaf.hafnian(mat2))
    end2=time.clock()
    row=[2*n,end2-init2,x2]
    
    print("{: >5} {: >15} {: >25}".format(*row))
