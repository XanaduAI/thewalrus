#!/usr/bin/env python3
import time
from math import factorial

import numpy as np
from scipy import diagonal, randn
from scipy.linalg import qr

import matplotlib.pyplot as plt

from hafnian.lib.librhaf import haf_int, haf_real


a0 = 100.
anm1 = 2.
n = 15
r = (anm1/a0)**(1./(n-1))
nreps = [(int)(a0*(r**((i)))) for i in range(n)]



times = np.empty([n, 2])
for ind, reps in enumerate(nreps):
    size = 2*(ind+1)
    print('\nTesting matrix size {}, with {} reps...'.format(size,reps))

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.float64(np.clip(matrix+matrix.T, -1, 1))
        res = haf_real(A)

    end = time.time()
    print('Mean time taken (real): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 0] = (end - start)/reps

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.clip(matrix+matrix.T, -1, 1)
        res = haf_int(A)

    end = time.time()
    print('Mean time taken (int): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 1] = (end - start)/reps


plt.semilogy(2*np.arange(1,n+1),times[:, 0], marker='.')
plt.semilogy(2*np.arange(1,n+1),times[:, 1], marker='.')
plt.xlabel(r"Matrix size $n$")
plt.ylabel(r"Time in seconds")
plt.savefig('scipy-timing.png')
