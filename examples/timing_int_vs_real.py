#!/usr/bin/env python3
import time
from math import factorial

import numpy as np
from scipy import diagonal, randn
from scipy.linalg import qr

import matplotlib.pyplot as plt

from thewalrus.libwalrus import haf_int, haf_real, haf_complex


a0 = 100.
anm1 = 2.
n = 20
r = (anm1/a0)**(1./(n-1))
nreps = [(int)(a0*(r**((i)))) for i in range(n)]



times = np.empty([n, 5])

for ind, reps in enumerate(nreps):
    size = 2*(ind+1)
    print('\nTesting matrix size {}, with {} reps...'.format(size,reps))

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.complex128(np.clip(matrix+matrix.T, -1, 1))
        res = haf_complex(A)

    end = time.time()
    print('Mean time taken (complex): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 0] = (end - start)/reps

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.complex128(np.clip(matrix+matrix.T, -1, 1))
        res = haf_complex(A, recursive=True)

    end = time.time()
    print('Mean time taken (complex, recursive): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 1] = (end - start)/reps

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.float64(np.clip(matrix+matrix.T, -1, 1))
        res = haf_real(A)

    end = time.time()
    print('Mean time taken (real): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 2] = (end - start)/reps

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.float64(np.clip(matrix+matrix.T, -1, 1))
        res = haf_real(A, recursive=True)

    end = time.time()
    print('Mean time taken (real, recursive): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 3] = (end - start)/reps

    start = time.time()
    for i in range(reps):
        matrix = np.random.randint(low=-1, high=2, size=[size, size])
        A = np.int64(np.clip(matrix+matrix.T, -1, 1))
        res = haf_int(A)

    end = time.time()
    print('Mean time taken (int): ', (end - start)/reps)
    # print('\t Haf result: ', res)
    times[ind, 4] = (end - start)/reps


np.save("hafnian++.npy", times)

fig, ax = plt.subplots(1, 1)

ax.semilogy(2*np.arange(1,n+1),times[:, 0], marker='.', label='haf_complex')
ax.semilogy(2*np.arange(1,n+1),times[:, 1], marker='.', label='haf_complex (recursive)')
ax.semilogy(2*np.arange(1,n+1),times[:, 2], marker='.', label='haf_real')
ax.semilogy(2*np.arange(1,n+1),times[:, 3], marker='.', label='haf_real (recursive)')
ax.semilogy(2*np.arange(1,n+1),times[:, 4], marker='.', label='haf_int')

ax.set_xlabel(r"Matrix size $n$")
ax.set_ylabel(r"Time in seconds")
ax.legend()

plt.savefig('hafnian++-timing.png')
