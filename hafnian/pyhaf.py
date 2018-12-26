import numpy as np
from scipy.linalg import schur


def powtrace(A: np.array, l: int)->np.array:
    T, _ = schur(A)
    return np.sum(np.diag(T).reshape(-1, 1)**np.arange(1, l+1).reshape(1, -1), axis=0)


def dec2bin(x: int)->np.array:
    return np.array([int(i) for i in list("{0:b}".format(x))])


def find2(x:int)->tuple:
    d = dec2bin(x)
    s = 2*np.sum(d)
    i = np.argwhere(d == 1)
    pos = np.hstack([2*i, 2*i+1]).flatten()
    return s, np.pad(pos, (0, s-len(pos)), 'constant')


def do_chunk(A:np.array, X:int, chunksize:int)->float:
    n = A.shape[0]
    m = n//2
    res = 0

    for x in range(X, X+chunksize):
        s, p = find2(x)

        summand = 0

        B = np.zeros([s, s], dtype=np.complex128)
        rows = p.reshape(-1, 1)
        cols = p.reshape(1, -1)^1
        B = A[rows, cols]

        if s != 0:
            trace = powtrace(B, m)
        else:
            trace = np.zeros([m])

        print(trace)

        comb = np.zeros([2, m+1], dtype=np.complex128)
        comb[0, 0] = 1.

        cnt = 1
        cntidx = 0

        for i in range(1, m+1):
            factor = trace[i-1]/(2*i)
            powfactor = 1

            cnt *= -1
            cntidx = (1+cnt)//2

            for j in range(0, m+1):
                comb[1-cntidx, j] = comb[cntidx, j]

            for j in range(1, n//(2*i)+1):
                powfactor *= factor/j
                for k in range(i*j+1, m+2):
                    comb[1-cntidx, k-1] += comb[cntidx, k-i*j-1] * powfactor

        if (s/2)%2 == (n/2)%2:
            summand = comb[1 - cntidx, n//2]
        else:
            summand = -comb[1 - cntidx, n//2]

        res += summand

    return summand


def hafnian(A:np.array)->float:
    n = A.shape[0]
    assert n%2 == 0
    return do_chunk(A, 0, 2**(n//2))


# A = np.complex128(np.random.random([4, 4]))
# A += 1j*np.random.random([4, 4])
# A += A.T
# expected = A[0, 1]*A[2, 3] + \
#     A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]

# haf = hafnian(A)
# print(haf, expected)



from math import factorial as fac
n = 3
A = np.ones([2*n, 2*n])
expected = fac(2*n)/(fac(n)*(2**n))

haf = hafnian(A)
print(haf, expected)

