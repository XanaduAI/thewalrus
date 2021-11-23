import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def get_reflection_vector(matrix, size, k):
    sizeH = size - k
    reflect_vector = np.zeros(sizeH, dtype=matrix.dtype)
    order = size - sizeH
    offset = order - 1

    matrix_column = np.zeros(sizeH, dtype=matrix.dtype)
    for i in range(0, sizeH):
        matrix_column[i] = matrix[(i + order) * size + offset]

    sigma = np.linalg.norm(matrix_column)
    if matrix_column[0] != 0:
        sigma *= matrix_column[0] / np.abs(matrix_column[0])

    for i in range(0, sizeH):
        reflect_vector[i] = matrix_column[i]

    reflect_vector[0] += sigma
    return reflect_vector


@jit(nopython=True, cache=True)
def apply_householder(A, v, size_A, k):
    sizeH = len(v)
    norm_v_sqr = np.linalg.norm(v) ** 2
    if norm_v_sqr == 0:
        return

    vHA = np.zeros(size_A - k + 1, dtype=A.dtype)
    Av = np.zeros(size_A, dtype=A.dtype)

    for j in range(0, size_A - k + 1):
        for l in range(0, sizeH):
            vHA[j] += np.conj(v[l]) * A[(k + l) * size_A + k - 1 + j]

    for i in range(0, sizeH):
        for j in range(0, size_A - k + 1):
            A[(k + i) * size_A + k - 1 + j] -= 2 * v[i] * vHA[j] / norm_v_sqr

    for i in range(0, size_A):
        for l in range(0, sizeH):
            Av[i] += A[(i) * size_A + k + l] * v[l]

    for i in range(0, size_A):
        for j in range(0, sizeH):
            A[(i) * size_A + k + j] -= 2 * Av[i] * np.conj(v[j]) / norm_v_sqr


@jit(nopython=True, cache=True)
def reduce_matrix_to_hessenberg(matrix, size):
    for i in range(1, size - 1):
        reflect_vector = get_reflection_vector(matrix, size, i)
        apply_householder(matrix, reflect_vector, size, i)


@jit(nopython=True, cache=True)
def beta(H, i, size):
    return H[(i - 1) * size + i - 2]


@jit(nopython=True, cache=True)
def alpha(H, i, size):
    return H[(i - 1) * size + i - 1]


@jit(nopython=True, cache=True)
def hij(H, i, j, size):
    return H[(i - 1) * size + j - 1]


@jit(nopython=True, cache=True)
def mlo(i, j, size):
    return (i - 1) * size + j - 1


@jit(nopython=True, cache=True)
def _charpoly_from_labudde(H, n, k):
    c = np.zeros(n * n, dtype=H.dtype)
    c[mlo(1, 1, n)] = -alpha(H, 1, n)
    c[mlo(2, 1, n)] = c[mlo(1, 1, n)] - alpha(H, 2, n)
    c[mlo(2, 2, n)] = alpha(H, 1, n) * alpha(H, 2, n) - hij(H, 1, 2, n) * beta(H, 2, n)

    for i in range(3, k + 1):
        c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n)
        for j in range(2, i):
            suma = 0
            beta_prod = 1
            for m in range(1, j - 1):
                beta_prod = 1
                for bm in range(i, i - m, -1):
                    beta_prod *= beta(H, bm, n)
                suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)]

            beta_prod = 1
            for bm in range(i, i - j + 1, -1):
                beta_prod *= beta(H, bm, n)

            c[mlo(i, j, n)] = (
                c[mlo(i - 1, j, n)]
                - alpha(H, i, n) * c[mlo(i - 1, j - 1, n)]
                - suma
                - hij(H, i - j + 1, i, n) * beta_prod
            )

        suma = 0
        beta_prod = 0

        for m in range(1, i - 1):
            beta_prod = 1
            for bm in range(i, i - m, -1):
                beta_prod *= beta(H, bm, n)
            suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, i - m - 1, n)]

        beta_prod = 1
        for bm in range(i, 1, -1):
            beta_prod *= beta(H, bm, n)

        c[mlo(i, i, n)] = (
            -alpha(H, i, n) * c[mlo(i - 1, i - 1, n)] - suma - hij(H, 1, i, n) * beta_prod
        )

    for i in range(k + 1, n + 1):
        c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n)

        if k >= 2:
            for j in range(2, k + 1):
                suma = 0.0
                beta_prod = 1
                for m in range(1, j - 1):
                    beta_prod = 1
                    for bm in range(i, i - m, -1):
                        beta_prod *= beta(H, bm, n)

                    suma += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)]

                beta_prod = 1
                for bm in range(i, i - j + 1, -1):
                    beta_prod *= beta(H, bm, n)

                c[mlo(i, j, n)] = (
                    c[mlo(i - 1, j, n)]
                    - alpha(H, i, n) * c[mlo(i - 1, j - 1, n)]
                    - suma
                    - hij(H, i - j + 1, i, n) * beta_prod
                )
    poly_list = [c[(n - 1) * n + i - 1] for i in range(1, n + 1)]

    return poly_list


@jit(nopython=True, cache=True)
def charpoly_from_labudde(H, method="ravel"):
    """
    Calculates the characteristic polynomial of the matrix H
    Args:
        H (array): square matrix
        method (string): pre-processing operation on H
    Returns
        (array): list of power traces from 0 to n-1
    """
    if method == "ravel":
        Hflat = H.ravel()
    elif method == "flatten":
        Hflat = H.flatten()
    elif method == "reshape":
        Hflat = H.reshape(-1)
    n = len(H)
    reduce_matrix_to_hessenberg(Hflat, n)
    coeff = _charpoly_from_labudde(Hflat, n, n)
    return coeff


@jit(nopython=True)
def power_trace_eigen_h(H, n):
    """
    Calculates the powertraces of the matrix H up to power n-1.
    Args:
        H (array): square matrix
        n (int): required order
        is_hermitian (boolean): whether the input matrix is hermitian
    Returns
        (array): list of power traces from 0 to n-1
    """
    pow_traces = np.zeros(n, dtype=np.float64)
    vals = np.linalg.eigvalsh(H)
    pow_traces[0] = H.shape[0]
    pow_traces[1] = vals.sum()
    pow_vals = vals
    for i in range(2, n):
        pow_vals = pow_vals * vals
        pow_traces[i] = np.sum(pow_vals)
    return pow_traces


@jit(nopython=True)
def power_trace_eigen(H, n):
    """
    Calculates the powertraces of the matrix H up to power n-1.
    Args:
        H (array): square matrix
        n (int): required order
        is_hermitian (boolean): whether the input matrix is hermitian
    Returns
        (array): list of power traces from 0 to n-1
    """
    pow_traces = np.zeros(n, dtype=np.complex128)
    vals = np.linalg.eigvals(H)
    pow_traces[0] = H.shape[0]
    pow_traces[1] = vals.sum()
    pow_vals = vals
    for i in range(2, n):
        pow_vals = pow_vals * vals
        pow_traces[i] = np.sum(pow_vals)
    return pow_traces


@jit(nopython=True, cache=True)
def power_trace_labudde(H, n):
    """
    Calculates the powertraces of the matrix H up to power n-1.
    Args:
        H (array): square matrix
        n (int): required order
    Returns
        (array): list of power traces from 0 to n-1
    """
    m = len(H)
    min_val = min(n, m)
    pow_traces = [m, np.trace(H)]
    A = H
    for i in range(min_val - 2):
        A = A @ H
        pow_traces.append(np.trace(A))
    if n <= m:
        return np.array(pow_traces, dtype=H.dtype)
    char_pol = charpoly_from_labudde(H)
    for i in range(min_val, n):
        ssum = 0
        for k in range(m):
            ssum -= char_pol[k] * pow_traces[-k - 1]
        pow_traces.append(ssum)
    return np.array(pow_traces, dtype=H.dtype)


@jit(nopython=True, cache=True)
def f_all_labudde(H, n):

    pow_traces = power_trace_labudde(H, n // 2 + 1)
    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = pow_traces[i] / (2 * i)
        powfactor = 1
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor *= factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, :]


@jit(nopython=True, cache=True)
def f_from_powtraces(pow_traces, n):

    count = 0
    comb = np.zeros((2, n // 2 + 1), dtype=np.complex128)
    comb[0, 0] = 1
    for i in range(1, n // 2 + 1):
        factor = pow_traces[i] / (2 * i)
        powfactor = 1.0
        count = 1 - count
        comb[count, :] = comb[1 - count, :]
        for j in range(1, n // (2 * i) + 1):
            powfactor = powfactor * factor / j
            for k in range(i * j + 1, n // 2 + 2):
                comb[count, k - 1] += comb[1 - count, k - i * j - 1] * powfactor

    return comb[count, n // 2]
