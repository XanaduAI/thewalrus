#pragma once

#include <type_traits>

namespace libwalrus {

// definition of enable_if_t for std=c++11
// already defined in std=c++14
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

// definition of is_complex for checking if scalar type is complex
template <class T> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

/**
 * Auxiliary function for Labudde algorithm. Returns
 * Euclidean norm squared of the complex vector.
 *
 * @param vec complex vector
 *
 * @return Euclidean norm squared
 */
template <typename T, enable_if_t<is_complex<T>{}> * = nullptr>
T norm_sqr(const std::vector<T> &vec) {
  T ns = static_cast<T>(0);
  size_t vec_size = vec.size();
  for (size_t i = 0; i < vec_size; i++) {
    ns += vec[i] * std::conj(vec[i]);
  }
  return ns;
}

/**
 * Auxiliary function for Labudde algorithm. Returns
 * Euclidean norm squared of the non-complex vector.
 *
 * @param vec non-complex vector
 *
 * @return Euclidean norm squared
 */
template <typename T, enable_if_t<!is_complex<T>{}> * = nullptr>
T norm_sqr(const std::vector<T> &vec) {
  T ns = static_cast<T>(0);
  size_t vec_size = vec.size();
  for (size_t i = 0; i < vec_size; i++) {
    ns += vec[i] * vec[i];
  }
  return ns;
}
/**
 * Auxiliary function for Labudde algorithm. Returns
 * conjugate of complex variable or variable if not complex.
 * Useful for functions agnostic to the scalar type.
 * Allows us to reuse code for complex and non-complex types.
 *
 * @param val complex scalar
 *
 * @return the complex conjugate
 */
template <typename T, enable_if_t<is_complex<T>{}> * = nullptr>
inline T conjugate(T val) {
  return std::conj(val);
}

/**
 * Auxiliary function for Labudde algorithm. Returns
 * conjugate of complex variable or variable if not complex.
 * Useful for functions agnostic to the scalar type.
 * Allows us to reuse code for complex and non-complex types.
 *
 * @param val non-complex scalar
 *
 * @return the non-complex scalar
 */
template <typename T, enable_if_t<!is_complex<T>{}> * = nullptr>
inline T conjugate(T val) {
  return val;
}

/**
 * Auxiliary function for Labudde algorithm. Returns
 * Euclidean norm of the complex vector.
 *
 * @param vec complex vector
 *
 * @return Euclidean norm
 */
template <typename T, enable_if_t<is_complex<T>{}> * = nullptr>
T norm(const std::vector<T> &vec) {
  return sqrt(std::real(norm_sqr(vec)));
}

/**
 * Auxiliary function for Labudde algorithm. Returns
 * Euclidean norm of the non-complex vector.
 *
 * @param vec non-complex vector
 *
 * @return Euclidean norm
 */
template <typename T, enable_if_t<!is_complex<T>{}> * = nullptr>
T norm(const std::vector<T> &vec) {
  return sqrt(norm_sqr(vec));
}

/**
 * Auxiliary function for Labudde algorithm.
 * See pg 10 of for definition of beta
 * [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
 *
 * @param H upper-Hessenberg matrix
 * @param i row
 * @param size size of matrix
 *
 * @return element of the lower-diagonal of matrix H
 */
template <typename T>
inline T beta(const std::vector<T> &H, size_t i, size_t size) {
  return H[(i - 1) * size + i - 2];
}

/**
 * Auxiliary function for Labudde algorithm.
 * See pg 10 of for definition of alpha
 * [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
 *
 * @param H upper-Hessenberg matrix
 * @param i row
 * @param size size of matrix
 *
 * @return element of the central-diagonal of matrix H
 */
template <typename T>
inline T alpha(const std::vector<T> &H, size_t i, size_t size) {
  return H[(i - 1) * size + i - 1];
}

/**
 * Auxiliary function for Labudde algorithm.
 * See pg 10 of for definition of hij
 * [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
 *
 * @param H upper-Hessenberg matrix
 * @param i row
 * @param j column
 * @param size size of matrix
 *
 * @return element of upper triangle of matrix H
 */
template <typename T>
inline T hij(const std::vector<T> &H, size_t i, size_t j, size_t size) {
  return H[(i - 1) * size + j - 1];
}

/**
 * Auxiliary function for Labudde algorithm.
 * The paper uses indices that start counting at 1
 * so this function lowers them to start counting at 0.
 *
 * @param H upper-Hessenberg matrix
 * @param i row
 * @param j column
 * @param size size of matrix
 *
 * @return linear matrix index lowered by 1
 */
inline size_t mlo(size_t i, size_t j, size_t size) {
  return (i - 1) * size + j - 1;
}

/**
 * Compute characteristic polynomial using the LaBudde algorithm.
 * See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1).
 * If the matrix is n by n but you only want coefficients k < n
 * set k below n. If you want all coefficients, set k = n.
 *
 * @param H matrix in Hessenberg form (RowMajor)
 * @param n size of matrix
 * @param k compute coefficients up to k (k must be <= n)
 * @return char-poly coeffs + auxiliary data (see comment in function)
 *
 */
template <typename T>
std::vector<T> charpoly_from_labudde(const std::vector<T> &H, size_t n,
                                     size_t k)
{
 // the matrix c holds not just the polynomial coefficients, but also auxiliary
 // data. To retrieve the characteristic polynomial coeffients from the matrix c, use
 // this map for characteristic polynomial coefficient c_j:
 // if j = 0, c_0 -> 1
 // if j > 0, c_j -> c[(n - 1) * n + j - 1]
  std::vector<T> c(n * n);
  c[mlo(1, 1, n)] = -alpha(H, 1, n);
  c[mlo(2, 1, n)] = c[mlo(1, 1, n)] - alpha(H, 2, n);
  c[mlo(2, 2, n)] =
      alpha(H, 1, n) * alpha(H, 2, n) - hij(H, 1, 2, n) * beta(H, 2, n);

  for (size_t i = 3; i <= k; i++) {
    c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n);

    for (size_t j = 2; j <= i - 1; j++) {
      T sum = 0.;
      T beta_prod;
      for (size_t m = 1; m <= j - 2; m++) {
        beta_prod = 1.;
        for (size_t bm = i; bm >= i - m + 1; bm--) {
          beta_prod *= beta(H, bm, n);
        }
        sum +=
            hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)];
      }
      beta_prod = 1.;
      for (size_t bm = i; bm >= i - j + 2; bm--) {
        beta_prod *= beta(H, bm, n);
      }
      c[mlo(i, j, n)] = c[mlo(i - 1, j, n)] -
                        alpha(H, i, n) * c[mlo(i - 1, j - 1, n)] - sum -
                        hij(H, i - j + 1, i, n) * beta_prod;
    }
    T sum = 0.;
    T beta_prod;
    for (size_t m = 1; m <= i - 2; m++) {
      beta_prod = 1.;
      for (size_t bm = i; bm >= i - m + 1; bm--) {
        beta_prod *= beta(H, bm, n);
      }
      sum += hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, i - m - 1, n)];
    }

    beta_prod = 1.;
    for (size_t bm = i; bm >= 2; bm--) {
      beta_prod *= beta(H, bm, n);
    }
    c[mlo(i, i, n)] = -alpha(H, i, n) * c[mlo(i - 1, i - 1, n)] - sum -
                      hij(H, 1, i, n) * beta_prod;
  }

  for (size_t i = k + 1; i <= n; i++) {
    c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha(H, i, n);
    if (k >= 2) {
      for (size_t j = 2; j <= k; j++) {
        T sum = 0.;
        T beta_prod;
        for (size_t m = 1; m <= j - 2; m++) {
          beta_prod = 1.;
          for (size_t bm = i; bm >= i - m + 1; bm--) {
            beta_prod *= beta(H, bm, n);
          }
          sum +=
              hij(H, i - m, i, n) * beta_prod * c[mlo(i - m - 1, j - m - 1, n)];
        }
        beta_prod = 1.;
        for (size_t bm = i; bm >= i - j + 2; bm--) {
          beta_prod *= beta(H, bm, n);
        }

        c[mlo(i, j, n)] = c[mlo(i - 1, j, n)] -
                          alpha(H, i, n) * c[mlo(i - 1, j - 1, n)] - sum -
                          hij(H, i - j + 1, i, n) * beta_prod;
      }
    }
  }
  return c;
}

/**
 * Compute reflection vector for householder transformation on
 * general complex matrices.
 * See Introduction to Numerical Analysis-Springer New York (2002)
 * (3rd Edition) by J. Stoer and R. Bulirsch Section 6.5.1
 *
 * @param size
 * @param matrix
 * @param sizeH size of reflection vector
 * @param reflect_vector householder reflection vector
 *
 */
template <typename T>
std::vector<T> get_reflection_vector(std::vector<T> &matrix, size_t size,
                                     size_t k) {
  size_t sizeH = size - k;
  std::vector<T> reflect_vector(sizeH);
  size_t order = size - sizeH;
  size_t offset = order - 1;

  std::vector<T> matrix_column(sizeH);
  for (size_t i = 0; i < sizeH; i++) {
    matrix_column[i] = matrix[(i + order) * size + offset];
  }

  T sigma = norm(matrix_column);
  if (matrix_column[0] != static_cast<T> (0.))
    sigma *= matrix_column[0] / std::abs(matrix_column[0]);

  for (size_t i = 0; i < sizeH; i++) {
    reflect_vector[i] = matrix_column[i];
  }
  reflect_vector[0] += sigma;
  return reflect_vector;
}

/**
 * Apply householder transformation on a matrix A
 * See  Matrix Computations by Golub and Van Loan
 * (4th Edition) Sections 5.1.4 and 7.4.2
 *
 * @param A matrix to apply householder on
 * @param v reflection vector
 * @param size_A size of matrix A
 * @param k start of submatrix
 *
 * @return coefficients
 */
template <typename T>
void apply_householder(std::vector<T> &A, std::vector<T> &v, size_t size_A,
                       size_t k) {
  size_t sizeH = v.size();

  auto norm_v_sqr = norm_sqr(v);
  if (norm_v_sqr == static_cast<T>(0.))
    return;

  std::vector<T> vHA(size_A - k + 1, 0.);
  std::vector<T> Av(size_A, 0.);
  for (size_t j = 0; j < size_A - k + 1; j++) {
    for (size_t l = 0; l < sizeH; l++) {
      vHA[j] += conjugate(v[l]) * A[(k + l) * size_A + k - 1 + j];
    }
  }
  for (size_t i = 0; i < sizeH; i++) {
    for (size_t j = 0; j < size_A - k + 1; j++) {
      A[(k + i) * size_A + k - 1 + j] -= static_cast<T>(2.) * v[i] * vHA[j] / norm_v_sqr;
    }
  }
  for (size_t i = 0; i < size_A; i++) {
    for (size_t l = 0; l < sizeH; l++) {
      Av[i] += A[(i)*size_A + k + l] * v[l];
    }
  }
  for (size_t i = 0; i < size_A; i++) {
    for (size_t j = 0; j < sizeH; j++) {
      A[(i)*size_A + k + j] -= static_cast<T>(2.) * Av[i] * conjugate(v[j]) / norm_v_sqr;
    }
  }
}

/**
 * Reduce the matrix to upper hessenberg form
 * without Lapack. This function only accepts
 * RowOrder matrices right now.
 *
 * @param matrix matrix to reduce
 * @param size size of matrix
 *
 */
template <typename T>
void reduce_matrix_to_hessenberg(std::vector<T> &matrix, size_t size) {
  for (size_t i = 1; i < size - 1; i++) {
    std::vector<T> &&reflect_vector = get_reflection_vector(matrix, size, i);
    apply_householder(matrix, reflect_vector, size, i);
  }
}


/**
 * Compute the trace of \f$ A^{p}\f$, where p is an integer
 * and A is a square matrix using its characteristic
 * polynomial. In the case that the power p is above
 * the size of the matrix we can use an optimization
 * described in Appendix B of
 * [arxiv:1805.12498](https://arxiv.org/pdf/1104.3769v1.pdf)
 *
 * @param c the characteristic polynomial coefficients
 * @param n size of matrix
 * @param pow exponent p
 *
 * @return coefficients
 */
template <typename T>
std::vector<T> powtrace_from_charpoly(const std::vector<T> &c, size_t n,
                                      size_t pow) {
  if (pow == 0)
    return std::vector<T>({static_cast<T>(n)});

  std::vector<T> traces(pow);
  traces[0] = -c[(n - 1) * n];

  // Calculate traces using the LeVerrier
  // recursion relation
  for (size_t k = 2; k <= pow; k++) {
    traces[k - 1] = -static_cast<T>(k) * c[(n - 1) * n + k - 1];
    for (size_t j = k - 1; j >= 1; j--) {
      traces[k - 1] -= c[(n - 1) * n + k - j - 1] * traces[j - 1];
    }
  }

  // Appendix B optimization
  if (pow > n) {
    for (size_t l = 1; l <= pow - n; l++) {
      traces[n + l - 1] = 0.;
      for (size_t j = 1; j <= n; j++) {
        traces[l + n - 1] -= traces[n - j + l - 1] * c[(n - 1) * n + j - 1];
      }
    }
  }
  return traces;
}

/**
 * Given a complex matrix \f$z\f$ of dimensions \f$n\times n\f$, it calculates
 * \f$Tr(z^j)~\forall~1\leq j\leq l\f$.
 *
 * @param z a flattened complex vector of size \f$n^2\f$, representing an
 *       \f$n\times n\f$ row-ordered matrix.
 * @param n size of the matrix `z`.
 * @param l maximum matrix power when calculating the power trace.
 * @return a vector containing the power traces of matrix `z` to power
 *       \f$1\leq j \leq l\f$.
 */
template <typename T>
std::vector<T> powtrace(std::vector<T> &z, size_t n, size_t l) {
  std::vector<T> z_copy = z;
  reduce_matrix_to_hessenberg(z_copy, n);
  std::vector<T> && coeffs_labudde = charpoly_from_labudde(z_copy, n, n);
  return powtrace_from_charpoly(coeffs_labudde, n, l);
}

} // namespace libwalrus
