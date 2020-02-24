#pragma once
#include <vector>

#if (defined(__GNUC__) || defined(__GNUG__)) && \
    !(defined(__clang__) || defined(__INTEL_COMPILER))
typedef __float128 qp;
//#include <quadmath.h>
#else
typedef long double qp;
#endif

namespace libwalrus {
  enum class StorageType { ColumnOrder, RowOrder };

  template <typename T>
  inline auto norm_sqr(const std::vector<T> &vec) {
    T ns = static_cast<T>(0);
    if constexpr (std::is_same<T, std::complex<long double>>::value ||
                  std::is_same<T, std::complex<double>>::value) {
      for (size_t i = 0; i < vec.size(); i++) {
        ns += vec[i] * std::conj(vec[i]);
      }
    } else {
      for (size_t i = 0; i < vec.size(); i++) {
        ns += vec[i] * vec[i];
      }
    }
    return ns;
  }

  template <typename T>
  inline auto norm(const std::vector<T> &vec) {
    T ns = norm_sqr(vec);

    if constexpr (std::is_same<T, std::complex<long double>>::value ||
                  std::is_same<T, std::complex<double>>::value) {
      return sqrt(std::real(ns));
    } else {
      return sqrt(ns);
    }
  }

  template <StorageType ST, typename T>
  inline T beta(const std::vector<T> &H, int i, int size) {
    if constexpr (ST == StorageType::RowOrder) {
      return H[(i - 1) * size + i - 2];
    } else {
      return H[(i - 2) * size + i - 1];
    }
  }

  template <StorageType ST, typename T>
  inline T alpha(const std::vector<T> &H, int i, int size) {
    return H[(i - 1) * size + i - 1];
  }

  template <StorageType ST, typename T>
  inline T hu(const std::vector<T> &H, int i, int j, int size) {
    if constexpr (ST == StorageType::RowOrder) {
      return H[(i - 1) * size + j - 1];
    } else {
      return H[(j - 1) * size + i - 1];
    }
  }

  inline int mlo(int i, int j, int size) { return (i - 1) * size + j - 1; }

  /**
   * Compute characteristic polynomial using the LaBudde algorithm.
   * See arxiv:11-4.3769v1
   *
   *
   * @param H matrix in Hessenberg form (RowMajor)
   * @param n size of matrix
   *
   */
  template <StorageType ST, typename T>
  std::vector<T> charpoly_from_labudde(const std::vector<T> &H, int n) {
    // integer k tells us how many coefficients we want to compute (k <= n)
    // I have set it here to n, because we typically want all the
    // coefficients, set it otherwise if you want less then n coefficients.
    int k = n;
    std::vector<T> c(n * n);
    c[mlo(1, 1, n)] = -alpha<ST>(H, 1, n);
    c[mlo(2, 1, n)] = c[mlo(1, 1, n)] - alpha<ST>(H, 2, n);
    c[mlo(2, 2, n)] = alpha<ST>(H, 1, n) * alpha<ST>(H, 2, n) -
                      hu<ST>(H, 1, 2, n) * beta<ST>(H, 2, n);

    for (int i = 3; i <= k; i++) {
      c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha<ST>(H, i, n);

      for (int j = 2; j <= i - 1; j++) {
        T sum = 0.;
        T beta_prod;
        for (int m = 1; m <= j - 2; m++) {
          beta_prod = 1.;
          for (int bm = i; bm >= i - m + 1; bm--) {
            beta_prod *= beta<ST>(H, bm, n);
          }
          sum += hu<ST>(H, i - m, i, n) * beta_prod *
                 c[mlo(i - m - 1, j - m - 1, n)];
        }
        beta_prod = 1.;
        for (int bm = i; bm >= i - j + 2; bm--) {
          beta_prod *= beta<ST>(H, bm, n);
        }
        c[mlo(i, j, n)] = c[mlo(i - 1, j, n)] -
                          alpha<ST>(H, i, n) * c[mlo(i - 1, j - 1, n)] - sum -
                          hu<ST>(H, i - j + 1, i, n) * beta_prod;
      }
      T sum = 0.;
      T beta_prod;
      for (int m = 1; m <= i - 2; m++) {
        beta_prod = 1.;
        for (int bm = i; bm >= i - m + 1; bm--) {
          beta_prod *= beta<ST>(H, bm, n);
        }
        sum += hu<ST>(H, i - m, i, n) * beta_prod *
               c[mlo(i - m - 1, i - m - 1, n)];
      }

      beta_prod = 1.;
      for (int bm = i; bm >= 2; bm--) {
        beta_prod *= beta<ST>(H, bm, n);
      }
      c[mlo(i, i, n)] = -alpha<ST>(H, i, n) * c[mlo(i - 1, i - 1, n)] - sum -
                        hu<ST>(H, 1, i, n) * beta_prod;
    }

    for (int i = k + 1; i <= n; i++) {
      c[mlo(i, 1, n)] = c[mlo(i - 1, 1, n)] - alpha<ST>(H, i, n);
      if (k >= 2) {
        for (int j = 2; j <= k; j++) {
          T sum = 0.;
          T beta_prod;
          for (int m = 1; m <= j - 2; m++) {
            beta_prod = 1.;
            for (int bm = i; bm >= i - m + 1; bm--) {
              beta_prod *= beta<ST>(H, bm, n);
            }
            sum += hu<ST>(H, i - m, i, n) * beta_prod *
                   c[mlo(i - m - 1, j - m - 1, n)];
          }
          beta_prod = 1.;
          for (int bm = i; bm >= i - j + 2; bm--) {
            beta_prod *= beta<ST>(H, bm, n);
          }

          c[mlo(i, j, n)] = c[mlo(i - 1, j, n)] -
                            alpha<ST>(H, i, n) * c[mlo(i - 1, j - 1, n)] - sum -
                            hu<ST>(H, i - j + 1, i, n) * beta_prod;
        }
      }
    }
    // get coeffs
    std::vector<T> coeffs_labudde(n + 1);
    coeffs_labudde[0] = 1;
    for (int j = 1; j <= n; j++) {
      coeffs_labudde[j] = c[(n - 1) * n + j - 1];
    }
    return coeffs_labudde;
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
  template <StorageType ST, typename T,
            typename = std::enable_if_t<
                std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, std::complex<long double>>::value ||
                std::is_same<T, double>::value ||
                std::is_same<T, long double>::value>>
  auto get_reflection_vector(std::vector<T> & matrix, size_t size, int k) {
    size_t sizeH = size - k;
    std::vector<T> reflect_vector(sizeH);
    size_t order = size - sizeH;
    size_t offset = order - 1;

    std::vector<T> matrix_column(sizeH);
    for (size_t i = 0; i < sizeH; i++) {
      matrix_column[i] = matrix[(i + order) * size + offset];
    }

    T sigma = norm(matrix_column);
    if (matrix_column[0] != 0.)
      sigma *= matrix_column[0] / std::abs(matrix_column[0]);

    for (size_t i = 0; i < sizeH; i++) {
      reflect_vector[i] = matrix_column[i];
    }
    if (std::abs(matrix_column[0] - sigma) <
        std::abs(matrix_column[0] + sigma)) {
      reflect_vector[0] -= sigma;
    } else {
      reflect_vector[0] += sigma;
    }
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
  template <StorageType ST, typename T,
            typename = std::enable_if_t<ST == StorageType::RowOrder>>
  void apply_householder(std::vector<T> & A, std::vector<T> & v, size_t size_A,
                         size_t k) {
    size_t sizeH = v.size();

    auto norm_v_sqr = norm_sqr(v);
    if (norm_v_sqr == 0.) return;

    std::vector<T> vHA(size_A - k + 1, 0.);
    std::vector<T> Av(size_A, 0.);
    for (size_t j = 0; j < size_A - k + 1; j++) {
      for (size_t l = 0; l < sizeH; l++) {
        if constexpr (std::is_same<T, std::complex<double>>::value ||
                      std::is_same<T, std::complex<long double>>::value) {
          vHA[j] += std::conj(v[l]) * A[(k + l) * size_A + k - 1 + j];
        } else {
          vHA[j] += v[l] * A[(k + l) * size_A + k - 1 + j];
        }
      }
    }

    for (size_t i = 0; i < sizeH; i++) {
      for (size_t j = 0; j < size_A - k + 1; j++) {
        A[(k + i) * size_A + k - 1 + j] -= 2. * v[i] * vHA[j] / norm_v_sqr;
      }
    }

    for (size_t i = 0; i < size_A; i++) {
      for (size_t l = 0; l < sizeH; l++) {
        Av[i] += A[(i)*size_A + k + l] * v[l];
      }
    }

    for (size_t i = 0; i < size_A; i++) {
      for (size_t j = 0; j < sizeH; j++) {
        if constexpr (std::is_same<T, std::complex<double>>::value ||
                      std::is_same<T, std::complex<long double>>::value) {
          A[(i)*size_A + k + j] -= 2. * Av[i] * std::conj(v[j]) / norm_v_sqr;
        } else {
          A[(i)*size_A + k + j] -= 2. * Av[i] * v[j] / norm_v_sqr;
        }
      }
    }
  }

  extern "C" {
  int dgehrd_(int *n, int *ilo, int *ihi, double *a, int *lda, double *tau,
              double *work, int *lwork, int *info);
  int zgehrd_(int *n, int *ilo, int *ihi, std::complex<double> *a, int *lda,
              std::complex<double> *tau, std::complex<double> *work, int *lwork,
              int *info);
  }

  /**
   * Reduce the matrix to upper hessenberg form
   * using Lapack. This function only accepts
   * RowOrder matrices right now.
   *
   * @param matrix matrix to reduce
   * @param size size of matrix
   *
   */
  template <StorageType ST, typename T,
            typename =
                std::enable_if_t<std::is_same<T, double>::value ||
                                 std::is_same<T, std::complex<double>>::value>>
  void reduce_matrix_to_hessenberg_lapack(std::vector<T> & A, size_t size_A) {
    int n = size_A;
    int ilo = 1;
    int ihi = n;
    int lda = n;
    std::vector<T> tau(n);
    std::vector<T> work(n);
    int lwork = n;
    int info;

    if constexpr (std::is_same<T, double>::value) {
      dgehrd_(&n, &ilo, &ihi, A.data(), &lda, tau.data(), work.data(), &lwork,
              &info);
    } else {
      zgehrd_(&n, &ilo, &ihi, A.data(), &lda, tau.data(), work.data(), &lwork,
              &info);
    }

    if constexpr (ST == StorageType::RowOrder) {
      for (int i = 0; i < n; i++) {
        if (std::abs(A[i * n + i]) < 1e-16) {
          A[i * n + i] = 0.;
        }
        for (int j = 0; j < i; j++) {
          T temp = A[i * n + j];
          A[i * n + j] = A[j * n + i];
          A[j * n + i] = temp;
          // Lapack stores reflection vector in here,
          // so we should zero
          if (i > j + 1) {
            A[i * n + j] = static_cast<T>(0);
          }
          if (std::abs(A[j * n + i]) < 1e-16) {
            A[j * n + i] = 0.;
          }
          if (std::abs(A[i * n + j]) < 1e-16) {
            A[i * n + j] = 0.;
          }
        }
      }
    }
    // std::cout << "A = " << A << std::endl;
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
  template <StorageType ST, typename T,
            typename = std::enable_if_t<ST == StorageType::RowOrder>>
  void reduce_matrix_to_hessenberg(std::vector<T> & matrix, size_t size) {
    for (size_t i = 1; i < size - 1; i++) {
      std::vector<T> reflect_vector =
          get_reflection_vector<ST>(matrix, size, i);
      apply_householder<ST>(matrix, reflect_vector, size, i);
    }
  }

  /**
   * Compute the trace of \f$ A^{p}\f$, where p is an integer
   * and A is a square matrix using its characteristic
   * polynomial. In the case that the power p is above
   * the size of the matrix we can use an optimization
   * described in Appendix B of arxiv:1805.12498
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
    if (pow == 0) return std::vector<T>({static_cast<T>(n)});

    std::vector<T> traces(pow);
    traces[0] = -c[1];

    if (pow < n) n = pow;
    // Calculate traces using the LeVerrier
    // recursion relation
    for (size_t k = 2; k <= n; k++) {
      traces[k - 1] = -static_cast<T>(k) * c[k];
      for (size_t j = k - 1; j >= 1; j--) {
        traces[k - 1] -= c[k - j] * traces[j - 1];
      }
    }

    // Appendix B optimization
    if (pow > n) {
      for (size_t l = 1; l <= pow - n; l++) {
        traces[n + l - 1] = 0.;
        for (size_t j = 1; j <= n; j++) {
          traces[l + n - 1] -= traces[n - j + l - 1] * c[j];
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
  template <StorageType ST, typename T>
  auto powtrace(std::vector<T> & z, int n, int l) {
    std::vector<T> z_copy = z;
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value) {
	if (n > 15)
	  reduce_matrix_to_hessenberg_lapack<ST>(z_copy, n);
	else
	  reduce_matrix_to_hessenberg<ST>(z_copy, n);
    } else {
      reduce_matrix_to_hessenberg<ST>(z_copy, n);
    }
    std::vector<T> coeffs_labudde = charpoly_from_labudde<ST>(z_copy, n);

    return powtrace_from_charpoly(coeffs_labudde, n, l);
  }
}
