// Copyright 2019 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>
#include <complex>
#include <assert.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef LAPACKE
    #define EIGEN_SUPERLU_SUPPORT
    #define EIGEN_USE_BLAS
    #define EIGEN_USE_LAPACKE

    #define LAPACK_COMPLEX_CUSTOM
    #define lapack_complex_float std::complex<float>
    #define lapack_complex_double std::complex<double>
#endif


#include <Eigen/Eigenvalues>

#include <version.hpp>

namespace hafnian {

typedef unsigned char Byte;
typedef std::complex<double> double_complex;
typedef std::vector<double_complex> vec_complex;
typedef std::vector<double> vec_double;


inline vec_complex powtrace(vec_complex &z, int n, int l) {
    // given a complex matrix z of dimensions n x n
    // it calculates traces[k] = tr(z^j) for 1 <= j <= l
    vec_complex traces(l, 0.0);
    vec_complex vals(n, 0.0);
    vec_complex pvals(n, 0.0);

    Eigen::MatrixXcd A = Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned>(z.data(), n, n);
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(A, false);
    Eigen::MatrixXcd evals = solver.eigenvalues();
    vals = vec_complex(evals.data(), evals.data()+evals.size());

    double_complex sum;
    int i, j;

    for(j = 0; j < n; j++) {
        pvals[j] = vals[j];
    }

    for(i = 0; i < l; i++) {
        sum = 0.0;
        for(j = 0; j < n; j++) {
            sum += pvals[j];
        }
        traces[i] = sum;
        for(j = 0; j < n; j++) {
            pvals[j] = pvals[j] * vals[j];
        }
    }

    return traces;
};


inline vec_double powtrace(vec_double &z, int n, int l) {
    // given a real matrix z of dimensions n x n
    // it calculates traces[k] = tr(z^j) for 1 <= j <= l
    vec_double traces(l, 0.0);
    vec_complex vals(n, 0.0);
    vec_complex pvals(n, 0.0);

    Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(z.data(), n, n);
    Eigen::EigenSolver<Eigen::MatrixXd> solver(A, false);
    Eigen::MatrixXcd evals = solver.eigenvalues();
    vals = vec_complex(evals.data(), evals.data()+evals.size());

    double_complex sum;
    int i, j;

    for(j = 0; j < n; j++) {
        pvals[j] = vals[j];
    }
    for(i = 0; i < l; i++) {
        sum = 0.0;
        for(j = 0; j < n; j++) {
            sum += pvals[j];
        }
        traces[i] = sum.real();
        for(j = 0; j < n; j++) {
            pvals[j] = pvals[j] * vals[j];
        }
    }

    return traces;
};


inline void dec2bin(char* dst, unsigned long long int x, Byte len) {
    // Convert decimal number in x to character vector dst of length len representing binary number
    char i; // this variable cannot be unsigned
    for (i = len - 1; i >= 0; --i) {
        *dst++ = x >> i & 1;
    }
}


inline Byte find2(char* dst, Byte len, Byte* pos) {
    // Given a string of length len it finds in which positions it has a one and stores its position i,
    // as 2*i and 2*i+1 in consecutive slots of the array pos.
    // It also returns (twice) the number of ones in array dst
    Byte j = 0;
    for (Byte i = 0; i < len; i++) {
        if(1 == dst[i]) {
            pos[2 * j] = 2 * i;
            pos[2 * j + 1] = 2 * i + 1;
            j++;
        }
    }
    return 2 * j;
}


template <typename T>
inline T do_chunk(std::vector<T> &mat, int n, unsigned long long int X, unsigned long long int chunksize) {
    // This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the
    // Hafnian of matrix mat

    T res = 0.0;

    #pragma omp parallel for
    for (unsigned long long int x = X; x < X + chunksize; x++) {
        T summand = 0.0;

        Byte m = n / 2;
        int i, j, k;
        T factor, powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);
        Byte sum = find2(dst, m, pos);

        std::vector<T> B(sum*sum, 0.0);

        for (i = 0; i < sum; i++) {
            for (j = 0; j < sum; j++) {
                B[i*sum+j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
        }

        std::vector<T> traces(m, 0.0);
        if (sum != 0) {
            traces = powtrace(B, sum, m);
        }

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<T> comb(2*(m+1), 0.0);
        comb[0] = 1.0;

        for (i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / (2.0 * i);
            powfactor = 1.0;

            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (j = 0; j < n / 2 + 1; j++) {
                comb[(m+1)*(1 - cntindex)+j] = comb[(m+1)*cntindex+j];
            }
            for (j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (1.0*j);
                for (k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[(m+1)*(1 - cntindex) + k - 1] += comb[(m+1)*cntindex + k - i * j - 1] * powfactor;
                }
            }
        }
        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[(m+1)*(1-cntindex) + n / 2];
        }
        else {
            summand = -comb[(m+1)*(1-cntindex) + n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}


template <typename T>
inline T do_chunk_loops(std::vector<T> &mat, std::vector<T> &C, std::vector<T> &D, int n, unsigned long long int X, unsigned long long int chunksize) {
    // This function calculates adds parts X to X+chunksize of Cygan and Pilipczuk formula for the
    // Hafnian of matrix mat

    T res = 0.0;

    #pragma omp parallel for
    for(unsigned long long int x = X * chunksize; x < (X + 1)*chunksize; x++) {
        T summand = 0.0;

        Byte m = n / 2;
        int i, j, k;
        T factor, powfactor;

        char* dst = new char[m];
        Byte* pos = new Byte[n];
        dec2bin(dst, x, m);
        Byte sum = find2(dst, m, pos);

        std::vector<T> B(sum*sum, 0.0), B_powtrace(sum*sum, 0.0);
        std::vector<T> C1(sum, 0.0), D1(sum, 0.0);

        for(i = 0; i < sum; i++) {
            for(j = 0; j < sum; j++) {
                B[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
                B_powtrace[i * sum + j] = mat[pos[i] * n + ((pos[j]) ^ 1)];
            }
            C1[i] = C[pos[i]];
            D1[i] = D[pos[i]];
        }

        std::vector<T> traces(m, 0.0);
        if (sum != 0) {
            traces = powtrace(B, sum, m);
        }

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<T> comb(2*(m+1), 0.0);
        comb[0] = 1.0;

        for(i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / (2.0 * i);
            T tmpn = 0.0;

            for(int i = 0; i < sum; i++) {
                tmpn += C1[i] * D1[i];
            }

            factor += 0.5*tmpn;
            std::vector<T> tmp_c1(sum, 0.0);

            T tmp = 0.0;

            for(int i = 0; i < sum; i++) {
                tmp = 0.0;

                for(int j = 0; j < sum; j++) {
                    tmp += C1[j] * B[j * sum + i];
                }

                tmp_c1[i] = tmp;
            }

            for(int i = 0; i < sum; i++) {
                C1[i] = tmp_c1[i];
            }

            powfactor = 1.0;

            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for(j = 0; j < n / 2 + 1; j++) {
                comb[(m+1)*(1-cntindex)+j] = comb[(m+1)*cntindex+j];
            }

            for(j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (1.0*j);
                for(k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[(m+1)*(1-cntindex) + k - 1] = comb[(m+1)*(1-cntindex) + k - 1] + comb[(m+1)*cntindex + k - i * j - 1] * powfactor;
                }
            }
        }

        if(((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[(m+1)*(1-cntindex) + n / 2];
        }
        else {
            summand = -comb[(m+1)*(1-cntindex) + n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}


template <typename T>
inline T recursive_chunk(std::vector<T> b, int s, int w, std::vector<T> g, int n) {
    // Recursive integer hafnian solver.
    if (s == 0) {
        return static_cast<T>(w)*g[n];
    }

    std::vector<T> c((s-2)*(s-3)/2*(n+1), 0.0);
    T h, h1, h2;
    int u, v, j, k, i = 0;

    for (j = 1; j < s-2; j++) {
        for (k = 0; k < j; k++) {
            for (u = 0; u < n+1; u++) {
                c[(n+1)*i+u] = b[(n+1)*((j+1)*(j+2)/2+k+2)+u];
            }
            i += 1;
        }
    }

    #pragma omp task shared(h1)
    h1 = recursive_chunk(c, s-2, -w, g, n);

    std::vector<T> e(n+1, 0);
    e = g;

    for (u = 0; u < n; u++) {
        for (v = 0; v < n-u; v++) {
            e[u+v+1] += g[u]*b[v];

            for (j = 1; j < s-2; j++) {
                for (k = 0; k < j; k++) {
                    c[(n+1)*(j*(j-1)/2+k)+u+v+1] += b[(n+1)*((j+1)*(j+2)/2)+u]*b[(n+1)*((k+1)*(k+2)/2+1)+v] + b[(n+1)*(k+1)*(k+2)/2+u]*b[(n+1)*((j+1)*(j+2)/2+1)+v];
                }
            }
        }
    }

    #pragma omp task shared(h2)
    h2 = recursive_chunk(c, s-2, w, e, n);

    #pragma omp taskwait
    h = h1+h2;

    return h;
}


template <typename T>
inline T hafnian_recursive(std::vector<T> &mat) {
    // Returns the hafnian of an integer matrix A via the C hafnian library.
    // Modified with permission from https://github.com/eklotek/Hafnian.
    int n = std::sqrt(static_cast<double>(mat.size()))/2;

    std::vector<T> z(n*(2*n-1)*(n+1), 0);
    std::vector<T> g(n+1, 0);

    g[0] = 1;

    #pragma omp parallel for
    for (int j = 1; j < 2*n; j++) {
        for (int k = 0; k < j; k++) {
            z[(n+1)*(j*(j-1)/2+k)] = mat[2*j*n + k];
        }
    }

    T result;

    #pragma omp parallel
    #pragma omp single nowait
    result = recursive_chunk(z, 2*n, 1, g, n);

    return result;
}


template <typename T>
inline T hafnian(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(n % 2 == 0);

    Byte m = n / 2;
    Byte mm = m / 2;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m));
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = std::min(workers, pow1);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    T haf;
    haf = do_chunk(mat, n, rank, chunksize);
    return  haf;
}


template <typename T>
inline T loop_hafnian(std::vector<T> &mat) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(n % 2 == 0);

    Byte m = n / 2;
    Byte mm = m / 2;
    unsigned long long int pow1 = ((unsigned long long int)pow(2.0, (double)m));
    unsigned long long int workers = ((unsigned long long int)pow(2.0, (double)mm));
    workers = std::min(workers, pow1);

    std::vector<T> D(n, 0.0), C(n, 0.0);

    unsigned long long int chunksize = pow1;
    unsigned long long int rank = 0;

    for(int i = 0; i < n; i++) {
        D[i] = mat[i * n + i];
    }

    for(int i = 0; i < n; i += 2) {
        C[i] = D[i + 1];
        C[i + 1] = D[i];
    }

    T haf;
    haf = do_chunk_loops(mat, C, D, n, rank, chunksize);
    return  haf;
}


template <typename T>
inline T hafnian_rpt(std::vector<T> &mat, std::vector<int> &rpt, bool use_eigen=true) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(static_cast<int>(rpt.size()) == n);

    long double p = 2;
    T y = 0.0, q = 0.0;

    if (use_eigen) {
        namespace eg = Eigen;

        eg::Matrix<T,eg::Dynamic,eg::Dynamic> A = eg::Map<eg::Matrix<T,eg::Dynamic,eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);

        eg::VectorXd X = eg::VectorXd::Zero(n);
        eg::VectorXd rows2 = eg::Map<eg::VectorXi, eg::Unaligned>(rpt.data(), rpt.size()).cast<double>();

        int s = rows2.sum();
        int s2 = s/2;
        int steps = (rows2+eg::VectorXd::Ones(n)).prod()/2;

        for (int i=1; i<s2; i++) {
            p /= i+1;
        }

        rows2 /= 2;
        q = 0.5*rows2.dot(A*rows2);

        for (int i=0; i < steps; i++) {
            y += static_cast<double>(p)*pow(q, s2);
            for (int j=0; j < n; j++) {
                if (X[j] < rpt[j]) {
                    X[j] += 1;
                    p *= -(rpt[j]+1-X[j])/X[j];
                    q -= A.col(j).conjugate().dot(rows2-X);
                    q -= 0.5*A(j, j);
                    break;
                }
                else {
                    X[j] = 0;
                    if (rpt[j] % 2 == 1) {
                        p *= -1;
                    }
                    q += static_cast<double>(rpt[j])*A.col(j).conjugate().dot(rows2-X);
                    q -= 0.5*rpt[j]*rpt[j]*A(j, j);
                }
            }
        }
    }
    else {
        std::vector<int> x(n, 0.0);
        int s = std::accumulate(rpt.begin(), rpt.end(), 0);
        int s2 = s/2;

        for (int i=1; i<s2; i++) {
            p /= i+1;
        }

        std::vector<double> nu2(n);
        std::transform(rpt.begin(), rpt.end(), nu2.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, 0.5));

        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                q += 0.5*nu2[j]*mat[i*n+j]*nu2[i];
            }
        }

        int steps = 1;

        for (auto i : rpt) {
            steps *= i+1;
        }

        steps /= 2;

        for (int i=0; i < steps; i++) {
            y += static_cast<double>(p)*pow(q, s2);

            for (int j=0; j < n; j++) {

                if (x[j] < rpt[j]) {
                    x[j] += 1;
                    p *= -static_cast<double>(rpt[j]+1-x[j])/x[j];

                    for (int k=0; k < n; k++) {
                        q -= mat[k*n+j]*(nu2[k]-x[k]);
                    }
                    q -= 0.5*mat[j*n+j];
                    break;
                }
                else {
                    x[j] = 0;
                    if (rpt[j] % 2 == 1) {
                        p *= -1;
                    }
                    for (int k=0; k < n; k++) {
                        q += (1.0*rpt[j])*mat[k*n+j]*(nu2[k]-x[k]);
                    }
                    q -= 0.5*rpt[j]*rpt[j]*mat[j*n+j];
                }
            }
        }
    }

    return y;
}


template <typename T>
inline T loop_hafnian_rpt(std::vector<T> &mat, std::vector<int> &rpt, bool use_eigen=true) {
    int n = std::sqrt(static_cast<double>(mat.size()));
    assert(static_cast<int>(rpt.size()) == n);

    long long int p = 2;
    T y = 0.0, q = 0.0, q1 = 0.0;

    if (use_eigen) {
        namespace eg = Eigen;

        eg::Matrix<T,eg::Dynamic,eg::Dynamic> A = eg::Map<eg::Matrix<T,eg::Dynamic,eg::Dynamic>, eg::Unaligned>(mat.data(), n, n);
        eg::Matrix<T,eg::Dynamic,1> mu = A.diagonal();

        eg::VectorXd X = eg::VectorXd::Zero(n);
        eg::VectorXd rows2 = eg::Map<eg::VectorXi, eg::Unaligned>(rpt.data(), rpt.size()).cast<double>();

        int s = rows2.sum();
        int steps = (rows2+eg::VectorXd::Ones(n)).prod()/2;

        rows2 /= 2;
        q = 0.5*rows2.dot(A*rows2);
        q1 = rows2.dot(mu);

        int s1 = std::floor(0.5*s)+1;
        eg::Matrix<T,eg::Dynamic,1> z1 = eg::VectorXd::Ones(s1).cast<T>();
        eg::Matrix<T,eg::Dynamic,1> z2 = eg::VectorXd::Ones(s1).cast<T>();

        for (int i=0; i < steps; i++) {
            for (int j=1; j < s1; j++) {
                z1[j] = z1[j-1]*q/(1.0*j);
            }

            if (s % 2 == 1) {
                z2[0] = q1;
                for (int j=1; j < s1; j++) {
                    z2[j] = z2[j-1]*pow(q1, 2)/(2.0*j)/(2.0*j+1);
                }
            }
            else {
                for (int j=1; j < s1; j++) {
                    z2[j] = z2[j-1]*pow(q1, 2)/(2.0*j)/(2.0*j-1);
                }
            }

            y += static_cast<double>(p)*z1.conjugate().dot(z2.reverse());

            for (int j=0; j < n; j++) {
                if (X[j] < rpt[j]) {
                    X[j] += 1;
                    p *= -(rpt[j]+1-X[j])/X[j];
                    q -= A.col(j).conjugate().dot(rows2-X);
                    q -= 0.5*A(j, j);
                    q1 -= mu[j];
                    break;
                }
                else {
                    X[j] = 0;
                    if (rpt[j] % 2 == 1) {
                        p *= -1;
                    }
                    q += static_cast<double>(rpt[j])*A.col(j).conjugate().dot(rows2-X);
                    q -= 0.5*rpt[j]*rpt[j]*A(j, j);
                    q1 += static_cast<double>(rpt[j])*mu[j];
                }
            }
        }
    }
    else {
        std::vector<int> x(n, 0.0);
        int s = std::accumulate(rpt.begin(), rpt.end(), 0);
        int s1 = std::floor(0.5*s)+1;
        std::vector<T> z1(s1, 1.0);
        std::vector<T> z2(s1, 1.0);

        // diagonal of matrix mat
        std::vector<T> mu(n);
        for (int i=0; i<n; i++) {
            mu[i] = mat[i*n+i];
        }

        std::vector<double> nu2(n);
        std::transform(rpt.begin(), rpt.end(), nu2.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, 0.5));

        for (int i=0; i<n; i++) {
            q1 += nu2[i]*mu[i];
            for (int j=0; j<n; j++) {
                q += 0.5*nu2[j]*mat[i*n+j]*nu2[i];
            }
        }

        int steps = 1;

        for (auto i : rpt) {
            steps *= i+1;
        }

        steps /= 2;

        for (int i=0; i < steps; i++) {
            for (int j=1; j < s1; j++) {
                z1[j] = z1[j-1]*q/(1.0*j);
            }

            if (s % 2 == 1) {
                z2[0] = q1;
                for (int j=1; j < s1; j++) {
                    z2[j] = z2[j-1]*pow(q1, 2)/(2.0*j)/(2.0*j+1);
                }
            }
            else {
                for (int j=1; j < s1; j++) {
                    z2[j] = z2[j-1]*pow(q1, 2)/(2.0*j)/(2.0*j-1);
                }
            }

            T z1z2prod = 0.0;
            for (int j=0; j<s1; j++) {
                z1z2prod += z1[j]*z2[s1-1-j];
            }

            y += static_cast<double>(p)*z1z2prod;

            for (int j=0; j < n; j++) {

                if (x[j] < rpt[j]) {
                    x[j] += 1;
                    p *= -static_cast<double>(rpt[j]+1-x[j])/x[j];

                    for (int k=0; k < n; k++) {
                        q -= mat[k*n+j]*(nu2[k]-x[k]);
                    }
                    q -= 0.5*mat[j*n+j];
                    q1 -= mu[j];
                    break;
                }
                else {
                    x[j] = 0;
                    if (rpt[j] % 2 == 1) {
                        p *= -1;
                    }
                    for (int k=0; k < n; k++) {
                        q += (1.0*rpt[j])*mat[k*n+j]*(nu2[k]-x[k]);
                    }
                    q -= 0.5*rpt[j]*rpt[j]*mat[j*n+j];
                    q1 += static_cast<double>(rpt[j])*mu[j];
                }
            }
        }
    }


    return y;
}


}
