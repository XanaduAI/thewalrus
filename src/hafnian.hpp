#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef _LAPACKE
    #define EIGEN_USE_LAPACKE
#endif

#include <Eigen/Eigenvalues>


typedef unsigned char Byte;
typedef std::complex<double> double_complex;

typedef std::vector<double_complex> vec_complex;
typedef std::vector<vec_complex> mat_complex;

typedef std::vector<double> vec_double;
typedef std::vector<double> mat_double;


vec_complex powtrace(vec_complex &z, int n, int l) {
    // given a complex matrix z of dimensions n x n
    // it calculates traces[k] = tr(z^j) for 1 <= j <= l
    vec_complex traces(l, 0.0);
    vec_complex vals(n, 0.0);
    vec_complex pvals(n, 0.0);

    if (n != 0) {
        Eigen::MatrixXcd A = Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned>(z.data(), n, n);
        Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(A, false);
        Eigen::MatrixXcd evals = solver.eigenvalues();
        vals = vec_complex(evals.data(), evals.data()+evals.size());
    }

    int i,j;
    double_complex sum;
    for(j = 0; j < n; j++)
    {
        pvals[j] = vals[j];
    }
    for(i = 0; i < l; i++)
    {
        sum = 0.0;
        for(j = 0; j < n; j++)
        {
            sum += pvals[j];
        }
        traces[i] = sum;
        for(j = 0; j < n; j++)
        {
            pvals[j] = pvals[j] * vals[j];
        }
    }

    return traces;
};


vec_double powtrace(vec_double &z, int n, int l) {
    // given a real matrix z of dimensions n x n
    // it calculates traces[k] = tr(z^j) for 1 <= j <= l
    vec_double traces(l, 0.0);
    vec_complex vals(n, 0.0);
    vec_complex pvals(n, 0.0);

    if (n != 0) {
        Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(z.data(), n, n);
        Eigen::ComplexEigenSolver<Eigen::MatrixXd> solver(A, false);
        Eigen::MatrixXcd evals = solver.eigenvalues();
        vals = vec_complex(evals.data(), evals.data()+evals.size());
    }

    int i,j;
    double_complex sum;
    for(j = 0; j < n; j++)
    {
        pvals[j] = vals[j];
    }
    for(i = 0; i < l; i++)
    {
        sum = 0.0;
        for(j = 0; j < n; j++)
        {
            sum += pvals[j];
        }
        traces[i] = sum.real();
        for(j = 0; j < n; j++)
        {
            pvals[j] = pvals[j] * vals[j];
        }
    }

    return traces;
};


void dec2bin(char* dst, unsigned long long int x, Byte len) {
    // Convert decimal number in x to character vector dst of length len representing binary number
    char i; // this variable cannot be unsigned
    for (i = len - 1; i >= 0; --i) {
        *dst++ = x >> i & 1;
    }
}


Byte find2(char* dst, Byte len, Byte* pos) {
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
T do_chunk(std::vector<T> &mat, int n, unsigned long long int X, unsigned long long int chunksize) {
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

        std::vector<T> traces = powtrace(B, sum, m);

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<std::vector<T>> comb(2, std::vector<T>(m+1, 0.0));
        comb[0][0] = 1.0;

        for (i = 1; i <= n / 2; i++) {
            factor = traces[i - 1] / (2.0 * i);
            powfactor = 1.0;

            cnt = -cnt;
            cntindex = (1 + cnt) / 2;
            for (j = 0; j < n / 2 + 1; j++) {
                comb[1 - cntindex][j] = comb[cntindex][j];
            }
            for (j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (1.0*j);
                for (k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[1 - cntindex][k - 1] += comb[cntindex][k - i * j - 1] * powfactor;
                }
            }
        }
        if (((sum / 2) % 2) == (n / 2 % 2)) {
            summand = comb[1 - cntindex][n / 2];
        }
        else {
            summand = -comb[1 - cntindex][n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}


template <typename T>
T do_chunk_loops(std::vector<T> &mat, std::vector<T> &C, std::vector<T> &D, int n, unsigned long long int X, unsigned long long int chunksize) {
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

        std::vector<T> traces = powtrace(B_powtrace, sum, m);

        char cnt = 1;
        Byte cntindex = 0;

        std::vector<std::vector<T>> comb(2, std::vector<T>(m+1, 0.0));
        comb[0][0] = 1.0;

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
                comb[1 - cntindex][j] = comb[cntindex][j];
            }

            for(j = 1; j <= (n / (2 * i)); j++) {
                powfactor = powfactor * factor / (1.0*j);
                for(k = i * j + 1; k <= n / 2 + 1; k++) {
                    comb[1 - cntindex][k - 1] = comb[1 - cntindex][k - 1] + comb[cntindex][k - i * j - 1] * powfactor;
                }
            }
        }

        if(((sum / 2) % 2) == (n / 2 % 2))
        {
            summand = comb[1 - cntindex][n / 2];
        }
        else
        {
            summand = -comb[1 - cntindex][n / 2];
        }
        #pragma omp critical
        res += summand;
    }

    return res;
}


template <typename T>
T hafnian(std::vector<T> &mat) {
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
T loop_hafnian(std::vector<T> &mat) {
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

