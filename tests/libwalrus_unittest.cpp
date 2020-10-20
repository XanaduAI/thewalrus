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
#include <math.h>
#include <iostream>
#include <libwalrus.hpp>
#include <random>
#include "gtest/gtest.h"

const double tol = 1.0e-10f;
const double tol2 = 1.0e-7f;

namespace permanent {

TEST(PermanentRealFsum, CompleteGraph) {
  std::vector<double> mat2(4, 1.0);
  std::vector<double> mat3(9, 1.0);
  std::vector<double> mat4(16, 1.0);

  EXPECT_NEAR(2, libwalrus::permanent_fsum(mat2), tol);
  EXPECT_NEAR(6, libwalrus::permanent_fsum(mat3), tol);
  EXPECT_NEAR(24, libwalrus::permanent_fsum(mat4), tol);
}

TEST(PermanentFsum, Random) {
  std::vector<double> mat(9, 1.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double randnum = distribution(generator);
      mat[i * 3 + j] = randnum;
    }
  }

  double expected = mat[2] * mat[4] * mat[6] + mat[1] * mat[5] * mat[6] +
                    mat[2] * mat[3] * mat[7] + mat[0] * mat[5] * mat[7] +
                    mat[1] * mat[3] * mat[8] + mat[0] * mat[4] * mat[8];

  EXPECT_NEAR(expected, libwalrus::permanent_fsum(mat), tol);
}

TEST(PermanentReal, CompleteGraph) {
  std::vector<double> mat2(4, 1.0);
  std::vector<double> mat3(9, 1.0);
  std::vector<double> mat4(16, 1.0);

  EXPECT_NEAR(2, libwalrus::permanent_quad(mat2), tol);
  EXPECT_NEAR(6, libwalrus::permanent_quad(mat3), tol);
  EXPECT_NEAR(24, libwalrus::permanent_quad(mat4), tol);
}

TEST(PermanentReal, Random) {
  std::vector<double> mat(9, 1.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double randnum = distribution(generator);
      mat[i * 3 + j] = randnum;
    }
  }

  double expected = mat[2] * mat[4] * mat[6] + mat[1] * mat[5] * mat[6] +
                    mat[2] * mat[3] * mat[7] + mat[0] * mat[5] * mat[7] +
                    mat[1] * mat[3] * mat[8] + mat[0] * mat[4] * mat[8];

  EXPECT_NEAR(expected, libwalrus::permanent_quad(mat), tol);
}

TEST(PermanentComplex, Random) {
  std::vector<std::complex<double>> mat(9, 1.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat[i * 3 + j] = std::complex<double>(randnum1, randnum2);
    }
  }

  std::complex<double> expected =
      mat[2] * mat[4] * mat[6] + mat[1] * mat[5] * mat[6] +
      mat[2] * mat[3] * mat[7] + mat[0] * mat[5] * mat[7] +
      mat[1] * mat[3] * mat[8] + mat[0] * mat[4] * mat[8];

  std::complex<double> perm = libwalrus::permanent_quad(mat);

  EXPECT_NEAR(std::real(expected), std::real(perm), tol);
  EXPECT_NEAR(std::imag(expected), std::imag(perm), tol);
}

}  // namespace permanent

namespace recursive_real {

// Unit tests for the real recursive_hafnian function
// Check hafnian of real complete graphs with even dimensions.
TEST(HafianRecursiveDouble, CompleteGraphEven) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);
  std::vector<double> mat8(64, 1.0);
  EXPECT_NEAR(3, libwalrus::hafnian_recursive_quad(mat4), tol);
  EXPECT_NEAR(15, libwalrus::hafnian_recursive_quad(mat6), tol);
  EXPECT_NEAR(105, libwalrus::hafnian_recursive_quad(mat8), tol);
}

// Check hafnian of real random matrix with size 4x4.
TEST(HafianRecursiveDouble, Random) {
  std::vector<double> mat(16, 1.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum = distribution(generator);
      mat[i * 4 + j] = randnum;
      mat[j * 4 + i] = mat[i * 4 + j];
    }
  }

  double expected = mat[1] * mat[11] + mat[2] * mat[7] + mat[3] * mat[6];

  EXPECT_NEAR(expected, libwalrus::hafnian_recursive_quad(mat), tol);
}

// Check hafnian of complete graphs with odd dimensions.
TEST(HafianRecursiveDouble, CompleteGraphOdd) {
  std::vector<double> mat5(25, 1.0);
  std::vector<double> mat7(49, 1.0);
  std::vector<double> mat9(81, 1.0);
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat5));
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat7));
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat9));
}

// Check hafnian of an empty matrix.
TEST(HafianRecursiveDouble, Empty) {
  std::vector<double> mat(0, 0);
  EXPECT_EQ(1, libwalrus::hafnian_recursive_quad(mat));
}

}  // namespace recursive_real

namespace recursive_complex {

// Unit tests for the complex recursive_hafnian function
// Check hafnian of complex complete graphs with even dimensions.
TEST(HafianRecursiveDoubleComplex, CompleteGraphEven) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat8(64, std::complex<double>(1.0, 1.0));

  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);
  std::complex<double> haf4 = libwalrus::hafnian_recursive_quad(mat4);
  std::complex<double> haf6 = libwalrus::hafnian_recursive_quad(mat6);
  std::complex<double> haf8 = libwalrus::hafnian_recursive_quad(mat8);

  double re = std::real(haf);
  double im = std::imag(haf);

  double re4 = std::real(haf4);
  double im4 = std::imag(haf4);

  double re6 = std::real(haf6);
  double im6 = std::imag(haf6);

  double re8 = std::real(haf8);
  double im8 = std::imag(haf8);

  EXPECT_NEAR(3, re, tol);
  EXPECT_NEAR(0, im, tol);

  EXPECT_NEAR(0, re4, tol);
  EXPECT_NEAR(6, im4, tol);

  EXPECT_NEAR(-30, re6, tol);
  EXPECT_NEAR(30, im6, tol);

  EXPECT_NEAR(-420, re8, tol);
  EXPECT_NEAR(0, im8, tol);
}

// Check hafnian of complex random matrix with size 4x4.
TEST(HafianRecursiveDoubleComplex, Random) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(0.0, 0.0));

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat[i * 4 + j] = std::complex<double>(randnum1, randnum2);
      mat[j * 4 + i] = mat[i * 4 + j];
    }
  }

  std::complex<double> expected =
      mat[1] * mat[11] + mat[2] * mat[7] + mat[3] * mat[6];

  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);

  double re_expected = std::real(expected);
  double im_expected = std::imag(expected);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(re_expected, re, tol);
  EXPECT_NEAR(im_expected, im, tol);
}

// Check hafnian of complex complete graphs with odd dimensions.
TEST(HafianRecursiveDoubleComplex, CompleteGraphOdd) {
  std::vector<std::complex<double>> mat(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(0, re, tol);
  EXPECT_NEAR(0, im, tol);
}

// Check hafnian of a complex empty matrix.
TEST(HafianRecursiveDoubleComplex, Empty) {
  std::vector<std::complex<double>> mat(0, std::complex<double>(0.0, 0.0));
  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
}

}  // namespace recursive_complex

namespace eigen_real {

// Unit tests for the real eigen_hafnian function
// Check hafnian of real complete graphs with even dimensions.
TEST(HafianEigenDouble, CompleteGraphEven) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);
  std::vector<double> mat8(64, 1.0);
  EXPECT_NEAR(3, libwalrus::hafnian_eigen(mat4), tol);
  EXPECT_NEAR(15, libwalrus::hafnian_eigen(mat6), tol);
  EXPECT_NEAR(105, libwalrus::hafnian_eigen(mat8), tol);
}

// Check hafnian of real random matrix with size 4x4.
TEST(HafianEigenDouble, Random) {
  std::vector<double> mat(16, 1.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum = distribution(generator);
      mat[i * 4 + j] = randnum;
      mat[j * 4 + i] = mat[i * 4 + j];
    }
  }

  double expected = mat[1] * mat[11] + mat[2] * mat[7] + mat[3] * mat[6];

  EXPECT_NEAR(expected, libwalrus::hafnian_eigen(mat), tol);
}

// Check hafnian of complete graphs with odd dimensions.
TEST(HafianEigenDouble, CompleteGraphOdd) {
  std::vector<double> mat5(25, 1.0);
  std::vector<double> mat7(49, 1.0);
  std::vector<double> mat9(81, 1.0);
  EXPECT_EQ(0, libwalrus::hafnian_eigen(mat5));
  EXPECT_EQ(0, libwalrus::hafnian_eigen(mat7));
  EXPECT_EQ(0, libwalrus::hafnian_eigen(mat9));
}

// Check hafnian of an empty matrix.
TEST(HafianEigenDouble, Empty) {
  std::vector<double> mat(0, 0);
  EXPECT_EQ(1, libwalrus::hafnian_eigen(mat));
}

}  // namespace eigen_real

namespace eigen_complex {

// Unit tests for the complex recursive_hafnian function
// Check hafnian of complex complete graphs with even dimensions.
TEST(HafianEigenDoubleComplex, CompleteGraphEven) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat8(64, std::complex<double>(1.0, 1.0));

  std::complex<double> haf = libwalrus::hafnian_eigen(mat);
  std::complex<double> haf4 = libwalrus::hafnian_eigen(mat4);
  std::complex<double> haf6 = libwalrus::hafnian_eigen(mat6);
  std::complex<double> haf8 = libwalrus::hafnian_eigen(mat8);

  double re = std::real(haf);
  double im = std::imag(haf);

  double re4 = std::real(haf4);
  double im4 = std::imag(haf4);

  double re6 = std::real(haf6);
  double im6 = std::imag(haf6);

  double re8 = std::real(haf8);
  double im8 = std::imag(haf8);

  EXPECT_NEAR(3, re, tol);
  EXPECT_NEAR(0, im, tol);

  EXPECT_NEAR(0, re4, tol);
  EXPECT_NEAR(6, im4, tol);

  EXPECT_NEAR(-30, re6, tol);
  EXPECT_NEAR(30, im6, tol);

  EXPECT_NEAR(-420, re8, tol);
  EXPECT_NEAR(0, im8, tol);
}

// Check hafnian of complex random matrix with size 4x4.
TEST(HafianEigenDoubleComplex, Random) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(0.0, 0.0));

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat[i * 4 + j] = std::complex<double>(randnum1, randnum2);
      mat[j * 4 + i] = mat[i * 4 + j];
    }
  }

  std::complex<double> expected =
      mat[1] * mat[11] + mat[2] * mat[7] + mat[3] * mat[6];

  std::complex<double> haf = libwalrus::hafnian_eigen(mat);

  double re_expected = std::real(expected);
  double im_expected = std::imag(expected);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(re_expected, re, tol);
  EXPECT_NEAR(im_expected, im, tol);
}

// Check hafnian of complex complete graphs with odd dimensions.
TEST(HafianEigenDoubleComplex, CompleteGraphOdd) {
  std::vector<std::complex<double>> mat(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf = libwalrus::hafnian_eigen(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(0, re, tol);
  EXPECT_NEAR(0, im, tol);
}

// Check hafnian of a complex empty matrix.
TEST(HafianEigenDoubleComplex, Empty) {
  std::vector<std::complex<double>> mat(0, std::complex<double>(0.0, 0.0));
  std::complex<double> haf = libwalrus::hafnian_eigen(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
}

}  // namespace eigen_complex

namespace approx_real {

// Unit tests for the real non negative hafnian_approx function
// Check approx hafnian for random matrices with even dimensions.
TEST(HafnianApproxNonngeative, Random) {
  std::vector<double> mat4(16, 0.0);
  std::vector<double> mat6(36, 0.0);
  std::vector<double> mat8(64, 0.0);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(1.0, 0.0);
  generator.seed(20);

  int n = 4;
  int nsamples = 20000;
  std::vector<double> x4(n, 0.0);

  for (int i = 0; i < n; i++) {
    double randnum = distribution(generator);
    x4[i] = randnum;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat4[i * n + j] = x4[i] * x4[j];
    }
  }

  n = 6;
  std::vector<double> x6(n, 0.0);

  for (int i = 0; i < n; i++) {
    double randnum = distribution(generator);
    x6[i] = randnum;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat6[i * n + j] = x6[i] * x6[j];
    }
  }

  n = 8;
  std::vector<double> x8(n, 0.0);

  for (int i = 0; i < n; i++) {
    double randnum = distribution(generator);
    x8[i] = randnum;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat8[i * n + j] = x8[i] * x8[j];
    }
  }

  double expected4 =
      3.0 * std::accumulate(begin(x4), end(x4), 1, std::multiplies<double>());
  double expected6 =
      15.0 * std::accumulate(begin(x6), end(x6), 1, std::multiplies<double>());
  double expected8 =
      105.0 * std::accumulate(begin(x8), end(x8), 1, std::multiplies<double>());

  double haf4 = libwalrus::hafnian_approx(mat4, nsamples);
  double haf6 = libwalrus::hafnian_approx(mat6, nsamples);
  double haf8 = libwalrus::hafnian_approx(mat8, nsamples);

  EXPECT_NEAR(expected4, haf4, haf4 / 15.0);
  EXPECT_NEAR(expected6, haf6, haf6 / 15.0);
  EXPECT_NEAR(expected8, haf8, haf8 / 15.0);
}

}  // namespace approx_real

namespace hafnian_repeated {

// Unit tests for the repeated hafnian function
// Check repeated hafnian for all zero matrices with even dimensions.
TEST(HafnianRepeatedDouble, ZeroRpt) {
  std::vector<double> mat(16, 1.0);
  std::vector<int> rpt(4, 0);

  double haf = libwalrus::hafnian_rpt_quad(mat, rpt);

  EXPECT_NEAR(1, haf, tol);
}

// Check repeated hafnian for all ones matrices with even dimensions.
TEST(HafnianRepeatedDouble, AllOneRpt) {
  std::vector<double> mat2rand(4, 1.0);
  std::vector<double> mat2(4, 1.0);
  std::vector<int> rpt2(2, 1);

  std::vector<double> mat4rand(16, 1.0);
  std::vector<double> mat4(16, 1.0);
  std::vector<int> rpt4(4, 1);

  double expected2 = mat2[1];
  double expected2rand = mat2rand[1];

  double expected4 = 3;
  double expected4rand = mat4rand[1] * mat4rand[11] +
                         mat4rand[2] * mat4rand[7] + mat4rand[3] * mat4rand[6];
  ;

  double haf2 = libwalrus::hafnian_rpt_quad(mat2, rpt2);
  double haf2rand = libwalrus::hafnian_rpt_quad(mat2rand, rpt2);
  double haf4 = libwalrus::hafnian_rpt_quad(mat4, rpt4);
  double haf4rand = libwalrus::hafnian_rpt_quad(mat4rand, rpt4);

  EXPECT_NEAR(expected2, haf2, tol);
  EXPECT_NEAR(expected2rand, haf2rand, tol);
  EXPECT_NEAR(expected4, haf4, tol);
  EXPECT_NEAR(expected4rand, haf4rand, tol);
}

// Check repeated hafnian for all zero complex matrices with even dimensions.
TEST(HafnianRepeatedComplex, ZeroRpt) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(1.0, 0.0));
  std::vector<int> rpt(4, 0);

  std::complex<double> haf = libwalrus::hafnian_rpt_quad(mat, rpt);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
}

// Check repeated hafnian for all ones matrices with even dimensions.
TEST(HafnianRepeatedComplex, AllOneRpt) {
  std::vector<std::complex<double>> mat2rand(4, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mat2(4, std::complex<double>(1.0, 0.0));
  std::vector<int> rpt2(2, 1);

  std::vector<std::complex<double>> mat4rand(16,
                                             std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 0.0));
  std::vector<int> rpt4(4, 1);

  double expected2_re = std::real(mat2[1]);
  double expected2_im = std::imag(mat2[1]);
  double expected2rand_re = std::real(mat2rand[1]);
  double expected2rand_im = std::imag(mat2rand[1]);

  double expected4_re = 3;
  double expected4_im = 0;
  std::complex<double> expected4rand = mat4rand[1] * mat4rand[11] +
                                       mat4rand[2] * mat4rand[7] +
                                       mat4rand[3] * mat4rand[6];
  ;
  double expected4rand_re = std::real(expected4rand);
  double expected4rand_im = std::imag(expected4rand);

  std::complex<double> haf2 = libwalrus::hafnian_rpt_quad(mat2, rpt2);
  std::complex<double> haf2rand = libwalrus::hafnian_rpt_quad(mat2rand, rpt2);
  std::complex<double> haf4 = libwalrus::hafnian_rpt_quad(mat4, rpt4);
  std::complex<double> haf4rand = libwalrus::hafnian_rpt_quad(mat4rand, rpt4);

  double haf2_re = std::real(haf2);
  double haf2_im = std::imag(haf2);
  double haf2rand_re = std::real(haf2rand);
  double haf2rand_im = std::imag(haf2rand);

  double haf4_re = std::real(haf4);
  double haf4_im = std::imag(haf4);
  double haf4rand_re = std::real(haf4rand);
  double haf4rand_im = std::imag(haf4rand);

  EXPECT_NEAR(expected2_re, haf2_re, tol);
  EXPECT_NEAR(expected2_im, haf2_im, tol);
  EXPECT_NEAR(expected2rand_re, haf2rand_re, tol);
  EXPECT_NEAR(expected2rand_im, haf2rand_im, tol);

  EXPECT_NEAR(expected4_re, haf4_re, tol);
  EXPECT_NEAR(expected4_im, haf4_im, tol);
  EXPECT_NEAR(expected4rand_re, haf4rand_re, tol);
  EXPECT_NEAR(expected4rand_im, haf4rand_im, tol);
}

}  // namespace hafnian_repeated

namespace loophafnian_eigen {

// Unit tests for the loop hafnian function using eigenvalues
// Check loop hafnian with eignevalues for all ones matrices with even
// dimensions.
TEST(LoopHafnianEigenDouble, EvenOnes) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);

  double haf4 = libwalrus::loop_hafnian_eigen(mat4);
  double haf6 = libwalrus::loop_hafnian_eigen(mat6);

  EXPECT_NEAR(10, haf4, tol);
  EXPECT_NEAR(76, haf6, tol);
}

// // Check loop hafnian with eignevalues for random matrices with even
// dimensions.
TEST(LoopHafnianEigenDouble, EvenRandom) {
  std::vector<double> mat2(4, 0.0);
  std::vector<double> mat4(16, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      mat2[i * 2 + j] = randnum1;
      mat2[j * 2 + i] = mat2[i * 2 + j];
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      mat4[i * 4 + j] = randnum1;
      mat4[j * 4 + i] = mat4[i * 4 + j];
    }
  }

  double haf2 = libwalrus::loop_hafnian_eigen(mat2);
  double haf4 = libwalrus::loop_hafnian_eigen(mat4);

  double expected2 = mat2[1] + mat2[0] * mat2[3];
  double expected4 =
      mat4[1] * mat4[11] + mat4[2] * mat4[7] + mat4[3] * mat4[6] +
      mat4[0] * mat4[5] * mat4[11] + mat4[1] * mat4[10] * mat4[15] +
      mat4[2] * mat4[5] * mat4[15] + mat4[0] * mat4[10] * mat4[7] +
      mat4[0] * mat4[15] * mat4[6] + mat4[3] * mat4[5] * mat4[10] +
      mat4[0] * mat4[5] * mat4[10] * mat4[15];

  EXPECT_NEAR(expected2, haf2, tol);
  EXPECT_NEAR(expected4, haf4, tol);
}

// Check loop hafnian with eignevalues for all ones matrices with odd
// dimensions.
TEST(LoopHafnianEigenDouble, Odd) {
  std::vector<double> mat3(9, 1.0);
  std::vector<double> mat5(25, 1.0);

  double haf3 = libwalrus::loop_hafnian_eigen(mat3);
  double haf5 = libwalrus::loop_hafnian_eigen(mat5);

  EXPECT_NEAR(4, haf3, tol);
  EXPECT_NEAR(26, haf5, tol);
}

// Check loop hafnian with eignevalues for all ones complex matrices with even
// dimensions.
TEST(LoopHafnianEigenComplex, EvenOnes) {
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 0.0));

  std::complex<double> haf4 = libwalrus::loop_hafnian_eigen(mat4);
  std::complex<double> haf6 = libwalrus::loop_hafnian_eigen(mat6);

  EXPECT_NEAR(10, std::real(haf4), tol);
  EXPECT_NEAR(0, std::imag(haf4), tol);
  EXPECT_NEAR(76, std::real(haf6), tol);
  EXPECT_NEAR(0, std::imag(haf6), tol);
}

// Check loop hafnian with eigenvalues for random complex matrices with even
// dimensions.
TEST(LoopHafnianEigenComplex, EvenRandom) {
  std::vector<std::complex<double>> mat2(4, 0.0);
  std::vector<std::complex<double>> mat4(16, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat2[i * 2 + j] = std::complex<double>(randnum1, randnum2);
      mat2[j * 2 + i] = mat2[i * 2 + j];
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat4[i * 4 + j] = std::complex<double>(randnum1, randnum2);
      mat4[j * 4 + i] = mat4[i * 4 + j];
    }
  }

  std::complex<double> haf2 = libwalrus::loop_hafnian_eigen(mat2);
  std::complex<double> haf4 = libwalrus::loop_hafnian_eigen(mat4);

  std::complex<double> expected2 = mat2[1] + mat2[0] * mat2[3];
  std::complex<double> expected4 =
      mat4[1] * mat4[11] + mat4[2] * mat4[7] + mat4[3] * mat4[6] +
      mat4[0] * mat4[5] * mat4[11] + mat4[1] * mat4[10] * mat4[15] +
      mat4[2] * mat4[5] * mat4[15] + mat4[0] * mat4[10] * mat4[7] +
      mat4[0] * mat4[15] * mat4[6] + mat4[3] * mat4[5] * mat4[10] +
      mat4[0] * mat4[5] * mat4[10] * mat4[15];

  EXPECT_NEAR(std::real(expected2), std::real(haf2), tol);
  EXPECT_NEAR(std::imag(expected2), std::imag(haf2), tol);
  EXPECT_NEAR(std::real(expected4), std::real(haf4), tol);
  EXPECT_NEAR(std::imag(expected4), std::imag(haf4), tol);
}

// Check loop hafnian with eigenvalues for complex matrices with odd dimensions.
TEST(LoopHafnianEigenComplex, Odd) {
  std::vector<std::complex<double>> mat3(9, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat5(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf3 = libwalrus::loop_hafnian_eigen(mat3);
  std::complex<double> haf5 = libwalrus::loop_hafnian_eigen(mat5);

  EXPECT_NEAR(4, std::real(haf3), tol);
  EXPECT_NEAR(0, std::imag(haf3), tol);
  EXPECT_NEAR(26, std::real(haf5), tol);
  EXPECT_NEAR(0, std::imag(haf5), tol);
}


// Check loop hafnian with power-traces for random complex matrices with even dims
// using directly the function, i.e., without pre-padding.
TEST(LoopHafnianComplex, OddNoPadding) {
  std::vector<std::complex<double>> mat3(9, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat3[i * 3 + j] = std::complex<double>(randnum1, randnum2);
      mat3[j * 3 + i] = mat3[i * 3 + j];
    }
  }

  std::complex<double> haf3 = libwalrus::loop_hafnian_eigen(mat3);
  std::complex<double> hafq3 = libwalrus::loop_hafnian_quad(mat3);
  std::complex<double> expected3 = mat3[0] * mat3[4] * mat3[8] + mat3[2] * mat3[4] + mat3[1] * mat3[8] + mat3[5] * mat3[0];

  EXPECT_NEAR(std::real(expected3), std::real(haf3), tol);
  EXPECT_NEAR(std::imag(expected3), std::imag(haf3), tol);
  EXPECT_NEAR(std::real(expected3), std::real(hafq3), tol);
  EXPECT_NEAR(std::imag(expected3), std::imag(hafq3), tol);
}

// Check loop hafnian with power-traces for random real matrices with even dims
// using directly the function, i.e., without pre-padding.

TEST(LoopHafnianDouble, EvenNoPadding) {
  std::vector<double> mat3(9, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      mat3[i * 3 + j] = randnum1;
      mat3[j * 3 + i] = mat3[i * 3 + j];
    }
  }

  double haf3 = libwalrus::loop_hafnian_eigen(mat3);
  double hafq3 = libwalrus::loop_hafnian_quad(mat3);
  double expected3 = mat3[0] * mat3[4] * mat3[8] + mat3[2] * mat3[4] + mat3[1] * mat3[8] + mat3[5] * mat3[0];

  EXPECT_NEAR(expected3, haf3, tol);
  EXPECT_NEAR(expected3, hafq3, tol);
}



}  // namespace loophafnian_eigen

namespace loophafnian_repeated {

// Unit tests for the loop hafnian function using repeated
// Check repeated hafnian for empty matrix.
TEST(LoopHafnianRepeatedDouble, Empty) {
  std::vector<double> mat(0, 1.0);
  std::vector<double> mu(0, 0);
  std::vector<int> rpt(0, 0);

  std::vector<double> mat4(16, 1.0);
  std::vector<double> mu4(4, 0);
  std::vector<int> rpt4(4, 0);

  double haf = libwalrus::loop_hafnian_rpt_quad(mat, mu, rpt);
  double haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);

  EXPECT_NEAR(1, haf, tol);
  EXPECT_NEAR(1, haf4, tol);
}

// Check repeated hafnian for all ones matrices with even dimensions.
TEST(LoopHafnianRepeatedDouble, EvenOnes) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);

  std::vector<double> mu4(4, 0);
  std::vector<double> mu6(6, 0);

  std::vector<int> rpt4(4, 1);
  std::vector<int> rpt6(6, 1);

  for (int i = 0; i < 4; i++) mu4[i] = mat4[i * 4 + i];

  for (int i = 0; i < 6; i++) mu6[i] = mat6[i * 6 + i];

  double haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);
  double haf6 = libwalrus::loop_hafnian_rpt_quad(mat6, mu6, rpt6);

  EXPECT_NEAR(10, haf4, tol);
  EXPECT_NEAR(76, haf6, tol);
}

// Check repeated hafnian for all random  matrices with even dimensions.
TEST(LoopHafnianRepeatedDouble, EvenRandom) {
  std::vector<double> mat2(4, 0.0);
  std::vector<double> mat4(16, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  std::vector<double> mu2(2, 0);
  std::vector<double> mu4(4, 0);

  std::vector<int> rpt2(2, 1);
  std::vector<int> rpt4(4, 1);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      mat2[i * 2 + j] = randnum1;
      mat2[j * 2 + i] = mat2[i * 2 + j];
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      mat4[i * 4 + j] = randnum1;
      mat4[j * 4 + i] = mat4[i * 4 + j];
    }
  }

  for (int i = 0; i < 2; i++) mu2[i] = mat2[i * 2 + i];

  for (int i = 0; i < 4; i++) mu4[i] = mat4[i * 4 + i];

  double haf2 = libwalrus::loop_hafnian_rpt_quad(mat2, mu2, rpt2);
  double haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);

  double expected2 = mat2[1] + mat2[0] * mat2[3];
  double expected4 =
      mat4[1] * mat4[11] + mat4[2] * mat4[7] + mat4[3] * mat4[6] +
      mat4[0] * mat4[5] * mat4[11] + mat4[1] * mat4[10] * mat4[15] +
      mat4[2] * mat4[5] * mat4[15] + mat4[0] * mat4[10] * mat4[7] +
      mat4[0] * mat4[15] * mat4[6] + mat4[3] * mat4[5] * mat4[10] +
      mat4[0] * mat4[5] * mat4[10] * mat4[15];

  EXPECT_NEAR(expected2, haf2, tol);
  EXPECT_NEAR(expected4, haf4, tol);
}

// Check repeated hafnian for all ones matrices with odd dimensions.
TEST(LoopHafnianRepeatedDouble, Odd) {
  std::vector<double> mat3(9, 1.0);
  std::vector<double> mat5(25, 1.0);

  std::vector<double> mu3(3, 0);
  std::vector<double> mu5(5, 0);

  std::vector<int> rpt3(3, 1);
  std::vector<int> rpt5(5, 1);

  for (int i = 0; i < 3; i++) mu3[i] = mat3[i * 3 + i];

  for (int i = 0; i < 5; i++) mu5[i] = mat5[i * 5 + i];

  double haf3 = libwalrus::loop_hafnian_rpt_quad(mat3, mu3, rpt3);
  double haf5 = libwalrus::loop_hafnian_rpt_quad(mat5, mu5, rpt5);

  EXPECT_NEAR(4, haf3, tol);
  EXPECT_NEAR(26, haf5, tol);
}

// Check repeated hafnian of a complex empty matrix.
TEST(HafianEigenDoubleComplex, Empty) {
  std::vector<std::complex<double>> mat(0, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mu(0, 0);
  std::vector<int> rpt(0, 0);

  std::vector<std::complex<double>> mat4(16, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> mu4(4, 0);
  std::vector<int> rpt4(4, 0);

  std::complex<double> haf = libwalrus::loop_hafnian_rpt_quad(mat, mu, rpt);
  std::complex<double> haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);

  double re = std::real(haf);
  double im = std::imag(haf);

  double re4 = std::real(haf4);
  double im4 = std::imag(haf4);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
  EXPECT_NEAR(1, re4, tol);
  EXPECT_NEAR(0, im4, tol);
}

// Check repeated hafnian for all ones complex matrices with even dimensions.
TEST(LoopHafnianRepeatedComplex, EvenOnes) {
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 0.0));

  std::vector<std::complex<double>> mu4(4, 0);
  std::vector<std::complex<double>> mu6(6, 0);

  std::vector<int> rpt4(4, 1);
  std::vector<int> rpt6(6, 1);

  for (int i = 0; i < 4; i++) mu4[i] = mat4[i * 4 + i];
  std::complex<double> haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);
  EXPECT_NEAR(10, std::real(haf4), tol);
  EXPECT_NEAR(0, std::imag(haf4), tol);

  for (int i = 0; i < 6; i++) mu6[i] = mat6[i * 6 + i];

  std::complex<double> haf6 = libwalrus::loop_hafnian_rpt_quad(mat6, mu6, rpt6);

  EXPECT_NEAR(76, std::real(haf6), tol);
  EXPECT_NEAR(0, std::imag(haf6), tol);
}

// Check repeated hafnian for all random complex matrices with even dimensions.
TEST(LoopHafnianRepeatedComplex, EvenRandom) {
  std::vector<std::complex<double>> mat2(4, 0.0);
  std::vector<std::complex<double>> mat4(16, 0.0);

  std::default_random_engine generator;
  generator.seed(20);
  std::normal_distribution<double> distribution(0.0, 1.0);

  std::vector<std::complex<double>> mu2(2, 0);
  std::vector<std::complex<double>> mu4(4, 0);

  std::vector<int> rpt2(2, 1);
  std::vector<int> rpt4(4, 1);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat2[i * 2 + j] = std::complex<double>(randnum1, randnum2);
      mat2[j * 2 + i] = mat2[i * 2 + j];
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j <= i; j++) {
      double randnum1 = distribution(generator);
      double randnum2 = distribution(generator);
      mat4[i * 4 + j] = std::complex<double>(randnum1, randnum2);
      mat4[j * 4 + i] = mat4[i * 4 + j];
    }
  }

  for (int i = 0; i < 2; i++) mu2[i] = mat2[i * 2 + i];

  for (int i = 0; i < 4; i++) mu4[i] = mat4[i * 4 + i];

  std::complex<double> haf2 = libwalrus::loop_hafnian_rpt_quad(mat2, mu2, rpt2);
  std::complex<double> haf4 = libwalrus::loop_hafnian_rpt_quad(mat4, mu4, rpt4);

  std::complex<double> expected2 = mat2[1] + mat2[0] * mat2[3];
  std::complex<double> expected4 =
      mat4[1] * mat4[11] + mat4[2] * mat4[7] + mat4[3] * mat4[6] +
      mat4[0] * mat4[5] * mat4[11] + mat4[1] * mat4[10] * mat4[15] +
      mat4[2] * mat4[5] * mat4[15] + mat4[0] * mat4[10] * mat4[7] +
      mat4[0] * mat4[15] * mat4[6] + mat4[3] * mat4[5] * mat4[10] +
      mat4[0] * mat4[5] * mat4[10] * mat4[15];

  EXPECT_NEAR(std::real(expected2), std::real(haf2), tol);
  EXPECT_NEAR(std::imag(expected2), std::imag(haf2), tol);
  EXPECT_NEAR(std::real(expected4), std::real(haf4), tol);
  EXPECT_NEAR(std::imag(expected4), std::imag(haf4), tol);
}

// Check repeated hafnian for all ones matrices with odd dimensions.
TEST(LoopHafnianRepeatedComplex, Odd) {
  std::vector<std::complex<double>> mat3(9, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat5(25, std::complex<double>(1.0, 0.0));

  std::vector<std::complex<double>> mu3(3, 0);
  std::vector<std::complex<double>> mu5(5, 0);

  std::vector<int> rpt3(3, 1);
  std::vector<int> rpt5(5, 1);

  for (int i = 0; i < 3; i++) mu3[i] = mat3[i * 3 + i];

  for (int i = 0; i < 5; i++) mu5[i] = mat5[i * 5 + i];

  std::complex<double> haf3 = libwalrus::loop_hafnian_rpt_quad(mat3, mu3, rpt3);
  std::complex<double> haf5 = libwalrus::loop_hafnian_rpt_quad(mat5, mu5, rpt5);

  EXPECT_NEAR(4, std::real(haf3), tol);
  EXPECT_NEAR(0, std::imag(haf3), tol);
  EXPECT_NEAR(26, std::real(haf5), tol);
  EXPECT_NEAR(0, std::imag(haf5), tol);
}

}  // namespace loophafnian_repeated

namespace torontonian {
// Calculates the torontonian of a two-mode squeezed vacuum state squeezed with
// mean photon number 1.0
TEST(TorontonianDouble, TMSV) {
  std::vector<double> mat4(16, 0.0);
  std::vector<double> mat8(64, 0.0);
  std::vector<double> mat16(256, 0.0);

  double mean_n = 1.0;
  double r = asinh(std::sqrt(mean_n));

  int n = 4;
  for (int i = 0; i < n; i++) mat4[i * n + n - i - 1] = tanh(r) * 1.0;

  n = 8;
  for (int i = 0; i < n; i++) mat8[i * n + n - i - 1] = tanh(r) * 1.0;

  n = 16;
  for (int i = 0; i < n; i++) mat16[i * n + n - i - 1] = tanh(r) * 1.0;

  double tor4 = libwalrus::torontonian_quad(mat4);
  double tor8 = libwalrus::torontonian_quad(mat8);
  double tor16 = libwalrus::torontonian_quad(mat16);

  EXPECT_NEAR(1, tor4, tol);
  EXPECT_NEAR(1, tor8, tol);
  EXPECT_NEAR(1, tor16, tol);
}

TEST(TorontonianDouble, Vacuum) {
  int n_modes = 5;

  std::vector<double> mat(2 * n_modes * 2 * n_modes, 0.0);

  double tor_val = libwalrus::torontonian_quad(mat);

  EXPECT_NEAR(0, tor_val, tol);
}

TEST(TorontonianDouble, Analytical) {
  int n = 1;
  double nbar = 0.25;
  std::vector<double> mat1(2 * n * 2 * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat1[i * 2 * n + j] = nbar / (static_cast<double>(n) * (1.0 + nbar));
      mat1[(i + n) * 2 * n + (j + n)] =
          nbar / (static_cast<double>(n) * (1.0 + nbar));
    }
  }

  double tor1 = std::real(libwalrus::torontonian_quad(mat1));
  double expect1 = 0.25;

  n = 2;
  nbar = 0.25;
  std::vector<double> mat2(2 * n * 2 * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat2[i * 2 * n + j] = nbar / (static_cast<double>(n) * (1.0 + nbar));
      mat2[(i + n) * 2 * n + (j + n)] =
          nbar / (static_cast<double>(n) * (1.0 + nbar));
    }
  }

  double tor2 = std::real(libwalrus::torontonian_quad(mat2));
  double expect2 = 0.0277777777777;

  EXPECT_NEAR(expect1, tor1, tol);
  EXPECT_NEAR(expect2, tor2, tol);
}

}  // namespace torontonian

namespace batchhafnian {
TEST(BatchHafnian, Clements) {
    std::vector<std::complex<double>> mat4{std::complex<double>(-0.28264629150778969, 0.39867701584672210), std::complex<double>(-0.06086128222348247, -0.12220227033305252), std::complex<double>(-0.22959477315790058, 0.00000000000000008), std::complex<double>(-0.00660678867199307, -0.09884501458235322), std::complex<double>(-0.06086128222348247, -0.12220227033305252), std::complex<double>(0.38245649793510783, -0.41413300040003126), std::complex<double>(-0.00660678867199307, 0.09884501458235322), std::complex<double>(-0.13684045954832844, 0.00000000000000006), std::complex<double>(-0.22959477315790058, -0.00000000000000008), std::complex<double>(-0.00660678867199307, 0.09884501458235322), std::complex<double>(-0.28264629150778969, -0.39867701584672210), std::complex<double>(-0.06086128222348247, 0.12220227033305252), std::complex<double>(-0.00660678867199307, -0.09884501458235322), std::complex<double>(-0.13684045954832844, -0.00000000000000006), std::complex<double>(-0.06086128222348247, +0.12220227033305252), std::complex<double>(0.38245649793510783, 0.41413300040003126)};
    std::vector<std::complex<double>> d4{std::complex<double>(0.66917130190858, -1.52776303400764), std::complex<double>(-2.95847055822102, -1.29582519437023), std::complex<double>(0.66917130190858, 1.52776303400764), std::complex<double>(-2.95847055822102, 1.29582519437023)};
    std::vector<double> expected_re{1.00000000e+00, -1.64614736e+00,  1.94351456e+00, -1.44618627e+00,
                                    4.35642368e-01, -1.32047906e+00,  2.23766490e+00, -1.86917564e+00,
                                    -6.76966967e-01,  5.73670333e-01, -7.33188149e-02, -1.21997190e-01,
                                    2.32161778e-01, -5.57198229e-01,  1.18563164e+00, -1.79235874e+00,
                                    -1.64614736e+00,  3.23047167e+00, -4.65694018e+00,  4.44401287e+00,
                                    -4.63159381e-02,  1.31073870e+00, -3.22177207e+00,  3.63237405e+00,
                                    1.23991893e+00, -1.44213928e+00,  8.01092161e-01,  2.28567603e-01,
                                    -6.99231782e-01,  1.51022665e+00, -2.91603997e+00,  4.30125549e+00,
                                    1.94351456e+00, -4.65694018e+00,  8.15053238e+00, -9.76981613e+00,
                                    -7.95376620e-01, -3.06685257e-01,  2.99900529e+00, -5.19576276e+00,
                                    -1.40243198e+00,  2.03208134e+00, -1.62929470e+00, -2.50514870e-01,
                                    1.12880996e+00, -2.69285454e+00,  5.33966585e+00, -8.00210813e+00,
                                    -1.44618627e+00,  4.44401287e+00, -9.76981613e+00,  1.52284285e+01,
                                    1.11641813e+00, -8.67834158e-01, -1.21356826e+00,  4.85970544e+00,
                                    4.82389499e-01, -9.16927653e-01,  9.19681681e-01,  5.20655922e-01,
                                    -8.34195372e-01,  2.92229672e+00, -6.78809904e+00,  1.10996783e+01,
                                    4.35642368e-01, -4.63159381e-02, -7.95376620e-01,  1.11641813e+00,
                                    1.56877658e+00, -2.31606077e+00,  2.25637651e+00, -1.05357461e+00,
                                    4.79157988e-01, -1.44851326e+00,  2.00611116e+00, -5.18263640e-01,
                                    -6.15836536e-01,  5.36430564e-01, -2.88165936e-01,  4.59048908e-01,
                                    -1.32047906e+00,  1.31073870e+00, -3.06685257e-01, -8.67834158e-01,
                                    -2.31606077e+00,  4.15988484e+00, -5.13015068e+00,  3.41102851e+00,
                                    1.55349628e-01,  1.12665346e+00, -2.55195698e+00,  1.37496136e+00,
                                    5.89607772e-01, -4.94801663e-01, -7.51210237e-02,  5.26517347e-01,
                                    2.23766490e+00, -3.22177207e+00,  2.99900529e+00, -1.21356826e+00,
                                    2.25637651e+00, -5.13015068e+00,  8.00781690e+00, -7.39736288e+00,
                                    -1.05510937e+00,  1.44993971e-01,  1.83226945e+00, -2.23251713e+00,
                                    2.06934935e-01, -6.05111407e-01,  1.55663126e+00, -2.69619939e+00,
                                    -1.86917564e+00,  3.63237405e+00, -5.19576276e+00,  4.85970544e+00,
                                    -1.05357461e+00,  3.41102851e+00, -7.39736288e+00,  1.03896013e+01,
                                    6.44552723e-01, -6.96168471e-01, -2.62839607e-01,  1.95309353e+00,
                                    -1.42746493e+00,  2.66108892e+00, -4.01938103e+00,  4.99610368e+00,
                                    -6.76966967e-01,  1.23991893e+00, -1.40243198e+00,  4.82389499e-01,
                                    4.79157988e-01,  1.55349628e-01, -1.05510937e+00,  6.44552723e-01,
                                    2.08027043e+00, -2.71815189e+00,  2.08749438e+00, -3.59011119e-01,
                                    2.39920807e-01, -1.06932525e+00,  1.14339407e+00,  9.25081052e-01,
                                    5.73670333e-01, -1.44213928e+00,  2.03208134e+00, -9.16927653e-01,
                                    -1.44851326e+00,  1.12665346e+00,  1.44993971e-01, -6.96168471e-01,
                                    -2.71815189e+00,  4.45234470e+00, -4.61154260e+00,  1.74127108e+00,
                                    6.25361244e-01,  3.75915531e-01, -1.21876790e+00, -5.13059479e-01,
                                    -7.33188149e-02,  8.01092161e-01, -1.62929470e+00,  9.19681681e-01,
                                    2.00611116e+00, -2.55195698e+00,  1.83226945e+00, -2.62839607e-01,
                                    2.08749438e+00, -4.61154260e+00,  6.55944518e+00, -4.66420500e+00,
                                    -1.28536521e+00,  7.81945188e-01,  5.63969365e-01, -7.03167365e-01,
                                    -1.21997190e-01,  2.28567603e-01, -2.50514870e-01,  5.20655922e-01,
                                    -5.18263640e-01,  1.37496136e+00, -2.23251713e+00,  1.95309353e+00,
                                    -3.59011119e-01,  1.74127108e+00, -4.66420500e+00,  7.36229153e+00,
                                    1.42777111e-02, -3.18771206e-01, -5.24341968e-02,  1.68507163e+00,
                                    2.32161778e-01, -6.99231782e-01,  1.12880996e+00, -8.34195372e-01,
                                    -6.15836536e-01,  5.89607772e-01,  2.06934935e-01, -1.42746493e+00,
                                    2.39920807e-01,  6.25361244e-01, -1.28536521e+00,  1.42777111e-02,
                                    2.99123653e+00, -3.75195474e+00,  2.82213119e+00, -7.81358057e-01,
                                    -5.57198229e-01,  1.51022665e+00, -2.69285454e+00,  2.92229672e+00,
                                    5.36430564e-01, -4.94801663e-01, -6.05111407e-01,  2.66108892e+00,
                                    -1.06932525e+00,  3.75915531e-01,  7.81945188e-01, -3.18771206e-01,
                                    -3.75195474e+00,  6.10393167e+00, -6.48641622e+00,  3.30150978e+00,
                                    1.18563164e+00, -2.91603997e+00,  5.33966585e+00, -6.78809904e+00,
                                    -2.88165936e-01, -7.51210237e-02,  1.55663126e+00, -4.01938103e+00,
                                    1.14339407e+00, -1.21876790e+00,  5.63969365e-01, -5.24341968e-02,
                                    2.82213119e+00, -6.48641622e+00,  9.98950443e+00, -9.09570553e+00,
                                    -1.79235874e+00,  4.30125549e+00, -8.00210813e+00,  1.10996783e+01,
                                    4.59048908e-01,  5.26517347e-01, -2.69619939e+00,  4.99610368e+00,
                                    9.25081052e-01, -5.13059479e-01, -7.03167365e-01,  1.68507163e+00,
                                    -7.81358057e-01,  3.30150978e+00, -9.09570553e+00,  1.56159162e+01};
    std::vector<double> expected_im{0.00000000e+00, -6.19540212e-01,  1.62557597e+00, -2.04268077e+00,
                                    -1.07209959e+00,  1.37273368e+00, -1.04855753e+00,  2.44875659e-01,
                                    -5.35426993e-01,  1.06382831e+00, -1.13167436e+00, -5.50895821e-02,
                                    2.33832577e-01, -3.78336056e-01,  7.66353380e-01, -1.42492084e+00,
                                    6.19540212e-01, -8.32667268e-17, -1.64140851e+00,  3.13391670e+00,
                                    1.93588686e+00, -3.06589811e+00,  3.30672773e+00, -1.86213343e+00,
                                    3.61695050e-01, -1.18989013e+00,  1.65240909e+00, -7.67200444e-02,
                                    -5.09572526e-02,  1.60560058e-01, -6.31216079e-01,  1.58488036e+00,
                                    -1.62557597e+00,  1.64140851e+00, -2.00406005e-15, -2.89715424e+00,
                                    -2.45819767e+00,  4.73270217e+00, -6.50126689e+00,  5.38692527e+00,
                                    1.31935650e-01,  6.45842049e-01, -1.46639188e+00,  1.74105305e-01,
                                    -7.03114482e-01,  9.82371295e-01, -7.30475300e-01, -4.69877164e-01,
                                    2.04268077e+00, -3.13391670e+00,  2.89715424e+00, -9.52005210e-15,
                                    1.83179431e+00, -4.41516458e+00,  7.85181522e+00, -9.37606710e+00,
                                    -2.10974632e-01,  5.27110853e-02,  3.02027970e-01,  3.02537735e-01,
                                    1.73508640e+00, -2.84340169e+00,  3.30111115e+00, -2.05599107e+00,
                                    1.07209959e+00, -1.93588686e+00,  2.45819767e+00, -1.83179431e+00,
                                    -5.55111512e-17, -9.23929364e-01,  2.07252046e+00, -1.72348979e+00,
                                    -1.45132762e+00,  1.63837310e+00, -9.25631147e-01, -8.65199492e-02,
                                    -1.80257925e-02, -4.95003549e-03,  7.10339932e-01, -2.21351707e+00,
                                    -1.37273368e+00,  3.06589811e+00, -4.73270217e+00,  4.41516458e+00,
                                    9.23929364e-01, -6.97358837e-16, -1.96212896e+00,  2.69923158e+00,
                                    2.26051161e+00, -3.21251263e+00,  2.71847065e+00, -3.91405380e-01,
                                    -4.80113320e-01,  7.08913327e-01, -1.77090980e+00,  4.06599993e+00,
                                    1.04855753e+00, -3.30672773e+00,  6.50126689e+00, -7.85181522e+00,
                                    -2.07252046e+00,  1.96212896e+00, -2.15452656e-15, -2.46721598e+00,
                                    -2.22778040e+00,  3.99810349e+00, -4.56372028e+00,  1.81520753e+00,
                                    8.17667372e-01, -1.51861788e+00,  3.08750334e+00, -5.85075242e+00,
                                    -2.44875659e-01,  1.86213343e+00, -5.38692527e+00,  9.37606710e+00,
                                    1.72348979e+00, -2.69923158e+00,  2.46721598e+00, -8.65973959e-15,
                                    7.76349098e-01, -2.07828888e+00,  3.67741704e+00, -3.39674623e+00,
                                    1.67268754e-03,  1.05174705e+00, -3.26705990e+00,  6.24576761e+00,
                                    5.35426993e-01, -3.61695050e-01, -1.31935650e-01,  2.10974632e-01,
                                    1.45132762e+00, -2.26051161e+00,  2.22778040e+00, -7.76349098e-01,
                                    -4.49459509e-16, -1.15371273e+00,  2.14384564e+00, -7.75127965e-01,
                                    -1.69420825e+00,  1.75565162e+00, -7.87144208e-01, -2.91297712e-01,
                                    -1.06382831e+00,  1.18989013e+00, -6.45842049e-01, -5.27110853e-02,
                                    -1.63837310e+00,  3.21251263e+00, -3.99810349e+00,  2.07828888e+00,
                                    1.15371273e+00, -2.03309591e-15, -1.93589426e+00,  1.65877008e+00,
                                    2.16797656e+00, -2.87072370e+00,  1.92413422e+00,  6.63359067e-01,
                                    1.13167436e+00, -1.65240909e+00,  1.46639188e+00, -3.02027970e-01,
                                    9.25631147e-01, -2.71847065e+00,  4.56372028e+00, -3.67741704e+00,
                                    -2.14384564e+00,  1.93589426e+00, -3.84414722e-15, -1.84525302e+00,
                                    -1.51949589e+00,  2.78597545e+00, -2.83838661e+00,  1.47024726e-02,
                                    5.50895821e-02,  7.67200444e-02, -1.74105305e-01, -3.02537735e-01,
                                    8.65199492e-02,  3.91405380e-01, -1.81520753e+00,  3.39674623e+00,
                                    7.75127965e-01, -1.65877008e+00,  1.84525302e+00, -8.71004657e-15,
                                    5.32280868e-02, -7.29284151e-01,  2.11417457e+00, -2.58202829e+00,
                                    -2.33832577e-01,  5.09572526e-02,  7.03114482e-01, -1.73508640e+00,
                                    1.80257925e-02,  4.80113320e-01, -8.17667372e-01, -1.67268754e-03,
                                    1.69420825e+00, -2.16797656e+00,  1.51949589e+00, -5.32280868e-02,
                                    -1.26663794e-15, -1.59424776e+00,  2.80268987e+00, -9.62872951e-01,
                                    3.78336056e-01, -1.60560058e-01, -9.82371295e-01,  2.84340169e+00,
                                    4.95003549e-03, -7.08913327e-01,  1.51861788e+00, -1.05174705e+00,
                                    -1.75565162e+00,  2.87072370e+00, -2.78597545e+00,  7.29284151e-01,
                                    1.59424776e+00, -3.49720253e-15, -2.61480295e+00,  2.53217806e+00,
                                    -7.66353380e-01,  6.31216079e-01,  7.30475300e-01, -3.30111115e+00,
                                    -7.10339932e-01,  1.77090980e+00, -3.08750334e+00,  3.26705990e+00,
                                    7.87144208e-01, -1.92413422e+00,  2.83838661e+00, -2.11417457e+00,
                                    -2.80268987e+00,  2.61480295e+00, -6.24500451e-15, -3.10967275e+00,
                                    1.42492084e+00, -1.58488036e+00,  4.69877164e-01,  2.05599107e+00,
                                    2.21351707e+00, -4.06599993e+00,  5.85075242e+00, -6.24576761e+00,
                                    2.91297712e-01, -6.63359067e-01, -1.47024726e-02,  2.58202829e+00,
                                    9.62872951e-01, -2.53217806e+00,  3.10967275e+00, -1.42559575e-14};
    int res = 4;
    std::vector<std::complex<double>> d4c{std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0)};
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            d4c[i] += mat4[4*i+j] * d4[j];
        }
    }
    // Note that internaly we are implementing the modified multidimensional
    // Hermite polynomials, which means that we have to mat4 * d4 as the
    // second argument, this is precisely what is done in the previous two loops
    std::complex<double> *out  = libwalrus::hermite_multidimensional_cpp(mat4, d4c, res);

    for (int i = 0; i < 256; i++) {
        EXPECT_NEAR(expected_re[i], std::real(out[i]), tol2);
        EXPECT_NEAR(expected_im[i], std::imag(out[i]), tol2);
    }
    free(out);
}


TEST(BatchHafnian, UnitRenormalization) {
    std::vector<std::complex<double>> B = {std::complex<double>(0, 0), std::complex<double>(-0.70710678, 0), std::complex<double>(-0.70710678, 0), std::complex<double>(0, 0)};
    std::vector<std::complex<double>> d(4, std::complex<double>(0.0, 0.0));
    int res = 4;
    std::vector<double> expected_re(res * res, 0);
    std::vector<double> expected_im(res * res, 0);

    for (int i = 0; i < res; i++)
        expected_re[i*res+i] = pow(0.5, static_cast<double>(i)/2.0);

    std::complex<double> *tout  = libwalrus::renorm_hermite_multidimensional_cpp(B, d, res);

    for (int i = 0; i < res*res; i++) {
        EXPECT_NEAR(expected_re[i], std::real(tout[i]), tol2);
        EXPECT_NEAR(expected_im[i], std::imag(tout[i]), tol2);
    }
    free(tout);
}

}  // namespace batchhafnian
