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

namespace recursive_real {

// Unit tests for the real recursive_hafnian function
// Check hafnian of real complete graphs with even dimensions.
TEST(HafnianRecursiveDouble, CompleteGraphEven) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);
  std::vector<double> mat8(64, 1.0);
  EXPECT_NEAR(3, libwalrus::hafnian_recursive_quad(mat4), tol);
  EXPECT_NEAR(15, libwalrus::hafnian_recursive_quad(mat6), tol);
  EXPECT_NEAR(105, libwalrus::hafnian_recursive_quad(mat8), tol);
}

// Check hafnian of real random matrix with size 4x4.
TEST(HafnianRecursiveDouble, Random) {
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
TEST(HafnianRecursiveDouble, CompleteGraphOdd) {
  std::vector<double> mat5(25, 1.0);
  std::vector<double> mat7(49, 1.0);
  std::vector<double> mat9(81, 1.0);
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat5));
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat7));
  EXPECT_EQ(0, libwalrus::hafnian_recursive_quad(mat9));
}

// Check hafnian of an empty matrix.
TEST(HafnianRecursiveDouble, Empty) {
  std::vector<double> mat(0, 0);
  EXPECT_EQ(1, libwalrus::hafnian_recursive_quad(mat));
}

}  // namespace recursive_real

namespace recursive_complex {

// Unit tests for the complex recursive_hafnian function
// Check hafnian of complex complete graphs with even dimensions.
TEST(HafnianRecursiveDoubleComplex, CompleteGraphEven) {
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
TEST(HafnianRecursiveDoubleComplex, Random) {
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
TEST(HafnianRecursiveDoubleComplex, CompleteGraphOdd) {
  std::vector<std::complex<double>> mat(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(0, re, tol);
  EXPECT_NEAR(0, im, tol);
}

// Check hafnian of a complex empty matrix.
TEST(HafnianRecursiveDoubleComplex, Empty) {
  std::vector<std::complex<double>> mat(0, std::complex<double>(0.0, 0.0));
  std::complex<double> haf = libwalrus::hafnian_recursive_quad(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
}

}  // namespace recursive_complex

namespace trace_real {

// Unit tests for the real trace_hafnian function
// Check hafnian of real complete graphs with even dimensions.
TEST(HafnianTraceDouble, CompleteGraphEven) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);
  std::vector<double> mat8(64, 1.0);
  EXPECT_NEAR(3, libwalrus::hafnian_trace(mat4), tol);
  EXPECT_NEAR(15, libwalrus::hafnian_trace(mat6), tol);
  EXPECT_NEAR(105, libwalrus::hafnian_trace(mat8), tol);
}

// Check hafnian of real random matrix with size 4x4.
TEST(HafnianTraceDouble, Random) {
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

  EXPECT_NEAR(expected, libwalrus::hafnian_trace(mat), tol);
}

// Check hafnian of complete graphs with odd dimensions.
TEST(HafnianTraceDouble, CompleteGraphOdd) {
  std::vector<double> mat5(25, 1.0);
  std::vector<double> mat7(49, 1.0);
  std::vector<double> mat9(81, 1.0);
  EXPECT_EQ(0, libwalrus::hafnian_trace(mat5));
  EXPECT_EQ(0, libwalrus::hafnian_trace(mat7));
  EXPECT_EQ(0, libwalrus::hafnian_trace(mat9));
}

// Check hafnian of an empty matrix.
TEST(HafnianTraceDouble, Empty) {
  std::vector<double> mat(0, 0);
  EXPECT_EQ(1, libwalrus::hafnian_trace(mat));
}

}  // namespace trace_real

namespace trace_complex {

// Unit tests for the complex recursive_hafnian function
// Check hafnian of complex complete graphs with even dimensions.
TEST(HafnianTraceDoubleComplex, CompleteGraphEven) {
  std::vector<std::complex<double>> mat(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 1.0));
  std::vector<std::complex<double>> mat8(64, std::complex<double>(1.0, 1.0));

  std::complex<double> haf = libwalrus::hafnian_trace(mat);
  std::complex<double> haf4 = libwalrus::hafnian_trace(mat4);
  std::complex<double> haf6 = libwalrus::hafnian_trace(mat6);
  std::complex<double> haf8 = libwalrus::hafnian_trace(mat8);

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
TEST(HafnianTraceDoubleComplex, Random) {
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

  std::complex<double> haf = libwalrus::hafnian_trace(mat);

  double re_expected = std::real(expected);
  double im_expected = std::imag(expected);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(re_expected, re, tol);
  EXPECT_NEAR(im_expected, im, tol);
}

// Check hafnian of complex complete graphs with odd dimensions.
TEST(HafnianTraceDoubleComplex, CompleteGraphOdd) {
  std::vector<std::complex<double>> mat(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf = libwalrus::hafnian_trace(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(0, re, tol);
  EXPECT_NEAR(0, im, tol);
}

// Check hafnian of a complex empty matrix.
TEST(HafnianTraceDoubleComplex, Empty) {
  std::vector<std::complex<double>> mat(0, std::complex<double>(0.0, 0.0));
  std::complex<double> haf = libwalrus::hafnian_trace(mat);

  double re = std::real(haf);
  double im = std::imag(haf);

  EXPECT_NEAR(1, re, tol);
  EXPECT_NEAR(0, im, tol);
}

}  // namespace trace_complex

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

namespace loophafnian_trace {

// Unit tests for the loop hafnian function using power-traces
// Check loop hafnian with eignevalues for all ones matrices with even
// dimensions.
TEST(LoopHafnianTraceDouble, EvenOnes) {
  std::vector<double> mat4(16, 1.0);
  std::vector<double> mat6(36, 1.0);

  double haf4 = libwalrus::loop_hafnian_trace(mat4);
  double haf6 = libwalrus::loop_hafnian_trace(mat6);

  EXPECT_NEAR(10, haf4, tol);
  EXPECT_NEAR(76, haf6, tol);
}

// // Check loop hafnian with power-traces for random matrices with even
// dimensions.
TEST(LoopHafnianTraceDouble, EvenRandom) {
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

  double haf2 = libwalrus::loop_hafnian_trace(mat2);
  double haf4 = libwalrus::loop_hafnian_trace(mat4);

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

// Check loop hafnian with power-traces for all ones matrices with odd
// dimensions.
TEST(LoopHafnianTraceDouble, Odd) {
  std::vector<double> mat3(9, 1.0);
  std::vector<double> mat5(25, 1.0);

  double haf3 = libwalrus::loop_hafnian_trace(mat3);
  double haf5 = libwalrus::loop_hafnian_trace(mat5);

  EXPECT_NEAR(4, haf3, tol);
  EXPECT_NEAR(26, haf5, tol);
}

// Check loop hafnian with power-traces for all ones complex matrices with even
// dimensions.
TEST(LoopHafnianTraceComplex, EvenOnes) {
  std::vector<std::complex<double>> mat4(16, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat6(36, std::complex<double>(1.0, 0.0));

  std::complex<double> haf4 = libwalrus::loop_hafnian_trace(mat4);
  std::complex<double> haf6 = libwalrus::loop_hafnian_trace(mat6);

  EXPECT_NEAR(10, std::real(haf4), tol);
  EXPECT_NEAR(0, std::imag(haf4), tol);
  EXPECT_NEAR(76, std::real(haf6), tol);
  EXPECT_NEAR(0, std::imag(haf6), tol);
}

// Check loop hafnian with power-traces for random complex matrices with even
// dimensions.
TEST(LoopHafnianTraceComplex, EvenRandom) {
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

  std::complex<double> haf2 = libwalrus::loop_hafnian_trace(mat2);
  std::complex<double> haf4 = libwalrus::loop_hafnian_trace(mat4);

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

// Check loop hafnian with power-traces for complex matrices with odd dimensions.
TEST(LoopHafnianTraceComplex, Odd) {
  std::vector<std::complex<double>> mat3(9, std::complex<double>(1.0, 0.0));
  std::vector<std::complex<double>> mat5(25, std::complex<double>(1.0, 0.0));

  std::complex<double> haf3 = libwalrus::loop_hafnian_trace(mat3);
  std::complex<double> haf5 = libwalrus::loop_hafnian_trace(mat5);

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

  std::complex<double> haf3 = libwalrus::loop_hafnian_trace(mat3);
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

  double haf3 = libwalrus::loop_hafnian_trace(mat3);
  double hafq3 = libwalrus::loop_hafnian_quad(mat3);
  double expected3 = mat3[0] * mat3[4] * mat3[8] + mat3[2] * mat3[4] + mat3[1] * mat3[8] + mat3[5] * mat3[0];

  EXPECT_NEAR(expected3, haf3, tol);
  EXPECT_NEAR(expected3, hafq3, tol);
}



}  // namespace loophafnian_trace

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
TEST(HafnianTraceDoubleComplex, Empty) {
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
