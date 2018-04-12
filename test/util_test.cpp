#include <cstdio>
#include <iostream>

#include "util/function.hpp"
#include "util/tensor.hpp"
#include "util/read_data.hpp"
#include "gtest/gtest.h"

void debug(float* a, int n) {
  for (int i = 0; i < n; ++i)
    std::cout << a[i] << " ";
  std::cout << std::endl;
}

int fmemcmp(float* a, float* b, size_t size) {
  for (size_t i = 0 ; i < size / sizeof(float); ++i) {
    if (std::abs(a[i] - b[i]) > 0.0001) {
      printf("a[%4lu]=%f, b[%4lu]=%f\n", i, a[i], i, b[i]);
      return i + 1;
    }
  }
  return 0;
}

TEST(ReadDataTest, OneHot) {
  float expected[10] = {};
  unsigned long t = 2;
  expected[t] = 1.0;

  float* actual = mnistOneHot<float>(t);
  EXPECT_EQ(fmemcmp(expected, actual, sizeof(expected)), 0);
}

TEST(MatmulTest, Matmul2D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2};
  Tensor2D<3, 2, float> x;
  x.set_v(x_raw);

  float y_raw[] = {1, 2,
                   2, 3,
                   3, 4};
  Tensor2D<2, 3, float> y;
  y.set_v(y_raw);

  float expected_raw[] = {6, 9,
                          12, 18};
  Tensor2D<2, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<2, 2, float> actual;

  Function::matmul(x, y, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(MatmulTest, Matmul3D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3,
                   1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  Tensor3D<3, 3, 2, float> x;
  x.set_v(x_raw);

  float y_raw[] = {1, 2,
                   2, 3,
                   3, 4,
                   1, 1,
                   1, 1,
                   1, 1};
  Tensor3D<2, 3, 2, float> y;
  y.set_v(y_raw);

  float expected_raw[] = {6, 9,
                          12, 18,
                          18, 27,
                          3, 3,
                          6, 6,
                          9, 9};
  Tensor3D<2, 3, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor3D<2, 3, 2, float> actual;

  Function::matmul(x, y, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}

TEST(MatmulTest, Matmul1Dand1D) {
  float x_raw[] = {1,
                   2,
                   3};
  Tensor2D<1, 3, float> x;
  x.set_v(x_raw);

  float y_raw[] = {1, 2, 3};
  Tensor2D<3, 1, float> y;
  y.set_v(y_raw);

  float expected_raw[] = {1, 2, 3,
                          2, 4, 6,
                          3, 6, 9};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<3, 3, float> actual;

  Function::matmul(x, y, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}


TEST(TransposeTest, Transpose2D) {
  float x_raw[] = {11, 12, 13,
                   21, 22, 23,
                   31, 32, 33};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {11, 21, 31,
                          12, 22, 32,
                          13, 23, 33};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<3, 3, float> actual = x.transpose();

  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}


TEST(PaddingTest, Padding1) {
  float x_raw[] = {1, 2, 3,
                   1, 2, 3,
                   1, 2, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0, 0, 0, 0, 0,
                          0, 1, 2, 3, 0,
                          0, 1, 2, 3, 0,
                          0, 1, 2, 3, 0,
                          0, 0, 0, 0, 0};
  Tensor2D<5, 5, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<5, 5, float> actual;

  Function::padding(x, &actual, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(PaddingTest, Padding2) {
  float x_raw[] = {1, 2, 3,
                   1, 2, 3,
                   1, 2, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 2, 3, 0, 0,
                          0, 0, 1, 2, 3, 0, 0,
                          0, 0, 1, 2, 3, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0};
  Tensor2D<7, 7, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<7, 7, float> actual;

  Function::padding(x, &actual, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(PaddingTest, Padding3D) {
  float x_raw[] = {1, 2, 3,
                   1, 2, 3,
                   1, 2, 3,
                   2, 4, 6,
                   2, 4, 6,
                   2, 4, 6};
  Tensor3D<3, 3, 2, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0, 0, 0, 0, 0,
                          0, 1, 2, 3, 0,
                          0, 1, 2, 3, 0,
                          0, 1, 2, 3, 0,
                          0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0,
                          0, 2, 4, 6, 0,
                          0, 2, 4, 6, 0,
                          0, 2, 4, 6, 0,
                          0, 0, 0, 0, 0};
  Tensor3D<5, 5, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor3D<5, 5, 2, float> actual;

  Function::padding(x, &actual, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(SoftmaxTest, Softmax2D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0.33333333,  0.33333333,  0.33333333,
                          0.33333333,  0.33333333,  0.33333333,
                          0.33333333,  0.33333333,  0.33333333};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SoftmaxTest, Softmax3D) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3,
                   1, 2, 3,
                   1, 2, 3,
                   1, 2, 3};
  Tensor3D<3, 3, 2, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0.33333333, 0.33333333, 0.33333333,
                          0.33333333, 0.33333333, 0.33333333,
                          0.33333333, 0.33333333, 0.33333333,
                          0.09003057, 0.24472847, 0.66524096,
                          0.09003057, 0.24472847, 0.66524096,
                          0.09003057, 0.24472847, 0.66524096};
  Tensor3D<3, 3, 2, float> expected;
  expected.set_v(expected_raw);

  Function::softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SoftmaxTest, DerivSoftmax) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0.22222222, 0.22222222, 0.22222222,
                          0.22222222, 0.22222222, 0.22222222,
                          0.22222222, 0.22222222, 0.22222222};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::deriv_softmax(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SigmoidTest, Sigmoid) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = { 0.7310586,  0.7310586, 0.7310586,
                           0.88079703, 0.88079703, 0.88079703,
                           0.95257413, 0.95257413, 0.95257413};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::sigmoid(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(SigmoidTest, DerivSigmoid) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2,
                   3, 3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0.19661193,  0.19661193,  0.19661193,
                          0.10499363,  0.10499363,  0.10499363,
                          0.04517666,  0.04517666,  0.04517666};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::deriv_sigmoid(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(ReLUTest, ReLU) {
  float x_raw[] = {0.1, -0.1, 0.1,
                   -2, 2, -2,
                   3, -3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {0.1, 0, 0.1,
                          0, 2, 0,
                          3, 0, 3};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::ReLU(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(ReLUTest, DerivReLU) {
  float x_raw[] = {0.1, -0.1, 0.1,
                   -2, 2, -2,
                   3, -3, 3};
  Tensor2D<3, 3, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {1, 0, 1,
                          0, 1, 0,
                          1, 0, 1};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);

  Function::deriv_ReLU(&x);
  EXPECT_EQ(fmemcmp(x.get_v(), expected.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Stride1) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1};
  Tensor2D<3, 3, float> w;
  w.set_v(w_raw);

  float expected_raw[] = {4, 3, 4,
                          2, 4, 3,
                          2, 3, 4};
  Tensor2D<3, 3, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<3, 3, float> actual;

  Function::conv2d(x, w, &actual, 0, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Strides2) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1};
  Tensor2D<3, 3, float> w;
  w.set_v(w_raw);

  float expected_raw[] = {4, 4,
                          2, 4};
  Tensor2D<2, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<2, 2, float> actual;

  Function::conv2d(x, w, &actual, 0, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Out3D) {
  float x_raw[] = {1, 1, 1, 0, 0,
                   0, 1, 1, 1, 0,
                   0, 0, 1, 1, 1,
                   0, 0, 1, 1, 0,
                   0, 1, 1, 0, 0};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float w_raw[] = {1, 0, 1,
                   0, 1, 0,
                   1, 0, 1,
                   2, 1, 2,
                   1, 2, 1,
                   2, 1, 2,
                   0.2, 0.1, 0.2,
                   0.1, 0.2, 0.1,
                   0.2, 0.1, 0.2};
  Tensor4D<3, 3, 1, 3, float> w;
  w.set_v(w_raw);

  float expected_raw[] = {4, 3, 4,
                          2, 4, 3,
                          2, 3, 4,
                          10, 10, 10,
                          6, 11, 10,
                          6, 9, 10,
                          1, 1, 1,
                          0.6, 1.1, 1,
                          0.6, 0.9, 1};
  Tensor3D<3, 3, 3, float> expected;
  expected.set_v(expected_raw);
  Tensor3D<3, 3, 3, float> actual;

  Function::conv2d(x, w, &actual, 0, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(Conv2dTest, Deconv) {
  float x_raw[] = {0, 1, 2, 3, 4,
                   5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float w_raw[] = {0, 1,
                   2, 3};
  Tensor2D<2, 2, float> w;
  w.set_v(w_raw);

  Tensor2D<7, 7, float> pad_conv;

  float expected_raw[] = {0, 0, 1, 2, 3, 4,
                          0, 7, 13, 19, 25, 21,
                          10, 37, 43, 49, 55, 41,
                          20, 67, 73, 79, 85, 61,
                          30, 97, 103, 109, 115, 81,
                          40, 102, 107, 112, 117, 72};
  Tensor2D<6, 6, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<6, 6, float> actual;

  Function::deconv2d(x, w, &actual, 0, 1);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Stride1) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {83, 82, 83, 83,
                          87, 82, 83, 83,
                          87, 87, 85, 80,
                          87, 87, 85, 78};
  Tensor2D<4, 4, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<4, 4, float> actual;
  Tensor1D<16, int> idx;

  Function::max_pool(x, 2, 2, &actual, &idx, 0, 1);
  EXPECT_EQ(memcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Stride2) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76};
  Tensor2D<5, 5, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {83, 83,
                          87, 85};
  Tensor2D<2, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<2, 2, float> actual;
  Tensor1D<4, int> idx;

  Function::max_pool(x, 2, 2, &actual, &idx, 0, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, Out3D) {
  float x_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76,
                   77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 7};
  Tensor3D<5, 5, 2, float> x;
  x.set_v(x_raw);

  float expected_raw[] = {83, 83,
                          87, 85,
                          83, 83,
                          87, 85};
  Tensor3D<2, 2, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor3D<2, 2, 2, float> actual;
  Tensor1D<8, int> idx;

  Function::max_pool(x, 2, 2, &actual, &idx, 0, 2);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(MaxPoolTest, DepoolStride2) {
  float before_raw[] = {77, 80, 82, 78, 70,
                   83, 78, 80, 83, 82,
                   87, 82, 81, 80, 74,
                   87, 87, 85, 77, 66,
                   84, 79, 77, 78, 76};
  Tensor2D<5, 5, float> before;
  before.set_v(before_raw);

  Tensor2D<2, 2, float> x;

  Tensor1D<4, int> idx;

  Function::max_pool(before, 2, 2, &x, &idx, 0, 2);

  float expected_raw[] = {0, 0, 0, 0, 0,
                          83, 0, 0, 83, 0,
                          87, 0, 0, 0, 0,
                          0, 0, 85, 0, 0,
                          0, 0, 0, 0, 0};
  Tensor2D<5, 5, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<5, 5, float> actual;

  Function::depool(x, idx, &actual);
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), expected.bytes()), 0);
}

TEST(AddTest, AddBias) {
  float x_raw[] = {1, 1, 1, 1,
                   2, 2, 2, 2,
                   3, 3, 3, 3};
  Tensor3D<2, 2, 3, float> x;
  x.set_v(x_raw);

  float b_raw[] = {3, 2, 1};
  Tensor1D<3, float> b;
  b.set_v(b_raw);

  float expected_raw[] = {4, 4, 4, 4,
                          4, 4, 4, 4,
                          4, 4, 4, 4};
  Tensor3D<2, 2, 3, float> expected;
  expected.set_v(expected_raw);

  Function::add_bias(&x, b);
  EXPECT_EQ(fmemcmp(expected.get_v(), x.get_v(), expected.bytes()), 0);
}

TEST(OperatorTest, Add) {
  float x_raw[] = {1, 1, 1,
                   2, 2, 2};
  Tensor2D<3, 2, float> x;
  x.set_v(x_raw);

  float y_raw[] = {1, 2, 3,
                   1, 2, 3};
  Tensor2D<3, 2, float> y;
  y.set_v(y_raw);

  float expected_raw[] = {2, 3, 4,
                          3, 4, 5};
  Tensor2D<3, 2, float> expected;
  expected.set_v(expected_raw);
  Tensor2D<3, 2, float> actual;

  actual = x + y;
  EXPECT_EQ(fmemcmp(expected.get_v(), actual.get_v(), actual.bytes()), 0);
}
