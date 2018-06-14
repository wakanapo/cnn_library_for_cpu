#include "util/block.hpp"
#include "gtest/gtest.h"

TEST(BoxQuantTest, MinumulNum) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = -4.0;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, -3.0);
}

TEST(BoxQuantTest, MaximumNum) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 4.0;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 3.0);
}

TEST(BoxQuantTest, Num1) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 1.2;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 1.5);
}

TEST(BoxQuantTest, Num2) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = -2.1;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, -2.5);
}

TEST(BoxQuantTest, Num3) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 2.7;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 2.5);
}

TEST(BoxQuantTest, Add) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  Box a = 1.2 + 0.9; // -> 1.5 + 0.5 = 2.0 -> 2.5
  float b = a.toFloat();
  EXPECT_EQ(b, 2.5);
}

TEST(BoxQuantTest, Sub) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  Box a = 1.2 - 1.8; // -> 1.5 - 1.5 = 0.0 -> -0.5
  float b = a.toFloat();
  EXPECT_EQ(b, -0.5);
}

TEST(BoxQuantTest, Mul) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  Box a = 2.5 * -2.0; // -> 2.5 * -1.5 = -3.75 -> -3.0
  float b = a.toFloat();
  EXPECT_EQ(b, -3.0);
}

TEST(BoxQuantTest, Div) {
  BlockParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  Box a = 1.0 / 2.0; // -> 1.5 / 2.5 = 0.6 -> 0.5
  float b = a.toFloat();
  EXPECT_EQ(b, 0.5);
}
