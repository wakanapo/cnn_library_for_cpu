#include "util/box_quant.hpp"
#include "gtest/gtest.h"

TEST(BoxQuantTest, MinumulNum) {
  GlobalParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = -4.0;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, -3.0);
}

TEST(BoxQuantTest, MaximumNum) {
  GlobalParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 4.0;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 3.0);
}

TEST(BoxQuantTest, Num1) {
  GlobalParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 1.2;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 1.5);
}

TEST(BoxQuantTest, Num2) {
  GlobalParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = -2.1;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, -2.5);
}

TEST(BoxQuantTest, Num3) {
  GlobalParams::setParams({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  float a = 2.7;
  float b = Box(a).toFloat();
  EXPECT_EQ(b, 2.5);
}

