#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "ga/genom.hpp"

template<typename T>
int fmemcmp(T a, T b) {
  for (int i = 0 ; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > 0.0001) {
      printf("a[%d]=%f, b[%d]=%f\n", i, a[i], i, b[i]);
      return i + 1;
    }
  }
  return 0;
}

TEST(GATest, crossover) {
  int center = 2;
  int range = 3;

  std::vector<float> genom_one = {-3, -2, -1, 0, 1, 2, 3};
  std::vector<float> genom_two = {-6, -4, -2, 0, 2, 4, 6};
  auto inc_itr = std::lower_bound(genom_two.begin(), genom_two.end(),
                                     genom_one[center]);
  auto dic_itr = inc_itr;
  dic_itr--;

  for (int i = 0; i < range; ++i) {
    if (inc_itr != genom_two.end()) {
      std::swap(*inc_itr, genom_one[center+i]);
      ++inc_itr;
    }
    if (i != 0 && dic_itr != genom_two.begin()) {
      std::swap(*dic_itr, genom_one[center-i]);
      --dic_itr;
    }
  }

  std::vector<float> expected_one = {-4, -2, 0, 2, 4, 2, 3};
  std::vector<float> expected_two = {-6, -3, -2, -1, 0, 1, 6};
  EXPECT_EQ(expected_one, genom_one);
  EXPECT_EQ(expected_two, genom_two);
}
