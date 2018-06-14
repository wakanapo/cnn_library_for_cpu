#pragma once

#include <vector>


class BlockParams {
public:
  const std::vector<float>& partition() const {
    return partition_;
  }
  const int add(int a, int b) const {
    return add_[a][b];
  }
  const int sub(int a, int b) const {
    return sub_[a][b];
  }
  const int mul(int a, int b) const {
    return mul_[a][b];
  }
  const int div(int a, int b) const {
    return div_[a][b];
  }
  int fromFloat(float val) const;
  float toFloat(int val) const;
  static BlockParams* getInstance();
  static void setParams(std::vector<float> partition);
private:
  BlockParams(std::vector<float> partition)
    : partition_(std::move(partition)) {};
  std::vector<float> partition_;
  std::vector<float> rep_;
  std::vector<std::vector<int>> add_;
  std::vector<std::vector<int>> sub_;
  std::vector<std::vector<int>> mul_;
  std::vector<std::vector<int>> div_;
};
