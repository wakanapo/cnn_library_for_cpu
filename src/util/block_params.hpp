#pragma once

#include <vector>


class BlockParams {
public:
  std::vector<float> partition() const {
    return partition_;
  }
  int add(int a, int b) const {
    return add_[a][b];
  }
  int sub(int a, int b) const {
    return sub_[a][b];
  }
  int mul(int a, int b) const {
    return mul_[a][b];
  }
  int div(int a, int b) const {
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
