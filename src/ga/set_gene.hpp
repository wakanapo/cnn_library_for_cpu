#pragma once

#include <vector>


class GlobalParams {
 public:  
  const std::vector<float>& partition() const {
    return partition_;
  }
  static GlobalParams* getInstance();
  static void setParams(std::vector<float> partition);
 private:
  GlobalParams(std::vector<float> partition)
    : partition_(std::move(partition)) {};
  std::vector<float> partition_;
};
