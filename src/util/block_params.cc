#include "util/block_params.hpp"
#include "util/color.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <thread>

namespace {
  thread_local static std::unique_ptr<BlockParams> g_block_params;
}  // namespace

int BlockParams::fromFloat(float val) const {
  return  std::upper_bound(partition_.begin(), partition_.end(), val) - partition_.begin();
}

float BlockParams::toFloat(int val) const {
  return rep_[val];
}

BlockParams* BlockParams::getInstance() {
  if (g_block_params == nullptr) {
    std::cerr << coloringText("ERROR!", Color::RED)
              << ": Please call BlockParams::setParams(partition) first."
              << std::endl;
    exit(1);
  }
  return g_block_params.get();
}

void BlockParams::setParams(std::vector<float> partition) {
  BlockParams* block_params = new BlockParams(std::move(partition));
  // Memorize representative value.
  block_params->rep_.push_back(block_params->partition_[0]);
  int size = block_params->partition_.size();
  for (int i = 1; i < size; ++i) {
    block_params->rep_.push_back((block_params->partition_[i] +
                                  block_params->partition_[i-1]) / 2.0);
  }
  block_params->rep_.push_back(block_params->partition_[size-1]);

  // Memorize add.
  block_params->add_ =
    std::vector<std::vector<int>>(size+1, std::vector<int>(size+1, 0));
  for (int i = 0; i < size + 1; ++i) {
    for (int j = 0; j < size + 1; ++j) {
      block_params->add_[i][j] =
        block_params->fromFloat(block_params->rep_[i] + block_params->rep_[j]);
    }
  }

  // Memorize sub.
  block_params->sub_ =
    std::vector<std::vector<int>>(size+1, std::vector<int>(size+1, 0));
  for (int i = 0; i < size + 1; ++i) {
    for (int j = 0; j < size + 1; ++j) {
      block_params->sub_[i][j] =
        block_params->fromFloat(block_params->rep_[i] - block_params->rep_[j]);
    }
  }

  // Memorize add.
  block_params->mul_ =
    std::vector<std::vector<int>>(size+1, std::vector<int>(size+1, 0));
  for (int i = 0; i < size + 1; ++i) {
    for (int j = 0; j < size + 1; ++j) {
      block_params->mul_[i][j] =
        block_params->fromFloat(block_params->rep_[i] * block_params->rep_[j]);
    }
  }

  // Memorize div.
  block_params->div_ =
    std::vector<std::vector<int>>(size+1, std::vector<int>(size+1, 0));
  for (int i = 0; i < size + 1; ++i) {
    for (int j = 0; j < size + 1; ++j) {
      block_params->div_[i][j] =
        block_params->fromFloat(block_params->rep_[i] / block_params->rep_[j]);
    }
  }
  g_block_params.reset(std::move(block_params));
}
