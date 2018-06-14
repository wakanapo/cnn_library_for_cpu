#pragma once

#include <algorithm>
#include <vector>

#include "util/block_params.hpp"

class Box {
public:
  Box() {};
  Box(const Box& other) : val_(other.get()) {};
  Box(const Box&& other) :val_(std::move(other.get())) {};
  Box(int other) : val_(BlockParams::getInstance()->fromFloat(other)) {};
  Box(float other) : val_(BlockParams::getInstance()->fromFloat(other)) {};
  Box(double other): val_(BlockParams::getInstance()->fromFloat(other)) {};

  static Box min() {
    Box val;
    val.val_ = 0;
    return val;
  };

  inline int get() const {return val_;}

  float toFloat() const {
    return BlockParams::getInstance()->toFloat(this->val_);
  }

  Box &operator=(float& fl) {
    val_ = BlockParams::getInstance()->fromFloat(fl);
    return *this;
  }

  Box &operator=(const int& in) {
    val_ = BlockParams::getInstance()->fromFloat(in);
    return *this;
  }

  Box &operator=(const Box& bx) {
    val_ = bx.get();
    return *this;
  }

  Box operator+(const Box& other) const {
    Box box;
    box.val_ = BlockParams::getInstance()->add(this->val_, other.get());
    return box;
  }

  Box operator*(const Box& other) const {
    Box box;
    box.val_ = BlockParams::getInstance()->mul(this->val_, other.get());
    return box;
  }

  Box operator/(const Box& other) const {

    Box box;
    box.val_ = BlockParams::getInstance()->div(this->val_, other.get());
    return box;
  }

  Box operator-(const Box& other) const {
    Box box;
    box.val_ = BlockParams::getInstance()->sub(this->val_, other.get());
    return box;
  }

  bool operator!=(const Box& bx) const {
    return val_ != bx.get();
  }

  bool operator==(const Box& bx) const {
    return val_ == bx.get();
  }

  bool operator>(const Box& bx) const {
    return val_ > bx.get();
  }

  bool operator<(const Box& bx) const {
    return val_ < bx.get();
  }

  bool operator>=(const Box& bx) const {
    return val_ >= bx.get();
  }

  bool operator<=(const Box& bx) const {
    return val_ <= bx.get();
  }
  
private:
  int val_;
};

namespace std {
  template<> class numeric_limits<Box> {
  public:
    static Box lowest() {return Box::min();};
  };
}
