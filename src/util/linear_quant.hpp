#pragma once

#include <algorithm>
#include <vector>

#include "util/linear_range.hpp"

#define kOffset 128

class Box {
public:
  Box() = default;
  Box(const Box& other) = default;
  Box(const Box&& other) : val_(other.get()) {}
  Box(int other) : val_(other) {}
  Box(float other) : val_(fromFloat(other)) {}
  Box(double other) : val_(fromFloat(other)) {}
  
  float toFloat() const {
    if (this->val_ == 0)
      return 0;
    return (this->val_ < 0 ? partation_[this->val_ + kOffset] : partation_[this->val_ + kOffset - 1]);
  }
  
  int fromFloat(float fl) const {
    return  std::upper_bound(partation_.begin(), partation_.end(), fl) - partation_.begin() - kOffset;
  }

  static Box min() {
    Box val;
    val.val_ = -kOffset;
    return val;
  };

  inline const int get() const {return val_;}

  Box &operator=(float& fl) {
    val_ = fromFloat(fl);
    return *this;
  }

  Box &operator=(const int& in) {
    val_ = in;
    return *this;
  }

  Box &operator=(const Box& bx) {
    val_ = bx.get();
    return *this;
  }
  
  Box operator+(const Box& other) const {
    return fromFloat(this->toFloat() + other.toFloat());
  }
  
  Box operator*(const Box& other) const {
    return fromFloat(this->toFloat() * other.toFloat());
  }
  
  Box operator/(const Box& other) const {
    return fromFloat(this->toFloat() / other.toFloat());
  }
  
  Box operator-(const Box& other) const {
    return fromFloat(this->toFloat() - other.toFloat());
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
  std::vector<float> partation_ = range;
  int val_;
};

namespace std {
  template<> class numeric_limits<Box> {
  public:
    static Box lowest() {return Box::min();};
  };
}
