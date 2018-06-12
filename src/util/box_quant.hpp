#pragma once

#include <algorithm>
#include <vector>

#include "ga/set_gene.hpp"

class Box {
public:
  Box() : partation_(GlobalParams::getInstance()->partition()) { offset_ = partation_.size() / 2; };
  Box(const Box& other) : partation_(GlobalParams::getInstance()->partition()),
                          val_(other.get()), offset_(partation_.size() / 2) {};
  Box(const Box&& other) : partation_(GlobalParams::getInstance()->partition()),
                           val_(std::move(other.get())),
                           offset_(partation_.size() / 2) {};
  Box(int other) : partation_(GlobalParams::getInstance()->partition()),
                   val_(other), offset_(partation_.size() / 2) {};
  Box(float other) : partation_(GlobalParams::getInstance()->partition()),
                     val_(fromFloat(other)), offset_(partation_.size() / 2) {};
  Box(double other) : partation_(GlobalParams::getInstance()->partition()),
                      val_(fromFloat(other)), offset_(partation_.size() / 2) {};
  
  float toFloat() const {
    if (this->val_ == 0)
      return 0;
    return (this->val_ < 0 ? partation_[this->val_ + offset_] : partation_[this->val_ + offset_ - 1]);
  }
  
  int fromFloat(float fl) const {
    return  std::upper_bound(partation_.begin(), partation_.end(), fl) - partation_.begin() - offset_;
  }

 static Box min() {
    Box val;
    val.val_ = -4;
    return val;
  };

  inline int get() const {return val_;}

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
  std::vector<float> partation_;
  int val_;
  int offset_;
};

namespace std {
  template<> class numeric_limits<Box> {
  public:
    static Box lowest() {return Box::min();};
  };
}
