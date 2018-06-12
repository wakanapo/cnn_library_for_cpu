#pragma once

#include <algorithm>
#include <vector>

#include "ga/set_gene.hpp"

class Box {
public:
  Box() : partation_(GlobalParams::getInstance()->partition()) {};
  Box(const Box& other) : partation_(GlobalParams::getInstance()->partition()),
                          val_(other.get()) {
    zero_ = std::lower_bound(partation_.begin(), partation_.end()-1, 0) -
      partation_.begin();
  };
  Box(const Box&& other) : partation_(GlobalParams::getInstance()->partition()),
                           val_(std::move(other.get())) {
    zero_ = std::lower_bound(partation_.begin(), partation_.end()-1, 0) -
      partation_.begin();
  };
  Box(int other) : partation_(GlobalParams::getInstance()->partition()),
                   val_(other) {
    zero_ = std::lower_bound(partation_.begin(), partation_.end()-1, 0) -
      partation_.begin();
  };
  Box(float other) : partation_(GlobalParams::getInstance()->partition()),
                     val_(fromFloat(other)) {
    zero_ = std::lower_bound(partation_.begin(), partation_.end()-1, 0) -
      partation_.begin();
  };
  Box(double other) : partation_(GlobalParams::getInstance()->partition()),
                      val_(fromFloat(other)) {
    zero_ = std::lower_bound(partation_.begin(), partation_.end()-1, 0) -
      partation_.begin();
  };
  float toFloat() const {
    if (this->val_ == 0)
      return partation_[0];
    if (this->val_ == partation_.size())
      return partation_[partation_.size()-1];
    return  (partation_[this->val_] + partation_[this->val_-1]) / 2.0;
  }
  
  int fromFloat(float fl) const {
    return  std::upper_bound(partation_.begin(), partation_.end(), fl) - partation_.begin();
  }

 static Box min() {
    Box val;
    val.val_ = 0;
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
  int zero_;
};

namespace std {
  template<> class numeric_limits<Box> {
  public:
    static Box lowest() {return Box::min();};
  };
}
