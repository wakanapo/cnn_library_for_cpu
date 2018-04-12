#pragma once

#include <iostream>
#include <limits>

typedef union UnFix16_{
  signed int iNum;
  struct StFix16{
    unsigned int fraction: 16;
    unsigned int decimal : 15;
    unsigned int sign : 1;
  } stFix16;
} UnFix16;

typedef union UnFloat_{
  signed int iNum;
  struct StFloat{
    unsigned int fraction: 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } stFloat;
} UnFloat;

class CFix16 {
public:
  CFix16() = default;
  CFix16(const CFix16& other) = default;
  CFix16(const CFix16&& other) : val_(other.get()) {}
  CFix16(UnFix16 other) : val_(other) {}
  CFix16(float other) : val_(fromFloat(other)) {}

  UnFix16 fromFloat(float fL) const;
  float toFloat() const;
  static UnFix16 min();
  inline const UnFix16 get() const { return val_; }

  /* Assignment */
  CFix16 &operator=(float& fn) {
    val_ = fromFloat(fn);
    return *this;
  }

  CFix16 &operator=(const UnFix16& uF) {
    val_ = uF;
    return *this;
  }

  CFix16 &operator=(const CFix16& cn) {
    val_ = cn.get();
    return *this;
  }

   /* 比較 */
  bool operator==(const float& fn) const {
    return val_.iNum == fromFloat(fn).iNum;
  }

  bool operator==(const CFix16& cF) const {
    return val_.iNum == cF.get().iNum;
  }

  bool operator>(const CFix16& cF) const {
    return val_.iNum > cF.get().iNum;
  }
  
  bool operator<(const CFix16& cF) const {
    return val_.iNum > cF.get().iNum;
  }

   bool operator>=(const CFix16& cF) const {
     return val_.iNum >= cF.get().iNum;
  }

  bool operator<=(const CFix16& cF) const {
    return val_.iNum >= cF.get().iNum;
  }
  /* 演算 */
  CFix16 operator+(const CFix16& cF) const {
    UnFix16 uF;
    uF.iNum = val_.iNum + cF.get().iNum;
     return uF;
  }

  CFix16 operator+=(const CFix16& cF) {
    val_.iNum = val_.iNum + cF.get().iNum;
    return val_;  
  }

  CFix16 operator-(const CFix16& cF) const {
    UnFix16 uF;
    uF.iNum = val_.iNum - cF.get().iNum;
    return uF;
  }
  
  CFix16 operator*(const CFix16& cF) const {
    UnFix16 uF;
    uF.iNum = ((int64_t)val_.iNum * (int64_t)cF.get().iNum) >> 16 ;
    return uF;
  }

  CFix16 operator*(const float& f) const {
    CFix16 cF = f;
    return *this * cF;
  }

  CFix16 operator/(const CFix16& cF) const {
    UnFix16 uF;
    uF.iNum = (((int64_t)val_.iNum << 32) / cF.get().iNum) >> 16 ;
    return uF;
  }

private:
  UnFix16 val_;
};

UnFix16 CFix16::fromFloat(float fL) const {
  UnFix16 uF;
  UnFloat *uFl = (UnFloat*)&fL;

  uF.iNum = 0;

  if( uFl->stFloat.sign ){
    uFl->stFloat.sign = 0;
    uF.stFix16.decimal = (unsigned int)fL;
    fL = ( fL - (unsigned int)fL  + 1.0);
    uF.stFix16.fraction |= uFl->iNum >> 7;
    uF.iNum = ( ~uF.iNum + 1);
  } else {
    uF.stFix16.decimal = (unsigned int)fL;
    fL = ( fL - (unsigned int)fL  + 1.0);
    uF.stFix16.fraction |= uFl->iNum >> 7;
  }

  return uF;
}

float CFix16::toFloat() const {
  UnFix16 uF = val_;
  float fL;
  UnFloat *uFl = (UnFloat*)&fL;

  uFl->iNum = 0;
  uFl->stFloat.exponent = 127;

  if( uF.stFix16.sign ){
    uF.iNum = ~( uF.iNum - 1 );
    uFl->stFloat.fraction |= uF.stFix16.fraction << 7;
    fL = fL + (float)uF.stFix16.decimal - 1.0;
    uFl->stFloat.sign = 1;
  }else{
    uFl->stFloat.fraction |= uF.stFix16.fraction << 7;
    fL = fL + (float)uF.stFix16.decimal - 1.0;
  }
  return fL;
}

UnFix16 CFix16::min() {
  UnFix16 ans;
  ans.iNum = -INT32_MAX;
  return ans;
}

namespace std {
  template<> class numeric_limits<CFix16> {
  public:
    static CFix16 lowest() {return CFix16::min();};
  };
}
