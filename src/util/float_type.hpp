#pragma once

#include <bitset>
#include <cmath>
#include <iostream>

long long overflow = 0;
long long underflow = 0;

float BitConverter(int e, int m, const float value) {
  if (value == 0)
    return 0;
  int exp = 0;
  float fr = std::frexp(value, &exp);
  if (exp >= std::pow(2, e-1)) {
    ++overflow;
    return INFINITY;
  }
  else if (exp <= -1 * std::pow(2, e-1)) {
    ++underflow;
    return 0;
  }
  
  union {float f; int i;} unF;
  unF.f = fr;
  unF.i = ((unF.i>>(23-m))<<(23-m));
  return std::ldexp(unF.f, exp);
}

