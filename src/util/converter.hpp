#pragma once

#include "half.hpp"

class Converter {
public:
  template<typename T>
  static float ToFloat(const T& other) {
    return other.toFloat();
  }
};

template<>
float Converter::ToFloat(const float& other);

using half_float::half;
template<>
float Converter::ToFloat(const half& other);

template<>
float Converter::ToFloat(const double& other);
