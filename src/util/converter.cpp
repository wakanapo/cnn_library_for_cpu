#include "util/converter.hpp"

// static
template<>
float Converter::ToFloat(const float& other) {
  return other;
}

template<>
float Converter::ToFloat(const half& other) {
  return (float)other;
}

template<>
float Converter::ToFloat(const double& other) {
  return (float)other;
}
