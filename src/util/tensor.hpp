#pragma once

#include <array>
#include <cstring>
#include <iostream>
#include <typeinfo>
#include <random>
#include <memory>

#include "util/float_macro.hpp"

using Shape = std::array<int, 5>;

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
class Tensor {
private:
  static constexpr int kSize = dim1*dim2*dim3*dim4*dim5;
  std::array<T, kSize> v_;
public:
  using Type = T;
  int size() const;
  int bytes() const;
  Shape shape() const;
  void init();
  void randomInit(float low, float high);
  template <typename T_prime>
  void set_v(T_prime* v_in);
  void set_v(std::array<T, kSize> v_in);
  std::array<T, kSize> get_v();
  typename std::array<T, kSize>::iterator begin();
  typename std::array<T, kSize>::const_iterator begin() const;
  typename std::array<T, kSize>::iterator end();
  typename std::array<T, kSize>::const_iterator end() const;
  Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T> flatten() const;
  const T& operator[](const int i) const;
  T &operator[](const int i);
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator+(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator-(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator*(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  bool operator==(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  times(const T& other) const;
  Tensor<dim2, dim1, dim3, dim4, dim5, T>
  transpose() const;
};

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::size() const {
  return kSize;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::bytes() const {
  return kSize * sizeof(T);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Shape Tensor<dim1, dim2, dim3, dim4, dim5, T>::shape() const {
  const std::array<int, 5> ans = {{dim1, dim2, dim3, dim4, dim5}};
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
template <typename T_prime>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::set_v(T_prime* v_in) {
  std::memmove(&(v_[0]), v_in, kSize*sizeof(T));
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::set_v(std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize> v_in) {
  v_ = v_in;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::randomInit(float low, float high) {
  // std::random is supported by C++11.
  // I'm not sure whether it's supported Vivado HLS or not.
  // std::random_device seed_gen;
  std::mt19937 engine(123);
  // std::uniform_real_distribution<> dist(low, high);
  // for (int i = 0; i < size_; ++i) {
  //   v_[i] = dist(engine);
  // }
  std::normal_distribution<> dist(0, 1);
  for (int i = 0; i < kSize; ++i) {
    v_[i] = std::abs(high)*dist(engine);
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::init() {
  v_.fill((T)0);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize> Tensor<dim1, dim2, dim3, dim4, dim5, T>::get_v() {
  return v_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
typename std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize>::iterator
Tensor<dim1, dim2, dim3, dim4, dim5, T>::begin() {
  return v_.begin();
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
typename std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize>::const_iterator
Tensor<dim1, dim2, dim3, dim4, dim5, T>::begin() const {
  return v_.begin();
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
typename std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize>::const_iterator
Tensor<dim1, dim2, dim3, dim4, dim5, T>::end() const {
  return v_.end();
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
typename std::array<T, Tensor<dim1, dim2, dim3, dim4, dim5, T>::kSize>::iterator
Tensor<dim1, dim2, dim3, dim4, dim5, T>::end() {
  return v_.end();
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const T &Tensor<dim1, dim2, dim3, dim4, dim5, T>::operator[](const int i) const {
  if (i < 0 || i >= kSize) {
    std::cerr << "Error!: Invalid index." << std::endl;
    std::cerr << i << " is out of range" << kSize << "(" << typeid(*this).name() << ")" << std::endl;
    exit(1);
  }
  return v_[i];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
T &Tensor<dim1, dim2, dim3, dim4, dim5, T>::operator[](const int i) {
  if (i < 0 || i >= kSize) {
    std::cerr << "Error!: Invalid index." << std::endl;
    std::cerr << i << " is out of range" << kSize << "(" << typeid(*this).name() << ")" << std::endl;
    exit(1);
  }
  return v_[i];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator+(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < kSize; ++i) {
    ans[i] = ADD(v_[i], y[i]);
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator-(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < kSize; ++i) {
    ans[i] = SUB(v_[i], y[i]);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator*(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < kSize; ++i) {
    ans[i] = MUL(v_[i], y[i]);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
bool Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator==(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const {
  for (int i = 0; i < kSize; ++i) {
    if (this->v_[i] != other.v_[i])
      return false;
  }
  return true;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::times(const T& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < kSize; ++i) {
    ans[i] = MUL(v_[i], y);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T>
Tensor<dim1, dim2, dim3, dim4, dim5, T>::flatten() const {
  Tensor<kSize, 1, 1, 1, 1, T> ans;
  ans.set_v(v_);
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim2, dim1, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>::transpose() const {
  // TODO(wakanapo): Be able to deal with the multi-dimensional array.
  Tensor<dim2, dim1, dim3, dim4, dim5, T> m;
  for (int k = 0; k < dim3 * dim4 * dim5; ++k) {
    for (int i = 0; i < dim1; ++i) {
      for (int j = 0 ; j < dim2; ++j) {
        m[k * dim1 * dim2 + i * dim2 + j] = v_[k * dim1 * dim2 + j * dim1 + i];
      }
    }
  }
  return std::move(m);
}

template<int dim1, typename T>
using Tensor1D = Tensor<dim1, 1, 1, 1, 1, T>;

template<int dim1, int dim2, typename T>
using Tensor2D = Tensor<dim1, dim2, 1, 1, 1, T>;

template<int dim1, int dim2, int dim3, typename T>
using Tensor3D = Tensor<dim1, dim2 ,dim3, 1, 1, T>;

template<int dim1, int dim2, int dim3, int dim4, typename T>
using Tensor4D = Tensor<dim1, dim2, dim3, dim4, 1, T>;

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
using Tensor5D = Tensor<dim1, dim2, dim3, dim4, dim5, T>;

