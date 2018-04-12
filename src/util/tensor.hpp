#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "util/float_macro.hpp"

template <typename T>
class Tensor {
public:
  Tensor(std::vector<int> shape);
  const int size() const;
  const std::vector<int> shape() const;
  void set(T* v);
  void random(float low, float high);
  std::unique_ptr<T> get() const;
  const T& operator[](const int i) const;
  T& operator[](const int i);
  Tensor<T> operator+(const Tensor<T>& y) const;
  Tensor<T> operator-(const Tensor<T>& y) const;
  Tensor<T> operator*(const Tensor<T>& y) const;
  Tensor<T> flatten();
  Tensor<T> times(const T& y);
  Tensor<T> transpose() const;
private:
  std::unique_ptr<T> v_;
  std::vector<int> shape_;
  uint size_;
};

template <typename T>
Tensor<T>::Tensor(std::vector<int> shape) {
  shape_ = shape;
  size_ = 1;
  for (auto s : shape) {
    size_ *= s;
  }
  v_ = std::move((T*)malloc(size_ * sizeof(T)));
}

template <typename T>
const int Tensor<T>::size() const {
  return size_;
}

template <typename T>
const std::vector<int> Tensor<T>::shape() const {
  return shape_;
}

template <typename T>
void Tensor<T>::set(T* v) {
  v_ = std::move(v);
}

template <typename T>
void Tensor<T>::random(float low, float high) {
  std::mt19937 engine(123);
  std::normal_distribution<> dist(0, 1);
  T* v = (T*)malloc(size_ * sizeof(T));
  for (int i = 0; i < size_; ++i) {
    v[i] = std::abs(high) * dist(engine);
  }
  v_ = std::move(v);
}

template <typename T>
std::unique_ptr<T> Tensor<T>::get() const {
  return std::move(v_);
}

template <typename T>
const T& Tensor<T>::operator[](const int i) const {
  if (i < 0 || i >= size_) {
    std::cerr << i << " is out of range(" << size_ << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v_[i];
}

template <typename T>
T& Tensor<T>::operator[](const int i) {
  if (i < 0 || i >= size_) {
    std::cerr << i << " is out of range(" << size_ << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v_[i];
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& y) const {
  Tensor<T> ans(shape_);
  for (int i = 0; i < size_; ++i) {
    ans[i] = ADD(v_[i], y[i]);
  }
  return ans;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& y) const {
  Tensor<T> ans(shape_);
  for (int i = 0; i < size_; ++i) {
    ans[i] = SUB(v_[i], y[i]);
  }
  return ans;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& y) const {
  Tensor<T> ans(shape_);
  for (int i = 0; i < size_; ++i) {
    ans[i] = MUL(v_[i], y[i]);
  }
  return ans;
}

template <typename T>
Tensor<T> Tensor<T>::flatten() {
  Tensor<T> ans({size_});
  ans.set(v_);
  return ans;
}

template <typename T>
Tensor<T> Tensor<T>::times(const T& y) {
  Tensor<T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = MUL(v_[i], y);
  }
  return ans;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
  // TODO(wakanapo): Be able to deal with the multi-dimensional array.
  Tensor<T> ans;
  for (int i = 0; i < shape_[0]; ++i) {
    for (int j = 0 ; j < shape_[1]; ++j) {
      ans[i * shape_[1] + j] = v_[j * shape_[0] + i];
    }
  }
  return ans;
}
