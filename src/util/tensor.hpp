#pragma once

#include <iostream>
#include <typeinfo>
#include <random>
#include <memory>

#include "util/float_macro.hpp"

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
class Tensor {
private:
  static constexpr int size_ = dim1*dim2*dim3*dim4*dim5;
  int shape_[5] = {dim1, dim2, dim3, dim4, dim5};
  std::unique_ptr<T[]> v_ = std::make_unique<T[]>(size_);;
public:
  int size(int axis) const;
  const int size() const;
  int bytes() const;
  const int* shape() const;
  template <typename T_prime>
    void set_v(const std::unique_ptr<T_prime>& v_in);
  template <typename T_prime>
    void set_v(T_prime* v_in);
  void init();
  void randomInit(float low, float high);
  T* get_v();
  Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T> flatten() const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T> make_clone() const;
  const T& operator[](const int i) const;
  T &operator[](const int i);
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator+(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator-(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  operator*(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& other) const;
  Tensor<dim1, dim2, dim3, dim4, dim5, T>
  times(const T& other) const;
  Tensor<dim2, dim1, dim3, dim4, dim5, T>
  transpose() const;
};

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::size(int axis) const {
  return shape_[axis];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const int Tensor<dim1, dim2, dim3, dim4, dim5, T>::size() const {
  return size_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
int Tensor<dim1, dim2, dim3, dim4, dim5, T>::bytes() const {
  return size_ * sizeof(T);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const int* Tensor<dim1, dim2, dim3, dim4, dim5, T>::shape() const {
  return shape_;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
template <typename T_prime>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::set_v(const std::unique_ptr<T_prime>& v_in) {
  for (int i = 0; i < size_; ++i) {
    v_[i] = v_in[i];
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
template <typename T_prime>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::set_v(T_prime* v_in) {
  for (int i = 0; i < size_; ++i) {
      v_[i] = v_in[i];
  }
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
  for (int i = 0; i < size_; ++i) {
    v_[i] = std::abs(high)*dist(engine);
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Tensor<dim1, dim2, dim3, dim4, dim5, T>::init() {
  for (int i = 0; i < size_; ++i)
    v_[i] = 0;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
T* Tensor<dim1, dim2, dim3, dim4, dim5, T>::get_v() {
  return v_.get();
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
const T &Tensor<dim1, dim2, dim3, dim4, dim5, T>::operator[](const int i) const {
  if (i < 0 || i >= size_) {
    std::cerr << i << " is out of range(" << size_ << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v_[i];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
T &Tensor<dim1, dim2, dim3, dim4, dim5, T>::operator[](const int i) {
  if (i < 0 || i >= size_) {
    std::cerr << i << " is out of range(" << size_ << ")." << std::endl;
    std::cerr << "Error!: Invalid index." << std::endl;
    abort(); // abort() is not supported by Vivado HLS.
  }
  return v_[i];
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator+(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = ADD(v_[i], y[i]);
  }
  return ans;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator-(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = SUB(v_[i], y[i]);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::operator*(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = MUL(v_[i], y[i]);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>
::times(const T& y) const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  for (int i = 0; i < size_; ++i) {
    ans[i] = MUL(v_[i], y);
  }
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, T>
Tensor<dim1, dim2, dim3, dim4, dim5, T>::flatten() const {
  Tensor<size_, 1, 1, 1, 1, T> ans;
  ans.set_v(v_);
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim1, dim2, dim3, dim4, dim5, T>
Tensor<dim1, dim2, dim3, dim4, dim5, T>::make_clone() const {
  Tensor<dim1, dim2, dim3, dim4, dim5, T> ans;
  ans.set_v(v_);
  return std::move(ans);
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
Tensor<dim2, dim1, dim3, dim4, dim5, T> Tensor<dim1, dim2, dim3, dim4, dim5, T>::transpose() const {
  // TODO(wakanapo): Be able to deal with the multi-dimensional array.
  Tensor<dim2, dim1, dim3, dim4, dim5, T> m;
  for (int i = 0; i < dim1; ++i) {
    for (int j = 0 ; j < dim2; ++j) {
      m[i * dim2 + j] = v_[j * dim1 + i];
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

