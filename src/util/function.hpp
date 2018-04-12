#pragma once

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <typeinfo>
#include <limits>

#include "util/tensor.hpp"
#include "util/converter.hpp"
#include "util/float_macro.hpp"

class Function {
public:
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void ReLU(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void deriv_ReLU(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void sigmoid(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void deriv_sigmoid(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void softmax(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
  static void deriv_softmax(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int dim1_p, int dim2_p>
  static void matmul(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                     const Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>& m,
                     Tensor<dim1_p, dim2, dim3, dim4, dim5, T>* ans);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int w_row, int w_col, int out, int a_row, int a_col>
  static void conv2d(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                     const Tensor<w_row, w_col, dim3, out, dim4, T>& w,
                     Tensor<a_row, a_col, out, dim4, dim5, T>* ans,
                     int p, int s);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int w_row, int w_col, int out, int a_row, int a_col>
  static void deconv2d(const Tensor<a_row, a_col, out, dim4, dim5, T>& conv,
                       const Tensor<w_row, w_col, dim3, out, dim5, T>& w,
                       Tensor<dim1, dim2, dim3, dim4, dim5, T>* ans,
                       int p, int s);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int a_row, int a_col>
  static void max_pool(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                       int k_row, int k_col,
                        Tensor<a_row, a_col, dim3, dim4, dim5, T>* ans,
                       Tensor<a_row*a_col*dim3*dim4*dim5, 1, 1, 1, 1, int>* idx,
                       int p, int s);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int dim1_p, int dim2_p>
  static void depool(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& pool,
                     const Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, int>& idx,
                     Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>* ans);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T, int dim1_p>
  static void add_bias(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t,
                       const Tensor<dim1_p, 1, 1, 1, 1, T>& b);
  template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
            int dim1_p, int dim2_p>
  static void padding(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& before,
                      Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>* ans, int p);
};

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::ReLU(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t) {
  for (int i = 0; i < t->size(); ++i) {
    if ((*t)[i] <= 0.0)
      (*t)[i] = 0.0;
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::deriv_ReLU(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t) {
  for (int i = 0; i < t->size(); ++i)
    if ((*t)[i] <= 0.0)
      (*t)[i] = 0.0;
    else
      (*t)[i] = 1.0;
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::sigmoid(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t) {
  for (int i = 0; i < t->size(); ++i)
    (*t)[i] = DIV(1.0, ADD(1.0, exp(-1.0 * Converter::ToFloat((*t)[i]))));
}

float uni_sigmoid(float v) {
  return DIV(1.0, ADD(1.0, exp(-1.0 * v)));
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::deriv_sigmoid(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t) {
  for (int i = 0; i < t->size(); ++i)
    (*t)[i] = MUL(uni_sigmoid((*t)[i]), SUB(1.0, uni_sigmoid((*t)[i])));
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::softmax(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t) {
  int col = t->size(1);
  int row = t->size(0);
  T* v = t->get_v();
  for (int l = 0; l < t->size() / (col * row); ++l) {
    for (int k = 0; k < col; ++k) {
      float sum = 0;
      for (int i = 0; i < row; ++i) {
        sum = ADD(sum, exp(v[l * (row * col) + k * row + i]));
      }
      for (int j = 0; j < row; ++j) {
        v[l * (row * col) + k * row + j] = DIV(exp(v[l * (row * col) + k * row + j]), sum);
      }
    }
  }
}

template<int dim1, int dim2, int dim3, int dim4, int dim5, typename T>
void Function::deriv_softmax(Tensor<dim1, dim2, dim3, dim4, dim5, T> *t) {
  int col = t->size(1);
  int row = t->size(0);
  T* v = t->get_v();
  for (int l = 0; l < t->size() / (col * row); ++l) {
    for (int k = 0; k < col; ++k) {
      float sum = 0;
      for (int i = 0; i < row; ++i) {
        sum = ADD(sum, exp(v[l * (row * col) + k * row + i]));
      }
      for (int j = 0; j < row; ++j) {
        int idx = l*(row*col) + k*row + j;
        v[idx] = exp(v[idx]) / sum;
        v[idx] = v[idx] * (1.0 - Converter::ToFloat(v[idx]));
      }
    }
  }
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int dim1_p, int dim2_p>
void Function::matmul(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                      const Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>& m,
                      Tensor<dim1_p, dim2, dim3, dim4, dim5, T>* ans) {
  int t_col = t.size(1);
  int t_row = t.size(0);
  int m_col = m.size(1);
  int m_row = m.size(0);
  if (m_col != t_row) {
    std::cout << "Dimensional Error!" << std::endl;
    abort();
  }
  ans->init();
  for (int l = 0; l < t.size() / (t_col * t_row); ++l)
    for (int i = 0; i < t_col; ++i)
      for (int k = 0; k < t_row; ++k)
        for (int j = 0; j < m_row; ++j)
            (*ans)[l * (t_col * m_row) + i * m_row + j]
              = ADD((*ans)[l * (t_col * m_row) + i * m_row + j], MUL(t[l * (t_col * t_row) + i * t_row + k],  m[l * (m_col * m_row) + k * m_row + j]));
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int w_row, int w_col, int out, int a_row, int a_col>
void Function::conv2d(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                      const Tensor<w_row, w_col, dim3, out, dim4, T>& w,
                      Tensor<a_row, a_col, out, dim4, dim5, T> *ans,
                      int p, int s) {
  ans->init();
  const int* ans_dim = ans->shape();
  const int* w_dim = w.shape();
  const int* dim = t.shape();
  for (int k = 0; k < ans_dim[2]; ++k)
    for (int i = 0; i < ans_dim[1]; ++i)
      for (int j = 0; j < ans_dim[0]; ++j)

        for (int ch = 0; ch < w_dim[2]; ++ch)
          for (int c = 0; c < w_dim[1]; ++c)
            for (int r = 0; r < w_dim[0]; ++r)
              (*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j] =
                ADD((*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j],
                    MUL(t[ch*(dim[1]*dim[0]) + (i*s+c)*dim[0] + (j*s+r)],
                        w[k*(w_dim[2]*w_dim[1]*w_dim[0]) + ch*(w_dim[1]*w_dim[0])
                          + c*w_dim[0] + r]));
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int w_row, int w_col, int out, int a_row, int a_col>
void Function::deconv2d(const Tensor<a_row, a_col, out, dim4, dim5, T>& conv,
                        const Tensor<w_row, w_col, dim3, out, dim5, T>& w,
                        Tensor<dim1, dim2, dim3, dim4, dim5, T>* ans,
                        int p, int s) {
  constexpr int pad_size = w_row - 1;
  Tensor<dim1+2*pad_size, dim2+2*pad_size, out, dim4, dim5, T> pad_conv;
  Function::padding(conv, &pad_conv, pad_size);

  ans->init();
  const int* ans_dim = ans->shape();
  const int* w_dim = w.shape();
  const int* dim = pad_conv.shape();
  for (int k = 0; k < ans_dim[2]; ++k)
    for (int i = 0; i < ans_dim[1]; ++i)
      for (int j = 0; j < ans_dim[0]; ++j)

        for (int ch = 0; ch < w_dim[3]; ++ch)
          for (int c = 0; c < w_dim[1]; ++c)
            for (int r = 0; r < w_dim[0]; ++r)
              (*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j] =
                ADD((*ans)[k*(ans_dim[0]*ans_dim[1]) + i*ans_dim[0] + j],
                    MUL(pad_conv[ch*(dim[1]*dim[0]) + (i*s+c)*dim[0] + (j*s+r)],
                        w[ch*(w_dim[2]*w_dim[1]*w_dim[0]) + k*(w_dim[1]*w_dim[0])
                          + (w_dim[1]-1-c)*w_dim[0] + (w_dim[0]-1-r)]));
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int a_row, int a_col>
void Function::max_pool(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& t,
                        int k_row, int k_col,
                        Tensor<a_row, a_col, dim3, dim4, dim5, T>* ans,
                        Tensor<a_row*a_col*dim3*dim4*dim5, 1, 1, 1, 1, int>* idx,
                        int p, int s) {
  const int* ans_dim = ans->shape();
  const int* dim = t.shape();
  for (int k = 0; k < ans_dim[2]; ++k) {
    for (int i = 0; i < ans_dim[1]; ++i) {
      for (int j = 0; j < ans_dim[0]; ++j){

        T max = std::numeric_limits<T>::lowest();
        for (int c = 0; c < k_col; ++c)
          for (int r = 0; r < k_row; ++r)
            if (max < t[k*(dim[1]*dim[0]) + (i*s+c)*dim[0] + (j*s+r)]) {
              max = (*ans)[k*(ans_dim[1]*ans_dim[0]) + i*ans_dim[0] + j] =
                t[k*(dim[1]*dim[0]) + (i*s+c)*dim[0] + (j*s+r)];
              (*idx)[k*(ans_dim[1]*ans_dim[0]) + i*ans_dim[0] + j] =
                k*(dim[1]*dim[0]) + (i*s+c)*dim[0] + (j*s+r);
            }

      }
    }
  }
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int dim1_p, int dim2_p>
void Function::depool(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& pool,
                      const Tensor<dim1*dim2*dim3*dim4*dim5, 1, 1, 1, 1, int>& idx,
                      Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>* ans) {
  ans->init();
  for (int i = 0; i < pool.size(); ++i) {
    (*ans)[idx[i]] = pool[i];
  }
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T, int dim1_p>
void Function::add_bias(Tensor<dim1, dim2, dim3, dim4, dim5, T>* t,
                        const Tensor<dim1_p, 1, 1, 1, 1, T>& b) {
  int len = t->size() / dim1_p;
  for (int j = 0; j < dim1_p; ++j)
    for (int i = 0; i < len; ++i)
      (*t)[j * len + i] = ADD((*t)[j * len + i], b[j]);
}

template <int dim1, int dim2, int dim3, int dim4, int dim5, typename T,
          int dim1_p, int dim2_p>
void Function::padding(const Tensor<dim1, dim2, dim3, dim4, dim5, T>& before,
                       Tensor<dim1_p, dim2_p, dim3, dim4, dim5, T>* ans, int p) {
  ans->init();
  int col = before.size(1);
  int row = before.size(0);
  for (int k = 0; k < before.size() / (col*row); ++k) {
    for (int i = 0; i < col; ++i) {
      for (int j = 0; j < row; ++j) {
        (*ans)[k*ans->size(1)*ans->size(0)
               + (i+p)*ans->size(0) + (j+p)]
          = before[k*col*row + i*row + j];
      }
    }
  }
}
