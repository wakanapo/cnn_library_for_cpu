#pragma once

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "cnn/for_test/cnn_weight.hpp"
#include "util/read_data.hpp"

enum activation {
  RELU,
  SOFTMAX,
  SIGMOID,
  NONE
};

class CNN {
private:
  Tensor4D<5, 5, 1, 30, float> w1;
  Tensor1D<30, float> b1;
  Tensor2D<12*12*30, 100, float> w2;
  Tensor1D<100, float> b2;
  Tensor2D<100, 10, float> w3;
  Tensor1D<10, float> b3;
public:
  void makeWeight();
  void randomWeight();
  void conv_layer(Tensor2D<28, 28, float>& x, Tensor4D<5, 5, 1, 30, float>& w,
                  Tensor1D<20, float>& b, Tensor3D<24, 24, 30, float>* ans);
  void pool_layer(Tensor3D<24, 24, 30, float>& x,
                  Tensor3D<12, 12, 30, float>* ans, Tensor1D<12*12*30, int>* idx);
  template <int N, int M>
  void affine_layer(Tensor1D<N, float>& x, Tensor2D<M, N, float>& w,
                Tensor1D<M, float>& b, Tensor1D<M, float>* ans);
  template <int N, int M, int L>
  void activate_layer(Tensor3D<N, M, L, float>* x, activation act);
  void deconv_layer(Tensor3D<24, 24, 30, float>& delta,
                    Tensor2D<28, 28, float>& x,
                    Tensor4D<5, 5, 1, 30, float>* w,
                    Tensor1D<30, float>* b,
                    Tensor3D<32, 32, 30, float>* pad_conv,
                    Tensor2D<28, 28, float>* ans,
                    const float& eps);
  void back_conv(Tensor3D<24, 24, 30, float>& delta,
                 Tensor4D<5, 5, 1, 30, float>& w,
                 Tensor3D<32, 32, 30, float>* pad_conv,
                 Tensor2D<28, 28, float>* ans);
  void depool_layer(Tensor3D<12, 12, 30, float>& delta,
                    Tensor1D<12*12*30, int>& idx,
                    Tensor3D<24, 24, 30, float>* depool);
  template <int N, int M>
  void deaffine_layer(Tensor1D<N, float>& delta, Tensor1D<M, float>& x,
                      Tensor2D<N, M, float>* w, Tensor1D<N, float>* b,
                      Tensor1D<M, float>* ans, const float& eps);
  template <int N, int M>
  void back_affine(Tensor1D<N, float>& delta, Tensor2D<N, M, float>& w,
                   Tensor1D<M, float>* ans);
  template <int N, int M, int L>
  void deactivate_layer(Tensor3D<N, M, L, float>* delta,
                        Tensor3D<N, M, L, float>& x, activation act);
  void deconv_w(Tensor3D<24, 24, 30, float>& delta,
                Tensor2D<28, 28, float>& x,
                Tensor4D<5, 5, 1, 30, float>* w, const float& eps);
  void deconv_b(Tensor3D<24, 24, 30, float>& delta,
                Tensor2D<28, 28, float>& x,
                Tensor1D<30, float>* b, const float& eps);
  template <int N, int M>
  void defc_w(Tensor1D<N, float>& delta, Tensor1D<M, float>& x,
              Tensor2D<N, M, float>* w, const float& eps);
  template <int N, int M>
  void defc_b(Tensor1D<N, float>& delta, Tensor1D<M, float>& x,
              Tensor1D<N, float>* b, const float& eps);
  void train(Tensor2D<28, 28, float>& x, Tensor1D<10, float>& t, const float& eps);
  unsigned long predict(Tensor2D<28, 28, float>& x);
  static void run(status st);
};

void CNN::makeWeight() {
  w1.set_v(w1_raw);
  b1.set_v(b1_raw);
  w2.set_v(w2_raw);
  b2.set_v(b2_raw);
  w3.set_v(w3_raw);
  b3.set_v(b3_raw);
}


void CNN::randomWeight() {
  w1.randomInit(0.0, 0.01);
  b1.init();
  w2.randomInit(0.0, 0.01);
  b2.init();
  w3.randomInit(0.0, 0.01);
  b3.init();
}

void CNN::conv_layer(Tensor2D<28, 28, float> &x, Tensor4D<5, 5, 1, 30, float> &w, Tensor1D<20, float> &b, Tensor3D<24, 24, 30, float> *ans) {
  Function::conv2d(x, w, ans, 0, 1);
  Function::add_bias(ans, b);
}

void CNN::pool_layer(Tensor3D<24, 24, 30, float> &x, Tensor3D<12, 12, 30, float> *ans, Tensor1D<12 * 12 * 30, int> *idx) {
  Function::max_pool(x, 2, 2, ans, idx, 0, 2);
}

template <int N, int M>
void CNN::affine_layer(Tensor1D<N, float> &x, Tensor2D<M, N, float> &w, Tensor1D<M, float> &b, Tensor1D<M, float> *ans) {
  Function::matmul(x, w, ans);
  (*ans) = (*ans) + b;
}

template <int N, int M, int L>
void CNN::activate_layer(Tensor3D<N, M, L, float> *x, activation act) {
  if (act == RELU)
    Function::ReLU(x);
  else if (act == SOFTMAX)
    Function::softmax(x);
}

void CNN::depool_layer(Tensor3D<12, 12, 30, float> &delta, Tensor1D<12 * 12 * 30, int> &idx, Tensor3D<24, 24, 30, float> *depool) {
  Function::depool(delta, idx, depool);
}

void CNN::deconv_w(Tensor3D<24, 24, 30, float> &delta, Tensor2D<28, 28, float> &x, Tensor4D<5, 5, 1, 30, float> *w, const float &eps) {
  Tensor4D<5, 5, 1, 30, float> delta_w;
  delta_w.init();
  const int* w_dim = w->shape();
  const int* d_dim = delta.shape();
  const int* x_dim = x.shape();
  for (int i = 0; i < w_dim[3]; ++i)
    for (int j = 0; j < w_dim[2]; ++j)
      for (int k = 0; k < w_dim[1]; ++k)
        for (int l = 0; l < w_dim[0]; ++l)

          for (int c = 0; c < d_dim[1]; ++c)
            for (int r = 0; r < d_dim[0]; ++r)
              delta_w[i*w_dim[0]*w_dim[1]*w_dim[2] +
                      j*w_dim[0]*w_dim[1] + k*w_dim[0] + l]
                += delta[i*d_dim[0]*d_dim[1] + c*d_dim[0] + r] *
                x[j*(x_dim[1]*x_dim[0]) + (k+c)*x_dim[0] + (l+r)];
  
  delta_w = delta_w.times(eps);
  (*w) = (*w) - delta_w;
}

void CNN::deconv_b(Tensor3D<24, 24, 30, float> &delta, Tensor2D<28, 28, float> &x, Tensor1D<30, float> *b, const float &eps) {
  Tensor1D<30, float> delta_b;
  delta_b.init();
  for (int i = 0; i < delta.size(2); ++i)
    for (int j = 0; j < delta.size(1); ++j)
      for (int h = 0; h < delta.size(0); ++h)
        delta_b[i] = delta_b[i] + delta[i*delta.size(0)*delta.size(1) + j*delta.size(0) + h];

  delta_b = delta_b.times(eps);
  (*b) = (*b) - delta_b;
}

void CNN::back_conv(Tensor3D<24, 24, 30, float> &delta, Tensor4D<5, 5, 1, 30, float> &w, Tensor3D<32, 32, 30, float> *pad_conv, Tensor2D<28, 28, float> *ans) {
  Function::deconv2d(delta, w, ans, 0, 1);
}

void CNN::deconv_layer(Tensor3D<24, 24, 30, float> &delta, Tensor2D<28, 28, float> &x, Tensor4D<5, 5, 1, 30, float> *w, Tensor1D<30, float> *b, Tensor3D<32, 32, 30, float> *pad_conv, Tensor2D<28, 28, float> *ans, const float &eps) {
  back_conv(delta, *w, pad_conv, ans);
  deconv_w(delta, x, w, eps);
  deconv_b(delta, x, b, eps);
}

template <int N, int M>
void CNN::defc_w(Tensor1D<N, float> &delta, Tensor1D<M, float> &x, Tensor2D<N, M, float> *w, const float &eps) {
  Tensor2D<N, M, float> dw;
  Tensor2D<1, M, float> x_t = x.transpose();
  Function::matmul(x_t, delta, &dw);
  dw = dw.times(eps);
  (*w) = (*w) - dw;
}

template <int N, int M>
void CNN::defc_b(Tensor1D<N, float> &delta, Tensor1D<M, float> &x, Tensor1D<N, float> *b, const float &eps) {
  Tensor1D<1, float> x_ones;
  for (int i = 0; i < 1; ++i)
    x_ones[i] = 1;
  Tensor1D<N, float> db;
  Function::matmul(x_ones, delta, &db);
  
  db = db.times(eps);
  (*b) = (*b) - db;
}

template <int N, int M>
void CNN::back_affine(Tensor1D<N, float> &delta, Tensor2D<N, M, float> &w, Tensor1D<M, float> *ans) {
  Tensor2D<M, N, float> w_t = w.transpose();
  Function::matmul(delta, w_t, ans);
}

template <int N, int M>
void CNN::deaffine_layer(Tensor1D<N, float> &delta, Tensor1D<M, float> &x, Tensor2D<N, M, float> *w, Tensor1D<N, float> *b, Tensor1D<M, float> *ans, const float &eps) {
  back_affine(delta, *w, ans);
  defc_w(delta, x, w, eps);
  defc_b(delta, x, b, eps);

}

template <int N, int M, int L>
void CNN::deactivate_layer(Tensor3D<N, M, L, float> *delta, Tensor3D<N, M, L, float> &x, activation act) {
  Tensor3D<N, M, L, float> tmp = x;
  if (act == RELU)
    Function::deriv_ReLU(&tmp);
  else if(act == SIGMOID)
    Function::deriv_sigmoid(&tmp);
  else if (act == SOFTMAX)
    Function::deriv_softmax(&tmp);
  (*delta) = (*delta) * tmp;
}

