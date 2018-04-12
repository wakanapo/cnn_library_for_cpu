#pragma once

#include "util/converter.hpp"
#include "util/function.hpp"
#include "util/tensor.hpp"
#include "protos/cnn_params.pb.h"

template<typename T>
class Layer {
public:
  virtual Tensor<T> forward();
  virtual Tensor<T> backword() const;
  virtual void update();
  virtual void loadParams();
  virtual void saveParams() const;
};

template<int S, int P, typename T>
class Convolution : public Layer<T> {
public:
  Convolution(const float low, const float high);
  void loadParams(CnnProto::Params* p, int idx);
  void saveParams(CnnProto::Params* p) const;
  Tensor<T> forward(const Tensor<T>&& x);
  Tensor<T> backward(const Tensor<T>&& delta) const;
  void update(const Tensor<T>& delta, const T& eps);

private:
  int stride_ = S;
  int padding_ = P;
  Tensor<T> w_;
  Tensor<T> b_;
  Tensor<T> x_;
  void update_w(const Tensor<T>& delta, const Tensor<T>& x, const T& eps);
  void update_b(const Tensor<T>& delta, const Tensor<T>& x, const T& eps);
};

template<int P, int S, typename T>
Convolution<P, S, T>::Convolution(const float low, const float high) {
  w_.randomInit(low, high);
  b_.init();
}

template<int P, int S, typename T>
void Convolution<P, S, T>::loadParams(CnnProto::Params* p, int idx) {
  for (int i = 0; i < p->weights(idx).w_size(); ++i)
    w_[i] = p->weights(idx).w(i);
  for (int i = 0; i < p->biases(idx).b_size(); ++i)
    b_[i] = p->biases(idx).b(i);
}


template<int P, int S, typename T>
void Convolution<P, S, T>::saveParams(CnnProto::Params* p) const {
  CnnProto::Weight* w = p->add_weights();
  CnnProto::Bias* b = p->add_biases();
  for (int i = 0; i < w_.size(); ++i)
    w->mutable_w()->Add(Converter::ToFloat(w_[i]));
  for (int i = 0; i < b_.size(); ++i)
    b->mutable_b()->Add(Converter::ToFloat(b_[i]));
}

template<int P, int S, typename T>
Tensor<T> Convolution<P, S, T>::forward(const Tensor<T> &x) {
  Tensor<T> ans;
  Function::conv2d(x, w_, ans, P, S);
  Function::add_bias(ans, b_);
  x_ = std::move(x);
  return std::move(ans);
}

template<int P, int S, typename T>
Tensor<T> Convolution<P, S, T>::backward(const Tensor<T> &delta) const {
  Tensor<T> ans;
  Function::deconv2d(delta, w_, ans, P, S);
  return std::move(ans);
}

template<int P, int S, typename T>
void Convolution<P, S, T>::update(const Tensor<T> &delta, const T& eps) {
  update_w(delta, x_, eps);
  update_b(delta, x_, eps);
};

template<int P, int S, typename T>
void Convolution<P, S, T>::update_w(const Tensor<T>& delta, const Tensor<T>& x,
                                    const T& eps) {
  Tensor<T> delta_w;
  delta_w.init();
  const int* w_dim = w_.shape();
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
                = ADD(delta_w[i*w_dim[0]*w_dim[1]*w_dim[2] +
                              j*w_dim[0]*w_dim[1] + k*w_dim[0] + l],
                      MUL(delta[i*d_dim[0]*d_dim[1] + c*d_dim[0] + r],
                          x[j*(x_dim[1]*x_dim[0]) + (k+c)*x_dim[0] + (l+r)]));
  
  delta_w = delta_w.times(eps);
  w_ = w_ - delta_w;
}

template<int P, int S, typename T>
void Convolution<P, S, T>::update_b(const Tensor<T>& delta, const Tensor<T>& x,
                                    const T& eps) {
  Tensor<T> delta_b;
  delta_b.init();
  for (int i = 0; i < delta.size(2); ++i)
    for (int j = 0; j < delta.size(1); ++j)
      for (int h = 0; h < delta.size(0); ++h)
        delta_b[i] = ADD(delta_b[i], delta[i*delta.size(0)*delta.size(1) + j*delta.size(0) + h]);

  delta_b = delta_b.times(eps);
  b_ = b_ - delta_b;
}

template<int P, int S, typename T, int K_ROW, int K_COL>
class Pooling : public Layer<T> {
public:
  Tensor<T> forward(const Tensor<T>& x);
  Tensor<T> backward(const Tensor<T>& delta) const;
  void update() {};
  void loadParams() {};
  void saveParams() {};
private:
  int stride_ = S;
  int padding_ = P;
  Tensor<int> idx_;
  Tensor<T> x_;
};

template<int P, int S, typename T, int K_ROW, int K_COL>
Tensor<T> Pooling<P, S, T, K_ROW, K_COL>::forward(const Tensor<T>& x) {
  Tensor<T> ans;
  Function::max_pool(x, K_ROW, K_COL, ans, idx_, P, S);
  x_ = std::move(x);
  return std::move(ans);
}

template<int P, int S, typename T, int K_ROW, int K_COL>
Tensor<T> Pooling<P, S, T, K_ROW, K_COL>::backward(const Tensor<T> &delta) const {
  Tensor<T> ans;
  Function::depool(delta, idx_, ans);
  return std::move(ans);
}

template<typename T>
class Affine : public Layer<T> {
public:
  Affine(const float low, const float high);
  void loadParams(CnnProto::Params* p, int idx);
  void saveParams(CnnProto::Params* p) const;
  Tensor<T> forward(const Tensor<T>& x);
  Tensor<T> backward(const Tensor<T>& delta) const;
  void update(const Tensor<T>& delta, const T& eps);
private:
  Tensor<T> w_;
  Tensor<T> b_;
  Tensor<T> x_;
  void update_w(const Tensor<T>& delta, const Tensor<T>& x,
                const T& eps);
  void update_b(const Tensor<T>& delta, const Tensor<T>& x,
                const T& eps);
};

template<typename T>
Affine<T>::Affine(const float low, const float high) {
  w_.randomInit(low, high);
  b_.init();
}

template<typename T>
void Affine<T>::loadParams(CnnProto::Params* p, int idx) {
  for (int i = 0; i < p->weights(idx).w_size(); ++i)
    w_[i] = p->weights(idx).w(i);
  for (int i = 0; i < p->biases(idx).b_size(); ++i)
    b_[i] = p->biases(idx).b(i);
}

template<typename T>
void Affine<T>::saveParams(CnnProto::Params *p) const {
  CnnProto::Weight* w = p->add_weights();
  CnnProto::Bias* b = p->add_biases();
  for (int i = 0; i < w_.size(); ++i)
    w->mutable_w()->Add(Converter::ToFloat(w_[i]));
  for (int i = 0; i < b_.size(); ++i)
    b->mutable_b()->Add(Converter::ToFloat(b_[i]));
}

template<typename T>
Tensor<T> Affine<T>::forward(const Tensor<T> &x) {
  Tensor<T> ans;
  Function::matmul(x, w_, &ans);
  ans = ans + b_;
  x_ = std::move(x);
  return std::move(ans);
}

template<typename T>
void Affine<T>::update_w(const Tensor<T>& delta, const Tensor<T>& x,
                         const T& eps) {
  Tensor<T> dw;
  Tensor<T> x_t = x.transpose();
  Function::matmul(x_t, delta, &dw);
  dw = dw.times(eps);
  w_ = w_ - dw;
}

template<typename T>
void Affine<T>::update_b(const Tensor<T>& delta, const Tensor<T>& x,
           const T& eps) {
  Tensor<T> x_ones;
  x_ones[0] = 1;
  Tensor<T> db;
  Function::matmul(x_ones, delta, &db);
  db = db.times(eps);
  b_ = b_ - db;
}

template<typename T>
Tensor<T> Affine<T>::backward(const Tensor<T> &delta) const {
  Tensor<T> ans;
  Tensor<T> w_t = w_.transpose();
  Function::matmul(delta, w_t, &ans);
  return std::move(ans);
}

template<typename T>
void Affine<T>::update(const Tensor<T> &delta, const T& eps) {
  update_w(delta, x_, eps);
  update_b(delta, x_, eps);
}

template<typename T>
class Relu : public Layer<T>{
public:
  Tensor<T> forward(Tensor<T>&& x);
  Tensor<T> backward(Tensor<T>&& delta) const;
  void uptate() {};
  void loadParams() {};
  void saveParams() {};
};

template<typename T>
void Relu<T>::forward(Tensor<T>* x) const {
  Function::ReLU(x);
};

template<typename T>
void Relu<T>::backward(Tensor<T> *delta,
                       const Tensor<T> &x) const {
  Tensor<T> tmp = x;
  Function::deriv_ReLU(&tmp);
  (*delta) = (*delta) * tmp;
}

template<typename T>
class Sigmoid : public Layer {
public:
  void forward(Tensor<T>* x) const;
  void backward(Tensor<T>* delta, const Tensor<T>& x) const; 
  void update() {};
  void loadParams() {};
  void saveParams() {};
};

template<typename T>
void Sigmoid<T>::forward(Tensor<T>* x) const {
  Function::sigmoid(x);
};

template<typename T>
void Sigmoid<T>::backward(Tensor<T> *delta, const Tensor<T> &x) const {
  Tensor<T> tmp = x;
  Function::deriv_sigmoid(&tmp);
  (*delta) = (*delta) * tmp;
}

template<typename T>
class Softmax : public Layer {
public:
  void forward(Tensor<T>* x) const;
  void backward(Tensor<T>* delta, const Tensor<T>& x) const;
  void update() {};
  void loadParams() {};
  void saveParams() {};
};

template<typename T>
void Softmax<T>::forward(Tensor<T>* x) const {
  Function::softmax(x);
};

template<typename T>
void Softmax<T>::backward(Tensor<T> *delta, const Tensor<T> &x) const {
  Tensor<T> tmp = x;
  Function::deriv_softmax(&tmp);
  (*delta) = (*delta) * tmp;
}
