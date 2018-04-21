#pragma once

#include <fstream>
#include <iostream>

#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "util/tensor.hpp"

template <typename T>
class SimpleConvNet {
public:
  using Type = T;
  using InputType = Tensor2D<28, 28, T>;
  using OutputType = Tensor1D<10, T>;
  SimpleConvNet() : Conv1(Convolution<5, 5, 1, 30, 0, 1, T>(-0.1, 0.1)),
                    Affine1(Affine<12*12*30, 100, T>(-0.1, 0.1)),
                    Affine2(Affine<100, 10, T>(-0.1, 0.1)) {};
  void train(const InputType& x, const OutputType& t, const T& eps);
  unsigned long predict(const InputType& x) const;
  void save();
  void load();
  Dataset<InputType, OutputType> readData(Status st);
private:
  Convolution<5, 5, 1, 30, 0, 1, T> Conv1;
  Relu<T> Relu1;
  Pooling<2, 2, 0, 2, T> Pool1;
  Affine<12*12*30, 100, T> Affine1;
  Relu<T> Relu2;
  Affine<100, 10, T> Affine2;
  Sigmoid<T> Last;
};


template <typename T>
void SimpleConvNet<T>::train(const typename SimpleConvNet<T>::InputType& x,
                             const typename SimpleConvNet<T>::OutputType& t,
                             const T& eps) {
  // forward
  Tensor3D<24, 24, 30, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);

  Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  Affine1.forward(dense1, &dense2);

  Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  Affine2.forward(dense2, &ans);

  Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta3 = ans - t;
  Tensor1D<100, T> delta2;
  Affine2.backward(delta3, dense2, &delta2, eps);
  Relu2.backward(&delta2, dense2);

  Tensor1D<12*12*30, T> delta1;
  Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<12, 12, 30, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<24, 24, 30, T> delta_pool;
  Pool1.backward(delta1_3D, idx, &delta_pool);

  Relu1.backward(&delta_pool, conv1_ans);

  Tensor2D<28, 28, T> delta_conv;
  Conv1.backward(delta_pool, x, &delta_conv, eps);
}

template <typename T>
unsigned long SimpleConvNet<T>::predict(const SimpleConvNet<T>::InputType& x) const{
  Tensor3D<24, 24, 30, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);

  Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  Affine1.forward(dense1, &dense2);

  Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  Affine2.forward(dense2, &ans);

  Last.forward(&ans);

  T max = (T)0;
  unsigned long argmax = 0;
  for (int i = 0; i < 10; ++i) {
    if (ans[i] > max) {
      max = ans[i];
      argmax = i;
    }
  }
  return argmax;
}

template <typename T>
void SimpleConvNet<T>::save() {
  CnnProto::Params params;
  Conv1.saveParams(&params);
  Affine1.saveParams(&params);
  Affine2.saveParams(&params);
  std::fstream output(Options::GetWeightsOutput(), std::ios::out | std::ios::trunc | std::ios::binary);
  if (!params.SerializeToOstream(&output)) {
    std::cerr << "Failed to write params." << std::endl;
  }
}

template <typename T>
void SimpleConvNet<T>::load() {
  CnnProto::Params params;
  std::fstream input(Options::GetWeightsInput(), std::ios::in | std::ios::binary);
  if (!params.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv1.loadParams(&params, 0);
  Affine1.loadParams(&params, 1);
  Affine2.loadParams(&params, 2);
}

template<typename T>
Dataset<typename SimpleConvNet<T>::InputType, typename SimpleConvNet<T>::OutputType>
SimpleConvNet<T>::readData(Status st) {
  return std::move(ReadMNISTData<InputType, OutputType>(st));
}
