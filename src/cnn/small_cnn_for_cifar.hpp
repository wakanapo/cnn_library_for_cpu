#pragma once

#include <fstream>
#include <iostream>

#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "util/tensor.hpp"

template <typename T>
class SmallCNNForCifar {
public:
  using Type = T;
  using InputType = Tensor3D<32, 32, 3, T>;
  using OutputType = Tensor1D<10, T>;
  SmallCNNForCifar() : Conv1(Convolution<3, 3, 3, 32, 0, 1, T>(-0.1, 0.1)),
                       Conv2(Convolution<3, 3, 32, 32, 0, 1, T>(-0.1, 0.1)),
                       Conv3(Convolution<3, 3, 32, 64, 0, 1, T>(-0.1, 0.1)),
                       Affine1(Affine<5*5*64, 10, T>(-0.1, 0.1)) {};
  void train(const InputType& x, const OutputType& t, const T& eps);
  unsigned long predict(const InputType& x) const;
  void save();
  void load();
  Dataset<InputType, OutputType> readData(Status st);
private:
  Convolution<3, 3, 3, 32, 0, 1, T> Conv1;
  Relu<T> Relu1;
  Pooling<2, 2, 0, 2, T> Pool1;
  Convolution<3, 3, 32, 32, 0, 1, T> Conv2;
  Relu<T> Relu2;
  Pooling<2, 2, 0, 1, T> Pool2;
  Convolution<3, 3, 32, 64, 0, 1, T> Conv3;
  Relu<T> Relu3;
  Pooling<2, 2, 0, 2, T> Pool3;
  Affine<5*5*64, 10, T> Affine1;
  Sigmoid<T> Last;
};


template <typename T>
void SmallCNNForCifar<T>::train(const typename SmallCNNForCifar<T>::InputType& x,
                                const typename SmallCNNForCifar<T>::OutputType& t,
                                const T& eps) {
  // forward
  Tensor3D<30, 30, 32, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);
  Relu1.forward(&conv1_ans);
  Tensor3D<15, 15, 32, T> pool1_ans;
  Tensor1D<15*15*32, int> idx1;
  Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<13, 13, 32, T> conv2_ans;
  Conv2.forward(pool1_ans, &conv2_ans);
  Relu2.forward(&conv2_ans);
  Tensor3D<12, 12, 32, T> pool2_ans;
  Tensor1D<12*12*32, int> idx2;
  Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor3D<10, 10, 64, T> conv3_ans;
  Conv3.forward(pool2_ans, &conv3_ans);
  Relu3.forward(&conv3_ans);
  Tensor3D<5, 5, 64, T> pool3_ans;
  Tensor1D<5*5*64, int> idx3;
  Pool3.forward(conv3_ans, &pool3_ans, &idx3);

  Tensor1D<5*5*64, T> dense1 = pool3_ans.flatten();
  Tensor1D<10, T> ans;
  Affine1.forward(dense1, &ans);
  Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta2 = ans - t;
  Tensor1D<5*5*64, T> delta1;
  Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<5, 5, 64, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  
  Tensor3D<10, 10, 64, T> delta_pool3;
  Pool3.backward(delta1_3D, idx3, &delta_pool3);
  Relu3.backward(&delta_pool3, conv3_ans);
  Tensor3D<12, 12, 32, T> delta_conv3;
  Conv3.backward(delta_pool3, pool2_ans, &delta_conv3, eps);

  Tensor3D<13, 13, 32, T> delta_pool2;
  Pool2.backward(delta_conv3, idx2, &delta_pool2);
  Relu2.backward(&delta_pool2, conv2_ans);
  Tensor3D<15, 15, 32, T> delta_conv2;
  Conv2.backward(delta_pool2, pool1_ans, &delta_conv2, eps);

  Tensor3D<30, 30, 32, T> delta_pool1;
  Pool1.backward(delta_conv2, idx1, &delta_pool1);
  Relu1.backward(&delta_pool1, conv1_ans);
  Tensor3D<32, 32, 3, T> delta_conv1;
  Conv1.backward(delta_pool1, x, &delta_conv1, eps);
}

template <typename T>
unsigned long SmallCNNForCifar<T>
::predict(const typename SmallCNNForCifar<T>::InputType& x) const {
  Tensor3D<30, 30, 32, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);
  Relu1.forward(&conv1_ans);
  Tensor3D<15, 15, 32, T> pool1_ans;
  Tensor1D<15*15*32, int> idx1;
  Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<13, 13, 32, T> conv2_ans;
  Conv2.forward(pool1_ans, &conv2_ans);
  Relu2.forward(&conv2_ans);
  Tensor3D<12, 12, 32, T> pool2_ans;
  Tensor1D<12*12*32, int> idx2;
  Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor3D<10, 10, 64, T> conv3_ans;
  Conv3.forward(pool2_ans, &conv3_ans);
  Relu3.forward(&conv3_ans);
  Tensor3D<5, 5, 64, T> pool3_ans;
  Tensor1D<5*5*64, int> idx3;
  Pool3.forward(conv3_ans, &pool3_ans, &idx3);

  Tensor1D<5*5*64, T> dense1 = pool3_ans.flatten();
  Tensor1D<10, T> ans;
  Affine1.forward(dense1, &ans);
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
void SmallCNNForCifar<T>::save() {
  CnnProto::Params params;
  Conv1.saveParams(&params);
  Conv2.saveParams(&params);
  Conv3.saveParams(&params);
  Affine1.saveParams(&params);
  std::fstream output(Options::GetWeightsOutput(), std::ios::out | std::ios::trunc | std::ios::binary);
  if (!params.SerializeToOstream(&output)) {
    std::cerr << "Failed to write params." << std::endl;
  }
}

template <typename T>
void SmallCNNForCifar<T>::load() {
  CnnProto::Params params;
  std::fstream input(Options::GetWeightsInput(), std::ios::in | std::ios::binary);
  if (!params.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv1.loadParams(&params, 0);
  Conv2.loadParams(&params, 1);
  Conv3.loadParams(&params, 2);
  Affine1.loadParams(&params, 3);
}

template<typename T>
Dataset<typename SmallCNNForCifar<T>::InputType, typename SmallCNNForCifar<T>::OutputType>
SmallCNNForCifar<T>::readData(Status st) {
  return ReadCifar10Data<InputType, OutputType>(st);
}
