#pragma once
#include <fstream>
#include <iostream>

#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "util/tensor.hpp"

template<typename T>
class HintonCifar10 {
public:
  using InputType = Tensor3D<32, 32, 3, T>;
  using OutputType = Tensor1D<10, T>;
  HintonCifar10() : Conv1(Convolution<5, 5, 3, 64, 2, 1, T>(-0.1, 0.1)),
                    Conv2(Convolution<5, 5, 64, 64, 2, 1, T>(-0.1, 0.1)),
                    Conv3(Convolution<5, 5, 64, 64, 2, 1, T>(-0.1, 0.1)),
                    Affine1(Affine<4*4*64, 10, T>(-0.1, 0.1)) {};
  void train(const InputType& x, const OutputType& t, const T& eps);
  unsigned long predict(const InputType& x) const;
  void save();
  void load();
  Dataset<InputType, OutputType> readData(Status st);
private:
  Convolution<5, 5, 3, 64, 2, 1, T> Conv1;
  Pooling<3, 3, 0, 2, T> Pool1;
  Convolution<5, 5, 64, 64, 2, 1, T> Conv2;
  Pooling<3, 3, 0, 2, T> Pool2;
  Convolution<5, 5, 64, 64, 2, 1, T> Conv3;
  Pooling<3, 3, 0, 2, T> Pool3;
  Affine<4*4*64, 10, T> Affine1;
  Sigmoid<T> Last;
};

template<typename T>
void HintonCifar10<T>::train(const typename HintonCifar10<T>::InputType& x,
                             const typename HintonCifar10<T>::OutputType& t,
                             const T& eps) {
  //Forward
  Tensor3D<32, 32, 64, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);
  Tensor3D<16, 16, 64, T> pool1_ans;
  Tensor1D<16*16*64, int> idx1;
  Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<16, 16, 64, T> conv2_ans;
  Conv2.forward(pool1_ans, &conv2_ans);
  Tensor3D<8, 8, 64, T> pool2_ans;
  Tensor1D<8*8*64, int> idx2;
  Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor3D<8, 8, 64, T> conv3_ans;
  Conv3.forward(pool2_ans, &conv3_ans);
  Tensor3D<4, 4, 64, T> pool3_ans;
  Tensor1D<4*4*64, int> idx3;
  Pool3.forward(conv3_ans, &pool3_ans, &idx3);

  Tensor1D<4*4*64, T> dense1 = pool3_ans.flatten();
  Tensor1D<10, T> ans;
  Affine1.forward(dense1, &ans);
  Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta2 = ans - t;
  Tensor1D<4*4*64, T> delta1;
  Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<4, 4, 64, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<8, 8, 64, T> delta_pool3;
  Pool3.backward(delta1_3D, idx3, &delta_pool3);
  Tensor3D<8, 8, 64, T> delta_conv3;
  Conv3.backward(delta_pool3, pool2_ans, &delta_conv3, eps);

  Tensor3D<16, 16, 64, T> delta_pool2;
  Pool2.backward(delta_conv3, idx2, &delta_pool2);
  Tensor3D<16, 16, 64, T> delta_conv2;
  Conv2.backward(delta_pool2, pool1_ans, &delta_conv2, eps);

  Tensor3D<32, 32, 64, T> delta_pool1;
  Pool1.backward(delta_conv2, idx1, &delta_pool1);
  Tensor3D<32, 32, 64, T> delta_conv1;
  Conv1.backward(delta_pool1, x, &delta_conv1, eps);
}

template<typename T>
unsigned long HintonCifar10<T>::predict(const typename HintonCifar10<T>::InputType & x) const {
  Tensor3D<32, 32, 64, T> conv1_ans;
  Conv1.forward(x, &conv1_ans);
  Tensor3D<16, 16, 64, T> pool1_ans;
  Tensor1D<16*16*64, int> idx1;
  Pool1.forward(conv1_ans, &pool1_ans, &idx1);

  Tensor3D<16, 16, 64, T> conv2_ans;
  Conv2.forward(pool1_ans, &conv2_ans);
  Tensor3D<8, 8, 64, T> pool2_ans;
  Tensor1D<8*8*64, int> idx2;
  Pool2.forward(conv2_ans, &pool2_ans, &idx2);

  Tensor3D<8, 8, 64, T> conv3_ans;
  Conv3.forward(pool2_ans, &conv3_ans);
  Tensor3D<4, 4, 64, T> pool3_ans;
  Tensor1D<4*4*64, int> idx3;
  Pool3.forward(conv3_ans, &pool3_ans, &idx3);

  Tensor1D<4*4*64, T> dense1 = pool3_ans.flatten();
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

template<typename T>
void HintonCifar10<T>::save() {
  CnnProto::Params params;
  Conv1.saveParams(&params);
  Conv2.saveParams(&params);
  Conv3.saveparams(&params);
  Affine1.saveParams(&params);
  std::fstream output(Options::GetWeightsOutput(),
                      std::ios::out | std::ios::trunc | std::ios::binary);
  if (!params.SerializeToOstream(&output)) {
    std::cerr << "Failes to write params." << std::endl;
  }
}

template<typename T>
void HintonCifar10<T>::load() {
  CnnProto::Params params;
  std::fstream input(Options::GetWeightsInput(),
                     std::ios::in | std::ios::binary);
  if (!params.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv1.loadParams(&params, 0);
  Conv2.loadParams(&params, 1);
  Conv3.loadParams(&params, 2);
  Affine1.loadParams(&params, 3);
}

template<typename T>
Dataset<typename HintonCifar10<T>::InputType, typename HintonCifar10<T>::OutputType>
HintonCifar10<T>::readData(Status st) {
  return ReadCifar10Data<InputType, OutputType>(st);
}
