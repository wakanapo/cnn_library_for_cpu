#pragma once

#include <fstream>
#include <iostream>

#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "util/tensor.hpp"

template<typename T>
class VGG16 {
public:
  using Type = T;
  using InputType = Tensor3D<32, 32, 3, T>;
  using OutputType = Tensor1D<100, T>;
  VGG16() :  Conv11(Convolution<3, 3, 3, 64, 1, 1, T>(-0.1, 0.1)),
              Conv12(Convolution<3, 3, 64, 64, 1, 1, T>(-0.1, 0.1)),
              Conv21(Convolution<3, 3, 64, 128, 1, 1, T>(-0.1, 0.1)),
              Conv22(Convolution<3, 3, 128, 128, 1, 1, T>(-0.1, 0.1)),
              Conv31(Convolution<3, 3, 128, 256, 1, 1, T>(-0.1, 0.1)),
              Conv32(Convolution<3, 3, 256, 256, 1, 1, T>(-0.1, 0.1)),
              Conv33(Convolution<3, 3, 256, 256, 1, 1, T>(-0.1, 0.1)),
              Conv34(Convolution<3, 3, 256, 256, 1, 1, T>(-0.1, 0.1)),
              Affine1(Affine<4*4*256, 1024, T>(-0.1, 0.1)),
              Affine2(Affine<1024, 1024, T>(-0.1, 0.1)),
              Affine3(Affine<1024, 100, T>(-0.1, 0.1)) {};
  void train(const InputType& x, const OutputType& t, const T& eps);
  unsigned long predict(const InputType& x) const;
  void save();
  void load();
  Dataset<InputType, OutputType> readData(Status st);
private:
  Convolution<3, 3, 3, 64, 1, 1, T> Conv11;
  Relu<T> Relu11;  
  Convolution<3, 3, 64, 64, 1, 1, T> Conv12;
  Relu<T> Relu12;
  Pooling<2, 2, 0, 2, T> Pool1;
  
  Convolution<3, 3, 64, 128, 1, 1, T> Conv21;
  Relu<T> Relu21;
  Convolution<3, 3, 128, 128, 1, 1, T> Conv22;
  Relu<T> Relu22;
  Pooling<2, 2, 0, 2, T> Pool2;
  
  Convolution<3, 3, 128, 256, 1, 1, T> Conv31;
  Relu<T> Relu31;
  Convolution<3, 3, 256, 256, 1, 1, T> Conv32;
  Relu<T> Relu32;
  Convolution<3, 3, 256, 256, 1, 1, T> Conv33;
  Relu<T> Relu33;
  Convolution<3, 3, 256, 256, 1, 1, T> Conv34;
  Relu<T> Relu34;
  Pooling<2, 2, 0, 2, T> Pool3;
  Affine<4*4*256, 1024, T> Affine1;
  Relu<T> Relu4;
  Affine<1024, 1024, T> Affine2;
  Relu<T> Relu5;
  Affine<1024, 100, T> Affine3;
  Sigmoid<T> Last;
};

template<typename T>
void VGG16<T>::train(const typename VGG16<T>::InputType& x,
                     const typename VGG16<T>::OutputType& t,
                     const T& eps) {
  // Forward
  Tensor3D<32, 32, 64, T> conv11_ans;
  Conv11.forward(x, &conv11_ans);
  Relu11.forward(&conv11_ans);
  Tensor3D<32, 32, 64, T> conv12_ans;
  Conv12.forward(conv11_ans, &conv12_ans);
  Relu12.forward(&conv12_ans);
  Tensor3D<16, 16, 64, T> pool1_ans;
  Tensor1D<16*16*64, int> idx1;
  Pool1.forward(conv12_ans, &pool1_ans, &idx1);

  Tensor3D<16, 16, 128, T> conv21_ans;
  Conv21.forward(pool1_ans, &conv21_ans);
  Relu21.forward(&conv21_ans);
  Tensor3D<16, 16, 128, T> conv22_ans;
  Conv22.forward(conv21_ans, &conv22_ans);
  Relu22.forward(&conv22_ans);
  Tensor3D<8, 8, 128, T> pool2_ans;
  Tensor1D<8*8*128, int> idx2;
  Pool2.forward(conv22_ans, &pool2_ans, &idx2);

  Tensor3D<8, 8, 256, T> conv31_ans;
  Conv31.forward(pool2_ans, &conv31_ans);
  Relu31.forward(&conv31_ans);
  Tensor3D<8, 8, 256, T> conv32_ans;
  Conv32.forward(conv31_ans, &conv32_ans);
  Relu32.forward(&conv32_ans);
  Tensor3D<8, 8, 256, T> conv33_ans;
  Conv33.forward(conv32_ans, &conv33_ans);
  Relu33.forward(&conv33_ans);
  Tensor3D<8, 8, 256, T> conv34_ans;
  Conv34.forward(conv33_ans, &conv34_ans);
  Relu34.forward(&conv34_ans);
  Tensor3D<4, 4, 256, T> pool3_ans;
  Tensor1D<4*4*256, int> idx3;
  Pool3.forward(conv34_ans, &pool3_ans, &idx3);

  Tensor1D<4*4*256, T> dense1 = pool3_ans.flatten();
  Tensor1D<1024, T> dense2;
  Affine1.forward(dense1, &dense2);
  Relu4.forward(&dense2);

  Tensor1D<1024, T> dense3;
  Affine2.forward(dense2, &dense3);
  Relu5.forward(&dense3);

  Tensor1D<100, T> ans;
  Affine3.forward(dense3, &ans);
  Last.forward(&ans);

  // Backward
  Tensor1D<100, T> delta4 = ans - t;
  Tensor1D<1024, T> delta3;
  Affine3.backward(delta4, dense3, &delta3, eps);
  Relu5.backward(&delta3, dense3);

  Tensor1D<1024, T> delta2;
  Affine2.backward(delta3, dense2, &delta2, eps);
  Relu4.backward(&delta2, dense2);
  
  Tensor1D<4*4*256, T> delta1;
  Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<4, 4, 256, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<8, 8, 256, T> delta_pool3;
  Pool3.backward(delta1_3D, idx3, &delta_pool3);
  Relu34.backward(&delta_pool3, conv34_ans);
  Tensor3D<8, 8, 256, T> delta_conv34;
  Conv34.backward(delta_pool3, conv33_ans, &delta_conv34, eps);
  Relu33.backward(&delta_conv34, conv33_ans);
  Tensor3D<8, 8, 256, T> delta_conv33;
  Conv33.backward(delta_conv34, conv32_ans, &delta_conv33, eps);
  Relu32.backward(&delta_conv33, conv32_ans);
  Tensor3D<8, 8, 256, T> delta_conv32;
  Conv32.backward(delta_conv33, conv31_ans, &delta_conv32, eps);
  Relu31.backward(&delta_conv32, conv31_ans);
  Tensor3D<8, 8, 128, T> delta_conv31;
  Conv31.backward(delta_conv32, pool2_ans, &delta_conv31, eps);

  Tensor3D<16, 16, 128, T> delta_pool2;
  Pool2.backward(delta_conv31, idx2, &delta_pool2);
  Relu22.backward(&delta_pool2, conv22_ans);
  Tensor3D<16, 16, 128, T> delta_conv22;
  Conv22.backward(delta_pool2, conv21_ans, &delta_conv22, eps);
  Relu21.backward(&delta_conv22, conv21_ans);
  Tensor3D<16, 16, 64, T> delta_conv21;
  Conv21.backward(delta_conv22, pool1_ans, &delta_conv21, eps);

  Tensor3D<32, 32, 64, T> delta_pool1;
  Pool1.backward(delta_conv21, idx1, &delta_pool1);
  Relu12.backward(&delta_pool1, conv12_ans);
  Tensor3D<32, 32, 64, T> delta_conv12;
  Conv12.backward(delta_pool1, conv12_ans, &delta_conv12, eps);
  Relu11.backward(&delta_conv12, conv11_ans);
  Tensor3D<32, 32, 3, T> delta_conv11;
  Conv11.backward(delta_conv12, x, &delta_conv11, eps);
}

template<typename T>
unsigned long VGG16<T>::predict(const typename VGG16<T>::InputType &x) const {
  Tensor3D<32, 32, 64, T> conv11_ans;
  Conv11.forward(x, &conv11_ans);
  Relu11.forward(&conv11_ans);
  Tensor3D<32, 32, 64, T> conv12_ans;
  Conv12.forward(conv11_ans, &conv12_ans);
  Relu12.forward(&conv12_ans);
  Tensor3D<16, 16, 64, T> pool1_ans;
  Tensor1D<16*16*64, int> idx1;
  Pool1.forward(conv12_ans, &pool1_ans, &idx1);

  Tensor3D<16, 16, 128, T> conv21_ans;
  Conv21.forward(pool1_ans, &conv21_ans);
  Relu21.forward(&conv21_ans);
  Tensor3D<16, 16, 128, T> conv22_ans;
  Conv22.forward(conv21_ans, &conv22_ans);
  Relu22.forward(&conv22_ans);
  Tensor3D<8, 8, 128, T> pool2_ans;
  Tensor1D<8*8*128, int> idx2;
  Pool2.forward(conv22_ans, &pool2_ans, &idx2);

  Tensor3D<8, 8, 256, T> conv31_ans;
  Conv31.forward(pool2_ans, &conv31_ans);
  Relu31.forward(&conv31_ans);
  Tensor3D<8, 8, 256, T> conv32_ans;
  Conv32.forward(conv31_ans, &conv32_ans);
  Relu32.forward(&conv32_ans);
  Tensor3D<8, 8, 256, T> conv33_ans;
  Conv33.forward(conv32_ans, &conv33_ans);
  Relu33.forward(&conv33_ans);
  Tensor3D<8, 8, 256, T> conv34_ans;
  Conv34.forward(conv33_ans, &conv34_ans);
  Relu34.forward(&conv34_ans);
  Tensor3D<4, 4, 256, T> pool3_ans;
  Tensor1D<4*4*256, int> idx3;
  Pool3.forward(conv34_ans, &pool3_ans, &idx3);

  Tensor1D<4*4*256, T> dense1 = pool3_ans.flatten();
  Tensor1D<1024, T> dense2;
  Affine1.forward(dense1, &dense2);
  Relu4.forward(&dense2);

  Tensor1D<1024, T> dense3;
  Affine2.forward(dense2, &dense3);
  Relu5.forward(&dense3);

  Tensor1D<100, T> ans;
  Affine3.forward(dense3, &ans);
  Last.forward(&ans);

  T max = (T)0;
  unsigned long argmax = 0;
  for (int i = 0; i < 100; ++i) {
    if (ans[i] > max) {
      max = ans[i];
      argmax = i;
    }
  }
  return argmax;
}

template<typename T>
void VGG16<T>::save() {
  CnnProto::Params params;
  Conv11.saveParams(&params);
  Conv12.saveParams(&params);
  Conv21.saveParams(&params);
  Conv22.saveParams(&params);
  Conv31.saveParams(&params);
  Conv32.saveParams(&params);
  Conv33.saveParams(&params);
  Conv34.saveParams(&params);
  Affine1.saveParams(&params);
  Affine2.saveParams(&params);
  Affine3.saveParams(&params);
  std::fstream output(Options::GetWeightsOutput(),
                      std::ios::out | std::ios::trunc | std::ios::binary);
  if (!params.SerializeToOstream(&output)) {
    std::cerr << "Failes to write params." << std::endl;
  }
}

template<typename T>
void VGG16<T>::load() {
  CnnProto::Params params;
  std::fstream input(Options::GetWeightsInput(),
                     std::ios::in | std::ios::binary);
  if (!params.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv11.loadParams(&params, 0);
  Conv12.loadParams(&params, 1);
  Conv21.loadParams(&params, 2);
  Conv22.loadParams(&params, 3);
  Conv31.loadParams(&params, 4);
  Conv32.loadParams(&params, 5);
  Conv33.loadParams(&params, 6);
  Conv34.loadParams(&params, 7);
  Affine1.loadParams(&params, 8);
  Affine2.loadParams(&params, 9);
  Affine3.loadParams(&params, 10);
}

template<typename T>
Dataset<typename VGG16<T>::InputType, typename VGG16<T>::OutputType>
VGG16<T>::readData(Status st) {
  return ReadCifar100Data<InputType, OutputType>(st, FINE);
}
