#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "util/flags.hpp"
#include "util/layers.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"
#include "protos/arithmatic.pb.h"


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
  void save(std::string fname);
  void load(std::string fname);
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
void SimpleConvNet<T>::save(std::string fname) {
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
void SimpleConvNet<T>::load(std::string fname) {
  CnnProto::Params p;
  std::fstream input(Options::GetWeightsInput(), std::ios::in | std::ios::binary);
  if (!p.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv1.loadParams(&p, 0);
  Affine1.loadParams(&p, 1);
  Affine2.loadParams(&p, 2);
}

template<typename T>
Dataset<typename SimpleConvNet<T>::InputType, typename SimpleConvNet<T>::OutputType>
SimpleConvNet<T>::readData(Status st) {
  return std::move(ReadMNISTData<InputType, OutputType>(st));
}

template <typename T>
class SmallCNNForCifar {
public:
  using Type = T;
  using InputType = Tensor3D<32, 32, 3, T>;
  using OutputType = Tensor1D<100, T>;
  SmallCNNForCifar() : Conv1(Convolution<3, 3, 3, 32, 0, 1, T>(-0.1, 0.1)),
                       Conv2(Convolution<3, 3, 32, 32, 0, 1, T>(-0.1, 0.1)),
                       Conv3(Convolution<3, 3, 32, 64, 0, 1, T>(-0.1, 0.1)),
                       Affine1(Affine<5*5*64, 100, T>(-0.1, 0.1)) {};
  void train(const InputType& x, const OutputType& t, const T& eps);
  unsigned long predict(const InputType& x) const;
  void save(std::string fname);
  void load(std::string fname);
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
  Affine<5*5*64, 100, T> Affine1;
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
  Tensor1D<100, T> ans;
  Affine1.forward(dense1, &ans);
  Last.forward(&ans);

  // Backward
  Tensor1D<100, T> delta2 = ans - t;
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
  Tensor1D<100, T> ans;
  Affine1.forward(dense1, &ans);
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

template <typename T>
void SmallCNNForCifar<T>::save(std::string fname) {
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
void SmallCNNForCifar<T>::load(std::string fname) {
  CnnProto::Params p;
  std::fstream input(Options::GetWeightsInput(), std::ios::in | std::ios::binary);
  if (!p.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  Conv1.loadParams(&p, 0);
  Conv2.loadParams(&p, 0);
  Conv3.loadParams(&p, 0);
  Affine1.loadParams(&p, 1);
}

template<typename T>
Dataset<typename SmallCNNForCifar<T>::InputType, typename SmallCNNForCifar<T>::OutputType>
SmallCNNForCifar<T>::readData(Status st) {
  return ReadCifar100Data<InputType, OutputType>(st, FINE);
}

template <typename ModelType>
class CNN {
public:
  using Type = typename ModelType::Type;
  using InputType = typename ModelType::InputType;
  using OutputType = typename ModelType::OutputType;
  static void training();
  static void inference();
};

template <typename ModelType>
void CNN<ModelType>::training() {
  ModelType model;
  Dataset<InputType, OutputType> train = model.readData(TRAIN);
  Dataset<InputType, OutputType> test = model.readData(TEST);

  Type eps = (Type)0.01;
  int epoch = 20;
  int image_num = 1000;

  for (int k = 0; k < epoch; ++k) {
    for (int i = image_num*k; i < image_num*(k+1); ++i) {
      if (Options::IsSaveArithmetic()) {
        std::stringstream sFile;
        sFile << Options::GetArithmaticOutput();
        model.train(train.images[i], train.labels[i], eps);
        using namespace google::protobuf::io;
        std::ofstream output(sFile.str(), std::ofstream::out | std::ofstream::trunc
                             | std::ofstream::binary);
        OstreamOutputStream outputFileStream(&output);
        GzipOutputStream::Options options;
        options.format = GzipOutputStream::GZIP;
        options.compression_level = 9;
        GzipOutputStream gzip_stream(&outputFileStream, options);
        if (!(p.SerializeToZeroCopyStream(&gzip_stream))) {
          std::cerr << "Failed to write values." << std::endl;
        }
        p.Clear();
      }
      else {
        model.train(train.images[i], train.labels[i], eps);
      }
    }
    int cnt = 0;
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 3000; ++i) {
      unsigned long y = model.predict(test.images[i]);
      if (Options::IsSaveArithmetic())
        p.Clear();
      if (OneHot<OutputType>(y) == test.labels[i])
        ++cnt;
    }
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    std::cout << "Inference time = "
              << std::chrono::duration_cast<std::chrono::microseconds>(diff).count()
              << " microsec."
              << std::endl;
    std::cout << "Epoc: " << k << std::endl;
    std::cout << "Accuracy: " << (float)cnt / (float)3000 << std::endl;
  }
  if (Options::IsSaveParams())
    model.save("float_params.pb");
}

template <typename ModelType>
void CNN<ModelType>::inference() {
  ModelType model;
  Dataset<InputType, OutputType> test = model.readData(TEST);

  model.load("float_params.pb");
  int cnt = 0;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 3000; ++i) {
    unsigned long y = model.predict(test.images[i]);
    if (OneHot<OutputType>(y) == test.labels[i])
      ++cnt;
  }
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::cout << "Inference time = "
            << std::chrono::duration_cast<std::chrono::microseconds>(diff).count()
            << " microsec."
            << std::endl;
  std::cout << "Accuracy: " << (float)cnt / (float)3000 << std::endl;
}


