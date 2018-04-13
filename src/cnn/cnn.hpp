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
#include "cnn/layers.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"
#include "protos/arithmatic.pb.h"

template <typename T>
class SimpleConvNet {
public:
  SimpleConvNet() : Conv1(Convolution<5, 5, 1, 30, 0, 1, T>(-0.1, 0.1)),
                    Affine1(Affine<12*12*30, 100, T>(-0.1, 0.1)),
                    Affine2(Affine<100, 10, T>(-0.1, 0.1)) {};
  Convolution<5, 5, 1, 30, 0, 1, T> Conv1;
  Relu<T> Relu1;
  Pooling<2, 2, 0, 2, T> Pool1;
  Affine<12*12*30, 100, T> Affine1;
  Relu<T> Relu2;
  Affine<100, 10, T> Affine2;
  Sigmoid<T> Last;
};

template <typename T>
class CNN {
public:
  void simple_train(const Tensor2D<28, 28, T>& x, const Tensor1D<10, T>& t,
                    const T& eps);
  unsigned long simple_predict(const Tensor2D<28, 28, T>& x) const;
  void simple_save(std::string fname);
  void simple_load(std::string fname);
  static void run();
  static void inference();
private:
  SimpleConvNet<T> simple;
};

template <typename T>
void CNN<T>::simple_train(const Tensor2D<28, 28, T>& x, const Tensor1D<10, T>& t,
                          const T& eps) {
  // forward
  Tensor3D<24, 24, 30, T> conv1_ans;
  simple.Conv1.forward(x, &conv1_ans);

  simple.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  simple.Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  simple.Affine1.forward(dense1, &dense2);

  simple.Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  simple.Affine2.forward(dense2, &ans);

  simple.Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta3 = ans - t;
  Tensor1D<100, T> delta2;
  simple.Affine2.backward(delta3, dense2, &delta2, eps);
  simple.Relu2.backward(&delta2, dense2);

  Tensor1D<12*12*30, T> delta1;
  simple.Affine1.backward(delta2, dense1, &delta1, eps);

  Tensor3D<12, 12, 30, T> delta1_3D;
  delta1_3D.set_v(delta1.get_v());
  Tensor3D<24, 24, 30, T> delta_pool;
  simple.Pool1.backward(delta1_3D, idx, &delta_pool);

  simple.Relu1.backward(&delta_pool, conv1_ans);

  Tensor2D<28, 28, T> delta_conv;
  simple.Conv1.backward(delta_pool, x, &delta_conv, eps);
}

template <typename T>
unsigned long CNN<T>::simple_predict(const Tensor2D<28, 28, T>& x) const {
  Tensor3D<24, 24, 30, T> conv1_ans;
  simple.Conv1.forward(x, &conv1_ans);

  simple.Relu1.forward(&conv1_ans);

  Tensor3D<12, 12, 30, T> pool1_ans;
  Tensor1D<12*12*30, int> idx;
  simple.Pool1.forward(conv1_ans, &pool1_ans, &idx);

  Tensor1D<12*12*30, T> dense1 = pool1_ans.flatten();
  Tensor1D<100, T> dense2;
  simple.Affine1.forward(dense1, &dense2);

  simple.Relu2.forward(&dense2);

  Tensor1D<10, T> ans;
  simple.Affine2.forward(dense2, &ans);

  simple.Last.forward(&ans);

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
void CNN<T>::simple_save(std::string fname) {
  CnnProto::Params params;
  simple.Conv1.saveParams(&params);
  simple.Affine1.saveParams(&params);
  simple.Affine2.saveParams(&params);
  std::fstream output(Options::GetWeightsOutput(), std::ios::out | std::ios::trunc | std::ios::binary);
  if (!params.SerializeToOstream(&output)) {
    std::cerr << "Failed to write params." << std::endl;
  }
}

template <typename T>
void CNN<T>::simple_load(std::string fname) {
  CnnProto::Params p;
  std::fstream input(Options::GetWeightsInput(), std::ios::in | std::ios::binary);
  if (!p.ParseFromIstream(&input)) {
    std::cerr << "Failed to load params." << std::endl;
  }
  simple.Conv1.loadParams(&p, 0);
  simple.Affine1.loadParams(&p, 1);
  simple.Affine2.loadParams(&p, 2);
}

template <typename T>
void CNN<T>::run() {
  const Data train_X = ReadMnistImages<T>(TRAIN);
  const Data train_y = ReadMnistLabels(TRAIN);

  const Data test_X = ReadMnistImages<T>(TEST);
  const Data test_y = ReadMnistLabels(TEST);

  Tensor2D<28, 28, T> x;
  Tensor1D<10, T> t;
  CNN<T> cnn;
 
  T eps = (T)0.01;
  int epoch = 20;
  int image_num = 1000;

  for (int k = 0; k < epoch; ++k) {
    for (int i = image_num*k; i < image_num*(k+1); ++i) {
      x.set_v((T*)train_X.ptr_ + i * x.size(1) * x.size(0));
      t.set_v(mnistOneHot<T>(((unsigned long*) train_y.ptr_)[i]));
      if (Options::IsSaveArithmetic()) {
        std::stringstream sFile;
        sFile << Options::GetArithmaticOutput();
        cnn.simple_train(x, t, eps);
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
        cnn.simple_train(x, t, eps);
      }
    }
    int cnt = 0;
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 3000; ++i) {
      x.set_v((T*)test_X.ptr_ + i * x.size(1) * x.size(0));
      unsigned long y = cnn.simple_predict(x);
      if (Options::IsSaveArithmetic())
        p.Clear();
      if (y == ((unsigned long*)test_y.ptr_)[i])
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
    cnn.simple_save("float_params.pb");
  free(train_X.ptr_);
  free(train_y.ptr_);

  free(test_X.ptr_);
  free(test_y.ptr_);
}

template <typename T>
void CNN<T>::inference() {
  const Data test_X = ReadMnistImages<T>(TEST);
  const Data test_y = ReadMnistLabels(TEST);

  Tensor2D<28, 28, T> x;
  CNN<T> cnn;
  cnn.simple_load("float_params.pb");
  int cnt = 0;
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 3000; ++i) {
    x.set_v((T*)test_X.ptr_ + i * x.size(1) * x.size(0));
    unsigned long y = cnn.simple_predict(x);
    if (y == ((unsigned long*)test_y.ptr_)[i])
      ++cnt;
  }
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::cout << "Inference time = "
            << std::chrono::duration_cast<std::chrono::microseconds>(diff).count()
            << " microsec."
            << std::endl;
  std::cout << "Accuracy: " << (float)cnt / (float)3000 << std::endl;
  
  free(test_X.ptr_);
  free(test_y.ptr_);
}


