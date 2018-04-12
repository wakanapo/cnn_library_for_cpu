#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "util/tensor.hpp"
#include "util/function.hpp"
#include "cnn/layers.hpp"
#include "util/read_data.hpp"
#include "protos/cnn_params.pb.h"

template <typename T>
class ThreeLayers {
public:
  ThreeLayers() : Affine1(Affine<28*28, 100, T>((T)-1.0, (T)1.0)),
                  Affine2(Affine<100, 10, T>((T)-1.0, (T)1.0)) {};
  Affine<28*28, 100, T> Affine1;
  Sigmoid<T> Sigmoid1;
  Affine<100, 10, T> Affine2;
  Softmax<T> Last;
};

template <typename T>
class MLP {
public:
  void three_train(Tensor1D<28*28, T>&x, Tensor1D<10, T>& t, const T& eps);
  unsigned long three_predict(Tensor1D<28*28, T>& x);
  void three_save(std::string fname);
  static void run();
private:
  ThreeLayers<T> three;
};

template <typename T>
void MLP<T>::three_train(Tensor1D<28*28, T>& x, Tensor1D<10, T>& t, const T& eps) {
  // forward
  Tensor1D<100, T> dense2;
  three.Affine1.forward(x, &dense2);

  three.Sigmoid1.forward(&dense2);

  Tensor1D<10, T> ans;
  three.Affine2.forward(dense2, &ans);

  three.Last.forward(&ans);

  // Backward
  Tensor1D<10, T> delta3 = ans - t;
  Tensor1D<100, T> delta2;
  three.Affine2.backward(delta3, dense2, &delta2, eps);
  three.Sigmoid1.backward(&delta2, dense2);

  Tensor1D<28*28, T> delta1;
  three.Affine1.backward(delta2, x, &delta1, eps);
}

template <typename T>
unsigned long MLP<T>::three_predict(Tensor1D<28*28, T>& x) {
  Tensor1D<100, T> dense2;
  three.Affine1.forward(x, &dense2);

  three.Sigmoid1.forward(&dense2);

  Tensor1D<10, T> ans;
  three.Affine2.forward(dense2, &ans);

  three.Last.forward(&ans);
  
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
void MLP<T>::three_save(std::string fname) {
  std::string home = getenv("HOME");
  CnnProto::Params p;
  three.Affine1.saveParams(&p);
  three.Affine2.saveParams(&p);
  std::fstream output(home+"/utokyo-kudohlab/cnn_cpp/data/"+fname, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!p.SerializeToOstream(&output)) {
    std::cerr << "Failed to write params." << std::endl;
  }
}

template <typename T>
void MLP<T>::run() {
  const data train_X = readMnistImages(TRAIN);
  const data train_y = readMnistLabels(TRAIN);

  const data test_X = readMnistImages(TEST);
  const data test_y = readMnistLabels(TEST);

  Tensor1D<28*28, T> x;
  Tensor1D<10, T> t;
  MLP<T> mlp;
 
  T eps = (T)0.01;
  int epoch = 15;
  for (int k = 0; k < epoch; ++k) {
    for (int i = 0; i < train_X.col; ++i) {
      x.set_v((float*)train_X.ptr + i * x.size());
      t.set_v(mnistOneHot(((unsigned long*) train_y.ptr)[i]));
      mlp.three_train(x, t, eps);
    }
    int cnt = 0;
    for (int i = 0; i < test_X.col; ++i) {
      x.set_v((float*)test_X.ptr + i * x.size());
      unsigned long y = mlp.three_predict(x);
      if (y == ((unsigned long*)test_y.ptr)[i])
        ++cnt;
    }
    std::cout << "Epoc: " << k << std::endl;
    std::cout << "Accuracy: " << (float)cnt / (float)test_X.col << std::endl;
  }

  mlp.three_save("three_uniform11.pb");
  free(train_X.ptr);
  free(train_y.ptr);

  free(test_X.ptr);
  free(test_y.ptr);
}
